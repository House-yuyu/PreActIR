from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from preactir.models.common import ImageEncoder


class ToolConditionedWorldModel(nn.Module):
    """Distributional predictor for restoration-tool outcomes."""

    def __init__(
        self,
        num_degradations: int,
        num_tools: int,
        num_quality_metrics: int = 4,
        base_channels: int = 32,
        feature_dim: int = 256,
        action_embed_dim: int = 48,
        hidden_dim: int = 384,
        dropout: float = 0.15,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        predict_harm: bool = False,
        predict_severe_harm: bool = False,
        predict_paper_quality: bool = False,
        trajectory_context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_degradations = num_degradations
        self.num_tools = num_tools
        self.num_quality_metrics = num_quality_metrics
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.predict_harm = bool(predict_harm)
        self.predict_severe_harm = bool(predict_severe_harm)
        self.predict_paper_quality = bool(predict_paper_quality)
        self.trajectory_context_dim = int(trajectory_context_dim)
        if self.trajectory_context_dim < 0:
            raise ValueError("trajectory_context_dim must be non-negative")

        self.encoder = ImageEncoder(base_channels=base_channels, feature_dim=feature_dim)
        deepest_channels = self.encoder.out_channels[-1]
        self.region_projection = nn.Linear(deepest_channels, feature_dim)
        self.tool_embedding = nn.Embedding(num_tools, action_embed_dim)
        self.target_embedding = nn.Embedding(num_degradations, action_embed_dim // 2)

        input_dim = (
            feature_dim * 2
            + 2 * num_degradations
            + action_embed_dim
            + action_embed_dim // 2
            + 3
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.delta_degradation_head = nn.Linear(hidden_dim, 2 * num_degradations)
        self.delta_quality_head = nn.Linear(hidden_dim, 2 * num_quality_metrics)
        self.damage_head = nn.Linear(hidden_dim, 2)
        self.accept_head = nn.Linear(hidden_dim, 1)
        # Kept optional so legacy v1-v4 checkpoints retain an identical state
        # dictionary.  The v5 paper configuration enables this pre-action head.
        self.harm_head = nn.Linear(hidden_dim, 1) if self.predict_harm else None
        # Severe harm is deliberately separate from the broad any-metric harm
        # label. It lets deployment reject catastrophic state/action pairs
        # without deleting an entire restoration-tool family.
        self.severe_harm_head = (
            nn.Linear(hidden_dim, 1) if self.predict_severe_harm else None
        )
        # The legacy quality vector is RGB/proxy based. The paper-quality head
        # directly predicts [YCbCr-Y PSNR gain / 10, YCbCr-Y SSIM gain], so
        # planning and within-state ranking optimize the binding evaluator.
        self.paper_quality_head = (
            nn.Linear(hidden_dim, 4) if self.predict_paper_quality else None
        )
        self.trajectory_projection = (
            nn.Sequential(
                nn.Linear(self.trajectory_context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.trajectory_context_dim
            else None
        )

    def _masked_pool(self, feature: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(action_mask, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        denominator = mask.sum(dim=(2, 3)).clamp_min(1e-4)
        pooled = (feature * mask).sum(dim=(2, 3)) / denominator
        return self.region_projection(pooled)

    def forward(
        self,
        image: torch.Tensor,
        belief: torch.Tensor,
        action_mask: torch.Tensor,
        tool_id: torch.Tensor,
        target_index: torch.Tensor,
        strength: torch.Tensor,
        mask_area: torch.Tensor,
        cost_prior: torch.Tensor,
        trajectory_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features, global_feature = self.encoder(image)
        region_feature = self._masked_pool(features[-1], action_mask)
        tool_feature = self.tool_embedding(tool_id)
        target_feature = self.target_embedding(target_index)
        scalar_features = torch.stack([strength, mask_area, cost_prior], dim=-1)
        if self.trajectory_context_dim:
            if trajectory_context is None:
                trajectory_context = torch.zeros(
                    image.shape[0],
                    self.trajectory_context_dim,
                    dtype=image.dtype,
                    device=image.device,
                )
            if trajectory_context.shape != (image.shape[0], self.trajectory_context_dim):
                raise ValueError(
                    "trajectory_context must have shape "
                    f"({image.shape[0]}, {self.trajectory_context_dim}), got "
                    f"{tuple(trajectory_context.shape)}"
                )
        else:
            trajectory_context = image.new_zeros((image.shape[0], 0))
        context = torch.cat(
            [
                global_feature,
                region_feature,
                belief,
                tool_feature,
                target_feature,
                scalar_features,
            ],
            dim=-1,
        )
        hidden = self.trunk(context)
        if self.trajectory_projection is not None:
            hidden = hidden + self.trajectory_projection(trajectory_context)

        degradation_params = self.delta_degradation_head(hidden)
        degradation_mu, degradation_logvar = degradation_params.chunk(2, dim=-1)
        degradation_logvar = degradation_logvar.clamp(self.logvar_min, self.logvar_max)

        quality_params = self.delta_quality_head(hidden)
        quality_mu, quality_logvar = quality_params.chunk(2, dim=-1)
        quality_logvar = quality_logvar.clamp(self.logvar_min, self.logvar_max)

        damage_params = self.damage_head(hidden)
        damage_mu, damage_logvar = damage_params.chunk(2, dim=-1)
        damage_mu = torch.sigmoid(damage_mu.squeeze(-1))
        damage_logvar = damage_logvar.squeeze(-1).clamp(self.logvar_min, self.logvar_max)
        result = {
            "delta_degradation_mu": degradation_mu,
            "delta_degradation_logvar": degradation_logvar,
            "delta_quality_mu": quality_mu,
            "delta_quality_logvar": quality_logvar,
            "damage_mu": damage_mu,
            "damage_logvar": damage_logvar,
            "accept_logit": self.accept_head(hidden).squeeze(-1),
        }
        if self.harm_head is not None:
            result["harm_logit"] = self.harm_head(hidden).squeeze(-1)
        if self.severe_harm_head is not None:
            result["severe_harm_logit"] = self.severe_harm_head(hidden).squeeze(-1)
        if self.paper_quality_head is not None:
            paper_params = self.paper_quality_head(hidden)
            paper_mu, paper_logvar = paper_params.chunk(2, dim=-1)
            result["paper_quality_mu"] = paper_mu
            result["paper_quality_logvar"] = paper_logvar.clamp(
                self.logvar_min, self.logvar_max
            )
        return result
