from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from preactir.models.common import ImageEncoder


class LearnedTransitionVerifier(nn.Module):
    """Independent post-action verifier.

    The verifier observes the actual before/after pair rather than the planner's
    predicted outcome. It estimates commit probability, transition status, target
    gain, cross-degradation side effect, and outside-region change.
    """

    def __init__(
        self,
        num_degradations: int,
        num_tools: int,
        base_channels: int = 24,
        feature_dim: int = 192,
        action_embed_dim: int = 48,
        hidden_dim: int = 384,
        dropout: float = 0.10,
        predict_paper_quality: bool = False,
        trajectory_context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.predict_paper_quality = bool(predict_paper_quality)
        self.trajectory_context_dim = int(trajectory_context_dim)
        if self.trajectory_context_dim < 0:
            raise ValueError("trajectory_context_dim must be non-negative")
        self.encoder = ImageEncoder(base_channels=base_channels, feature_dim=feature_dim)
        deepest_channels = self.encoder.out_channels[-1]
        self.region_projection = nn.Linear(deepest_channels, feature_dim)
        self.tool_embedding = nn.Embedding(num_tools, action_embed_dim)
        self.target_embedding = nn.Embedding(num_degradations, action_embed_dim // 2)

        input_dim = feature_dim * 6 + action_embed_dim + action_embed_dim // 2 + 3
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
        self.accept_head = nn.Linear(hidden_dim, 1)
        self.status_head = nn.Linear(hidden_dim, 3)  # Rejected / Progress / Resolved
        self.target_gain_head = nn.Linear(hidden_dim, 1)
        self.side_effect_head = nn.Linear(hidden_dim, 1)
        self.outside_change_head = nn.Linear(hidden_dim, 1)
        self.paper_quality_head = (
            nn.Linear(hidden_dim, 4) if self.predict_paper_quality else None
        )
        self.paper_accept_head = (
            nn.Linear(hidden_dim, 1) if self.predict_paper_quality else None
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
        if self.trajectory_projection is not None:
            # Loading a legacy verifier should initially reproduce its trunk
            # representation exactly; the residual context branch then learns
            # away from zero during V8 fine-tuning.
            nn.init.zeros_(self.trajectory_projection[-1].weight)
            nn.init.zeros_(self.trajectory_projection[-1].bias)

    def _masked_pool(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        denominator = mask.sum(dim=(2, 3)).clamp_min(1e-4)
        pooled = (feature * mask).sum(dim=(2, 3)) / denominator
        return self.region_projection(pooled)

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        action_mask: torch.Tensor,
        tool_id: torch.Tensor,
        target_index: torch.Tensor,
        strength: torch.Tensor,
        mask_area: torch.Tensor,
        cost_prior: torch.Tensor,
        trajectory_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        before_features, before_global = self.encoder(before)
        after_features, after_global = self.encoder(after)
        deepest_delta = after_features[-1] - before_features[-1]
        region_delta = self._masked_pool(deepest_delta, action_mask)
        outside_delta = self._masked_pool(deepest_delta, 1.0 - action_mask)
        scalar_features = torch.stack([strength, mask_area, cost_prior], dim=-1)
        context = torch.cat(
            [
                before_global,
                after_global,
                after_global - before_global,
                torch.abs(after_global - before_global),
                region_delta,
                outside_delta,
                self.tool_embedding(tool_id),
                self.target_embedding(target_index),
                scalar_features,
            ],
            dim=-1,
        )
        hidden = self.trunk(context)
        if self.trajectory_projection is not None:
            if trajectory_context is None:
                trajectory_context = before.new_zeros(
                    (before.shape[0], self.trajectory_context_dim)
                )
            if trajectory_context.shape != (
                before.shape[0],
                self.trajectory_context_dim,
            ):
                raise ValueError(
                    "trajectory_context must have shape "
                    f"({before.shape[0]}, {self.trajectory_context_dim}), got "
                    f"{tuple(trajectory_context.shape)}"
                )
            hidden = hidden + self.trajectory_projection(trajectory_context)
        result = {
            "accept_logit": self.accept_head(hidden).squeeze(-1),
            "status_logits": self.status_head(hidden),
            "target_gain": torch.tanh(self.target_gain_head(hidden).squeeze(-1)),
            "side_effect": torch.sigmoid(self.side_effect_head(hidden).squeeze(-1)),
            "outside_change": torch.sigmoid(self.outside_change_head(hidden).squeeze(-1)),
        }
        if self.paper_quality_head is not None and self.paper_accept_head is not None:
            paper_params = self.paper_quality_head(hidden)
            paper_mu, paper_logvar = paper_params.chunk(2, dim=-1)
            result["paper_quality_mu"] = paper_mu
            result["paper_quality_logvar"] = paper_logvar.clamp(-6.0, 2.0)
            result["paper_accept_logit"] = self.paper_accept_head(hidden).squeeze(-1)
        return result
