from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from preactir.models.common import ConvNormAct, ImageEncoder


class SpatialBeliefEncoder(nn.Module):
    """Predict degradation presence, severity distribution, and spatial masks."""

    def __init__(
        self,
        num_degradations: int,
        base_channels: int = 32,
        dropout: float = 0.1,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        global_backbone: str = "none",
        clip_checkpoint: str | None = None,
        clip_fusion_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.num_degradations = num_degradations
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.global_backbone = str(global_backbone)
        self.clip_fusion_mode = str(clip_fusion_mode)
        feature_dim = base_channels * 8
        self.encoder = ImageEncoder(base_channels=base_channels, feature_dim=feature_dim)
        global_input_dim = feature_dim
        self.clip_visual: nn.Module | None = None
        if self.global_backbone == "clip_vit_b32":
            if clip_checkpoint is None:
                raise ValueError("clip_checkpoint is required for global_backbone=clip_vit_b32")
            checkpoint_path = Path(clip_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            import clip

            clip_model, _ = clip.load(str(checkpoint_path), device="cpu", jit=False)
            self.clip_visual = clip_model.visual
            self.clip_visual.requires_grad_(False)
            self.clip_visual.eval()
            if self.clip_fusion_mode == "concat":
                global_input_dim += int(self.clip_visual.output_dim)
            elif self.clip_fusion_mode in {"replace", "linear"}:
                global_input_dim = int(self.clip_visual.output_dim)
            else:
                raise ValueError(
                    "clip_fusion_mode must be 'concat', 'replace', or 'linear', got "
                    f"{self.clip_fusion_mode!r}"
                )
            self.register_buffer(
                "clip_mean",
                torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "clip_std",
                torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
                persistent=False,
            )
        elif self.global_backbone != "none":
            raise ValueError(f"Unsupported belief global_backbone={self.global_backbone!r}")
        if self.clip_fusion_mode == "linear" and self.clip_visual is not None:
            self.global_head = nn.Identity()
            head_feature_dim = global_input_dim
        else:
            self.global_head = nn.Sequential(
                nn.LayerNorm(global_input_dim),
                nn.Dropout(dropout),
                nn.Linear(global_input_dim, feature_dim),
                nn.SiLU(),
            )
            head_feature_dim = feature_dim
        self.presence_head = nn.Linear(head_feature_dim, num_degradations)
        self.severity_mu_head = nn.Linear(head_feature_dim, num_degradations)
        self.severity_logvar_head = nn.Linear(head_feature_dim, num_degradations)

        fpn_channels = base_channels * 2
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, fpn_channels, 1) for channels in self.encoder.out_channels]
        )
        self.smooth = nn.ModuleList([ConvNormAct(fpn_channels, fpn_channels) for _ in range(3)])
        self.mask_head = nn.Sequential(
            ConvNormAct(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, num_degradations, 1),
        )

    def extract_clip_feature(self, image: torch.Tensor) -> torch.Tensor:
        if self.clip_visual is None:
            raise RuntimeError("CLIP features require global_backbone=clip_vit_b32")
        clip_image = F.interpolate(
            image,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        clip_image = (clip_image - self.clip_mean) / self.clip_std
        with torch.no_grad():
            clip_feature = self.clip_visual(clip_image).float()
        return F.normalize(clip_feature, dim=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        features, global_feature = self.encoder(image)
        clip_feature = None
        if self.clip_visual is not None:
            clip_feature = self.extract_clip_feature(image)
            global_feature = (
                torch.cat([global_feature, clip_feature], dim=1)
                if self.clip_fusion_mode == "concat"
                else clip_feature
            )
        hidden = self.global_head(global_feature)
        presence_logits = self.presence_head(hidden)
        severity_mu = torch.sigmoid(self.severity_mu_head(hidden))
        severity_logvar = self.severity_logvar_head(hidden).clamp(self.logvar_min, self.logvar_max)

        pyramid = [layer(feature) for layer, feature in zip(self.lateral, features)]
        x = pyramid[-1]
        for level in range(2, -1, -1):
            x = F.interpolate(x, size=pyramid[level].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[level](x + pyramid[level])
        mask_logits = self.mask_head(x)
        mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)
        output = {
            "presence_logits": presence_logits,
            "severity_mu": severity_mu,
            "severity_logvar": severity_logvar,
            "mask_logits": mask_logits,
        }
        if clip_feature is not None:
            output["clip_feature"] = clip_feature
        return output

    def train(self, mode: bool = True) -> SpatialBeliefEncoder:
        super().train(mode)
        if self.clip_visual is not None:
            self.clip_visual.eval()
        return self

    @property
    def external_state_prefixes(self) -> tuple[str, ...]:
        """Frozen weights reconstructed from an explicit external checkpoint."""

        return ("clip_visual.",) if self.clip_visual is not None else ()

    @torch.inference_mode()
    def predict_belief(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.forward(image)
        output["presence_prob"] = torch.sigmoid(output["presence_logits"])
        output["severity_std"] = torch.exp(0.5 * output["severity_logvar"])
        output["mask_prob"] = torch.sigmoid(output["mask_logits"])
        return output
