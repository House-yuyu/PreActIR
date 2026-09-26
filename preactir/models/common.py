from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, 1),
            ConvNormAct(channels, channels, 3, 1, activation=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ImageEncoder(nn.Module):
    def __init__(self, base_channels: int = 32, feature_dim: int = 256) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(ConvNormAct(3, channels[0], 5, 2), ResidualBlock(channels[0]))
        self.stage2 = nn.Sequential(ConvNormAct(channels[0], channels[1], 3, 2), ResidualBlock(channels[1]))
        self.stage3 = nn.Sequential(ConvNormAct(channels[1], channels[2], 3, 2), ResidualBlock(channels[2]))
        self.stage4 = nn.Sequential(ConvNormAct(channels[2], channels[3], 3, 2), ResidualBlock(channels[3]))
        self.global_projection = nn.Linear(channels[3], feature_dim)
        self.out_channels = channels
        self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        f1 = self.stem(image)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        pooled = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        global_feature = self.global_projection(pooled)
        return [f1, f2, f3, f4], global_feature
