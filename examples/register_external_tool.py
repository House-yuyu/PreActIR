"""Minimal example for replacing a built-in tool with a pretrained PyTorch model."""
from __future__ import annotations

import torch

from preactir.tools.neural_adapter import TorchModuleTool
from preactir.tools.registry import build_default_registry


class YourRestorationNetwork(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Replace this body with the actual checkpointed restoration network.
        return image


registry = build_default_registry()
model = YourRestorationNetwork()
registry.register(
    TorchModuleTool(
        name="denoise",
        targets=("noise",),
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cost_prior=1.0,
    ),
    overwrite=True,
)
print("Registered tools:", registry.names())
