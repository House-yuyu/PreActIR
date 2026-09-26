from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += int(count)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            return value
        return [move_to_device(item, device) for item in value]
    return value


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type in {"cuda", "cpu"},
    )


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create a GradScaler across recent PyTorch AMP APIs."""
    effective = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=effective)
    except (AttributeError, TypeError):  # Compatibility with older supported PyTorch releases.
        return torch.cuda.amp.GradScaler(enabled=effective)
