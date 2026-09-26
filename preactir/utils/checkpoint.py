from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    external_prefixes = tuple(getattr(model, "external_state_prefixes", ()))
    if external_prefixes:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(external_prefixes)
        }
    payload: dict[str, Any] = {
        "model": state,
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    external_prefixes = tuple(getattr(model, "external_state_prefixes", ()))
    if external_prefixes:
        incompatible = model.load_state_dict(state, strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(external_prefixes)
        ]
        if strict and (invalid_missing or incompatible.unexpected_keys):
            raise RuntimeError(
                "Checkpoint state mismatch: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
    else:
        model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
