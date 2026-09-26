from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ToolResult:
    image: np.ndarray
    elapsed_ms: float
    metadata: dict[str, float | str]


class RestorationTool(ABC):
    name: str
    targets: tuple[str, ...]
    cost_prior: float = 1.0
    # Multiplicative spatial scale produced by the tool. Most restoration
    # tools preserve shape; paper super-resolution tools override this with 4.
    output_scale: float = 1.0

    @abstractmethod
    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        raise NotImplementedError

    def run(
        self,
        image: np.ndarray,
        strength: float = 0.5,
        region_mask: np.ndarray | None = None,
    ) -> ToolResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
        strength = float(np.clip(strength, 0.0, 1.0))
        start = time.perf_counter()
        restored = np.clip(self._apply(image.astype(np.float32), strength), 0.0, 1.0)
        if region_mask is not None:
            # A full-image action needs no blending. This also permits tools such
            # as x4 super-resolution to change the spatial resolution.
            mask = np.clip(region_mask.astype(np.float32), 0.0, 1.0)
            if float(mask.min(initial=1.0)) < 1.0 - 1e-6:
                out_height, out_width = restored.shape[:2]
                if mask.shape != (out_height, out_width):
                    mask = cv2.resize(mask, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                original = image
                if original.shape[:2] != restored.shape[:2]:
                    original = cv2.resize(
                        original,
                        (out_width, out_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
                alpha = mask[..., None]
                restored = original * (1.0 - alpha) + restored * alpha
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ToolResult(
            image=np.clip(restored, 0.0, 1.0).astype(np.float32),
            elapsed_ms=float(elapsed_ms),
            metadata={"tool": self.name, "strength": strength},
        )
