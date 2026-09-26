from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from preactir.utils.metrics import gradient_magnitude


@dataclass
class DiagnosticOutput:
    scores: dict[str, float]


class DiagnosticProbes:
    """Cheap, non-destructive probes used only when belief uncertainty is high.

    These are deliberately lightweight priors, not ground-truth degradation
    estimators. Their contribution is uncertainty-gated and softly blended.
    """

    def __init__(self, degradation_names: list[str]) -> None:
        self.degradation_names = list(degradation_names)

    def run(self, image: np.ndarray) -> DiagnosticOutput:
        gray = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        smooth = cv2.GaussianBlur(gray_f, (5, 5), 1.0)
        high = np.abs(gray_f - smooth)
        edge = gradient_magnitude(image)

        noise = float(np.clip(np.mean(high) / 0.08, 0.0, 1.0))
        blur = float(np.clip(1.0 - np.mean(edge) / 0.18, 0.0, 1.0))
        lowlight = float(np.clip((0.42 - np.mean(gray_f)) / 0.42, 0.0, 1.0))
        laplacian = float(np.var(cv2.Laplacian(gray_f, cv2.CV_32F)))
        defocus_blur = float(np.clip(1.0 - laplacian / 0.012, 0.0, 1.0))
        horizontal_edge = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
        vertical_edge = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))
        anisotropy = abs(float(horizontal_edge.mean() - vertical_edge.mean()))
        motion_blur = float(np.clip(0.75 * blur + 0.25 * anisotropy / 0.05, 0.0, 1.0))
        low_resolution = float(np.clip(0.65 * blur + 0.35 * (1.0 - np.mean(high) / 0.06), 0.0, 1.0))
        contrast = float(np.std(gray_f))
        channel_min = float(np.mean(np.min(image, axis=2)))
        haze = float(np.clip(0.6 * (0.20 - contrast) / 0.20 + 0.4 * channel_min / 0.65, 0.0, 1.0))

        vertical_boundary = np.zeros_like(gray_f)
        horizontal_boundary = np.zeros_like(gray_f)
        x_boundaries = np.arange(8, gray_f.shape[1], 8)
        y_boundaries = np.arange(8, gray_f.shape[0], 8)
        if x_boundaries.size:
            vertical_boundary[:, x_boundaries] = np.abs(
                gray_f[:, x_boundaries] - gray_f[:, x_boundaries - 1]
            )
        if y_boundaries.size:
            horizontal_boundary[y_boundaries, :] = np.abs(
                gray_f[y_boundaries, :] - gray_f[y_boundaries - 1, :]
            )
        jpeg = float(np.clip((vertical_boundary.mean() + horizontal_boundary.mean()) / 0.035, 0.0, 1.0))

        vertical_gradient = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))
        horizontal_gradient = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
        rain = float(
            np.clip(
                (vertical_gradient.mean() - 0.65 * horizontal_gradient.mean()) / 0.16,
                0.0,
                1.0,
            )
        )
        available = {
            "noise": noise,
            "blur": blur,
            "motion_blur": motion_blur,
            "defocus_blur": defocus_blur,
            "lowlight": lowlight,
            "dark": lowlight,
            "low_resolution": low_resolution,
            "haze": haze,
            "jpeg": jpeg,
            "rain": rain,
        }
        return DiagnosticOutput({name: float(available.get(name, 0.0)) for name in self.degradation_names})

    def refine(
        self,
        presence: np.ndarray,
        severity: np.ndarray,
        uncertainty: np.ndarray,
        output: DiagnosticOutput,
        uncertainty_threshold: float = 0.30,
        blend: float = 0.30,
    ) -> tuple[np.ndarray, np.ndarray]:
        presence = presence.copy().astype(np.float32)
        severity = severity.copy().astype(np.float32)
        for index, name in enumerate(self.degradation_names):
            if float(uncertainty[index]) < uncertainty_threshold:
                continue
            proxy = float(output.scores.get(name, 0.0))
            severity[index] = (1.0 - blend) * severity[index] + blend * proxy
            presence[index] = max(float(presence[index]), 0.8 * proxy)
        return np.clip(presence, 0.0, 1.0), np.clip(severity, 0.0, 1.0)
