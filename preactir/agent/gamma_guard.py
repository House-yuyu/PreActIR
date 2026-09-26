from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


GAMMA_FEATURE_NAMES = [
    "value_q01",
    "value_q10",
    "value_q25",
    "value_q50",
    "value_q75",
    "value_q90",
    "value_q99",
    "value_mean",
    "value_std",
    "saturation_mean",
    "saturation_std",
    "gray_mean",
    "gray_std",
    "dark_fraction",
    "bright_fraction",
    "gradient_mean",
    "value_gain_mean",
    "gray_gain_mean",
]


def _hsv_and_gray(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(np.floor(np.clip(image, 0.0, 1.0) * 255.0 + 0.5), 0, 255).astype(
        np.uint8
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return hsv[..., 1] / 255.0, hsv[..., 2] / 255.0, gray


def extract_gamma_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_saturation, before_value, before_gray = _hsv_and_gray(before)
    _, after_value, after_gray = _hsv_and_gray(after)
    quantiles = np.quantile(before_value, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    gradient_x = cv2.Sobel(before_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(before_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mean = float(np.sqrt(gradient_x**2 + gradient_y**2).mean())
    values = [
        *quantiles.tolist(),
        float(before_value.mean()),
        float(before_value.std()),
        float(before_saturation.mean()),
        float(before_saturation.std()),
        float(before_gray.mean()),
        float(before_gray.std()),
        float((before_value <= 0.10).mean()),
        float((before_value >= 0.90).mean()),
        gradient_mean,
        float((after_value - before_value).mean()),
        float((after_gray - before_gray).mean()),
    ]
    return np.asarray(values, dtype=np.float32)


class GammaSafetyGuard:
    def __init__(
        self,
        *,
        mean: np.ndarray,
        scale: np.ndarray,
        coefficients: np.ndarray,
        intercept: float,
        threshold: float,
    ) -> None:
        expected = len(GAMMA_FEATURE_NAMES)
        for name, value in {
            "mean": mean,
            "scale": scale,
            "coefficients": coefficients,
        }.items():
            if np.asarray(value).shape != (expected,):
                raise ValueError(f"{name} must have shape ({expected},)")
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-8)
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> GammaSafetyGuard:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("feature_names") != GAMMA_FEATURE_NAMES:
            raise ValueError("Gamma guard feature vocabulary does not match runtime")
        return cls(
            mean=np.asarray(payload["standardizer"]["mean"], dtype=np.float32),
            scale=np.asarray(payload["standardizer"]["scale"], dtype=np.float32),
            coefficients=np.asarray(payload["logistic_regression"]["coefficients"], dtype=np.float32),
            intercept=float(payload["logistic_regression"]["intercept"]),
            threshold=float(payload["selected_threshold"]),
        )

    def predict_probability(self, before: np.ndarray, after: np.ndarray) -> float:
        features = extract_gamma_features(before, after)
        standardized = (features - self.mean) / self.scale
        logit = float(np.dot(standardized, self.coefficients) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def accepts(self, before: np.ndarray, after: np.ndarray) -> tuple[bool, float]:
        probability = self.predict_probability(before, after)
        return probability >= self.threshold, probability
