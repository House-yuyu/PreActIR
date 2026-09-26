from __future__ import annotations

import io
from collections.abc import Mapping

import cv2
import numpy as np
from PIL import Image

from preactir.utils.image import gaussian_soften_mask
from preactir.utils.metrics import gradient_magnitude, masked_mean

SUPPORTED_DEGRADATIONS = (
    "noise",
    "blur",
    "motion_blur",
    "defocus_blur",
    "lowlight",
    "dark",
    "low_resolution",
    "haze",
    "jpeg",
    "rain",
)


def random_spatial_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    spatial_probability: float = 0.45,
) -> np.ndarray:
    if rng.random() >= spatial_probability:
        return np.ones((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:
            x1, x2 = sorted(int(v) for v in rng.integers(0, width, size=2))
            y1, y2 = sorted(int(v) for v in rng.integers(0, height, size=2))
            cv2.rectangle(mask, (x1, y1), (max(x1 + 1, x2), max(y1 + 1, y2)), 1.0, -1)
        else:
            center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            axes = (
                int(rng.integers(max(4, width // 10), max(5, width // 2))),
                int(rng.integers(max(4, height // 10), max(5, height // 2))),
            )
            cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = gaussian_soften_mask(mask, sigma=float(rng.uniform(3.0, 10.0)))
    return mask if float(mask.mean()) >= 0.08 else np.ones((height, width), dtype=np.float32)


def _blend(original: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(mask[..., None], 0.0, 1.0)
    return np.clip(original * (1.0 - alpha) + degraded * alpha, 0.0, 1.0).astype(np.float32)


def _noise(image: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    sigma = 0.015 + 0.16 * severity
    return np.clip(image + rng.normal(0.0, sigma, image.shape).astype(np.float32), 0.0, 1.0)


def _blur(image: np.ndarray, severity: float) -> np.ndarray:
    sigma = 0.4 + 3.2 * severity
    kernel = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def _motion_blur(image: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    length = max(3, int(round(3 + 20 * severity)) | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    angle = float(rng.uniform(-75.0, 75.0))
    radians = np.deg2rad(angle)
    dx = int(round(center * np.cos(radians)))
    dy = int(round(center * np.sin(radians)))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, 1)
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _low_resolution(image: np.ndarray, severity: float) -> np.ndarray:
    height, width = image.shape[:2]
    scale = 1.0 + 7.0 * severity
    small_width = max(2, int(round(width / scale)))
    small_height = max(2, int(round(height / scale)))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def _lowlight(image: np.ndarray, severity: float) -> np.ndarray:
    gamma = 1.15 + 2.4 * severity
    exposure = 1.0 - 0.35 * severity
    return np.clip((image**gamma) * exposure, 0.0, 1.0)


def _haze(image: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    direction = float(rng.uniform(0, 2 * np.pi))
    depth = np.cos(direction) * xx / max(w - 1, 1) + np.sin(direction) * yy / max(h - 1, 1)
    depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
    transmission = np.clip(1.0 - (0.28 + 0.58 * severity) + 0.18 * (1.0 - depth), 0.15, 0.95)[..., None]
    atmosphere = np.asarray([0.88, 0.91, 0.96], dtype=np.float32)
    return np.clip(image * transmission + atmosphere[None, None, :] * (1.0 - transmission), 0.0, 1.0)


def _jpeg(image: np.ndarray, severity: float) -> np.ndarray:
    quality = int(np.clip(round(96 - 86 * severity), 8, 95))
    buffer = io.BytesIO()
    Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
        buffer, format="JPEG", quality=quality, subsampling=2
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def _rain(image: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)
    count = int((70 + 420 * severity) * h * w / (256 * 256))
    length = int(6 + 20 * severity)
    angle = float(rng.uniform(-0.35, 0.35))
    dx, dy = int(round(length * np.sin(angle))), int(round(length * np.cos(angle)))
    for _ in range(max(1, count)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(layer, (x, y), (x + dx, y + dy), float(rng.uniform(0.35, 0.95)), int(rng.integers(1, 3)))
    layer = cv2.GaussianBlur(layer, (3, max(3, int(3 + severity * 8) | 1)), 0.8)[..., None]
    return np.clip(image * (0.98 - 0.08 * severity) + layer * (0.35 + 0.45 * severity) + 0.02, 0.0, 1.0)


def apply_degradation(
    image: np.ndarray,
    name: str,
    severity: float,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    severity = float(np.clip(severity, 0.0, 1.0))
    if name == "noise":
        degraded = _noise(image, severity, rng)
    elif name in {"blur", "defocus_blur"}:
        degraded = _blur(image, severity)
    elif name == "motion_blur":
        degraded = _motion_blur(image, severity, rng)
    elif name in {"lowlight", "dark"}:
        degraded = _lowlight(image, severity)
    elif name == "low_resolution":
        degraded = _low_resolution(image, severity)
    elif name == "haze":
        degraded = _haze(image, severity, rng)
    elif name == "jpeg":
        degraded = _jpeg(image, severity)
    elif name == "rain":
        degraded = _rain(image, severity, rng)
    else:
        raise KeyError(f"Unsupported degradation: {name}")
    return _blend(image, degraded, mask)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _high_pass(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    return gray - cv2.GaussianBlur(gray, (5, 5), 1.0)


def _blockiness(gray: np.ndarray) -> np.ndarray:
    score = np.zeros_like(gray, dtype=np.float32)
    x_boundaries = np.arange(8, gray.shape[1], 8)
    y_boundaries = np.arange(8, gray.shape[0], 8)
    if x_boundaries.size:
        score[:, x_boundaries] += np.abs(gray[:, x_boundaries] - gray[:, x_boundaries - 1])
    if y_boundaries.size:
        score[y_boundaries, :] += np.abs(gray[y_boundaries, :] - gray[y_boundaries - 1, :])
    return cv2.GaussianBlur(score, (5, 5), 1.0)


def degradation_score(image: np.ndarray, clean: np.ndarray, name: str, mask: np.ndarray) -> float:
    mask = np.clip(mask.astype(np.float32), 0, 1)
    l1_map = np.mean(np.abs(image - clean), axis=2)
    if name == "noise":
        value = masked_mean(np.abs(_high_pass(image) - _high_pass(clean)), mask) * 4.0
    elif name in {"blur", "motion_blur"}:
        value = masked_mean(np.maximum(gradient_magnitude(clean) - gradient_magnitude(image), 0.0), mask) * 4.0
    elif name == "defocus_blur":
        gradient_loss = masked_mean(
            np.maximum(gradient_magnitude(clean) - gradient_magnitude(image), 0.0), mask
        )
        lap_image = np.abs(cv2.Laplacian(_gray(image), cv2.CV_32F))
        lap_clean = np.abs(cv2.Laplacian(_gray(clean), cv2.CV_32F))
        laplacian_loss = masked_mean(np.maximum(lap_clean - lap_image, 0.0), mask)
        value = 2.5 * gradient_loss + 1.5 * laplacian_loss
    elif name in {"lowlight", "dark"}:
        value = masked_mean(np.maximum(_gray(clean) - _gray(image), 0.0), mask) * 2.0
    elif name == "low_resolution":
        edge_loss = masked_mean(
            np.maximum(gradient_magnitude(clean) - gradient_magnitude(image), 0.0), mask
        )
        value = 3.0 * edge_loss + 0.75 * masked_mean(l1_map, mask)
    elif name == "haze":
        gi, gc = _gray(image), _gray(clean)
        ci = np.abs(gi - cv2.GaussianBlur(gi, (15, 15), 3.0))
        cc = np.abs(gc - cv2.GaussianBlur(gc, (15, 15), 3.0))
        value = 2.0 * masked_mean(np.maximum(cc - ci, 0.0), mask) + masked_mean(l1_map, mask)
    elif name == "jpeg":
        value = 5.0 * masked_mean(np.abs(_blockiness(_gray(image)) - _blockiness(_gray(clean))), mask) + 0.5 * masked_mean(l1_map, mask)
    elif name == "rain":
        diff = np.abs(_gray(image) - _gray(clean))
        vertical = np.abs(cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3))
        value = 1.5 * masked_mean(diff, mask) + 0.5 * masked_mean(vertical, mask)
    else:
        raise KeyError(name)
    return float(np.clip(value, 0.0, 1.0))


def compute_degradation_scores(
    image: np.ndarray,
    clean: np.ndarray,
    masks_by_name: Mapping[str, np.ndarray],
    degradation_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    h, w = image.shape[:2]
    scores = [
        degradation_score(image, clean, name, masks_by_name.get(name, np.ones((h, w), dtype=np.float32)))
        for name in degradation_names
    ]
    return np.asarray(scores, dtype=np.float32)
