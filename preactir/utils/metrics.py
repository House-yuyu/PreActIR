from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
from skimage.metrics import structural_similarity


QUALITY_NAMES = ["psnr_scaled", "ssim", "l1_similarity", "edge_similarity"]


def psnr(image: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((image.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range**2) / mse)


def ssim(image: np.ndarray, target: np.ndarray) -> float:
    min_side = min(image.shape[0], image.shape[1])
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    win_size = max(3, win_size)
    return float(
        structural_similarity(
            image,
            target,
            data_range=1.0,
            channel_axis=2,
            win_size=win_size,
        )
    )


def ycbcr_y(image: np.ndarray) -> np.ndarray:
    """Return the pyiqa YCbCr luminance channel for an RGB [0, 1] image.

    This matches ``pyiqa.utils.color_util.to_y_channel(..., color_space="ycbcr")``
    used by the frozen SkillIR comparison protocol.
    """

    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image, found shape={rgb.shape}")
    weights = np.asarray([65.481, 128.553, 24.966], dtype=np.float32)
    return (rgb @ weights + np.float32(16.0)) / np.float32(255.0)


def skillir_y_psnr(image: np.ndarray, target: np.ndarray) -> float:
    """PSNR on YCbCr-Y under the frozen pyiqa SkillIR protocol."""

    image_y = ycbcr_y(image)
    target_y = ycbcr_y(target)
    mse = np.mean((image_y - target_y) ** 2, dtype=np.float32)
    return float(10.0 * np.log10(1.0 / (float(mse) + 1e-8)))


def skillir_y_ssim(image: np.ndarray, target: np.ndarray) -> float:
    """SSIM on YCbCr-Y under the frozen pyiqa SkillIR protocol.

    pyiqa rounds the Y channel to the [0, 255] MATLAB-compatible domain and
    applies an 11x11 Gaussian window with sigma 1.5 in valid mode.  The
    implementation below mirrors those operations without adding pyiqa as a
    training-environment dependency.
    """

    image_y = np.round(ycbcr_y(image) * 255.0).astype(np.float64)
    target_y = np.round(ycbcr_y(target) * 255.0).astype(np.float64)
    coordinates = np.arange(11, dtype=np.float64) - 5.0
    gaussian = np.exp(-(coordinates**2) / (2.0 * 1.5**2))
    gaussian /= gaussian.sum()
    # pyiqa constructs the 2-D kernel in float32 before promoting it to the
    # float64 SSIM computation.
    window = np.outer(gaussian, gaussian).astype(np.float32).astype(np.float64)

    def valid_filter(values: np.ndarray) -> np.ndarray:
        filtered = cv2.filter2D(values, cv2.CV_64F, window, borderType=cv2.BORDER_CONSTANT)
        return filtered[5:-5, 5:-5]

    mu_image = valid_filter(image_y)
    mu_target = valid_filter(target_y)
    mu_image_sq = mu_image * mu_image
    mu_target_sq = mu_target * mu_target
    mu_product = mu_image * mu_target
    sigma_image_sq = valid_filter(image_y * image_y) - mu_image_sq
    sigma_target_sq = valid_filter(target_y * target_y) - mu_target_sq
    covariance = valid_filter(image_y * target_y) - mu_product
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    contrast = np.maximum(
        (2.0 * covariance + c2) / (sigma_image_sq + sigma_target_sq + c2),
        0.0,
    )
    structure = (2.0 * mu_product + c1) / (mu_image_sq + mu_target_sq + c1)
    return float(np.mean(structure * contrast))


def skillir_y_psnr_ssim(image: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return the two full-reference metrics from the frozen SkillIR protocol."""

    return skillir_y_psnr(image, target), skillir_y_ssim(image, target)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def quality_vector(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    psnr_value = min(psnr(image, target), 50.0) / 50.0
    ssim_value = np.clip(ssim(image, target), 0.0, 1.0)
    l1_similarity = 1.0 - float(np.mean(np.abs(image - target)))
    edge_error = float(np.mean(np.abs(gradient_magnitude(image) - gradient_magnitude(target))))
    edge_similarity = float(np.exp(-5.0 * edge_error))
    return np.asarray(
        [psnr_value, ssim_value, np.clip(l1_similarity, 0, 1), np.clip(edge_similarity, 0, 1)],
        dtype=np.float32,
    )


def masked_mean(values: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> float:
    if values.ndim == 3 and mask.ndim == 2:
        mask = mask[..., None]
    denominator = float(mask.sum())
    if values.ndim == 3 and mask.shape[-1] == 1:
        denominator *= values.shape[-1]
    if denominator <= eps:
        return float(values.mean())
    return float((values * mask).sum() / (denominator + eps))


def outside_damage(
    before: np.ndarray,
    after: np.ndarray,
    clean: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    outside = np.clip(1.0 - target_mask, 0.0, 1.0)
    if float(outside.sum()) <= 1e-6:
        return 0.0
    before_error = np.mean(np.abs(before - clean), axis=2)
    after_error = np.mean(np.abs(after - clean), axis=2)
    error_increase = masked_mean(np.maximum(after_error - before_error, 0.0), outside)

    before_edge = gradient_magnitude(before)
    after_edge = gradient_magnitude(after)
    clean_edge = gradient_magnitude(clean)
    edge_before_error = np.abs(before_edge - clean_edge)
    edge_after_error = np.abs(after_edge - clean_edge)
    edge_increase = masked_mean(np.maximum(edge_after_error - edge_before_error, 0.0), outside)
    return float(np.clip(error_increase + 0.5 * edge_increase, 0.0, 1.0))


def aggregate_metric_dicts(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    result: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(row[key])]
        if values:
            result[key] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result
