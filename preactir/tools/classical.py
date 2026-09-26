from __future__ import annotations

import cv2
import numpy as np

from preactir.tools.base import RestorationTool


def _u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _float(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


class DenoiseTool(RestorationTool):
    name = "denoise"
    targets = ("noise",)
    cost_prior = 0.7

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        h = float(2.0 + 18.0 * strength)
        output = cv2.fastNlMeansDenoisingColored(_u8(image), None, h, h, 7, 21)
        return _float(output)


class DeblurTool(RestorationTool):
    name = "deblur"
    targets = ("blur",)
    cost_prior = 0.45

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        sigma = 0.7 + 1.6 * strength
        kernel = max(3, int(round(sigma * 6)) | 1)
        blurred = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)
        amount = 0.35 + 1.65 * strength
        return np.clip(image + amount * (image - blurred), 0.0, 1.0)


class MotionDeblurTool(DeblurTool):
    name = "motion_deblur"
    targets = ("motion_blur",)
    cost_prior = 0.55


class DefocusDeblurTool(DeblurTool):
    name = "defocus_deblur"
    targets = ("defocus_blur",)
    cost_prior = 0.50


class BrightenTool(RestorationTool):
    name = "brighten"
    targets = ("lowlight",)
    cost_prior = 0.2

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        gamma = 1.0 / (1.0 + 1.8 * strength)
        exposure = 1.0 + 0.18 * strength
        output = np.clip((image**gamma) * exposure, 0.0, 1.0)
        lab = cv2.cvtColor(_u8(output), cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + 1.5 * strength, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = _float(cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB))
        return np.clip(0.65 * output + 0.35 * enhanced, 0.0, 1.0)


class DarkEnhanceTool(BrightenTool):
    name = "dark_enhance"
    targets = ("dark",)


class SuperResolutionTool(RestorationTool):
    """Shape-preserving bicubic enhancement placeholder for low-resolution inputs."""

    name = "super_resolve"
    targets = ("low_resolution",)
    cost_prior = 0.65

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        height, width = image.shape[:2]
        scale = 1.25 + 1.75 * strength
        enlarged = cv2.resize(
            image,
            (max(width + 1, int(round(width * scale))), max(height + 1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        restored = cv2.resize(enlarged, (width, height), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(restored, (3, 3), 0.65 + 0.35 * strength)
        return np.clip(restored + (0.25 + 0.75 * strength) * (restored - smooth), 0.0, 1.0)


class DehazeTool(RestorationTool):
    name = "dehaze"
    targets = ("haze",)
    cost_prior = 0.35

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        lab = cv2.cvtColor(_u8(image), cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2 + 2.5 * strength, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        contrast = _float(cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB))
        means = np.mean(contrast, axis=(0, 1), keepdims=True)
        gray_mean = float(np.mean(means))
        balanced = np.clip(contrast * (gray_mean / (means + 1e-6)), 0.0, 1.0)
        amount = 0.45 + 0.45 * strength
        return np.clip((1.0 - amount) * image + amount * balanced, 0.0, 1.0)


class DejpegTool(RestorationTool):
    name = "dejpeg"
    targets = ("jpeg",)
    cost_prior = 0.35

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        diameter = int(5 + 4 * strength)
        if diameter % 2 == 0:
            diameter += 1
        smooth = cv2.bilateralFilter(_u8(image), diameter, 20 + 45 * strength, 20 + 45 * strength)
        smooth_f = _float(smooth)
        blur = cv2.GaussianBlur(smooth_f, (3, 3), 0.8)
        return np.clip(smooth_f + (0.15 + 0.45 * strength) * (smooth_f - blur), 0.0, 1.0)


class DerainTool(RestorationTool):
    name = "derain"
    targets = ("rain",)
    cost_prior = 0.5

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        size = 3 if strength < 0.6 else 5
        median = _float(cv2.medianBlur(_u8(image), size))
        residual = np.maximum(image - median, 0.0)
        output = image - (0.35 + 0.55 * strength) * residual
        return np.clip(0.65 * output + 0.35 * median, 0.0, 1.0)
