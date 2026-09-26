from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, image_size: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            image = _resize_and_center_crop(image, image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB float image to ``(height, width)`` with suitable filtering."""

    height, width = (int(size[0]), int(size[1]))
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)
    shrinking = height < image.shape[0] or width < image.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return np.clip(cv2.resize(image, (width, height), interpolation=interpolation), 0.0, 1.0).astype(
        np.float32
    )


def resize_and_center_crop_image(image: np.ndarray, size: int) -> np.ndarray:
    """Match the aspect-preserving preprocessing used by ``load_image``.

    Agent inference must use the same transform as model training and offline
    calibration. Converting through uint8 also matches the PNG-backed rollout
    datasets used to train the belief, world, and verifier models.
    """

    image_u8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    processed = _resize_and_center_crop(Image.fromarray(image_u8, mode="RGB"), int(size))
    return np.asarray(processed, dtype=np.float32) / 255.0


def align_to_reference(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return resize_image(image, reference.shape[:2])


def align_agenticir_matlab_x4(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Apply the frozen AgenticIR geometry rule used by the MiO100 metrics.

    Matching shapes are returned unchanged.  A mismatch is accepted only when
    the input is exactly one quarter of the reference along both axes, in which
    case BasicSR's MATLAB-compatible x4 resize is used.
    """

    if image.shape[:2] == reference.shape[:2]:
        return image.astype(np.float32, copy=False)
    image_height, image_width = image.shape[:2]
    reference_height, reference_width = reference.shape[:2]
    if image_height * 4 != reference_height or image_width * 4 != reference_width:
        raise ValueError(
            "AgenticIR metric alignment only permits exact x4 mismatches: "
            f"image={(image_width, image_height)}, "
            f"reference={(reference_width, reference_height)}"
        )
    try:
        from basicsr.utils.matlab_functions import imresize
    except ImportError as exc:
        raise RuntimeError(
            "BasicSR is required for the frozen AgenticIR MATLAB-style x4 alignment"
        ) from exc
    tensor = image_to_tensor(image)
    aligned = torch.clamp(imresize(tensor, scale=4), 0.0, 1.0)
    return tensor_to_image(aligned).astype(np.float32, copy=False)


def _resize_and_center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    new_width = max(size, int(round(width * scale)))
    new_height = max(size, int(round(height * scale)))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return image.crop((left, top, left + size, top + size))


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def load_mask(path: str | Path, image_size: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, ...]
    return torch.from_numpy(np.ascontiguousarray(mask)).float()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    return tensor.permute(1, 2, 0).numpy()


def gaussian_soften_mask(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    if sigma <= 0:
        return np.clip(mask, 0, 1)
    kernel = max(3, int(round(sigma * 6)) | 1)
    softened = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), sigmaX=sigma)
    maximum = float(softened.max())
    if maximum > 0:
        softened /= maximum
    return np.clip(softened, 0, 1)
