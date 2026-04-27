import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .params import RGB_CORRUPTIONS, TIR_CORRUPTIONS


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    float_img = image.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, std, image.shape).astype(np.float32)
    return _clip_uint8((float_img + noise) * 255.0)


def _motion_blur(image: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    kernel_size = 2 * radius + 1
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)  # shape (k, 1)
    kernel_2d = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel_2d[radius] = kernel_1d[:, 0]  # horizontal row through centre
    blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel_2d)
    return _clip_uint8(blurred.astype(np.float32))


def _brightness_shift(image: np.ndarray, offset: float) -> np.ndarray:
    if image.ndim == 2:
        # Grayscale (TIR) — shift directly in intensity space
        float_img = image.astype(np.float32) / 255.0
        return _clip_uint8((float_img + offset) * 255.0)
    # RGB — shift V channel in HSV space (same parameterisation as imagecorruptions)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + offset * 255.0, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _low_contrast(image: np.ndarray, scale: float) -> np.ndarray:
    float_img = image.astype(np.float32) / 255.0
    return _clip_uint8(float_img * scale * 255.0)


def _gaussian_blur_tir(image: np.ndarray, sigma: float) -> np.ndarray:
    blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    return _clip_uint8(blurred)


def _complete_dropout(image: np.ndarray) -> np.ndarray:
    return np.zeros_like(image)


# ── Dispatch tables ───────────────────────────────────────────────────────────

_RGB_DISPATCH = {
    "gaussian_noise":   lambda img, p: _gaussian_noise(img, p["std"]),
    "motion_blur":      lambda img, p: _motion_blur(img, p["radius"], p["std"]),
    "brightness_shift": lambda img, p: _brightness_shift(img, p["offset"]),
    "low_contrast":     lambda img, p: _low_contrast(img, p["scale"]),
    "complete_dropout": lambda img, p: _complete_dropout(img),
}

_TIR_DISPATCH = {
    "sensor_noise":    lambda img, p: _gaussian_noise(img, p["std"]),
    "blur":            lambda img, p: _gaussian_blur_tir(img, p["std"]),
    "intensity_shift": lambda img, p: _brightness_shift(img, p["offset"]),
    "complete_dropout": lambda img, p: _complete_dropout(img),
}


# ── Public API ────────────────────────────────────────────────────────────────

def apply_corruption(
    image: np.ndarray,
    modality: str,
    corruption_type: str,
    severity: int | None,
) -> np.ndarray:
    """Apply one corruption to an image using exact Table 2 parameters.

    Parameters come exclusively from params.py — not from the imagecorruptions
    library severity system — ensuring exact compliance with the proposal.

    Args:
        image:          uint8 numpy array, shape HxW (grayscale) or HxWxC (RGB).
        modality:       'rgb' or 'tir'.
        corruption_type: name matching keys in RGB_CORRUPTIONS / TIR_CORRUPTIONS.
        severity:       1, 2, or 3 for graded corruptions; None for dropout.

    Returns:
        Corrupted image, same shape and dtype (uint8) as input.
    """
    if modality == "rgb":
        params_table, dispatch = RGB_CORRUPTIONS, _RGB_DISPATCH
    elif modality == "tir":
        params_table, dispatch = TIR_CORRUPTIONS, _TIR_DISPATCH
    else:
        raise ValueError(f"Unknown modality '{modality}'. Expected 'rgb' or 'tir'.")

    if corruption_type not in dispatch:
        raise ValueError(
            f"Unknown corruption '{corruption_type}' for modality '{modality}'."
        )

    entry = params_table[corruption_type]
    params = entry[severity] if (entry is not None and severity is not None) else {}
    return dispatch[corruption_type](image, params)


def zero_modality(image: np.ndarray) -> np.ndarray:
    """Return all-zeros array of the same shape — used for modality removal experiments."""
    return np.zeros_like(image)
