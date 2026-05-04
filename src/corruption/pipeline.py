import numpy as np
from scipy.ndimage import gaussian_filter

from imagecorruptions.corruptions import (
    gaussian_noise as _ic_gaussian_noise,
    motion_blur as _ic_motion_blur,
    brightness as _ic_brightness,
    contrast as _ic_contrast,
)

from .params import RGB_CORRUPTIONS, TIR_CORRUPTIONS


# ── Utilities ─────────────────────────────────────────────────────────────────

def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _apply_ic(func, image: np.ndarray, severity: int) -> np.ndarray:
    """Call an imagecorruptions function and guarantee uint8 output."""
    return _clip_uint8(np.array(func(image, severity=severity), dtype=np.float32))


# ── TIR-specific helpers ──────────────────────────────────────────────────────

def _tir_sensor_noise(image: np.ndarray, std: float) -> np.ndarray:
    # TIR sigma values (0.15/0.20/0.35) are larger than the imagecorruptions
    # library's gaussian_noise defaults (0.08/0.12/0.18) to reflect thermal
    # sensor noise characteristics. Algorithm is identical; only sigma differs.
    float_img = image.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, std, image.shape).astype(np.float32)
    return _clip_uint8((float_img + noise) * 255.0)


def _tir_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    # Simple Gaussian blur; imagecorruptions has no equivalent (only defocus/
    # motion/glass blur, none of which match σ=1/2/3 from Table 2).
    return _clip_uint8(gaussian_filter(image.astype(np.float32), sigma=sigma))


def _tir_intensity_shift(image: np.ndarray, severity: int) -> np.ndarray:
    # imagecorruptions brightness uses skimage.color.rgb2hsv which requires
    # 3-channel input. Expand grayscale TIR to RGB, apply, take one channel back.
    rgb = np.stack([image, image, image], axis=-1)
    result = _apply_ic(_ic_brightness, rgb, severity)
    return result[:, :, 0]


def _complete_dropout(image: np.ndarray) -> np.ndarray:
    return np.zeros_like(image)


# ── Dispatch tables ───────────────────────────────────────────────────────────
# RGB uses imagecorruptions library with severity=1/2/3 — the library's built-in
# severity scale maps exactly to Table 2 parameters.

_RGB_DISPATCH = {
    "gaussian_noise":   lambda img, sev: _apply_ic(_ic_gaussian_noise, img, sev),
    "motion_blur":      lambda img, sev: _apply_ic(_ic_motion_blur, img, sev),
    "brightness_shift": lambda img, sev: _apply_ic(_ic_brightness, img, sev),
    "low_contrast":     lambda img, sev: _apply_ic(_ic_contrast, img, sev),
    "complete_dropout": lambda img, _:   _complete_dropout(img),
}

_TIR_DISPATCH = {
    "sensor_noise":    lambda img, sev: _tir_sensor_noise(img, TIR_CORRUPTIONS["sensor_noise"][sev]["std"]),
    "blur":            lambda img, sev: _tir_blur(img, TIR_CORRUPTIONS["blur"][sev]["std"]),
    "intensity_shift": lambda img, sev: _tir_intensity_shift(img, sev),
    "complete_dropout": lambda img, _:  _complete_dropout(img),
}


# ── Public API ────────────────────────────────────────────────────────────────

def apply_corruption(
    image: np.ndarray,
    modality: str,
    corruption_type: str,
    severity: int | None,
) -> np.ndarray:
    """Apply one corruption to an image per the proposal's methodology (Section 4.2).

    RGB corruptions use the imagecorruptions library at severity 1/2/3, which
    maps to Table 2 parameters exactly. TIR sensor_noise uses a custom sigma
    (Table 2: 0.15/0.20/0.35) because thermal sensors are noisier than RGB;
    the algorithm is identical to the library's gaussian_noise. TIR blur uses
    scipy gaussian_filter (σ=1/2/3; no imagecorruptions equivalent exists).

    Args:
        image:           uint8 numpy array, HxW (grayscale/TIR) or HxWxC (RGB).
        modality:        'rgb' or 'tir'.
        corruption_type: key matching RGB_CORRUPTIONS / TIR_CORRUPTIONS in params.py.
        severity:        1, 2, or 3 for graded corruptions; None for dropout.

    Returns:
        Corrupted image, same shape and dtype (uint8) as input.
    """
    if modality == "rgb":
        dispatch = _RGB_DISPATCH
    elif modality == "tir":
        dispatch = _TIR_DISPATCH
    else:
        raise ValueError(f"Unknown modality '{modality}'. Expected 'rgb' or 'tir'.")

    if corruption_type not in dispatch:
        raise ValueError(
            f"Unknown corruption '{corruption_type}' for modality '{modality}'."
        )

    return dispatch[corruption_type](image, severity)


def zero_modality(image: np.ndarray) -> np.ndarray:
    """Return all-zeros array of same shape — used for modality removal experiments."""
    return np.zeros_like(image)
