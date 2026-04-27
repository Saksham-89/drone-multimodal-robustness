import numpy as np
from imagecorruptions import corrupt
from .params import RGB_CORRUPTIONS, TIR_CORRUPTIONS


def apply_corruption(image: np.ndarray, modality: str, corruption_type: str, severity: int | None) -> np.ndarray:
    """Apply a single corruption to an image array (uint8, HxW or HxWxC).

    For complete_dropout, severity is ignored and the image is zeroed.
    Intensity shift for TIR uses the brightness corruption on the grayscale channel,
    matching the HSV-offset parameterisation from the proposal (Medeiros et al. precedent).
    """
    if corruption_type == "complete_dropout":
        return np.zeros_like(image)

    params_table = RGB_CORRUPTIONS if modality == "rgb" else TIR_CORRUPTIONS
    params = params_table[corruption_type][severity]

    # Map thesis corruption names to imagecorruptions function names
    CORRUPTION_NAME_MAP = {
        # RGB
        "gaussian_noise": "gaussian_noise",
        "motion_blur": "motion_blur",
        "brightness_shift": "brightness",
        "low_contrast": "contrast",
        # TIR
        "sensor_noise": "gaussian_noise",
        "blur": "gaussian_blur",
        "intensity_shift": "brightness",
    }

    lib_name = CORRUPTION_NAME_MAP[corruption_type]
    return corrupt(image, corruption_name=lib_name, severity=params["severity"])


def zero_modality(image: np.ndarray) -> np.ndarray:
    """Return an all-zeros tensor of the same shape — used for modality removal experiments."""
    return np.zeros_like(image)
