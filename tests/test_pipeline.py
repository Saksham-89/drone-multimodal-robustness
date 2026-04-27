"""Unit tests for the corruption pipeline.

No GPU, no dataset — uses synthetic numpy images only.
Run with: pytest tests/test_pipeline.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corruption.params import ALL_CONDITIONS, GRADED_CONDITIONS, TIR_CORRUPTIONS
from src.corruption.pipeline import apply_corruption, zero_modality

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

@pytest.fixture
def rgb_image():
    """Synthetic HxWxC uint8 RGB image."""
    return RNG.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)

@pytest.fixture
def tir_image():
    """Synthetic HxW uint8 grayscale TIR image."""
    return RNG.integers(30, 220, size=(64, 64), dtype=np.uint8)


# ── Test 1: ALL_CONDITIONS completeness ──────────────────────────────────────

def test_all_conditions_count():
    """Exactly 23 corruption conditions must be defined (21 graded + 2 dropouts)."""
    assert len(ALL_CONDITIONS) == 23, (
        f"Expected 23 conditions, got {len(ALL_CONDITIONS)}"
    )


# ── Test 2: Modality coverage ─────────────────────────────────────────────────

def test_modality_coverage():
    """Both 'rgb' and 'tir' must appear in ALL_CONDITIONS."""
    modalities = {mod for mod, _, _ in ALL_CONDITIONS}
    assert "rgb" in modalities
    assert "tir" in modalities


# ── Test 3: Shape preservation ────────────────────────────────────────────────

@pytest.mark.parametrize("modality,corruption_type,severity", ALL_CONDITIONS)
def test_shape_preserved_rgb(modality, corruption_type, severity, rgb_image, tir_image):
    """Output shape must match input shape for every condition."""
    image = rgb_image if modality == "rgb" else tir_image
    result = apply_corruption(image, modality, corruption_type, severity)
    assert result.shape == image.shape, (
        f"Shape changed for {modality}/{corruption_type}/s{severity}: "
        f"{image.shape} → {result.shape}"
    )


# ── Test 4: Dtype preservation ────────────────────────────────────────────────

@pytest.mark.parametrize("modality,corruption_type,severity", ALL_CONDITIONS)
def test_dtype_uint8(modality, corruption_type, severity, rgb_image, tir_image):
    """Output must always be uint8."""
    image = rgb_image if modality == "rgb" else tir_image
    result = apply_corruption(image, modality, corruption_type, severity)
    assert result.dtype == np.uint8, (
        f"Dtype is {result.dtype} for {modality}/{corruption_type}/s{severity}"
    )


# ── Test 5: Value range ───────────────────────────────────────────────────────

@pytest.mark.parametrize("modality,corruption_type,severity", ALL_CONDITIONS)
def test_value_range(modality, corruption_type, severity, rgb_image, tir_image):
    """All pixels must remain in [0, 255] after corruption."""
    image = rgb_image if modality == "rgb" else tir_image
    result = apply_corruption(image, modality, corruption_type, severity)
    assert result.min() >= 0, f"Pixel below 0 for {modality}/{corruption_type}/s{severity}"
    assert result.max() <= 255, f"Pixel above 255 for {modality}/{corruption_type}/s{severity}"


# ── Test 6: Complete dropout produces all zeros ───────────────────────────────

def test_rgb_dropout_all_zeros(rgb_image):
    result = apply_corruption(rgb_image, "rgb", "complete_dropout", None)
    assert np.all(result == 0), "RGB dropout should produce all-zero image"

def test_tir_dropout_all_zeros(tir_image):
    result = apply_corruption(tir_image, "tir", "complete_dropout", None)
    assert np.all(result == 0), "TIR dropout should produce all-zero image"


# ── Test 7: zero_modality produces all zeros ──────────────────────────────────

def test_zero_modality_rgb(rgb_image):
    result = zero_modality(rgb_image)
    assert result.shape == rgb_image.shape
    assert np.all(result == 0)

def test_zero_modality_tir(tir_image):
    result = zero_modality(tir_image)
    assert result.shape == tir_image.shape
    assert np.all(result == 0)


# ── Test 8: Corruption actually modifies the image ───────────────────────────

@pytest.mark.parametrize("modality,corruption_type,severity", GRADED_CONDITIONS)
def test_corruption_modifies_image(modality, corruption_type, severity, rgb_image, tir_image):
    """Every graded corruption must change at least some pixels."""
    image = rgb_image if modality == "rgb" else tir_image
    np.random.seed(0)  # fix seed so noise is deterministic
    result = apply_corruption(image, modality, corruption_type, severity)
    assert not np.array_equal(result, image), (
        f"Corruption had no effect: {modality}/{corruption_type}/s{severity}"
    )


# ── Test 9: Severity monotonicity ─────────────────────────────────────────────

@pytest.mark.parametrize("modality,corruption_type", [
    ("rgb", "gaussian_noise"),
    ("rgb", "low_contrast"),
    ("tir", "sensor_noise"),
    ("tir", "blur"),
])
def test_severity_monotonicity(modality, corruption_type, rgb_image, tir_image):
    """Higher severity must produce greater mean absolute deviation from original."""
    image = rgb_image if modality == "rgb" else tir_image
    deviations = []
    for sev in [1, 2, 3]:
        np.random.seed(0)
        result = apply_corruption(image, modality, corruption_type, sev)
        mae = np.mean(np.abs(result.astype(np.float32) - image.astype(np.float32)))
        deviations.append(mae)
    assert deviations[0] < deviations[1] < deviations[2], (
        f"Severity not monotone for {modality}/{corruption_type}: {deviations}"
    )


# ── Test 10: TIR sensor noise uses custom sigma (not library defaults) ────────

def test_tir_sensor_noise_sigma():
    """
    TIR sensor noise sigma values (0.15/0.20/0.35) differ from the imagecorruptions
    library's gaussian_noise defaults (0.08/0.12/0.18). Verify our params are used
    by checking that higher-sigma conditions produce proportionally more noise.
    """
    # Flat grey image so all deviation comes from noise
    image = np.full((128, 128), 128, dtype=np.uint8)
    std_values = [TIR_CORRUPTIONS["sensor_noise"][s]["std"] for s in [1, 2, 3]]
    measured_stds = []
    for sev in [1, 2, 3]:
        np.random.seed(0)
        result = apply_corruption(image, "tir", "sensor_noise", sev)
        diff = result.astype(np.float32) - image.astype(np.float32)
        measured_stds.append(diff.std() / 255.0)  # normalise back to [0,1]

    # Each severity must produce more noise than the previous
    assert measured_stds[0] < measured_stds[1] < measured_stds[2], (
        f"TIR sensor noise std not monotone: {measured_stds}"
    )
    # Measured std should be in the right ballpark of Table 2 params
    for measured, expected in zip(measured_stds, std_values):
        assert abs(measured - expected) < 0.05, (
            f"TIR sensor noise: measured std {measured:.4f} too far from "
            f"expected {expected:.4f}"
        )


# ── Test 11: Invalid inputs raise cleanly ─────────────────────────────────────

def test_invalid_modality(rgb_image):
    with pytest.raises(ValueError, match="Unknown modality"):
        apply_corruption(rgb_image, "lidar", "gaussian_noise", 1)

def test_invalid_corruption_type(rgb_image):
    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_corruption(rgb_image, "rgb", "nonexistent_corruption", 1)
