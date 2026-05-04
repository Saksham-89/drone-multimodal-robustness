# Corruption severity parameters — Table 2 from research proposal.
# All pixel values normalized to [0, 1].
#
# RGB params are documentation only: imagecorruptions library severity 1/2/3
# maps to these values exactly, so pipeline.py passes severity directly to the
# library rather than reading RGB params at runtime.
#
# TIR sensor_noise and blur params are actively used by pipeline.py because:
#   - sensor_noise sigma (0.15/0.20/0.35) differs from the library's defaults
#     (0.08/0.12/0.18); thermal sensors are noisier than RGB sensors.
#   - blur has no imagecorruptions equivalent; scipy gaussian_filter uses these
#     sigma values directly.

RGB_CORRUPTIONS = {
    "gaussian_noise": {
        1: {"severity": 1, "std": 0.08},
        2: {"severity": 2, "std": 0.12},
        3: {"severity": 3, "std": 0.18},
    },
    "motion_blur": {
        1: {"severity": 1, "radius": 10, "std": 3},
        2: {"severity": 2, "radius": 15, "std": 5},
        3: {"severity": 3, "radius": 15, "std": 8},
    },
    "brightness_shift": {
        1: {"severity": 1, "offset": 0.10},
        2: {"severity": 2, "offset": 0.20},
        3: {"severity": 3, "offset": 0.30},
    },
    "low_contrast": {
        1: {"severity": 1, "scale": 0.4},
        2: {"severity": 2, "scale": 0.3},
        3: {"severity": 3, "scale": 0.2},
    },
    "complete_dropout": None,  # binary, no severity levels
}

TIR_CORRUPTIONS = {
    "sensor_noise": {
        1: {"severity": 1, "std": 0.15},
        2: {"severity": 2, "std": 0.20},
        3: {"severity": 3, "std": 0.35},
    },
    "blur": {
        1: {"severity": 1, "std": 1},
        2: {"severity": 2, "std": 2},
        3: {"severity": 3, "std": 3},
    },
    "intensity_shift": {
        1: {"severity": 1, "offset": 0.10},
        2: {"severity": 2, "offset": 0.20},
        3: {"severity": 3, "offset": 0.30},
    },
    "complete_dropout": None,  # binary, no severity levels
}

SEVERITY_LEVELS = [1, 2, 3]

# All graded corruption conditions as (modality, corruption_type, severity) tuples.
# Complete dropouts are separate since they have no severity.
GRADED_CONDITIONS = (
    [(mod, name, sev)
     for mod, corruptions in [("rgb", RGB_CORRUPTIONS), ("tir", TIR_CORRUPTIONS)]
     for name, params in corruptions.items()
     if params is not None
     for sev in SEVERITY_LEVELS]
)

DROPOUT_CONDITIONS = [
    ("rgb", "complete_dropout", None),
    ("tir", "complete_dropout", None),
]

ALL_CONDITIONS = GRADED_CONDITIONS + DROPOUT_CONDITIONS  # 23 total
