"""Export the corrupted DroneVehicle test split as a reusable dataset.

Applies all 23 corruption conditions to every test image pair and saves the
result to an organised directory structure that mirrors DroneVehicle's layout.
Annotations (OBB labels) are copied once into a shared labels/ directory since
they are identical across all corruption conditions.

Output structure:
  <out-root>/
    labels/                   # original test annotations (DOTA format .txt)
    clean/rgb/  clean/ir/     # unmodified test images (convenience copy)
    rgb_gaussian_noise_s1/rgb/  rgb_gaussian_noise_s1/ir/   # RGB corrupted, TIR clean
    ...
    tir_sensor_noise_s3/rgb/  tir_sensor_noise_s3/ir/       # RGB clean, TIR corrupted
    ...
    metadata.json             # condition descriptions and parameter values
    README.md

Resume: existing output files are skipped, so the script can be safely restarted.

Run from project root on HPC (no GPU needed):
    PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness \\
    python scripts/export_corrupted_dataset.py \\
        --data-root /home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle \\
        --out-root  /home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle_C
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.corruption.pipeline import apply_corruption
from src.corruption.params import ALL_CONDITIONS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_pair(img_dir: Path, stem: str):
    for rgb_p, ir_p in [
        (img_dir / f'{stem}.jpg',          img_dir / f'{stem}_tir.jpg'),   # actual convention
        (img_dir / f'{stem}.jpg',          img_dir / f'{stem}_ir.jpg'),
        (img_dir / 'rgb' / f'{stem}.jpg',  img_dir / 'ir' / f'{stem}.jpg'),
        (img_dir / f'{stem}.png',          img_dir / f'{stem}_tir.png'),
    ]:
        if rgb_p.exists() and ir_p.exists():
            return rgb_p, ir_p
    return None, None


def condition_name(modality, corruption_type, severity):
    if severity is None:
        return f'{modality}_{corruption_type}'  # e.g., rgb_complete_dropout
    return f'{modality}_{corruption_type}_s{severity}'  # e.g., rgb_gaussian_noise_s1


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f'Failed to read {path}')
    return img  # BGR


def load_tir(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f'Failed to read {path}')
    return img


def save_image(img: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ── Metadata ───────────────────────────────────────────────────────────────────

METADATA = {
    "dataset": "DroneVehicle-C (Corrupted)",
    "source": "DroneVehicle test split (Sun et al., IEEE TCSVT 2022)",
    "description": (
        "Corrupted version of the DroneVehicle test split. "
        "Each condition applies one corruption type to one modality (RGB or TIR) "
        "while leaving the other modality clean. "
        "Annotations are identical across all conditions (labels/ directory). "
        "Generated for the thesis: 'Evaluating Multimodal Fusion Robustness in "
        "Drone-Based Object Detection Under Sensor Degradation' (TSCiT 2025)."
    ),
    "classes": ["car", "truck", "freight_car", "bus", "van"],
    "annotation_format": "DOTA oriented bounding box (x1 y1 x2 y2 x3 y3 x4 y4 class difficult)",
    "image_size": "840x712 (original DroneVehicle resolution, 100px white border each side)",
    "conditions": {}
}

CONDITION_DESCRIPTIONS = {
    ("rgb", "gaussian_noise"):   "Additive Gaussian noise on RGB channel",
    ("rgb", "motion_blur"):      "Motion blur on RGB channel",
    ("rgb", "brightness_shift"): "Global brightness increase on RGB channel",
    ("rgb", "low_contrast"):     "Global contrast reduction on RGB channel",
    ("rgb", "complete_dropout"): "RGB channel zeroed (complete sensor failure)",
    ("tir", "sensor_noise"):     "Additive Gaussian noise on TIR channel (thermal sensor noise)",
    ("tir", "blur"):             "Gaussian blur on TIR channel",
    ("tir", "intensity_shift"):  "Global intensity shift on TIR channel (thermal baseline drift)",
    ("tir", "complete_dropout"): "TIR channel zeroed (complete sensor failure)",
}

SEVERITY_PARAMS = {
    ("rgb", "gaussian_noise",   1): {"std": 0.08},
    ("rgb", "gaussian_noise",   2): {"std": 0.12},
    ("rgb", "gaussian_noise",   3): {"std": 0.18},
    ("rgb", "motion_blur",      1): {"radius": 10, "std": 3},
    ("rgb", "motion_blur",      2): {"radius": 15, "std": 5},
    ("rgb", "motion_blur",      3): {"radius": 15, "std": 8},
    ("rgb", "brightness_shift", 1): {"offset": 0.10},
    ("rgb", "brightness_shift", 2): {"offset": 0.20},
    ("rgb", "brightness_shift", 3): {"offset": 0.30},
    ("rgb", "low_contrast",     1): {"scale": 0.4},
    ("rgb", "low_contrast",     2): {"scale": 0.3},
    ("rgb", "low_contrast",     3): {"scale": 0.2},
    ("tir", "sensor_noise",     1): {"std": 0.15},
    ("tir", "sensor_noise",     2): {"std": 0.20},
    ("tir", "sensor_noise",     3): {"std": 0.35},
    ("tir", "blur",             1): {"gaussian_sigma": 1},
    ("tir", "blur",             2): {"gaussian_sigma": 2},
    ("tir", "blur",             3): {"gaussian_sigma": 3},
    ("tir", "intensity_shift",  1): {"offset": 0.10},
    ("tir", "intensity_shift",  2): {"offset": 0.20},
    ("tir", "intensity_shift",  3): {"offset": 0.30},
}


def build_metadata(all_conditions):
    meta = dict(METADATA)
    meta["n_conditions"] = len(all_conditions) + 1  # +1 for clean
    meta["conditions"]["clean"] = {
        "modality": None,
        "corruption_type": None,
        "severity": None,
        "description": "Unmodified original test images"
    }
    for mod, ctype, sev in all_conditions:
        name = condition_name(mod, ctype, sev)
        entry = {
            "modality": mod,
            "corruption_type": ctype,
            "severity": sev,
            "description": CONDITION_DESCRIPTIONS.get((mod, ctype), ""),
        }
        if sev is not None:
            entry["parameters"] = SEVERITY_PARAMS.get((mod, ctype, sev), {})
        meta["conditions"][name] = entry
    return meta


README_TEXT = """\
# DroneVehicle-C — Corrupted DroneVehicle Test Split

This dataset is a corrupted version of the DroneVehicle test split, generated
for benchmarking multimodal RGB+TIR fusion robustness in drone-based vehicle detection.

## Source

Original dataset: Sun et al., "Drone-based RGB-Infrared Cross-Modality Vehicle Detection
via Uncertainty-Aware Learning," IEEE TCSVT 2022.

Corruptions applied by: Saksham Singh Birla, University of Twente (TSCiT 2025 thesis).

## Structure

```
DroneVehicle_C/
  labels/                    # DOTA-format .txt annotations (same for all conditions)
  clean/rgb/  clean/ir/      # Original unmodified images
  rgb_gaussian_noise_s1/rgb/
  rgb_gaussian_noise_s1/ir/  # TIR unchanged
  ...                        # (24 condition directories in total)
  metadata.json              # Condition descriptions and parameters
  README.md
```

Each condition directory has:
- `rgb/` — RGB images (corrupted for rgb_* conditions, clean for tir_* conditions)
- `ir/`  — TIR images (clean for rgb_* conditions, corrupted for tir_* conditions)

## Conditions (23 total)

### RGB Corruptions (TIR clean)
- rgb_gaussian_noise_s1/s2/s3
- rgb_motion_blur_s1/s2/s3
- rgb_brightness_shift_s1/s2/s3
- rgb_low_contrast_s1/s2/s3
- rgb_complete_dropout

### TIR Corruptions (RGB clean)
- tir_sensor_noise_s1/s2/s3
- tir_blur_s1/s2/s3
- tir_intensity_shift_s1/s2/s3
- tir_complete_dropout

## Annotations

Labels are in DOTA oriented bounding box format:
  `x1 y1 x2 y2 x3 y3 x4 y4 class difficult`

Five vehicle classes: car, truck, freight_car, bus, van.
Image size: 840×712 pixels (100px white border each side, as in original DroneVehicle).

## Citation

If you use this dataset, please cite:
1. Sun et al. (IEEE TCSVT 2022) for the original DroneVehicle dataset.
2. Saksham Singh Birla (TSCiT 2025) for the corruption benchmark.

See metadata.json for full condition parameters.
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',
        default='/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle')
    parser.add_argument('--out-root',
        default='/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle_C')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    label_dir = data_root / 'test' / 'testMatchedLabel'
    img_dir   = data_root / 'test' / 'testMatchedImg'

    # Discover all image stems (labels are {id}_tir.txt → strip suffix to get {id})
    stems = sorted(
        p.stem[:-4] if p.stem.endswith('_tir') else p.stem
        for p in label_dir.glob('*.txt')
    )
    print(f'Found {len(stems)} test images.')

    # Write metadata and README
    out_root.mkdir(parents=True, exist_ok=True)
    meta = build_metadata(ALL_CONDITIONS)
    with open(out_root / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    (out_root / 'README.md').write_text(README_TEXT)
    print('Written metadata.json and README.md')

    # Copy labels
    out_labels = out_root / 'labels'
    out_labels.mkdir(exist_ok=True)
    for stem in tqdm(stems, desc='Copying labels'):
        src = label_dir / f'{stem}_tir.txt'   # actual label filename convention
        dst = out_labels / f'{stem}.txt'       # strip _tir in output for cleanliness
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    # ── Process conditions ──────────────────────────────────────────────────
    all_conds = [('clean', None, None, None)] + [
        (condition_name(m, c, s), m, c, s) for m, c, s in ALL_CONDITIONS
    ]

    for cond_name, mod, ctype, sev in all_conds:
        out_rgb_dir = out_root / cond_name / 'rgb'
        out_ir_dir  = out_root / cond_name / 'ir'
        out_rgb_dir.mkdir(parents=True, exist_ok=True)
        out_ir_dir.mkdir(parents=True, exist_ok=True)

        n_done   = sum(1 for s in stems if (out_rgb_dir / f'{s}.jpg').exists())
        n_remain = len(stems) - n_done
        if n_remain == 0:
            print(f'[SKIP] {cond_name} — already complete')
            continue

        desc = f'{cond_name} ({n_remain} remaining)'
        for stem in tqdm(stems, desc=desc, leave=False):
            out_rgb = out_rgb_dir / f'{stem}.jpg'
            out_ir  = out_ir_dir  / f'{stem}.jpg'
            if out_rgb.exists() and out_ir.exists():
                continue

            rgb_p, ir_p = _find_pair(img_dir, stem)
            if rgb_p is None:
                print(f'  WARNING: no pair for {stem}, skipping')
                continue

            # Load
            rgb_bgr = load_rgb(rgb_p)  # BGR for cv2
            tir_gray = load_tir(ir_p)

            # Apply corruption
            if mod is None:
                # clean — save as-is
                rgb_out  = rgb_bgr
                tir_out  = tir_gray
            elif mod == 'rgb':
                rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                corrupted_rgb = apply_corruption(rgb_rgb, 'rgb', ctype, sev)
                rgb_out  = cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR)
                tir_out  = tir_gray  # unchanged
            else:  # tir
                rgb_out  = rgb_bgr   # unchanged
                tir_out  = apply_corruption(tir_gray, 'tir', ctype, sev)

            cv2.imwrite(str(out_rgb), rgb_out)
            cv2.imwrite(str(out_ir),  tir_out)

        print(f'[DONE] {cond_name}')

    print(f'\nExport complete: {out_root}')
    total_images = len(stems) * len(all_conds) * 2  # ×2 for rgb+ir
    size_est_gb  = total_images * 200 / 1e6  # ~200KB per image
    print(f'Approx output: {total_images:,} images, ~{size_est_gb:.0f} GB')


if __name__ == '__main__':
    main()
