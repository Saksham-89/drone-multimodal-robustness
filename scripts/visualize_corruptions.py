"""Corruption visualisation — thesis figures.

Creates three output figures in --out-dir:
  rgb_corruptions.png   — rows=4 RGB types, cols=clean/S1/S2/S3  (shows RGB channel)
  tir_corruptions.png   — rows=3 TIR types, cols=clean/S1/S2/S3  (shows TIR channel)
  overview.png          — side-by-side RGB+TIR for 6 representative conditions

Run from project root on HPC (no GPU needed):
    PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness \\
    python scripts/visualize_corruptions.py \\
        --data-root /home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle \\
        --out-dir figures/corruption_viz
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.corruption.pipeline import apply_corruption


# ── Image loading ──────────────────────────────────────────────────────────────

def _find_pair(img_dir: Path, stem: str):
    """Try known DroneVehicle naming conventions. Returns (rgb_path, ir_path) or (None, None)."""
    candidates = [
        (img_dir / f'{stem}.jpg',          img_dir / f'{stem}_tir.jpg'),   # actual convention
        (img_dir / f'{stem}.jpg',          img_dir / f'{stem}_ir.jpg'),
        (img_dir / 'rgb' / f'{stem}.jpg',  img_dir / 'ir' / f'{stem}.jpg'),
        (img_dir / f'{stem}.png',          img_dir / f'{stem}_tir.png'),
    ]
    for rgb_p, ir_p in candidates:
        if rgb_p.exists() and ir_p.exists():
            return rgb_p, ir_p
    return None, None


def load_pairs(data_root: Path, n: int):
    """Return up to n (stem, rgb_uint8, tir_uint8) tuples from the test split."""
    label_dir = data_root / 'test' / 'testMatchedLabel'
    img_dir   = data_root / 'test' / 'testMatchedImg'
    pairs = []
    for label_file in sorted(label_dir.glob('*.txt')):
        # Labels are named {id}_tir.txt; image stem is just {id}
        raw_stem = label_file.stem
        stem = raw_stem[:-4] if raw_stem.endswith('_tir') else raw_stem
        rgb_p, ir_p = _find_pair(img_dir, stem)
        if rgb_p is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
        tir = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE)
        pairs.append((stem, rgb, tir))
        if len(pairs) == n:
            break
    if not pairs:
        sys.exit(f'ERROR: no image pairs found in {img_dir}. Check naming convention.')
    return pairs


# ── Figure builders ────────────────────────────────────────────────────────────

def _severity_label(sev):
    return 'Dropout' if sev is None else f'S{sev}'


def make_modality_grid(rgb, tir, modality: str, conditions: list, out_path: Path):
    """
    Grid: rows = corruption types, cols = Clean / S1 / S2 / S3 (or Dropout).
    Each cell shows only the affected channel.
    """
    assert modality in ('rgb', 'tir')
    src = rgb if modality == 'rgb' else tir
    cmap = None if modality == 'rgb' else 'inferno'

    rows = []
    for ctype, severities in conditions:
        images = [('Clean', src.copy())]
        for sev in severities:
            corrupted = apply_corruption(src.copy(), modality, ctype, sev)
            images.append((_severity_label(sev), corrupted))
        rows.append((ctype.replace('_', ' ').title(), images))

    n_rows = len(rows)
    n_cols = max(len(r[1]) for r in rows)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 2.8),
                             squeeze=False)

    for r_idx, (row_label, images) in enumerate(rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if c_idx < len(images):
                col_label, img = images[c_idx]
                ax.imshow(img, cmap=cmap)
                if r_idx == 0:
                    ax.set_title(col_label, fontsize=11, fontweight='bold', pad=6)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, rotation=90,
                              labelpad=8, va='center')

    modality_str = 'RGB' if modality == 'rgb' else 'TIR'
    fig.suptitle(f'{modality_str} Corruption Benchmark — DroneVehicle Test Image',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close(fig)


def make_overview(rgb, tir, out_path: Path):
    """
    2-row × 6-col overview: RGB row and TIR row across 6 representative conditions.
    Useful as a thesis intro figure.
    """
    conditions = [
        ('Clean',               None,  None,              None),
        ('RGB Noise S3',        'rgb', 'gaussian_noise',  3),
        ('RGB Low Contrast S3', 'rgb', 'low_contrast',    3),
        ('RGB Dropout',         'rgb', 'complete_dropout',None),
        ('TIR Noise S3',        'tir', 'sensor_noise',    3),
        ('TIR Dropout',         'tir', 'complete_dropout',None),
    ]

    fig, axes = plt.subplots(2, len(conditions),
                             figsize=(len(conditions) * 3.0, 6.2),
                             squeeze=False)

    for col, (title, mod, ctype, sev) in enumerate(conditions):
        rgb_show = rgb.copy()
        tir_show = tir.copy()
        if mod == 'rgb' and ctype:
            rgb_show = apply_corruption(rgb_show, 'rgb', ctype, sev)
        elif mod == 'tir' and ctype:
            tir_show = apply_corruption(tir_show, 'tir', ctype, sev)

        axes[0, col].imshow(rgb_show)
        axes[0, col].set_title(title, fontsize=9, fontweight='bold', pad=5)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        axes[1, col].imshow(tir_show, cmap='inferno')
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel('RGB', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('TIR', fontsize=12, fontweight='bold')

    fig.suptitle('DroneVehicle Sensor Corruption Examples', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

RGB_CONDITIONS = [
    ('gaussian_noise',   [1, 2, 3]),
    ('motion_blur',      [1, 2, 3]),
    ('brightness_shift', [1, 2, 3]),
    ('low_contrast',     [1, 2, 3]),
    ('complete_dropout', [None]),
]

TIR_CONDITIONS = [
    ('sensor_noise',    [1, 2, 3]),
    ('blur',            [1, 2, 3]),
    ('intensity_shift', [1, 2, 3]),
    ('complete_dropout',[None]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',
        default='/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle')
    parser.add_argument('--out-dir', default='figures/corruption_viz')
    parser.add_argument('--image-index', type=int, default=0,
        help='Index of the test image to visualise (default: 0 = first)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)

    pairs = load_pairs(data_root, n=args.image_index + 1)
    stem, rgb, tir = pairs[args.image_index]
    print(f'Using image: {stem}  (RGB {rgb.shape}, TIR {tir.shape})')

    make_modality_grid(rgb, tir, 'rgb', RGB_CONDITIONS,
                       out_dir / 'rgb_corruptions.png')
    make_modality_grid(rgb, tir, 'tir', TIR_CONDITIONS,
                       out_dir / 'tir_corruptions.png')
    make_overview(rgb, tir, out_dir / 'overview.png')

    print(f'\nDone. Figures in: {out_dir}/')
    print('Tip: if the image looks uninteresting, try --image-index 50 or 100.')


if __name__ == '__main__':
    main()
