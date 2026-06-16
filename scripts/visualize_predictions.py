"""C2Former prediction visualisation — thesis qualitative figures.

For each selected test image, runs C2Former inference under 5 conditions
(clean + 4 corruptions) and saves a figure showing:
  - Left column:  RGB image with predicted OBBs overlaid
  - Right column: TIR image with predicted OBBs overlaid
  - One row per condition

Output per image: figures/predictions/{stem}_predictions.png

Run from project root on HPC (GPU required):
    PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness \\
    python scripts/visualize_predictions.py \\
        --config  experiments/configs/c2former_dronevehicle.py \\
        --ckpt    work_dirs/c2former/epoch_24.pth \\
        --indices 0 50 200 500 1000 \\
        --score-thr 0.35 \\
        --out-dir figures/predictions
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import mmcv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.transforms  # registers ApplyCorruption + ZeroModality  # noqa: F401
from src.corruption.pipeline import apply_corruption

# Patch for PyTorch >= 2.0 / older mmcv: _get_stream expects torch.device not int
import torch.nn.parallel._functions as _torch_pf
_orig_get_stream = _torch_pf._get_stream
def _patched_get_stream(device):
    if isinstance(device, int):
        device = torch.device('cuda', device)
    return _orig_get_stream(device)
_torch_pf._get_stream = _patched_get_stream


# ── Class metadata ─────────────────────────────────────────────────────────────

CLASSES = ['car', 'truck', 'freight_car', 'bus', 'van']

# BGR for cv2, RGB for matplotlib — stored as RGB here
CLASS_COLORS = {
    'car':         (0,   200,  50),
    'truck':       (255, 140,   0),
    'freight_car': (160,  32, 240),
    'bus':         (220,  20,  60),
    'van':         (255, 215,   0),
}


# ── OBB drawing ────────────────────────────────────────────────────────────────

def obb_corners(cx, cy, w, h, angle_deg):
    """Return (4, 2) int32 corner array for an oriented bounding box."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    hw, hh = w / 2.0, h / 2.0
    offsets = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = offsets @ rot.T + np.array([cx, cy])
    return corners.astype(np.int32)


def draw_detections(img_rgb, detections, score_thr: float):
    """
    Draw OBBs on a copy of img_rgb (H×W×3 uint8).

    detections: list of 5 arrays (one per class), each (N,6): cx,cy,w,h,angle,score
    """
    canvas = img_rgb.copy()
    for cls_idx, dets in enumerate(detections):
        if len(dets) == 0:
            continue
        cls_name = CLASSES[cls_idx]
        color_rgb = CLASS_COLORS[cls_name]
        color_bgr = color_rgb[::-1]  # cv2 uses BGR

        for cx, cy, w, h, angle, score in dets:
            if score < score_thr:
                continue
            corners = obb_corners(cx, cy, w, h, angle)
            cv2.polylines(canvas, [corners.reshape(-1, 1, 2)],
                          isClosed=True, color=color_bgr, thickness=2)
            # Label above the top-left corner
            top = corners[np.argmin(corners[:, 1])]
            label = f'{cls_name} {score:.2f}'
            font_scale = 0.45
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx, ty = int(top[0]), int(top[1]) - 4
            cv2.rectangle(canvas, (tx, ty - th - 2), (tx + tw, ty + 2),
                          color_bgr, -1)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return canvas


# ── Image loading ──────────────────────────────────────────────────────────────

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


def load_raw_pair(img_dir: Path, stem: str):
    """Load (rgb uint8 H×W×3, tir uint8 H×W) for display."""
    rgb_p, ir_p = _find_pair(img_dir, stem)
    if rgb_p is None:
        return None, None
    rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
    tir = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE)
    return rgb, tir


# ── Inference ──────────────────────────────────────────────────────────────────

def build_model(config_path: str, ckpt_path: str, device_id: int = 0):
    from mmrotate.models import build_detector
    from mmcv.runner import load_checkpoint
    from mmcv.parallel import MMDataParallel

    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.cfg = cfg
    model = model.cuda(device_id)
    model.eval()
    return MMDataParallel(model, device_ids=[device_id])


def run_inference(wrapped_model, cfg, selected_indices: list,
                  corruption_type=None, modality=None, severity=None,
                  zero_modality=None):
    """
    Run the model over the test set; collect per-image detections for
    the requested image indices only.

    Returns dict: {dataset_index: detections}
    """
    from mmdet.datasets import build_dataset, build_dataloader

    test_cfg = mmcv.Config.fromfile(cfg)
    test_cfg.data.test.test_mode = True

    pos = 1  # inject after LoadPairedImageFromFile, before MultiScaleFlipAug
    if corruption_type is not None:
        test_cfg.data.test.pipeline.insert(pos, dict(
            type='ApplyCorruption',
            modality=modality,
            corruption_type=corruption_type,
            severity=severity))
        pos += 1
    if zero_modality is not None:
        test_cfg.data.test.pipeline.insert(pos, dict(
            type='ZeroModality', modality=zero_modality))

    dataset = build_dataset(test_cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2,
        dist=False, shuffle=False)

    target_set = set(selected_indices)
    collected = {}
    for i, data in enumerate(data_loader):
        if i in target_set:
            with torch.no_grad():
                result = wrapped_model(return_loss=False, rescale=True, **data)
            collected[i] = result[0]
        if len(collected) == len(target_set):
            break

    img_infos = dataset.img_infos
    return collected, img_infos


# ── Figure builder ─────────────────────────────────────────────────────────────

CONDITIONS = [
    ('Clean',                      None,   None,              None,   None),
    ('RGB Gaussian Noise S3',      'rgb',  'gaussian_noise',  3,      None),
    ('TIR Sensor Noise S3',        'tir',  'sensor_noise',    3,      None),
    ('RGB Dropout (TIR only)',     None,   None,              None,   'rgb'),
    ('TIR Dropout (RGB only)',     None,   None,              None,   'tir'),
]


def make_prediction_figure(stem, rgb_raw, tir_raw,
                           results_by_condition: dict,
                           score_thr: float, out_path: Path):
    """
    Grid: 5 rows (one per condition) × 2 cols (RGB | TIR).
    OBBs from the model are drawn on the raw image.
    """
    n_conds = len(CONDITIONS)
    fig, axes = plt.subplots(n_conds, 2,
                             figsize=(12, n_conds * 3.5),
                             squeeze=False)

    tir_display = cv2.cvtColor(tir_raw, cv2.COLOR_GRAY2RGB)

    for row, (cond_label, *_) in enumerate(CONDITIONS):
        detections = results_by_condition[cond_label]

        # Determine which modality was corrupted to decide what to show
        _, mod, ctype, sev, zero_mod = CONDITIONS[row]

        rgb_show = rgb_raw.copy()
        tir_show = tir_display.copy()

        if mod == 'rgb' and ctype:
            corrupted = apply_corruption(rgb_raw.copy(), 'rgb', ctype, sev)
            rgb_show = corrupted
        elif mod == 'tir' and ctype:
            corrupted = apply_corruption(tir_raw.copy(), 'tir', ctype, sev)
            tir_show = cv2.cvtColor(corrupted, cv2.COLOR_GRAY2RGB)

        if zero_mod == 'rgb':
            rgb_show = np.zeros_like(rgb_raw)
        elif zero_mod == 'tir':
            tir_show = np.zeros_like(tir_display)

        rgb_drawn = draw_detections(rgb_show, detections, score_thr)
        tir_drawn = draw_detections(tir_show, detections, score_thr)

        axes[row, 0].imshow(rgb_drawn)
        axes[row, 1].imshow(tir_drawn)

        n_dets = sum(np.sum(d[:, -1] >= score_thr) for d in detections if len(d))
        axes[row, 0].set_ylabel(f'{cond_label}\n({n_dets} dets)',
                                fontsize=9, rotation=90, labelpad=6, va='center')

        for col in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_title('RGB Channel', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('TIR Channel', fontsize=11, fontweight='bold')

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1,
                              facecolor=np.array(CLASS_COLORS[c]) / 255,
                              label=c)
               for c in CLASSES]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=9, framealpha=0.9)

    fig.suptitle(f'C2Former Predictions — Image {stem}  (score ≥ {score_thr})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'  Saved {out_path}')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
        default='experiments/configs/c2former_dronevehicle.py')
    parser.add_argument('--ckpt',
        default='work_dirs/c2former/epoch_24.pth')
    parser.add_argument('--data-root',
        default='/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle')
    parser.add_argument('--indices', type=int, nargs='+',
        default=[0, 50, 200, 500, 1000],
        help='Test set indices to visualise')
    parser.add_argument('--score-thr', type=float, default=0.35,
        help='Minimum confidence score to display (default: 0.35)')
    parser.add_argument('--out-dir', default='figures/predictions')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    img_dir   = data_root / 'test' / 'testMatchedImg'
    out_dir   = Path(args.out_dir)

    print('Loading C2Former...')
    model = build_model(args.config, args.ckpt, args.device)

    # Run inference for each condition separately
    results_per_cond = {}
    for cond_label, mod, ctype, sev, zero_mod in CONDITIONS:
        print(f'Running inference: {cond_label}...')
        collected, img_infos = run_inference(
            model, args.config, args.indices,
            corruption_type=ctype, modality=mod, severity=sev,
            zero_modality=zero_mod)
        results_per_cond[cond_label] = collected

    # Build per-image figures
    for idx in args.indices:
        img_info = img_infos[idx]
        # img_info['filename'] is the RGB filename within img_prefix
        stem = Path(img_info['filename']).stem
        # Strip _ir suffix if present (some datasets include both in filename)
        if stem.endswith('_ir'):
            stem = stem[:-3]

        print(f'\nImage {idx}: {stem}')
        rgb_raw, tir_raw = load_raw_pair(img_dir, stem)
        if rgb_raw is None:
            print(f'  WARNING: could not load raw image for stem={stem}, skipping.')
            continue

        results_by_condition = {
            cond_label: results_per_cond[cond_label].get(idx, [np.zeros((0, 6))] * 5)
            for cond_label, *_ in CONDITIONS
        }

        make_prediction_figure(
            stem, rgb_raw, tir_raw,
            results_by_condition,
            args.score_thr,
            out_dir / f'{stem}_predictions.png')

    print(f'\nDone. Figures saved to {out_dir}/')
    print('Tip: adjust --score-thr (try 0.25–0.50) and --indices to find interesting scenes.')


if __name__ == '__main__':
    main()
