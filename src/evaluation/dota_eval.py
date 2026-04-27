"""Thin wrapper around the DOTA devkit mAP evaluation.

Assumes DOTA_devkit is installed or available on sys.path.
Call after cloning: https://github.com/CAPTAIN-WHU/DOTA_devkit
"""
import subprocess
from pathlib import Path


def evaluate_dota(det_path: Path, gt_path: Path, iou_threshold: float = 0.5) -> float:
    """Run DOTA OBB evaluation and return mAP@IoU.

    det_path: directory of per-class detection result txt files
    gt_path:  directory of per-image ground-truth txt files
    Returns mAP as a float in [0, 1].
    """
    # Placeholder — replace with actual DOTA devkit API call once the toolkit is cloned.
    raise NotImplementedError("Wire up DOTA devkit after cloning the repo")
