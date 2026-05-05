"""mAP evaluation for oriented bounding box detection.

For Early Fusion and C2Former (mmrotate-based), evaluation is handled
internally by mmrotate's DroneVehicleDataset.evaluate() — this wrapper
is not needed for those models.

This module is kept for UA-CMDet compatibility and as a standalone
fallback that calls the DOTA devkit directly.
"""

import subprocess
import tempfile
from pathlib import Path


def evaluate_dota(det_path: Path, gt_path: Path, iou_threshold: float = 0.5) -> float:
    """Run DOTA OBB evaluation and return mAP@IoU.

    det_path: directory of per-class detection txt files
              format per line: <img_id> <score> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    gt_path:  directory of per-image ground-truth txt files (DOTA format)
    Returns mAP as a float in [0, 1].
    """
    devkit = Path(__file__).parents[2] / 'models' / 'dota_devkit'
    eval_script = devkit / 'dota_evaluation_task1.py'
    if not eval_script.exists():
        raise FileNotFoundError(
            f"DOTA devkit not found at {devkit}. "
            "Clone it and build the polyiou extension first.")

    result = subprocess.run(
        ['python', str(eval_script),
         '--detpath', str(det_path),
         '--annopath', str(gt_path),
         '--imagesetfile', str(gt_path / 'imagesetfile.txt')],
        capture_output=True, text=True, check=True)

    # Parse mAP from stdout — format: "mAP: 0.4850"
    for line in result.stdout.splitlines():
        if 'mAP' in line:
            try:
                return float(line.split(':')[-1].strip())
            except ValueError:
                pass

    raise ValueError(
        f"Could not parse mAP from DOTA devkit output:\n{result.stdout}")
