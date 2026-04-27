"""Experiment 0 — Clean baseline validation.

Runs all three models on the unmodified DroneVehicle test split and saves mAP results.
mAP must match published figures for UA-CMDet and C2Former before proceeding.

Published baselines (DroneVehicle test split, mAP@0.5):
  UA-CMDet: verify against Sun et al. (TCSVT 2022) Table results
  C2Former:  verify against Yuan & Wei (TGRS 2024) Table results
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp0_baseline"
RESULTS_DIR.mkdir(exist_ok=True)


def run_baseline(model_name: str, runner, test_rgb_dir: Path, test_tir_dir: Path, gt_dir: Path):
    from src.evaluation.dota_eval import evaluate_dota
    # TODO: loop over test images, run inference, save detections, call evaluate_dota
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: instantiate runners and call run_baseline for each model
    raise NotImplementedError("Wire up model runners before running this script")
