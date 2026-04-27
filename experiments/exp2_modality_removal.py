"""Experiment 2 — Modality removal (RQ2).

For each model x each of the 23 corruption conditions:
  - Run A: zero out RGB stream (TIR-only inference)
  - Run B: zero out TIR stream (RGB-only inference)
  Compare both against full-modality mAP from Exp 1 to isolate which sensor
  carries detection performance under each degradation.

Produces: results/exp2_modality_removal/{model}_{modality}_{corruption}_{severity}_{config}.json
  where config in {rgb_only, tir_only}
"""
import json
from pathlib import Path

from src.corruption.params import ALL_CONDITIONS

RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp2_modality_removal"
RESULTS_DIR.mkdir(exist_ok=True)


def run_modality_removal(model_name: str, runner,
                          test_rgb_dir: Path, test_tir_dir: Path, gt_dir: Path):
    from src.corruption.pipeline import apply_corruption
    from src.evaluation.dota_eval import evaluate_dota

    for modality, corruption_type, severity in ALL_CONDITIONS:
        base_key = f"{model_name}__{modality}__{corruption_type}__s{severity}"

        for config in ("rgb_only", "tir_only"):
            out_file = RESULTS_DIR / f"{base_key}__{config}.json"
            if out_file.exists():
                print(f"[SKIP] {base_key}__{config} already done")
                continue

            # TODO: corrupt the relevant stream, zero the other, run inference, eval mAP
            map_val = None  # replace with actual mAP
            result = {"model": model_name, "modality": modality,
                      "corruption": corruption_type, "severity": severity,
                      "config": config, "map": map_val}
            out_file.write_text(json.dumps(result, indent=2))
            print(f"[DONE] {base_key}__{config} | mAP={map_val:.4f}")


if __name__ == "__main__":
    raise NotImplementedError("Wire up model runners before running")
