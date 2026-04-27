"""Experiment 1 — Corruption benchmark (RQ1).

For each model x each of the 23 corruption conditions:
  - Apply corruption to the relevant modality stream
  - Run full dual-modality inference
  - Save mAP and compute RA = corrupted_mAP / clean_mAP

Produces: results/exp1_corruption/{model}_{modality}_{corruption}_{severity}.json
"""
import json
from pathlib import Path

from src.corruption.params import ALL_CONDITIONS
from src.evaluation.metrics import compute_ra

RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp1_corruption"
RESULTS_DIR.mkdir(exist_ok=True)


def run_corruption_benchmark(model_name: str, runner, clean_map: float,
                              test_rgb_dir: Path, test_tir_dir: Path, gt_dir: Path):
    from src.corruption.pipeline import apply_corruption
    from src.evaluation.dota_eval import evaluate_dota

    for modality, corruption_type, severity in ALL_CONDITIONS:
        key = f"{model_name}__{modality}__{corruption_type}__s{severity}"
        out_file = RESULTS_DIR / f"{key}.json"
        if out_file.exists():
            print(f"[SKIP] {key} already done")
            continue

        # TODO: loop test images, corrupt the relevant stream, run inference, eval mAP
        corrupted_map = None  # replace with actual mAP
        ra = compute_ra(corrupted_map, clean_map)

        result = {"model": model_name, "modality": modality,
                  "corruption": corruption_type, "severity": severity,
                  "map": corrupted_map, "ra": ra}
        out_file.write_text(json.dumps(result, indent=2))
        print(f"[DONE] {key} | mAP={corrupted_map:.4f} | RA={ra:.4f}")


if __name__ == "__main__":
    raise NotImplementedError("Wire up model runners and clean_map before running")
