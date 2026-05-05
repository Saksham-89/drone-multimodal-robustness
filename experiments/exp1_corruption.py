"""Experiment 1 — Corruption benchmark (RQ1).

For each model × each of the 29 corruption conditions:
  - Apply corruption to the relevant modality stream
  - Run full dual-modality inference on the test split
  - Save mAP and RA = corrupted_mAP / clean_mAP

Produces: results/exp1_corruption/{model}__{modality}__{corruption}__s{severity}.json

Usage (run from project root, one model at a time):
    python experiments/exp1_corruption.py --model early_fusion
    PYTHONPATH=models/ua_cmddet:$PYTHONPATH python experiments/exp1_corruption.py --model ua_cmddet
"""

import argparse
import json
from pathlib import Path

import yaml

from src.corruption.params import ALL_CONDITIONS
from src.evaluation.metrics import compute_ra

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'exp1_corruption'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).parent.parent


def build_runner(model_name: str, cfg: dict):
    mmdet_config = str(ROOT / cfg['mmdet_config'])
    checkpoint = str(ROOT / cfg['checkpoint'])

    if model_name == 'early_fusion':
        from src.inference.early_fusion import EarlyFusionRunner
        runner = EarlyFusionRunner(config_path=mmdet_config)
    elif model_name == 'c2former':
        from src.inference.c2former import C2FormerRunner
        runner = C2FormerRunner(config_path=mmdet_config)
    elif model_name == 'ua_cmddet':
        from src.inference.ua_cmddet import UACMDetRunner
        runner = UACMDetRunner(config_path=mmdet_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    runner.load_model(checkpoint)
    return runner


def load_clean_map(model_name: str) -> float:
    baseline_file = ROOT / 'results' / 'exp0_baseline' / f'{model_name}.json'
    if not baseline_file.exists():
        raise FileNotFoundError(
            f"Clean baseline not found for {model_name}. "
            "Run exp0_baseline.py first.")
    return json.loads(baseline_file.read_text())['map']


def run_corruption_benchmark(model_name: str, runner, clean_map: float):
    total = len(ALL_CONDITIONS)
    for i, (modality, corruption_type, severity) in enumerate(ALL_CONDITIONS, 1):
        sev_str = str(severity) if severity is not None else 'dropout'
        key = f'{model_name}__{modality}__{corruption_type}__s{sev_str}'
        out_file = RESULTS_DIR / f'{key}.json'

        if out_file.exists():
            print(f'[SKIP] ({i}/{total}) {key}')
            continue

        print(f'[RUN ] ({i}/{total}) {key}...')
        corrupted_map = runner.evaluate(
            corruption_type=corruption_type,
            modality=modality,
            severity=severity)

        ra = compute_ra(corrupted_map, clean_map)
        result = {
            'model': model_name,
            'modality': modality,
            'corruption': corruption_type,
            'severity': severity,
            'map': corrupted_map,
            'clean_map': clean_map,
            'ra': ra,
        }
        out_file.write_text(json.dumps(result, indent=2))
        print(f'[DONE] ({i}/{total}) {key} | mAP={corrupted_map:.4f} | RA={ra:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['early_fusion', 'c2former', 'ua_cmddet'])
    args = parser.parse_args()

    cfg_path = ROOT / 'experiments' / 'configs' / f'{args.model}.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    runner = build_runner(args.model, cfg)
    clean_map = load_clean_map(args.model)
    print(f'Clean baseline mAP for {args.model}: {clean_map:.4f}')
    run_corruption_benchmark(args.model, runner, clean_map)


if __name__ == '__main__':
    main()
