"""Experiment 2 — Modality removal (RQ2).

For each model × each of the 23 corruption conditions × 2 modality configs:
  - rgb_only: corruption applied to relevant stream, TIR zeroed
  - tir_only: corruption applied to relevant stream, RGB zeroed

Comparing exp2 results against exp1 (full dual-modality) isolates which
sensor carries detection performance under each degradation type.

Produces:
  results/exp2_modality_removal/{model}__{modality}__{corruption}__s{sev}__{config}.json

Usage (run from project root, one model at a time):
    python experiments/exp2_modality_removal.py --model early_fusion
    PYTHONPATH=models/ua_cmddet:$PYTHONPATH python experiments/exp2_modality_removal.py --model ua_cmddet
"""

import argparse
import json
from pathlib import Path

import yaml

from src.corruption.params import ALL_CONDITIONS

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'exp2_modality_removal'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).parent.parent

# rgb_only → TIR is zeroed (only RGB stream active)
# tir_only → RGB is zeroed (only TIR stream active)
_ZERO_FOR_CONFIG = {'rgb_only': 'tir', 'tir_only': 'rgb'}


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


def run_modality_removal(model_name: str, runner):
    conditions = [(m, c, s, cfg)
                  for m, c, s in ALL_CONDITIONS
                  for cfg in ('rgb_only', 'tir_only')]
    total = len(conditions)

    for i, (modality, corruption_type, severity, config) in enumerate(conditions, 1):
        sev_str = str(severity) if severity is not None else 'dropout'
        key = f'{model_name}__{modality}__{corruption_type}__s{sev_str}__{config}'
        out_file = RESULTS_DIR / f'{key}.json'

        if out_file.exists():
            print(f'[SKIP] ({i}/{total}) {key}')
            continue

        zero_mod = _ZERO_FOR_CONFIG[config]
        print(f'[RUN ] ({i}/{total}) {key}...')
        mAP = runner.evaluate(
            corruption_type=corruption_type,
            modality=modality,
            severity=severity,
            zero_modality=zero_mod)

        result = {
            'model': model_name,
            'modality': modality,
            'corruption': corruption_type,
            'severity': severity,
            'config': config,
            'zeroed_stream': zero_mod,
            'map': mAP,
        }
        out_file.write_text(json.dumps(result, indent=2))
        print(f'[DONE] ({i}/{total}) {key} | mAP={mAP:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['early_fusion', 'c2former', 'ua_cmddet'])
    args = parser.parse_args()

    cfg_path = ROOT / 'experiments' / 'configs' / f'{args.model}.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    runner = build_runner(args.model, cfg)
    run_modality_removal(args.model, runner)


if __name__ == '__main__':
    main()
