"""Experiment 0 — Clean baseline validation.

Runs all three models on the unmodified DroneVehicle test split and saves mAP.
Must be run before any corruption experiments. mAP should match (within ~1%)
published figures for UA-CMDet and C2Former before proceeding to Exp 1/2.

Usage (run from project root, one model at a time):
    # Early Fusion / C2Former — use system mmrotate env
    python experiments/exp0_baseline.py --model early_fusion
    python experiments/exp0_baseline.py --model c2former

    # UA-CMDet — must have UA-CMDet's mmdet first on PYTHONPATH
    PYTHONPATH=models/ua_cmddet:$PYTHONPATH python experiments/exp0_baseline.py --model ua_cmddet

Published baselines (DroneVehicle test split, mAP@0.5):
    UA-CMDet : ~0.412  (Sun et al., TCSVT 2022)
    C2Former  : see Yuan & Wei (TGRS 2024) Table results
    Early Fusion: 0.485 (our run, job 491917)
"""

import argparse
import json
from pathlib import Path

import yaml

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'exp0_baseline'
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


def run_baseline(model_name: str, runner) -> float:
    out_file = RESULTS_DIR / f'{model_name}.json'
    if out_file.exists():
        result = json.loads(out_file.read_text())
        print(f'[SKIP] {model_name} already done -- mAP={result["map"]:.4f}')
        return result['map']

    print(f'[RUN ] {model_name} clean baseline...')
    mAP = runner.evaluate()

    result = {'model': model_name, 'map': mAP, 'split': 'test', 'corruption': 'none'}
    out_file.write_text(json.dumps(result, indent=2))
    print(f'[DONE] {model_name} | clean mAP={mAP:.4f}')
    return mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['early_fusion', 'c2former', 'ua_cmddet'])
    args = parser.parse_args()

    cfg_path = ROOT / 'experiments' / 'configs' / f'{args.model}.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    runner = build_runner(args.model, cfg)
    run_baseline(args.model, runner)


if __name__ == '__main__':
    main()
