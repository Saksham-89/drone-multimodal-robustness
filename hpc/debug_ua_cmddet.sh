#!/bin/bash
#SBATCH --job-name=dbg_ua_cmddet
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/debug_ua_cmddet_%j.out
#SBATCH --error=logs/debug_ua_cmddet_%j.err

cd ~/thesis/drone-multimodal-robustness
conda activate thesis
PYTHONPATH=models/ua_cmddet:$PYTHONPATH python scripts/debug_ua_cmddet_result.py
