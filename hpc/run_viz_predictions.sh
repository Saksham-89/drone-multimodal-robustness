#!/bin/bash
#SBATCH --job-name=viz_predictions
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_predictions_%j.log

PROJECT=/home/s3165582/thesis/drone-multimodal-robustness
cd $PROJECT

PYTHON=/home/s3165582/.conda/envs/thesis/bin/python

PYTHONPATH=$PROJECT $PYTHON scripts/visualize_predictions.py \
    --config  experiments/configs/c2former_dronevehicle.py \
    --ckpt    work_dirs/c2former/epoch_24.pth \
    --data-root $PROJECT/data/DroneVehicle \
    --indices 0 50 200 500 1000 2000 \
    --score-thr 0.35 \
    --out-dir figures/predictions
