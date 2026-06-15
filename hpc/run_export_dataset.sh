#!/bin/bash
#SBATCH --job-name=export_corrupted
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/export_corrupted_%j.log

PROJECT=/home/s3165582/thesis/drone-multimodal-robustness
cd $PROJECT

PYTHONPATH=$PROJECT python scripts/export_corrupted_dataset.py \
    --data-root $PROJECT/data/DroneVehicle \
    --out-root  $PROJECT/data/DroneVehicle_C
