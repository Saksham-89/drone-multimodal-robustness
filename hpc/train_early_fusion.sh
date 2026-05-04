#!/bin/bash
#SBATCH --job-name=early_fusion_train
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/early_fusion_train_%j.out
#SBATCH --error=logs/early_fusion_train_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
conda activate thesis

cd $SLURM_SUBMIT_DIR
python experiments/train_early_fusion.py --config experiments/configs/early_fusion.yaml
