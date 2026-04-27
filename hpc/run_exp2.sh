#!/bin/bash
#SBATCH --job-name=exp2_modality_removal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

module load cuda
source activate thesis

cd $SLURM_SUBMIT_DIR
python experiments/exp2_modality_removal.py
