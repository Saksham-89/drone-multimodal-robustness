#!/bin/bash
#SBATCH --job-name=exp1_corruption
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

module load cuda
source activate thesis

cd $SLURM_SUBMIT_DIR
python experiments/exp1_corruption.py
