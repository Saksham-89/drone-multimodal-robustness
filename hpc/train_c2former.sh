#!/bin/bash
#SBATCH --job-name=c2former_train
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/c2former_train_%j.out
#SBATCH --error=logs/c2former_train_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
conda activate thesis

cd $SLURM_SUBMIT_DIR
python models/c2former/train.py --config experiments/configs/c2former.yaml
