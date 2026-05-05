#!/bin/bash
#SBATCH --job-name=c2former_test
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/c2former_test_%j.out
#SBATCH --error=logs/c2former_test_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
python models/c2former/tools/test.py \
    experiments/configs/c2former_dronevehicle.py \
    work_dirs/c2former/epoch_24.pth \
    --eval mAP
