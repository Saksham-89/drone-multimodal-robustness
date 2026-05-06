#!/bin/bash
#SBATCH --job-name=exp1_c2former
#SBATCH --partition=main-gpu
#SBATCH --exclude=ctit085
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp1_c2former_%j.out
#SBATCH --error=logs/exp1_c2former_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness:$PYTHONPATH
python experiments/exp1_corruption.py --model c2former
