#!/bin/bash
#SBATCH --job-name=exp0_early_fusion
#SBATCH --partition=ps,main-gpu
#SBATCH --exclude=ctit084,ctit085
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp0_early_fusion_%j.out
#SBATCH --error=logs/exp0_early_fusion_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
python experiments/exp0_baseline.py --model early_fusion
