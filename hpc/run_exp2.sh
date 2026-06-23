#!/bin/bash
#SBATCH --job-name=exp2_modality
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

# Runs modality removal for all three models in sequence.
# For a single model, use the model-specific scripts:
#   hpc/run_exp2_early_fusion.sh, hpc/run_exp2_c2former.sh, hpc/run_exp2_ua_cmddet.sh

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

for model in early_fusion c2former ua_cmddet; do
    echo "=== exp2: $model ==="
    python experiments/exp2_modality_removal.py --model "$model"
done
