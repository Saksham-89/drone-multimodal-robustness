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
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness:$PYTHONPATH
python models/c2former/tools/train.py \
    experiments/configs/early_fusion_dronevehicle.py \
    --work-dir work_dirs/early_fusion
