#!/bin/bash
#SBATCH --job-name=early_fusion_test
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/early_fusion_test_%j.out
#SBATCH --error=logs/early_fusion_test_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
python models/c2former/tools/test.py \
    experiments/configs/early_fusion_dronevehicle.py \
    work_dirs/early_fusion/epoch_24.pth \
    --eval mAP
