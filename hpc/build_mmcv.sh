#!/bin/bash
#SBATCH --job-name=build_mmcv
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_mmcv_%j.out
#SBATCH --error=logs/build_mmcv_%j.err

source /etc/profile.d/modules.sh
module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

MMCV_WITH_OPS=1 pip install /home/s3165582/mmcv-build --no-build-isolation
echo "Done: mmcv build finished"
