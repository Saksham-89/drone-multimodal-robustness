#!/bin/bash
#SBATCH --job-name=build_mmcv
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_mmcv_%j.out
#SBATCH --error=logs/build_mmcv_%j.err

source /etc/profile.d/modules.sh
module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

# Compile for all GPU families present on this cluster:
#   7.0 = V100 (Volta), 7.5 = T4/RTX20xx (Turing),
#   8.0 = A100 (Ampere), 8.6 = RTX30xx/A40 (Ampere GA102)
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
MMCV_WITH_OPS=1 pip install /home/s3165582/mmcv-build --no-build-isolation --no-cache-dir
echo "Done: mmcv build finished"
python -c "from mmcv.ops import DeformConv2d; print('deform_conv import OK')"
