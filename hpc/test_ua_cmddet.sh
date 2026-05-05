#!/bin/bash
#SBATCH --job-name=ua_cmddet_test
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ua_cmddet_test_%j.out
#SBATCH --error=logs/ua_cmddet_test_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness/models/ua_cmddet:$PYTHONPATH

# Step 1: run inference, write predictions to pickle
python models/ua_cmddet/tools/test.py \
    models/ua_cmddet/configs/DroneVehicle/UACMDet.py \
    work_dirs/ua_cmddet/latest.pth \
    --out work_dirs/ua_cmddet/test_results.pkl

# Step 2: evaluate predictions with DroneVehicle eval (polyiou-based mAP)
python models/ua_cmddet/eval/DroneVehicleEval.py \
    --result work_dirs/ua_cmddet/test_results.pkl \
    --ann /home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle/test/testMatchedLabel
