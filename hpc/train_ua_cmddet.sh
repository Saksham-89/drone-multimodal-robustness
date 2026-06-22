#!/bin/bash
#SBATCH --job-name=ua_cmddet_train
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ua_cmddet_train_%j.out
#SBATCH --error=logs/ua_cmddet_train_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR/models/ua_cmddet:$PYTHONPATH
python models/ua_cmddet/tools/train.py models/ua_cmddet/configs/DroneVehicle/UACMDet.py --work_dir work_dirs/ua_cmddet
