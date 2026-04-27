#!/bin/bash
#SBATCH --job-name=ua_cmddet_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ua_cmddet_train_%j.out
#SBATCH --error=logs/ua_cmddet_train_%j.err

module load cuda
source activate thesis  # update with your conda env name

cd $SLURM_SUBMIT_DIR
python models/ua_cmddet/train.py --config experiments/configs/ua_cmddet.yaml
