#!/bin/bash
#SBATCH --job-name=exp1_ua_cmddet
#SBATCH --partition=ps,main-gpu
#SBATCH --exclude=ctit085
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp1_ua_cmddet_%j.out
#SBATCH --error=logs/exp1_ua_cmddet_%j.err

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR/models/ua_cmddet:$SLURM_SUBMIT_DIR:$PYTHONPATH
python experiments/exp1_corruption.py --model ua_cmddet
