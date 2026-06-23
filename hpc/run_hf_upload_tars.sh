#!/bin/bash
#SBATCH --job-name=hf_upload_tars
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/hf_upload_tars_%j.log

PYTHON=/home/s3165582/.conda/envs/thesis/bin/python

$PYTHON -c "
from huggingface_hub import HfApi
HfApi().upload_large_folder(
    repo_id='Saksham224/DroneVehicle-C',
    repo_type='dataset',
    folder_path='/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle_C_tars',
    num_workers=2,
)
"
