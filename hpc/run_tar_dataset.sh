#!/bin/bash
#SBATCH --job-name=tar_dataset
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/tar_dataset_%j.log

DATA=/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle_C
OUT=/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle_C_tars

mkdir -p $OUT

# Copy flat files
cp $DATA/metadata.json $OUT/
cp $DATA/README.md $OUT/

# Tar labels once
echo "Tarring labels..."
tar -czf $OUT/labels.tar.gz -C $DATA labels/

# Tar each condition directory
for dir in $DATA/*/; do
    name=$(basename "$dir")
    out_file=$OUT/${name}.tar.gz
    if [ -f "$out_file" ]; then
        echo "[SKIP] $name"
        continue
    fi
    echo "Tarring $name..."
    tar -czf "$out_file" -C $DATA "$name/"
    echo "[DONE] $name"
done

echo "All tars complete: $OUT"
ls -lh $OUT
