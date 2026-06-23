#!/bin/bash
# Apply all reproducibility fixes to drone-multimodal-robustness.
# Run this from the repo root (the folder containing README.md, experiments/, hpc/).
#
#   bash apply_fixes.sh
#
# It is idempotent-ish: safe to read top to bottom before running.
# Nothing here changes any result VALUES; only the 8 orphan ssdropout files are removed.

set -e

# Sanity check we're in the right place
if [ ! -f README.md ] || [ ! -d experiments ] || [ ! -d hpc ]; then
    echo "ERROR: run this from the repo root (where README.md and hpc/ live)."
    exit 1
fi

echo "==> 1/11  HPC partitions: main-gpu -> ps,main-gpu, main-cpu -> ps,main-cpu"
for f in hpc/*.sh; do
    sed -i 's/#SBATCH --partition=main-gpu/#SBATCH --partition=ps,main-gpu/' "$f"
    sed -i 's/#SBATCH --partition=main-cpu/#SBATCH --partition=ps,main-cpu/' "$f"
done

echo "==> 2/11  Hardcoded PYTHONPATH/paths in core scripts -> \$SLURM_SUBMIT_DIR"
# Only the experiment/test/train scripts (not the peripheral upload/build utilities)
for f in \
    hpc/run_exp0_c2former.sh hpc/run_exp0_early_fusion.sh hpc/run_exp0_ua_cmddet.sh \
    hpc/run_exp1_c2former.sh hpc/run_exp1_early_fusion.sh hpc/run_exp1_ua_cmddet.sh \
    hpc/run_exp2_c2former.sh hpc/run_exp2_early_fusion.sh hpc/run_exp2_ua_cmddet.sh \
    hpc/test_c2former.sh hpc/test_early_fusion.sh hpc/test_ua_cmddet.sh \
    hpc/train_c2former.sh hpc/train_early_fusion.sh hpc/train_ua_cmddet.sh ; do
    sed -i 's#/home/s3165582/thesis/drone-multimodal-robustness#$SLURM_SUBMIT_DIR#g' "$f"
done

echo "==> 3/11  Config data_root -> env-driven with local default (both configs)"
for cfg in experiments/configs/early_fusion_dronevehicle.py experiments/configs/c2former_dronevehicle.py ; do
    python3 - "$cfg" << 'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()

# 3a. data_root: hardcoded -> env var with cwd-relative default
old_root = "data_root = '/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle/'"
new_root = ("import os\n"
            "data_root = os.environ.get(\n"
            "    'DRONEVEHICLE_ROOT',\n"
            "    os.path.join(os.getcwd(), 'data', 'DroneVehicle')\n"
            ") + '/'")
assert old_root in text, f"data_root pattern not found in {path}"
text = text.replace(old_root, new_root)

# 3b. pretrained weights path: hardcoded prefix -> os.getcwd()-based prefix.
#     The original uses implicit string concatenation across two lines:
#         pretrained='/home/.../'
#                    'pretrain_weights/resnet50.pth'
#     We replace only the first literal with an os.getcwd()-based expression,
#     keeping the implicit concatenation with the second literal intact.
#     (Only used during training; inference sets pretrained=None.)
old_pre = "pretrained='/home/s3165582/thesis/drone-multimodal-robustness/'"
new_pre = "pretrained=os.getcwd() + '/'"
assert old_pre in text, f"pretrained pattern not found in {path}"
text = text.replace(old_pre, new_pre)

open(path, 'w').write(text)
print(f"    patched {path}")
PYEOF
done

# 3c. Fix the remaining hardcoded path that lives in a COMMENT in early_fusion config
sed -i 's#PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness#PYTHONPATH=$PROJECT_ROOT#' \
    experiments/configs/early_fusion_dronevehicle.py

# Validate both configs still parse
for cfg in experiments/configs/early_fusion_dronevehicle.py experiments/configs/c2former_dronevehicle.py ; do
    python3 -c "import ast; ast.parse(open('$cfg').read())" && echo "    $cfg parses OK"
done

echo "==> 4/11  exp2 docstring: 29 -> 23 conditions"
sed -i 's/each of the 29 corruption conditions/each of the 23 corruption conditions/' \
    experiments/exp2_modality_removal.py

echo "==> 5/11  Rewrite broken generic scripts run_exp1.sh / run_exp2.sh"
cat > hpc/run_exp1.sh << 'SCRIPT'
#!/bin/bash
#SBATCH --job-name=exp1_corruption
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

# Runs the corruption benchmark for all three models in sequence.
# For a single model, use the model-specific scripts:
#   hpc/run_exp1_early_fusion.sh, hpc/run_exp1_c2former.sh, hpc/run_exp1_ua_cmddet.sh

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

for model in early_fusion c2former ua_cmddet; do
    echo "=== exp1: $model ==="
    python experiments/exp1_corruption.py --model "$model"
done
SCRIPT

cat > hpc/run_exp2.sh << 'SCRIPT'
#!/bin/bash
#SBATCH --job-name=exp2_modality
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

# Runs modality removal for all three models in sequence.
# For a single model, use the model-specific scripts:
#   hpc/run_exp2_early_fusion.sh, hpc/run_exp2_c2former.sh, hpc/run_exp2_ua_cmddet.sh

module load anaconda3/2025.06
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate thesis

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

for model in early_fusion c2former ua_cmddet; do
    echo "=== exp2: $model ==="
    python experiments/exp2_modality_removal.py --model "$model"
done
SCRIPT

echo "==> 6/11  README: fix C2Former clone URL"
sed -i 's#https://github.com/YuxiangTAT/C2Former#https://github.com/yuanmaoxun/C2Former#' README.md

echo "==> 7/11  README: fix low-contrast description"
sed -i 's/| Low Contrast | 0.4× | 0.3× | 0.2× |/| Low Contrast | scale 0.4 around mean | scale 0.3 around mean | scale 0.2 around mean |/' README.md

echo "==> 8/11  environment.yml: add opencv"
if ! grep -q "  - opencv" environment.yml; then
    sed -i 's/  - pillow/  - opencv                        # required by imagecorruptions (cv2 import) and image IO\n  - pillow/' environment.yml
fi
python3 -c "import yaml; yaml.safe_load(open('environment.yml'))" && echo "    environment.yml valid YAML"

echo "==> 9/11  README: document model-specific run scripts"
python3 - << 'PYEOF'
text = open('README.md').read()
old = """On the HPC cluster, use the SLURM scripts in `hpc/`:
```bash
sbatch hpc/run_exp1.sh
sbatch hpc/run_exp2.sh
```"""
new = """On the HPC cluster, use the SLURM scripts in `hpc/`. The generic scripts run all
three models in sequence:
```bash
sbatch hpc/run_exp1.sh
sbatch hpc/run_exp2.sh
```
To run a single model (recommended, so jobs are smaller and can be parallelised):
```bash
sbatch hpc/run_exp1_early_fusion.sh
sbatch hpc/run_exp1_c2former.sh
sbatch hpc/run_exp1_ua_cmddet.sh
```"""
if old in text:
    text = text.replace(old, new)
    open('README.md', 'w').write(text)
    print("    README run section updated")
else:
    print("    (run section already updated or not found, skipping)")
PYEOF

echo "==> 10/11  README: dataset download + prepare instructions"
python3 - << 'PYEOF'
text = open('README.md').read()
old = """**3. Download the dataset**

Download DroneVehicle from the link in the [UA-CMDet repo](https://github.com/SunYM2020/UA-CMDet) and place it under `data/DroneVehicle/`."""
new = """**3. Download and prepare the dataset**

Download the DroneVehicle dataset from HuggingFace (RGB + TIR pairs with OBB
annotations). The HuggingFace mirror works from most HPC clusters; Kaggle is
typically blocked on compute nodes.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \\
    snapshot_download(repo_id='McCheng/DroneVehicle', repo_type='dataset', \\
    local_dir='data/DroneVehicle', allow_patterns=['*test*'])"

# The download arrives as a zip per split. Unzip it:
cd data/DroneVehicle && unzip test.zip && cd ../..
```

The raw download has separate `testimg/` (TIR), `testimgr/` (RGB) and
`testlabel/` (XML) folders. The inference configs expect paired
`testMatchedImg/` and `testMatchedLabel/` directories instead. Generate them
with the bundled preparation script (symlinks images, converts XML to DOTA
TXT, so no data is duplicated):

```bash
python scripts/prepare_c2former_data.py --split test --data-root data/DroneVehicle
```

To run training as well, repeat the download and prepare steps for the
`train` and `val` splits.

By default the configs look for the dataset at `./data/DroneVehicle/`. To use
a different location, set `DRONEVEHICLE_ROOT`:

```bash
export DRONEVEHICLE_ROOT=/path/to/DroneVehicle
```"""
if old in text:
    text = text.replace(old, new)
    open('README.md', 'w').write(text)
    print("    README dataset section updated")
else:
    print("    (dataset section already updated or not found, skipping)")
PYEOF

echo "==> 11/11  Remove 8 orphan ssdropout result files"
if ls results/exp2_modality_removal/*ssdropout* >/dev/null 2>&1; then
    git rm -q results/exp2_modality_removal/*ssdropout*
    echo "    removed"
else
    echo "    (already removed, skipping)"
fi

echo ""
echo "All fixes applied. Review with:  git diff"
echo "Then:  git add -A && git commit -m 'Fix reproducibility' && git push origin <branch>"
