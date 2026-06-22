# Evaluating Multimodal Fusion Robustness in Drone-Based Object Detection Under Sensor Degradation

> **TSCiT 2025** · University of Twente · Saksham Singh Birla

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-DroneVehicle-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

Autonomous drone perception systems fuse **RGB** and **thermal infrared (TIR)** camera streams for object detection. Despite this, published models are exclusively benchmarked on clean data — leaving their real-world robustness unknown.

This thesis fills that gap. We apply a **9-type corruption suite at 3 severity levels** to the DroneVehicle RGB-IR dataset and evaluate three fusion architectures using the **Resistance Ability (RA)** metric adapted from the MultiCorrupt benchmark. Through targeted modality removal experiments we also quantify which sensor stream carries detection performance under each degradation condition.

---

## Research Questions

| | Question | Type |
|---|---|---|
| **RQ1** | How does detection performance degrade under different types and severity levels of sensor corruption? | Diagnostic |
| **RQ2** | Which modality — RGB or TIR — contributes most to model resilience under each specific degradation? | Attributional |

---

## Models

All three models use a **ResNet-50** backbone (ImageNet pretrained) and are trained on the DroneVehicle dataset.

| Model | Fusion Strategy | Description |
|---|---|---|
| **UA-CMDet** | Late / uncertainty-guided | Separate RGB and TIR branches fused via uncertainty-weighted features. Primary baseline from the DroneVehicle paper. |
| **Early Fusion** | Early (concatenation) | RGB and TIR feature maps concatenated before the main encoder. Architecturally simplest — measures how much plain concatenation buys. |
| **C2Former** | Intermediate (cross-attention) | Inter-modality Cross-Attention (ICA) at backbone level. Dynamically attends to cross-modal feature relationships. |

---

## Dataset

**DroneVehicle** (Sun et al., IEEE TCSVT 2022)

- 28,439 synchronized RGB + TIR image pairs
- Captured from UAVs over urban/suburban scenes, day and night
- 5 vehicle classes: car, truck, bus, van, freight car
- Oriented bounding box (OBB) annotations
- Separate sensors per modality → corruptions applied independently

---

## Corruption Suite

Corruptions are applied to the **test split only**, independently per modality. Implemented via the [`imagecorruptions`](https://github.com/bethgelab/imagecorruptions) library.

### RGB Corruptions

| Type | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|
| Gaussian Noise | σ = 0.08 | σ = 0.12 | σ = 0.18 |
| Motion Blur | r=10, σ=3 | r=15, σ=5 | r=15, σ=8 |
| Brightness Shift | +0.10 | +0.20 | +0.30 |
| Low Contrast | scale 0.4 around mean | scale 0.3 around mean | scale 0.2 around mean |
| Complete Dropout | — | — | — |

### TIR Corruptions

| Type | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|
| Sensor Noise | σ = 0.15 | σ = 0.20 | σ = 0.35 |
| Blur | σ = 1 | σ = 2 | σ = 3 |
| Intensity Shift | +0.10 | +0.20 | +0.30 |
| Complete Dropout | — | — | — |

> All pixel values normalised to [0, 1].

---

## Experiment Design

### Experiment 0 — Clean Baseline
Validates the evaluation pipeline. Clean mAP must match published figures for UA-CMDet and C2Former before corruption experiments begin.

### Experiment 1 — Corruption Benchmark (RQ1)
3 models × 23 conditions = **69 inference runs**

For each condition: apply corruption to the relevant modality, run full dual-modality inference, compute `RA = corrupted_mAP / clean_mAP`.

### Experiment 2 — Modality Removal (RQ2)
3 models × 23 conditions × 2 configs = **138 inference runs**

For each condition: run with RGB zeroed (TIR-only) and TIR zeroed (RGB-only). Isolates which sensor carries performance when the other is degraded.

| Block | Runs |
|---|---|
| Clean baseline | 3 |
| Corruption benchmark | 69 |
| Modality removal | 138 |
| **Total** | **210** |

---

## Project Structure

```
├── src/
│   ├── corruption/        # Corruption pipeline + Table 2 parameters
│   ├── evaluation/        # RA metric, DOTA eval wrapper
│   └── inference/         # Model-agnostic inference interface + per-model wrappers
├── experiments/
│   ├── exp0_baseline.py   # Clean baseline validation
│   ├── exp1_corruption.py # RQ1 — corruption benchmark
│   └── exp2_modality_removal.py  # RQ2 — modality removal
├── data/
│   └── DroneVehicle/      # Dataset (not tracked — download separately)
├── models/                # Cloned model repos (not tracked — clone separately)
├── results/               # JSON outputs from every inference run (tracked)
├── analysis/              # Notebooks for tables + figures
├── hpc/                   # SLURM job scripts for EEMCS HPC cluster
└── docs/                  # Research proposal and documentation
```

---

## Setup

**1. Clone the repo and model dependencies**
```bash
git clone <this-repo>
cd drone-multimodal-robustness

# Model repos (into models/)
git clone https://github.com/SunYM2020/UA-CMDet models/ua_cmddet
git clone https://github.com/yuanmaoxun/C2Former  models/c2former

# DOTA evaluation toolkit
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit
```

**2. Create the environment**
```bash
conda env create -f environment.yml
conda activate thesis
```

> Update `pytorch-cuda` version in `environment.yml` to match your CUDA version before creating the environment.

**3. Download and prepare the dataset**

Download the DroneVehicle dataset from HuggingFace (RGB + TIR pairs with OBB
annotations). The HuggingFace mirror works from most HPC clusters; Kaggle is
typically blocked on compute nodes.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='McCheng/DroneVehicle', repo_type='dataset', \
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
```

---

## Running Experiments

```bash
# Step 1 — validate clean baselines (must pass before proceeding)
python experiments/exp0_baseline.py

# Step 2 — corruption benchmark (RQ1)
python experiments/exp1_corruption.py

# Step 3 — modality removal (RQ2)
python experiments/exp2_modality_removal.py
```

On the HPC cluster, use the SLURM scripts in `hpc/`. The generic scripts run all
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
```

---

## Evaluation

- **Metric**: mAP @ IoU 0.5 via the DOTA evaluation toolkit (OBB-aware)
- **Robustness**: Resistance Ability (RA) = `corrupted_mAP / clean_mAP`
- RA is reported per `(model, corruption_type, severity, modality_config)`

---

## References

1. Ma et al. — *Are Multimodal Transformers Robust to Missing Modality?* CVPR 2022
2. Chu & Liu — *MT-DETR: Robust End-to-End Multimodal Detection with Confidence Fusion.* WACV 2023
3. Beemelmanns et al. — *MultiCorrupt: A Multimodal Robustness Dataset and Benchmark of LiDAR-Camera Fusion.* IV 2024
4. Sun et al. — *Drone-Based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning.* IEEE TCSVT 2022
5. Michaelis et al. — *Benchmarking Robustness in Object Detection.* arXiv 2019
6. Medeiros et al. — *WiSE-OD: Benchmarking Robustness in Infrared Object Detection.* arXiv 2025
7. Yuan & Wei — *C²Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection.* IEEE TGRS 2024

---

*University of Twente · Faculty of EEMCS · TSCiT 2025*
