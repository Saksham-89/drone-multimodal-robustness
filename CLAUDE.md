# Thesis: Evaluating Multimodal Fusion Robustness in Drone-Based Object Detection Under Sensor Degradation

**Author**: Saksham Singh Birla  
**Institution**: University of Twente, Enschede, The Netherlands  
**Venue**: TSCiT 2025  
**Status**: Research proposal approved; experimental work in progress

---

## What This Thesis Does

Drone-based object detection systems fuse RGB and thermal infrared (TIR) camera streams. Published models are benchmarked only on clean data — this thesis fills the gap by systematically applying sensor corruptions and measuring how robustness varies across fusion architectures and modalities.

---

## Research Questions

**RQ1 (Diagnostic):** How does detection performance degrade under different types and severity levels of sensor corruption (noise, blur, missing modality) in drone-based object detection?

**RQ2 (Attributional):** Which sensor modality — RGB or thermal IR — contributes most to model resilience under specific degradation types, and which is most critical to preserve?

---

## Dataset: DroneVehicle

- **28,439** synchronized RGB + TIR image pairs
- Captured from UAVs over urban/suburban scenes, day and night
- **5 vehicle classes**: car, truck, bus, van, freight car
- **Oriented bounding boxes** (OBB annotations)
- Modalities captured by separate sensors → corruptions can be applied independently
- Test split used exclusively for evaluation — **no fine-tuning**
- Paper: Sun et al., IEEE TCSVT 2022 [4]

---

## Models (3 Fusion Architectures)

All use **ResNet-50** backbone (ImageNet pretrained). Trained on EEMCS HPC cluster.

| Model | Fusion Type | Description |
|---|---|---|
| **UA-CMDet** | Late/uncertainty | Primary baseline. Separate RGB+TIR branches fused via uncertainty-guided feature weighting. From the DroneVehicle paper. Clean mAP verified against published figures before corruption experiments. |
| **Early Fusion** | Early (concat) | RGB and TIR feature maps concatenated before main encoder. Simplest architecture. Sets a baseline for how much robustness plain concatenation provides. |
| **C2Former** | Intermediate (cross-attn) | Inter-modality Cross-Attention (ICA) module at backbone level. Dynamically attends to cross-modal feature relationships. Benchmarked on DroneVehicle in original paper — published mAP available for validation. |

---

## Corruption Pipeline

Adapted from **MultiCorrupt** codebase [3], using the **`imagecorruptions`** Python library [5]. Grayscale-native so same functions apply to both RGB and TIR.

Applied to **test split only**. Corruptions per modality are independent.

### RGB Corruptions (severity 1/2/3)
| Type | Sev. 1 | Sev. 2 | Sev. 3 |
|---|---|---|---|
| Gaussian Noise | σ = 0.08 | σ = 0.12 | σ = 0.18 |
| Motion Blur | r=10, σ=3 | r=15, σ=5 | r=15, σ=8 |
| Brightness Shift | +0.10 | +0.20 | +0.30 |
| Low Contrast | 0.4× | 0.3× | 0.2× |
| Complete Dropout | — (binary) | | |

### TIR Corruptions (severity 1/2/3)
| Type | Sev. 1 | Sev. 2 | Sev. 3 |
|---|---|---|---|
| Sensor Noise | σ = 0.15 | σ = 0.20 | σ = 0.35 |
| Blur | σ = 1 | σ = 2 | σ = 3 |
| Intensity Shift | +0.10 | +0.20 | +0.30 |
| Complete Dropout | — (binary) | | |

All values normalized to [0, 1] pixel range. Intensity shift for TIR uses library's `brightness` function on the grayscale channel with same additive HSV offset parameterization as RGB brightness shift.

**Total corruption conditions**: 9 graded types × 3 severities + 2 binary dropouts = **29 conditions per model**

---

## Modality Removal Experiments (RQ2)

For every corruption condition, run **3 inference passes**:
1. Full dual-modality (RGB + TIR)
2. RGB zeroed out (TIR only)
3. TIR zeroed out (RGB only)

Comparing mAP across all three isolates which sensor carries detection performance when the other is degraded.

---

## Evaluation Metrics

- **mAP @ IoU 0.5** — primary detection metric
- **DOTA evaluation toolkit** — used for oriented bounding box annotations (not standard COCO eval)
- **Resistance Ability (RA)** = corrupted mAP / clean mAP (adapted from MultiCorrupt [3])
  - Reported per: corruption type × severity level × modality condition × model

---

## Expected Outcomes

- **RQ1**: Degradation profiles per model. Complete dropout = sharpest drops. TIR noise expected to be more damaging than RGB noise (consistent with WiSE-OD findings). RA decline rate differs across fusion strategies.
- **RQ2**: TIR expected to be load-bearing under RGB corruptions (brightness, contrast). RGB expected to contribute more to fine-grained class discrimination. Output: per-corruption modality priority map for fault-tolerant system design.

---

## Project Timeline (8 Weeks from Proposal Approval)

| Week | Tasks |
|---|---|
| 1 | Proposal submission |
| 2–3 | Literature review + environment/dataset setup + corruption pipeline (parallel) |
| 3 | Clean baseline evaluation (validation checkpoint) |
| 4–5 | Corruption benchmark (RQ1) + Modality removal (RQ2) |
| 6–8 | Analysis, writing, final submission |

---

## Key References

1. Ma et al. (CVPR 2022) — Multimodal transformers robustness to missing modality
2. Chu et al. (WACV 2023) — MT-DETR confidence fusion module
3. Beemelmanns et al. (IV 2024) — **MultiCorrupt** benchmark (LiDAR-camera, nuScenes)
4. Sun et al. (IEEE TCSVT 2022) — **UA-CMDet** + **DroneVehicle** dataset
5. Michaelis et al. (arXiv 2019) — **`imagecorruptions`** Python library
6. Medeiros et al. (arXiv 2025) — **WiSE-OD** / LLVIP-C / FLIR-C (TIR corruption benchmarks)
7. Yuan & Wei (IEEE TGRS 2024) — **C2Former** (calibrated + complementary transformer)

---

## Cloned Repos (models/)

| Repo | Path | Framework | Notes |
|---|---|---|---|
| UA-CMDet | `models/ua_cmddet/` | AerialDetection + mmdetection (old fork) | Has `configs/DroneVehicle/UACMDet.py`. Own eval at `eval/DroneVehicleEval.py` with polyiou. Old deps (PyTorch 1.1, Python 3.7) — may need compat fixes. |
| C2Former | `models/c2former/` | MMRotate | **No DroneVehicle config shipped.** Must write `configs/dronevehicle/c2former_dronevehicle.py` ourselves based on existing S2ANet/DOTA configs. |
| MultiCorrupt | `models/multicorrupt/` | Reference only | `converter/img.py` — image corruption implementation to reference. `evaluation/` — RA CSVs for format reference. Do not run directly. |
| DOTA devkit | `models/dota_devkit/` | Standalone | `dota_evaluation_task1.py` — main OBB eval script. Requires SWIG + polyiou C extension build (`polyiou.cpp`). |

---

## Implementation Notes for Claude

### Key Libraries
- `imagecorruptions` — corruption transforms
- `models/dota_devkit/dota_evaluation_task1.py` — OBB-aware mAP (requires SWIG build)
- `models/ua_cmddet/eval/DroneVehicleEval.py` — UA-CMDet's own eval (also uses polyiou)
- PyTorch — model training/inference

### Critical Implementation Details
- Corruptions are applied **independently** per modality — do not apply one corruption to both streams simultaneously unless it's testing joint degradation
- Severity parameters must match Table 2 exactly for reproducibility
- Modality zeroing (dropout) means setting the tensor to all zeros, not dropping the channel dimension — the model architecture stays fixed
- RA = corrupted_mAP / clean_mAP, computed per (corruption_type, severity, modality_condition, model) cell
- Clean baseline mAP must be validated against published figures for UA-CMDet and C2Former before proceeding with corruption experiments
- DOTA eval toolkit required — standard COCO mAP will give wrong results on oriented boxes
- **C2Former DroneVehicle config must be written from scratch** — use `models/c2former/configs/s2anet/` as template and adapt dataset paths and class names to DroneVehicle

### HPC Context
- Training runs on EEMCS HPC cluster (University of Twente)
- Inference for corruption experiments: 3 models × 23 conditions × 3 modality configs = 207 + 3 baseline = 210 inference runs total

### Architecture Constraints
- All three models use ResNet-50 backbone
- C2Former uses Inter-modality Cross-Attention (ICA) at backbone level; built on MMRotate
- UA-CMDet uses separate branches + uncertainty-guided fusion weights; built on AerialDetection
- Early fusion: concatenation before encoder; our own implementation

### What NOT to Do
- Do not fine-tune models on corrupted data — evaluation only
- Do not apply corruptions to the training split
- Do not use standard COCO mAP for OBB annotations
- Do not run MultiCorrupt scripts directly — reference only for image corruption logic and RA output format
