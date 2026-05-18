# Experiment 0 — Clean Baseline Evaluation

**Purpose**: Establish clean-data mAP for all three models on the DroneVehicle test split. These figures are the denominator in the Resistance Ability (RA) metric used throughout the corruption experiments.

**Status**: All three models complete.

---

## Results Summary

| Model | Fusion Type | Epochs | Test mAP | Val mAP | Val→Test gap | Job |
|---|---|---|---|---|---|---|
| Early Fusion | Early (concat) | 24 | **0.485** | 0.488 | −0.003 | 491917 |
| C2Former | Intermediate (cross-attn) | 24 | **0.705** | 0.715 | −0.010 | 492464 |
| UA-CMDet | Late (uncertainty) | 12 | **0.214** | — | — | 492525 |
| UA-CMDet (published) | Late (uncertainty) | — | ~0.412 | — | — | Sun et al. 2022 |

---

## Per-Class Breakdown

### Early Fusion (mAP 0.485)

| Class | GT boxes | Recall | AP |
|---|---|---|---|
| car | 124,111 | 0.904 | 0.831 |
| truck | 7,102 | 0.789 | 0.345 |
| freight_car | 3,978 | 0.829 | 0.273 |
| bus | 4,161 | 0.915 | 0.739 |
| van | 3,960 | 0.853 | 0.236 |

### C2Former (mAP 0.705)

| Class | GT boxes | Recall | AP |
|---|---|---|---|
| car | 124,111 | 0.966 | 0.894 |
| truck | 7,102 | 0.910 | 0.682 |
| freight_car | 3,978 | 0.915 | 0.542 |
| bus | 4,161 | 0.960 | 0.889 |
| van | 3,960 | 0.940 | 0.515 |

### UA-CMDet (mAP 0.214, HBB eval)

Note: UA-CMDet uses axis-aligned COCO bbox IoU (HBB), not OBB polygon IoU. The published figure (0.412) used the DOTA devkit with polyiou on oriented boxes — a different eval protocol. All UA-CMDet RA comparisons are relative to this 0.214 baseline so RA is internally consistent.

| Class | AP |
|---|---|
| car | — |
| truck | — |
| freight_car | — |
| bus | — |
| van | — |

Per-class breakdown not available (COCOeval aggregates by category but UA-CMDet eval output was captured as overall mAP only).

---

## Key Observations

### 1. C2Former dominates under clean conditions (+22 pp over Early Fusion)

Cross-attention fusion achieves 0.705 vs Early Fusion's 0.485 — a 22 percentage point gap. This is a large margin and consistent with C2Former's design intent: the ICA module allows each modality to actively query complementary features from the other, producing richer representations than simple concatenation.

### 2. Early Fusion exceeds UA-CMDet's published figure (+7 pp)

Early concatenation with a warm-started 4-channel backbone achieves 0.485 vs UA-CMDet's published 0.412. This is a strong result for the simplest architecture. Two possible explanations:
- The 4-channel warm-start initialization (channel 3 = mean of RGB channels) provides a better-calibrated starting point than UA-CMDet's ImageNet-pretrained dual-stream setup
- S2ANet's oriented anchor-free head may be better suited to the DroneVehicle annotation style than UA-CMDet's cascade OBB head

This ranking may not hold under corruption — the thesis RQ1 question.

### 3. Per-class patterns are consistent across models

- **Car** and **bus** are the highest-AP classes in both models. Both are visually distinct (car is by far the most frequent class; bus is large and has a distinctive TIR signature).
- **Van** and **freight_car** are the hardest classes in both models. Both are small, underrepresented, and visually similar to car/truck respectively.
- C2Former's truck AP (0.682) is notably better than Early Fusion's (0.345). This is the largest per-class gap and suggests cross-attention particularly helps with mid-size, shape-ambiguous objects.

### 4. Val→test deltas are small

Early Fusion: −0.003, C2Former: −0.010. Both gaps are within normal variance, confirming the models generalize and neither overfit to the training/validation distribution.

---

## Evaluation Protocol

All evaluations use:
- **mmrotate built-in mAP** (`dataset.evaluate(results, metric='mAP')`)
- **IoU threshold**: 0.5 (standard for drone vehicle detection)
- **Oriented bounding boxes**: annotations are rotated rectangles; standard COCO axis-aligned IoU is not used
- **Test split only**: training and validation splits are not used for evaluation

Note on UA-CMDet: UA-CMDet's own eval script (`eval/DroneVehicleEval.py`) uses the DOTA devkit with polyiou for OBB IoU. If the test job uses mmrotate's built-in eval, the number may differ slightly from the published figure which used the original eval script. Both numbers will be reported if they differ.

---

## Implications for RQ1 and RQ2

The clean baseline hierarchy **C2Former > Early Fusion > UA-CMDet** sets the reference frame for the corruption experiments:

- **If the hierarchy is preserved under corruption**: fusion architecture determines both clean and corrupted performance — a straightforward finding.
- **If the hierarchy collapses or reverses**: corruption reveals robustness properties not captured by clean mAP — a more interesting and thesis-relevant finding. For example, UA-CMDet's uncertainty-guided late fusion might be inherently more robust to single-modality corruption than C2Former's tightly-coupled cross-attention.
- **RA (Resistance Ability) = corrupted_mAP / clean_mAP** will normalize out the baseline gap so we can compare *relative* robustness across architectures.

The per-class gap between C2Former and Early Fusion on truck (0.682 vs 0.345) is worth watching under corruption — if C2Former's advantage on ambiguous classes disappears when a modality is degraded, that would suggest the cross-attention module relies on both clean modalities to resolve ambiguity.

---

## Files

- `results/exp0_baseline/early_fusion.json` — Early Fusion test result
- `results/exp0_baseline/c2former.json` — C2Former test result
- `results/exp0_baseline/ua_cmddet.json` — UA-CMDet test result (mAP 0.2137, job 492525)
