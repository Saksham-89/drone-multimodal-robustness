# Experiment 1: Corruption Benchmark — C2Former (RQ1)

**Date**: 2026-05-07 (re-run intensity_shift: 2026-05-08)  
**Model**: C2Former (Inter-modality Cross-Attention, ResNet-50 backbone)  
**Experiment**: exp1_corruption.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5; Resistance Ability (RA) = corrupted_mAP / clean_mAP  
**Status**: 23/23 valid. TIR intensity_shift re-run completed after pipeline.py fix.

---

## 1. Overview

C2Former uses an Inter-modality Cross-Attention (ICA) module at the backbone feature level. Each modality stream (RGB and TIR, processed by separate ResNet-50 branches) can dynamically query complementary features from the other. This is an intermediate fusion strategy — deeper integration than concatenation, but not deferred to the prediction level like UA-CMDet.

**Clean baseline mAP**: 0.7046

---

## 2. Results Tables

### 2.1 RGB Corruptions

| Corruption | S1 mAP | S1 RA | S2 mAP | S2 RA | S3 mAP | S3 RA | Mean RA |
|---|---|---|---|---|---|---|---|
| gaussian_noise | 0.6525 | 0.926 | 0.6129 | 0.870 | 0.5549 | 0.788 | **0.861** |
| motion_blur | 0.6853 | 0.973 | 0.6628 | 0.941 | 0.6262 | 0.889 | **0.934** |
| brightness_shift | 0.6854 | 0.973 | 0.6540 | 0.928 | 0.6227 | 0.884 | **0.928** |
| low_contrast | 0.6122 | 0.869 | 0.5788 | 0.822 | 0.5367 | 0.762 | **0.817** |
| complete_dropout | — | — | — | — | 0.3875 | — | **0.550** |

**RGB mean RA (graded only, 12 conditions)**: 0.885

### 2.2 TIR Corruptions

| Corruption | S1 mAP | S1 RA | S2 mAP | S2 RA | S3 mAP | S3 RA | Mean RA |
|---|---|---|---|---|---|---|---|
| sensor_noise | 0.6566 | 0.932 | 0.6293 | 0.893 | 0.5410 | 0.768 | **0.864** |
| blur | 0.6901 | 0.979 | 0.6359 | 0.902 | 0.5811 | 0.825 | **0.902** |
| intensity_shift | 0.7011 | 0.995 | 0.6898 | 0.979 | 0.6701 | 0.951 | **0.975** |
| complete_dropout | — | — | — | — | 0.4339 | — | **0.616** |

**TIR mean RA (graded only, 9 conditions)**: 0.914

### 2.3 Full Ranking (23 valid conditions)

| Rank | Corruption | Modality | S3 RA | Mean RA |
|---|---|---|---|---|
| 1 (worst) | complete_dropout | RGB | n/a | **0.550** |
| 2 | complete_dropout | TIR | n/a | **0.616** |
| 3 | low_contrast | RGB | 0.762 | **0.817** |
| 4 | sensor_noise | TIR | 0.768 | **0.864** |
| 5 | gaussian_noise | RGB | 0.788 | **0.861** |
| 6 | blur | TIR | 0.825 | **0.902** |
| 7 | brightness_shift | RGB | 0.884 | **0.928** |
| 8 | motion_blur | RGB | 0.889 | **0.934** |
| 9 (best graded) | intensity_shift | TIR | 0.951 | **0.975** |

---

## 3. Key Findings

### 3.1 Both Dropout Conditions Are Damaging — Unlike Early Fusion

C2Former RGB dropout: RA **0.550**. C2Former TIR dropout: RA **0.616**.

This is a qualitatively different pattern from Early Fusion, where TIR dropout (RA 0.309) was far more catastrophic than RGB dropout (RA 0.614). In C2Former, both modalities are load-bearing to a similar degree. The cross-attention ICA module couples the two streams bidirectionally — each queries the other — so neither can be removed without significant loss.

### 3.2 C2Former Handles TIR Dropout Far Better Than Early Fusion

| Condition | Early Fusion | C2Former | Difference |
|---|---|---|---|
| TIR complete_dropout | 0.309 | **0.616** | +0.307 |
| RGB complete_dropout | **0.614** | 0.550 | −0.064 |

Early Fusion's TIR dropout RA of 0.309 reflects catastrophic channel-dominance: the concatenated feature representation is so TIR-dependent that zeroing the single TIR channel collapses performance. C2Former's RGB stream, when its cross-attention to TIR is cut off, still retains the full signal from its own ResNet branch.

### 3.3 C2Former Is More Robust to All RGB Graded Corruptions

| RGB Corruption | EF S3 RA | C2F S3 RA | Advantage |
|---|---|---|---|
| low_contrast | 0.483 | **0.762** | +0.279 |
| gaussian_noise | 0.726 | **0.788** | +0.062 |
| brightness_shift | 0.750 | **0.884** | +0.134 |
| motion_blur | 0.863 | **0.889** | +0.026 |

Cross-attention compensates for a degraded RGB stream by drawing on the clean TIR stream.

### 3.4 TIR Robustness Is Corruption-Type Dependent

For structural TIR corruptions (blur, noise), Early Fusion is more robust. For photometric TIR corruption (intensity shift), C2Former is substantially more robust:

| TIR Corruption | EF S3 RA | C2F S3 RA | Advantage |
|---|---|---|---|
| blur | **0.928** | 0.825 | EF +0.103 |
| sensor_noise | **0.819** | 0.768 | EF +0.051 |
| intensity_shift | 0.799 | **0.951** | C2F +0.152 |

**Mechanistic explanation**: Cross-attention queries TIR for spatial structural features (edges, thermal boundaries). Blur and sensor noise destroy those structures, so corrupted TIR features propagate degradation through the attention mechanism. A global intensity shift, by contrast, preserves all spatial structure — the ICA module continues to attend to the correct locations with full structural fidelity. Early Fusion dilutes single-channel degradation across a 4-channel concatenated tensor regardless of corruption type.

### 3.5 TIR Intensity Shift Is the Most Benign Corruption

C2Former TIR intensity_shift mean RA: **0.975**. The near-unity RA across all three severities (0.995/0.979/0.951) indicates that global brightness shifts to the TIR stream impose negligible performance cost in C2Former. The ICA module is robust to photometric variation when spatial structure is preserved.

### 3.6 Severity Gradients Are Monotone for All Conditions

All 7 graded corruption types show strictly decreasing mAP with increasing severity. This validates that the corruption parameterisation is well-ordered and that the RA ranking is meaningful.

---

## 4. Cross-Model Comparison (Early Fusion vs C2Former)

| Metric | Early Fusion | C2Former | Winner |
|---|---|---|---|
| Clean mAP | 0.485 | **0.705** | C2Former |
| RGB graded mean RA | 0.794 | **0.885** | C2Former |
| TIR graded mean RA | **0.914** | **0.914** | Tie |
| — blur + noise only | **0.914** | 0.883 | Early Fusion |
| — intensity_shift only | 0.883 | **0.975** | C2Former |
| RGB dropout RA | **0.614** | 0.550 | Early Fusion |
| TIR dropout RA | 0.309 | **0.616** | C2Former |
| Worst-case RA | 0.309 (TIR dropout) | **0.550** (RGB dropout) | C2Former |
| Most balanced modalities | No (TIR-dominant) | **Yes** | C2Former |

The TIR graded mean RA tie (both 0.914) conceals an important qualitative difference: Early Fusion leads on structural TIR corruptions; C2Former leads on photometric TIR corruption. For any mission-critical deployment scenario, C2Former's higher floor (worst case RA 0.550 vs 0.309) is a decisive advantage.

---

## 5. Implications for RQ1 (C2Former Component)

**RQ1 asks**: How does detection performance degrade under corruption?

For C2Former:
1. **No single modality dominates**: both dropouts produce RA ~0.55–0.62, unlike Early Fusion's extreme TIR dependence.
2. **Cross-attention compensates for RGB degradation**: RGB corruption RA consistently higher than Early Fusion.
3. **Cross-attention propagates structural TIR degradation**: TIR blur and noise RA are worse than Early Fusion.
4. **Cross-attention is robust to photometric TIR shift**: TIR intensity_shift RA (0.975) substantially exceeds Early Fusion (0.883).
5. **The worst observed condition is RGB dropout (RA 0.550)**, not TIR dropout — the opposite of Early Fusion.
6. **Severity gradients are monotone** for all valid conditions.

The architecture-dependent robustness profile is a core thesis finding: **fusion architecture choice determines not just clean accuracy but the shape and modality-asymmetry of the degradation curve.**

---

## 6. Data Reference

Result files: `results/exp1_corruption/c2former__*.json` (23 files, all valid)  
Clean baseline: `results/exp0_baseline/c2former.json` (mAP = 0.7046)  
Model checkpoint: `work_dirs/c2former/epoch_24.pth`  
Experiment script: `experiments/exp1_corruption.py`
