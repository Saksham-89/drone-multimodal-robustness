# Experiment 1: Corruption Benchmark — C2Former (RQ1)

**Date**: 2026-05-07  
**Model**: C2Former (Inter-modality Cross-Attention, ResNet-50 backbone)  
**Experiment**: exp1_corruption.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5; Resistance Ability (RA) = corrupted_mAP / clean_mAP  
**Status**: 20/23 valid. TIR intensity_shift (3 conditions) invalid — same pipeline bug as Early Fusion, re-run required.

---

## 1. Overview

C2Former uses an Inter-modality Cross-Attention (ICA) module at the backbone feature level. Each modality stream (RGB and TIR, processed by separate ResNet-50 branches) can dynamically query complementary features from the other. This is an intermediate fusion strategy — deeper integration than concatenation, but not deferred to the prediction level like UA-CMDet.

**Clean baseline mAP**: 0.7046

### TIR Intensity Shift — Invalid

The three `tir__intensity_shift` conditions ran before the `_tir_intensity_shift()` pipeline fix was applied on the HPC. They show RA = 1.000 (map identical to clean), which is impossible. Must be deleted and re-collected after `git fetch && git checkout origin/master -- src/corruption/pipeline.py`.

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
| intensity_shift | *(invalid)* | — | *(invalid)* | — | *(invalid)* | — | pending |
| complete_dropout | — | — | — | — | 0.4339 | — | **0.616** |

**TIR mean RA (graded only, 6 valid conditions)**: 0.883

### 2.3 Full Ranking (20 valid conditions)

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
| 9 (best graded) | *(intensity_shift TIR)* | — | pending | pending |

---

## 3. Key Findings

### 3.1 Both Dropout Conditions Are Damaging — Unlike Early Fusion

C2Former RGB dropout: RA **0.550**. C2Former TIR dropout: RA **0.616**.

This is a qualitatively different pattern from Early Fusion, where TIR dropout (RA 0.309) was far more catastrophic than RGB dropout (RA 0.614). In C2Former, both modalities are load-bearing to a similar degree. The cross-attention ICA module couples the two streams bidirectionally — each queries the other — so neither can be removed without significant loss.

This is mechanistically sensible: if stream A cannot attend to stream B (because B is zero), A loses the cross-modal context it was trained to exploit. Both streams degrade similarly.

### 3.2 C2Former Handles TIR Dropout Far Better Than Early Fusion

| Condition | Early Fusion | C2Former | Difference |
|---|---|---|---|
| TIR complete_dropout | 0.309 | **0.616** | +0.307 |
| RGB complete_dropout | **0.614** | 0.550 | −0.064 |

This is the single largest difference between the two models. Early Fusion's TIR dropout RA of 0.309 reflects the catastrophic channel-dominance effect: the concatenated feature representation is so TIR-dependent that zeroing the single TIR channel collapses performance. C2Former's RGB stream, when its cross-attention to TIR is cut off (TIR=zeros), still retains the full signal from its own ResNet branch — it degrades gracefully rather than collapsing.

### 3.3 C2Former Is More Robust to All RGB Graded Corruptions

| RGB Corruption | EF S3 RA | C2F S3 RA | Advantage |
|---|---|---|---|
| low_contrast | 0.483 | **0.762** | +0.279 |
| gaussian_noise | 0.726 | **0.788** | +0.062 |
| brightness_shift | 0.750 | **0.884** | +0.134 |
| motion_blur | 0.863 | **0.889** | +0.026 |

C2Former is substantially more robust to RGB corruption across all four types. The cross-attention mechanism can compensate for a degraded RGB stream by drawing on the clean TIR stream — the ICA module learns to weight TIR features more heavily when RGB features are noisy or low-amplitude.

### 3.4 C2Former Is More Sensitive to TIR Graded Corruptions

| TIR Corruption | EF S3 RA | C2F S3 RA | Advantage |
|---|---|---|---|
| blur | **0.928** | 0.825 | EF +0.103 |
| sensor_noise | **0.819** | 0.768 | EF +0.051 |

Early Fusion is more robust to TIR graded corruption. When TIR is blurred or noisy, C2Former's cross-attention module propagates the degraded TIR features into the RGB stream — the corruption "bleeds through" the attention mechanism. In Early Fusion, TIR features enter only through the single TIR channel in the concatenated tensor; degradation to one channel in a 4-channel input is diluted.

This is the robustness tradeoff of cross-attention: it enables better feature sharing under clean conditions (hence the higher clean mAP), but also enables more corruption sharing when one stream is degraded.

### 3.5 TIR Blur Is Significantly More Damaging in C2Former

C2Former TIR blur S3: RA **0.825**. Early Fusion TIR blur S3: RA **0.928**. The gap (0.103) is the largest TIR graded corruption difference between models. C2Former's attention to TIR is severely disrupted when TIR spatial frequencies are smoothed — the ICA module relies on structural TIR features (edges, thermal boundaries) for cross-modal queries, and blur destroys those.

### 3.6 RGB Graded Mean RA Is Substantially Better Than Early Fusion's

C2Former RGB graded mean RA: **0.885**. Early Fusion: **0.794**. A 9 pp gap in favour of C2Former under RGB corruption. This confirms that cross-attention provides meaningful RGB corruption robustness through TIR compensation.

---

## 4. Cross-Model Comparison (Early Fusion vs C2Former)

| Metric | Early Fusion | C2Former | Winner |
|---|---|---|---|
| Clean mAP | 0.485 | **0.705** | C2Former |
| RGB graded mean RA | 0.794 | **0.885** | C2Former |
| TIR graded mean RA | **0.914** | 0.883 | Early Fusion |
| RGB dropout RA | **0.614** | 0.550 | Early Fusion |
| TIR dropout RA | 0.309 | **0.616** | C2Former |
| Worst-case RA | 0.309 (TIR dropout) | 0.550 (RGB dropout) | **C2Former** |
| Most balanced modalities | No (TIR-dominant) | **Yes** | C2Former |

**Summary**: C2Former dominates on both clean performance and worst-case robustness. Its cross-attention provides a natural compensation mechanism for single-modality degradation. Early Fusion has a specific advantage only for TIR graded corruption (where concatenation dilutes rather than propagates degradation). For any mission-critical deployment scenario, C2Former's higher floor (worst case RA 0.550 vs 0.309) is a decisive advantage.

---

## 5. Implications for RQ1 (C2Former Component)

**RQ1 asks**: How does detection performance degrade under corruption?

For C2Former:
1. **No single modality dominates**: both dropouts produce RA ~0.55–0.62, unlike Early Fusion's extreme TIR dependence.
2. **Cross-attention compensates for RGB degradation**: RGB corruption RA is consistently higher than in Early Fusion.
3. **Cross-attention propagates TIR degradation**: TIR corruption RA is worse than in Early Fusion — the coupling that enables compensation also enables propagation.
4. **The worst observed condition is RGB dropout (RA 0.550)**, not TIR dropout — the opposite of Early Fusion.
5. **Severity gradients are monotone** for all valid conditions.

The architecture-dependent robustness profile is a core thesis finding: **fusion architecture choice determines not just clean accuracy but the shape of the degradation curve.**

---

## 6. Actions Required

### Re-run TIR Intensity Shift (3 conditions)

```bash
git fetch origin
git checkout origin/master -- src/corruption/pipeline.py
rm results/exp1_corruption/c2former__tir__intensity_shift__s{1,2,3}.json
sbatch hpc/run_exp1_c2former.sh
```

### Submit exp2_c2former (modality removal)

```bash
sbatch hpc/run_exp2_c2former.sh
```

---

## 7. Data Reference

Result files: `results/exp1_corruption/c2former__*.json` (23 files; 3 intensity_shift marked INVALID)  
Clean baseline: `results/exp0_baseline/c2former.json` (mAP = 0.7046)  
Model checkpoint: `work_dirs/c2former/epoch_24.pth`  
Experiment script: `experiments/exp1_corruption.py`
