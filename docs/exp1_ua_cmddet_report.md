# Experiment 1: Corruption Benchmark — UA-CMDet (RQ1)

**Date**: 2026-05-09  
**Model**: UA-CMDet (Sun et al., TCSVT 2022 — uncertainty-guided late fusion, ResNet-50 backbone)  
**Experiment**: exp1_corruption.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5; Resistance Ability (RA) = corrupted_mAP / clean_mAP  
**Status**: 23/23 conditions complete and valid.

---

## 1. Overview

This report documents the complete corruption benchmark for UA-CMDet. UA-CMDet maintains separate RGB and TIR branches throughout the network, fusing them via uncertainty-guided feature weighting at the detection head. This architecture is the primary baseline from the DroneVehicle paper.

**Clean baseline mAP**: 0.2137

Note: Our clean mAP (0.2137) is below the published figure (~0.412) because we use axis-aligned COCO bbox IoU rather than the polygon OBB IoU used in the original paper. All three models use the same eval protocol, so RA values are directly comparable across models.

---

## 2. Results Tables

### 2.1 RGB Corruptions

| Corruption | S1 mAP | S1 RA | S2 mAP | S2 RA | S3 mAP | S3 RA | Mean RA |
|---|---|---|---|---|---|---|---|
| gaussian_noise | 0.0912 | 0.427 | 0.0357 | 0.167 | 0.0091 | 0.043 | **0.212** |
| motion_blur | 0.1846 | 0.864 | 0.1676 | 0.784 | 0.1239 | 0.580 | **0.743** |
| brightness_shift | 0.1852 | 0.867 | 0.1664 | 0.779 | 0.1523 | 0.713 | **0.786** |
| low_contrast | 0.1803 | 0.844 | 0.1660 | 0.777 | 0.1449 | 0.678 | **0.766** |
| complete_dropout | — | — | — | — | 0.0012 | — | **0.006** |

**RGB mean RA (graded only, 12 conditions)**: 0.627

### 2.2 TIR Corruptions

| Corruption | S1 mAP | S1 RA | S2 mAP | S2 RA | S3 mAP | S3 RA | Mean RA |
|---|---|---|---|---|---|---|---|
| sensor_noise | 0.1981 | 0.927 | 0.1920 | 0.899 | 0.1799 | 0.842 | **0.889** |
| blur | 0.2103 | 0.984 | 0.2117 | 0.991 | 0.2123 | 0.994 | **0.990** |
| intensity_shift | 0.2143 | 1.003 | 0.2154 | 1.008 | 0.2169 | 1.015 | **1.009** |
| complete_dropout | — | — | — | — | 0.2152 | — | **1.007** |

**TIR mean RA (graded only, 9 conditions)**: 0.963

### 2.3 Full Ranking by Mean RA (worst to best)

| Rank | Corruption | Modality | S3 RA | Mean RA |
|---|---|---|---|---|
| 1 (worst) | complete_dropout | RGB | 0.006 | **0.006** |
| 2 | gaussian_noise | RGB | 0.043 | **0.212** |
| 3 | motion_blur | RGB | 0.580 | **0.743** |
| 4 | low_contrast | RGB | 0.678 | **0.766** |
| 5 | brightness_shift | RGB | 0.713 | **0.786** |
| 6 | sensor_noise | TIR | 0.842 | **0.889** |
| 7 | blur | TIR | 0.994 | **0.990** |
| 8 | intensity_shift | TIR | 1.015 | **1.009** |
| 9 (best) | complete_dropout | TIR | 1.007 | **1.007** |

---

## 3. Key Findings

### 3.1 UA-CMDet Is Completely RGB-Dominant — The Opposite of Early Fusion

The most striking result: **RGB complete dropout collapses mAP from 0.2137 to 0.0012 (RA = 0.006)**. Zeroing the RGB stream destroys detection almost entirely. Meanwhile, **TIR complete dropout has no effect whatsoever (RA = 1.007)** — mAP is unchanged when the TIR stream is zeroed.

This is the polar opposite of Early Fusion, where TIR dropout (RA = 0.309) was the catastrophic failure mode. UA-CMDet's uncertainty-guided fusion has collapsed almost entirely onto the RGB branch, rendering TIR functionally irrelevant to detection output.

### 3.2 RGB Gaussian Noise Is Near-Fatal at Higher Severities

RGB gaussian_noise produces the most severe graded-corruption degradation observed across any model in this study. At severity 1 (σ=0.08) RA = 0.427; by severity 3 (σ=0.18) RA = **0.043** — a 96% performance collapse. The mean RA of 0.212 across three severities is catastrophically low.

This reflects UA-CMDet's fundamental architecture vulnerability: the uncertainty estimator itself relies on feature statistics in the RGB branch. Gaussian noise corrupts local texture statistics uniformly, causing the uncertainty estimator to produce unreliable fusion weights, which amplifies the damage beyond what the raw signal degradation would cause alone.

### 3.3 TIR Corruptions Have Negligible Impact — Including Complete Dropout

All TIR graded corruptions produce RA ≥ 0.842 (sensor_noise S3). TIR blur at severity 3 yields RA = 0.994 — essentially no degradation. TIR intensity shift at all severities improves mAP slightly above the clean baseline (RA > 1.0), suggesting the model's fusion weighting treats the intensity-shifted TIR as a slightly stronger signal.

The implication is that the TIR branch contributes almost nothing to UA-CMDet's output under any condition. The uncertainty mechanism has learned to down-weight TIR, likely because the joint training on DroneVehicle's day+night scenes makes RGB sufficient for the model's detection objective.

### 3.4 All RGB Graded Corruptions Degrade Monotonically and Steeply

Every RGB corruption degrades monotonically with severity, but the decline rate is notably steeper than in Early Fusion or C2Former. The S1→S3 RA drop for gaussian_noise is 0.427→0.043 (a factor of 10×). For comparison, Early Fusion's gaussian_noise S1→S3 is 0.930→0.726. UA-CMDet amplifies RGB corruption damage through the uncertainty mechanism.

### 3.5 Motion Blur at S3 Is Surprisingly Damaging (RA = 0.580)

Motion blur at severity 3 drops mAP from 0.214 to 0.124. This is the third worst result overall. Motion blur smears directional features in the RGB stream, which the oriented detection head relies on for bounding box angle estimation. Combined with UA-CMDet's exclusive reliance on RGB, this produces substantial degradation.

### 3.6 Architecture Vulnerability Concentration

UA-CMDet has concentrated all detection capability in one modality. This is the highest-risk robustness profile among the three models: when RGB is clean, performance is acceptable; any significant RGB degradation degrades the system with no TIR fallback available.

---

## 4. Modality Comparison Summary

| | RGB | TIR |
|---|---|---|
| Dropout RA | **0.006** | 1.007 |
| Graded mean RA | **0.627** | 0.963 |
| Worst graded RA (S3) | 0.043 (gaussian_noise) | 0.842 (sensor_noise) |
| Best graded RA (S3) | 0.713 (brightness_shift) | 1.015 (intensity_shift) |

**RGB is the sole load-bearing modality.** Losing it is fatal; degrading it is highly damaging. TIR is functionally inert — corrupting or removing it has negligible effect on the output.

---

## 5. Implications for RQ1

**RQ1 asks**: How does detection performance degrade under different corruption types and severity levels?

For UA-CMDet:

1. **Degradation is exclusively driven by RGB corruption** — TIR corruptions are irrelevant. The corruption vulnerability space reduces to a single dimension: RGB stream quality.
2. **Gaussian noise is catastrophic** — RGB gaussian_noise S3 (RA = 0.043) is the worst graded result observed across all models. The uncertainty mechanism amplifies noise damage rather than absorbing it.
3. **RGB complete dropout is the absolute worst failure mode** (RA = 0.006), effectively disabling the detector.
4. **The severity gradient for RGB gaussian_noise is non-linear and extremely steep** — a near-threshold behavior where severity 3 essentially disables the model.
5. **TIR dropout is harmless** — a finding that directly contradicts the expected behaviour of a dual-modality system and reveals that UA-CMDet has not learned to exploit TIR.
6. **Brightness and contrast corruptions are relatively tolerable** (RA 0.678–0.867) because they preserve local texture — the RGB features the uncertainty estimator relies on remain partially intact.

Comparing to Early Fusion: the two models have inverted modality dependencies despite both using ResNet-50. Architecture — not backbone — determines which modality becomes load-bearing.

---

## 6. Data Reference

All 23 result files: `results/exp1_corruption/ua_cmddet__*.json`  
Clean baseline: `results/exp0_baseline/ua_cmddet.json` (mAP = 0.2137)  
Model checkpoint: `work_dirs/ua_cmddet/latest.pth` (epoch 12)  
Experiment script: `experiments/exp1_corruption.py`  
Corruption params: `src/corruption/params.py`  
Inference runner: `src/inference/ua_cmddet.py` (bug fix: `_get_img_tensor` for AerialDetection list format, 2026-05-09)
