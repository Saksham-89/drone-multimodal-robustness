# Experiment 1: Corruption Benchmark — Early Fusion (RQ1)

**Date**: 2026-05-07  
**Model**: Early Fusion (RGB-TIR channel concatenation, ResNet-50 backbone)  
**Experiment**: exp1_corruption.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5; Resistance Ability (RA) = corrupted_mAP / clean_mAP  
**Status**: 23/23 conditions complete and valid.

---

## 1. Overview

This report documents the complete corruption benchmark for the Early Fusion model. Corruptions are applied independently to one modality at a time at three severity levels, plus two binary dropout conditions. The Early Fusion architecture concatenates RGB (3-channel) and TIR (1-channel) feature maps before the shared ResNet-50 encoder.

**Clean baseline mAP**: 0.4848

### Pipeline Bug — Fixed

TIR intensity_shift initially produced identical mAP at all severities (RA ≈ 1.000) due to a shape bug in `_tir_intensity_shift()`: stacking a `(H,W,3)` TIR image with `np.stack` created a `(H,W,3,3)` array, causing skimage's `rgb2hsv` to return the original image unchanged. Fixed 2026-05-07 by replacing the imagecorruptions path with a direct additive pixel shift (`image + c*255`), which is mathematically identical for grayscale images and shape-agnostic. Results re-collected and confirmed valid.

---

## 2. Results Tables

### 2.1 RGB Corruptions

| Corruption | S1 mAP | S1 RA | S2 mAP | S2 RA | S3 mAP | S3 RA | Mean RA |
|---|---|---|---|---|---|---|---|
| gaussian_noise | 0.4511 | 0.930 | 0.4138 | 0.853 | 0.3518 | 0.726 | **0.836** |
| motion_blur | 0.4607 | 0.950 | 0.4454 | 0.919 | 0.4186 | 0.863 | **0.911** |
| brightness_shift | 0.4644 | 0.958 | 0.4220 | 0.870 | 0.3634 | 0.750 | **0.859** |
| low_contrast | 0.3102 | 0.640 | 0.2818 | 0.581 | 0.2340 | 0.483 | **0.568** |
| complete_dropout | — | — | — | — | 0.2977 | — | **0.614** |

**RGB mean RA (graded only, 12 conditions)**: 0.794

### 2.2 TIR Corruptions

| Corruption | S1 mAP | S1 RA | S2 mAP | S2 RA | S3 mAP | S3 RA | Mean RA |
|---|---|---|---|---|---|---|---|
| sensor_noise | 0.4634 | 0.956 | 0.4507 | 0.929 | 0.3972 | 0.819 | **0.901** |
| blur | 0.4821 | 0.994 | 0.4624 | 0.954 | 0.4498 | 0.928 | **0.959** |
| intensity_shift | 0.4653 | 0.960 | 0.4310 | 0.889 | 0.3875 | 0.799 | **0.883** |
| complete_dropout | — | — | — | — | 0.1498 | — | **0.309** |

**TIR mean RA (graded only, 9 conditions)**: 0.914

### 2.3 Full Ranking by Severity-3 RA (all 21 graded conditions + 2 dropouts)

| Rank | Corruption | Modality | S3 RA | Mean RA |
|---|---|---|---|---|
| 1 (worst) | complete_dropout | TIR | n/a | **0.309** |
| 2 | low_contrast | RGB | 0.483 | **0.568** |
| 3 | complete_dropout | RGB | n/a | **0.614** |
| 4 | gaussian_noise | RGB | 0.726 | **0.836** |
| 5 | brightness_shift | RGB | 0.750 | **0.859** |
| 6 | intensity_shift | TIR | 0.799 | **0.883** |
| 7 | sensor_noise | TIR | 0.819 | **0.901** |
| 8 | motion_blur | RGB | 0.863 | **0.911** |
| 9 (best) | blur | TIR | 0.928 | **0.959** |

---

## 3. Key Findings

### 3.1 TIR Dropout is the Most Catastrophic Condition (RA = 0.309)

Zeroing the TIR stream collapses mAP from 0.485 to **0.150** — a 69% performance loss. This is the worst observed result, worse than every graded RGB corruption, and reveals that TIR is the functionally dominant modality for Early Fusion on DroneVehicle despite contributing only 1 of the 4 concatenated channels. DroneVehicle includes many night-time scenes where RGB provides negligible signal; the model has learned to rely on TIR thermal contrast as its primary detection cue.

### 3.2 A Degraded Modality Can Be Worse Than an Absent One

RGB low contrast at S3 (RA **0.483**) is worse than RGB dropout (RA **0.614**). Severe contrast reduction doesn't produce zero information — it produces misleading low-amplitude signals that corrupt the joint feature representation more than clean zeros. This is a non-obvious result with practical implications: in a fault-tolerant system, it may be better to zero a heavily degraded sensor than to pass its signal through.

### 3.3 TIR Intensity Shift is More Damaging Than TIR Sensor Noise

TIR intensity_shift at S3: RA **0.799**. TIR sensor_noise at S3: RA **0.819**. A uniform brightness shift (which moves the entire thermal baseline) is slightly more damaging than additive noise (which preserves the baseline but adds variance). This is sensible: the model likely relies on absolute thermal intensity ranges for class discrimination (e.g., vehicle engine heat vs. road surface), so shifting the baseline disrupts this more than adding noise.

By contrast, the RGB analog (brightness_shift S3: RA 0.750) is more damaging than RGB gaussian_noise S3 (RA 0.726) for the same reason.

### 3.4 TIR Blur is Nearly Innocuous (RA = 0.928 at S3)

Gaussian blur at σ=3 on TIR reduces mAP by only 0.35 absolute points. Thermal sensors naturally produce lower spatial resolution than RGB; vehicle-scale detection tolerates smooth TIR streams because detection depends on thermal contrast (hot vehicle vs. cool background), not fine edge detail.

### 3.5 RGB Motion Blur Is Well-Tolerated

Motion blur at S3 (RA **0.863**) is the least damaging RGB corruption. Motion blur preserves spatial frequency information along the perpendicular axis to the blur direction, allowing oriented bounding box regression to partially recover vehicle orientation. Gaussian noise isotropically destroys local texture and produces more damage.

### 3.6 RGB Low Contrast Is the Dominant Vulnerability

Among graded corruptions, low contrast stands out with a mean RA of **0.568** — over 0.2 below the next worst (gaussian_noise at 0.836). This is the critical failure mode for this architecture: any significant reduction in RGB contrast (from glare, overexposure, or lens fogging) causes near-catastrophic performance collapse even at mild severity.

### 3.7 Severity Gradients Are Monotone Throughout

Every corruption type degrades monotonically with severity. No anomalies. The 3-level severity protocol is well-calibrated for this model.

---

## 4. Modality Comparison Summary

| | RGB | TIR |
|---|---|---|
| Dropout RA | 0.614 | **0.309** |
| Graded mean RA | 0.794 | **0.914** |
| Worst graded RA (S3) | 0.483 (low_contrast) | 0.799 (intensity_shift) |
| Best graded RA (S3) | 0.863 (motion_blur) | 0.928 (blur) |

**TIR is the dominant modality**: losing it is catastrophic (RA 0.309), but degrading it gradually is less harmful than degrading RGB. RGB is the vulnerable modality for graded corruptions: its larger channel footprint (3 vs 1) makes it more sensitive to corruption, and low contrast in particular is devastating.

---

## 5. Implications for RQ1

**RQ1 asks**: How does detection performance degrade under different corruption types and severity levels?

For Early Fusion the answer is:

1. **Degradation is strongly corruption-type dependent** — a factor of ~3 separates the worst (TIR dropout, RA 0.309) from the best (TIR blur, RA 0.928) observed condition.
2. **Modality determines degradation mode** — RGB corruptions cause graded decline proportional to channel count; TIR corruptions are milder under graded conditions but catastrophic under dropout.
3. **Severity gradients are monotone and well-behaved** — no threshold effects except a steeper S2→S3 drop for TIR sensor_noise (0.929 → 0.819).
4. **A degraded sensor can be worse than an absent sensor** for RGB low contrast — a result relevant to system-level fault tolerance design.
5. **The safety-critical failure mode** for drone operations using this architecture is TIR sensor malfunction, not RGB degradation.

These findings will be compared against C2Former and UA-CMDet results once those experiments complete, directly addressing whether fusion architecture affects robustness profiles.

---

## 6. Data Reference

All 23 result files: `results/exp1_corruption/early_fusion__*.json`  
Clean baseline: `results/exp0_baseline/early_fusion.json` (mAP = 0.4848)  
Model checkpoint: `work_dirs/early_fusion/epoch_24.pth`  
Experiment script: `experiments/exp1_corruption.py`  
Corruption params: `src/corruption/params.py`  
Pipeline bug fix: `src/corruption/pipeline.py` — `_tir_intensity_shift()` rewritten 2026-05-07
