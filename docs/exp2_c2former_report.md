# Experiment 2: Modality Removal — C2Former (RQ2)

**Date**: 2026-05-08  
**Model**: C2Former (Inter-modality Cross-Attention, ResNet-50 backbone)  
**Experiment**: exp2_modality_removal.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5  
**Status**: 46/46 valid (23 conditions × 2 modality configs: rgb_only, tir_only)

---

## 1. Overview

Experiment 2 answers RQ2 for C2Former: under each corruption condition, which modality stream carries the detection performance? For every one of the 23 exp1 conditions, two additional inference passes are run:

- **rgb_only**: TIR stream zeroed to all zeros; RGB stream receives the corrupted (or dropout) image
- **tir_only**: RGB stream zeroed to all zeros; TIR stream receives the corrupted (or dropout) image

**Dual-modality clean baseline mAP**: 0.7046

---

## 2. Single-Modality Clean Baselines

| Condition | mAP | Share of dual-modality clean |
|---|---|---|
| RGB-only (TIR=zeros) | **0.4339** | 61.6% |
| TIR-only (RGB=zeros) | **0.3875** | 55.0% |

**RGB-only (0.4339) slightly exceeds TIR-only (0.3875)** in C2Former — the opposite of Early Fusion. Both streams contribute substantially, with neither dominating. The cross-attention training has developed comparably capable single-stream paths for both modalities.

The combined clean mAP (0.7046) substantially exceeds both single-modality values, confirming that the ICA module adds genuine value when both streams are present.

---

## 3. Results: Under RGB Corruption

When RGB is corrupted, tir_only (clean TIR, RGB zeroed) provides a constant reference of **0.3875**.

| Corruption | S1 rgb_only | S2 rgb_only | S3 rgb_only | tir_only | TIR wins from |
|---|---|---|---|---|---|
| brightness_shift | 0.4302 | 0.4072 | 0.3560 | 0.3875 | S3 |
| gaussian_noise | 0.3263 | 0.2595 | 0.1700 | 0.3875 | S1 |
| low_contrast | 0.3503 | 0.3044 | 0.2577 | 0.3875 | S1 |
| motion_blur | 0.3886 | 0.3391 | 0.2646 | 0.3875 | S1–S2 boundary |
| complete_dropout | 0.000 | — | — | 0.3875 | — |

**Key findings:**
- For brightness_shift, RGB-only retains the advantage at S1 (0.4302) and S2 (0.4072), with TIR winning only at S3 (0.3560 < 0.3875). Brightness shift mildly degrades RGB's structural information.
- For gaussian_noise, low_contrast, and motion_blur, TIR-only immediately outperforms rgb_only from S1. These corruptions disrupt RGB spatial structure enough that even the clean TIR stream becomes more informative than the corrupted RGB.
- Gaussian_noise S3 rgb_only (0.1700) is the most severe RGB degradation, yet TIR-only (0.3875) provides 2.3× better detection.
- The crossover point (where corrupted RGB-only falls below clean TIR-only) depends on corruption type and is a useful design threshold.

---

## 4. Results: Under TIR Corruption

When TIR is corrupted, rgb_only (clean RGB, TIR zeroed) provides a constant reference of **0.4339**.

| Corruption | S1 tir_only | S2 tir_only | S3 tir_only | rgb_only | RGB advantage at S3 |
|---|---|---|---|---|---|
| blur | 0.3477 | 0.2515 | 0.1522 | 0.4339 | +0.282 |
| intensity_shift | 0.3980 | 0.3685 | 0.3267 | 0.4339 | +0.107 |
| sensor_noise | 0.2817 | 0.2145 | 0.0819 | 0.4339 | +0.352 |
| complete_dropout | 0.000 | — | — | 0.4339 | — |

**Key findings:**
- RGB-only (0.4339) outperforms corrupted TIR at all severity levels and for all TIR corruption types.
- TIR sensor_noise collapses severely: S3 tir_only = 0.0819. C2Former's ICA module, which queries TIR for spatial structural features, is acutely sensitive to per-pixel random noise that destroys those features.
- TIR blur: S3 tir_only = 0.1522. Blur removes high-frequency TIR edges (thermal object boundaries), which are the features the attention mechanism relies on.
- TIR intensity_shift is relatively benign: even at S3, tir_only = 0.3267 (vs rgb_only = 0.4339). Global brightness preservation of spatial structure partially sustains cross-attention feature quality.
- Under TIR corruption, RGB is the clear dominant fallback stream at every severity level.

---

## 5. Key Analysis

### 5.1 Symmetric Compensation Under Corruption

C2Former's ICA module provides symmetric fault tolerance: when RGB is corrupted, TIR compensates; when TIR is corrupted, RGB compensates. This is qualitatively different from Early Fusion's TIR-dominant profile.

The non-corrupted stream is always the better predictor — regardless of which modality is degraded, the clean stream outperforms the corrupted one above moderate severity. This is the operational definition of a balanced fusion architecture.

### 5.2 RGB Is the Stronger Single-Modality Stream

Clean RGB-only (0.4339) exceeds clean TIR-only (0.3875) by 12%. Under corruption, this means:
- When TIR is corrupted, the RGB fallback (0.4339) is relatively strong
- When RGB is corrupted, the TIR fallback (0.3875) is somewhat weaker in absolute terms

However, the corrupted stream degrades much faster in TIR (sensor_noise s3: 0.0819) than in RGB (gaussian_noise s3: 0.1700), so the effective RGB advantage under TIR corruption is larger than the clean baseline gap suggests.

### 5.3 Cross-Attention Creates a Structural TIR Dependency

Under TIR structural corruption (blur, sensor_noise), tir_only collapses much more severely than rgb_only under comparable RGB structural corruptions:

| Condition | Corrupted stream only mAP | Clean fallback mAP | Ratio |
|---|---|---|---|
| TIR sensor_noise S3 (tir_only) | 0.0819 | 0.4339 (RGB) | 0.19 |
| TIR blur S3 (tir_only) | 0.1522 | 0.4339 (RGB) | 0.35 |
| RGB gaussian_noise S3 (rgb_only) | 0.1700 | 0.3875 (TIR) | 0.44 |
| RGB low_contrast S3 (rgb_only) | 0.2577 | 0.3875 (TIR) | 0.66 |

The cross-attention mechanism is more sensitive to structural TIR degradation than to structural RGB degradation. TIR sensor_noise at S3 reduces tir_only to 19% of the clean RGB fallback — a near-total collapse of the TIR stream. This mirrors the exp1 finding that C2Former is less robust to structural TIR corruptions than Early Fusion.

### 5.4 Brightness Shift — RGB Retains Advantage at Low–Moderate Severity

For brightness_shift, rgb_only at S1 (0.4302) and S2 (0.4072) exceeds tir_only (0.3875). This is the one RGB corruption where RGB remains the better stream at low severity. Brightness shift preserves spatial structure (edges, shapes, relative contrasts) while only shifting the absolute intensity level — the model can still detect objects from shape and relative-contrast features even under this corruption.

This finding connects to the exp1 result that C2Former has high RA for RGB brightness_shift (0.973/0.928/0.884): the RGB stream degrades gracefully under photometric shifts, and the cross-attention to clean TIR provides only marginal additional benefit at low severity.

### 5.5 Fault Tolerance Profile

| Failure scenario | Fallback mAP | Recovery rate vs dual |
|---|---|---|
| RGB complete failure | 0.3875 (TIR-only) | 55.0% |
| TIR complete failure | 0.4339 (RGB-only) | 61.6% |
| TIR blur severity 3 | 0.1522 (corrupted TIR) | 21.6% |
| TIR sensor_noise severity 3 | 0.0819 (corrupted TIR) | 11.6% |
| RGB gaussian_noise severity 3 | 0.1700 (corrupted RGB) | 24.1% |
| RGB low_contrast severity 3 | 0.2577 (corrupted RGB) | 36.6% |

Sensor complete failure: TIR failure (recovery 61.6%) is slightly better than RGB failure (recovery 55.0%) — the opposite of Early Fusion's profile. Under severe structural corruption, TIR stream collapses more catastrophically than RGB.

---

## 6. Implications for RQ2 (C2Former Component)

**RQ2 asks**: Which sensor modality contributes most to model resilience under specific degradation types, and which is most critical to preserve?

For C2Former, the answer depends on which modality is under threat:

1. **Under RGB corruption**: TIR is the more resilient backup stream (wins from S1 for gaussian_noise, low_contrast, motion_blur; wins at S3 for brightness_shift). Preserve TIR when RGB is threatened.
2. **Under TIR corruption**: RGB is the more resilient backup stream (wins at all severities). Preserve RGB when TIR is threatened.
3. **Under complete failure**: TIR complete failure (fallback = RGB = 0.4339) is more survivable than RGB complete failure (fallback = TIR = 0.3875).
4. **Under structural TIR corruption (blur, noise)**: the TIR stream collapses severely (0.08–0.15 at S3). RGB is the critical modality to preserve when TIR is exposed to structural degradation.
5. **No single modality is permanently dominant**: the architecture provides symmetric compensation that makes both sensors equally important in principle, but RGB provides a stronger fallback under structural TIR corruption.

---

## 7. Data Reference

Result files: `results/exp2_modality_removal/c2former__*.json` (46 files)  
Dual-modality baseline: `results/exp0_baseline/c2former.json` (mAP = 0.7046)  
Model checkpoint: `work_dirs/c2former/epoch_24.pth`  
Experiment script: `experiments/exp2_modality_removal.py`
