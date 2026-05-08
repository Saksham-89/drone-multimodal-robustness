# Experiment 2: Modality Removal — Cross-Model Comparison (RQ2)

**Date**: 2026-05-08  
**Models**: Early Fusion (concat) vs C2Former (cross-attention)  
**Experiment**: exp2_modality_removal.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5  
**Status**: 92/92 files valid (46 per model)

---

## 1. Overview

Exp2 runs two single-modality inference passes for each of the 23 corruption conditions:

- **rgb_only**: TIR zeroed, RGB active (corrupted or clean)
- **tir_only**: RGB zeroed, TIR active (corrupted or clean)

Comparing these passes across Early Fusion (EF) and C2Former (C2F) directly answers RQ2: which modality is most critical to preserve, and does the answer depend on the fusion architecture?

---

## 2. Single-Modality Clean Baselines

The complete_dropout conditions give clean single-modality references — one stream receives total dropout while the other is clean and active:

| | RGB-only mAP | TIR-only mAP | Dominant modality | TIR/RGB ratio |
|---|---|---|---|---|
| **Early Fusion** | 0.1499 | 0.2977 | TIR | **1.99×** |
| **C2Former** | 0.4339 | 0.3875 | RGB (slight) | 0.89× |
| **Dual clean** | 0.4848 (EF) | 0.7046 (C2F) | — | — |

This single table captures the fundamental difference in modality dependence:

- **Early Fusion is TIR-centric**: TIR is 2× more informative in single-modality mode; RGB provides only 31% of dual-modality performance when isolated.
- **C2Former is balanced**: RGB and TIR contribute similarly, with RGB slightly ahead. C2Former's cross-attention training has developed strong pathways for both modalities.

---

## 3. Under RGB Corruption: TIR as Safety Net

When RGB is corrupted, the tir_only value (constant = clean TIR baseline) serves as the safety net.

| Corruption | EF rgb_only S3 | EF tir_only | C2F rgb_only S3 | C2F tir_only |
|---|---|---|---|---|
| brightness_shift | 0.0945 | 0.2977 | 0.3560 | 0.3875 |
| gaussian_noise | 0.0778 | 0.2977 | 0.1700 | 0.3875 |
| low_contrast | **0.000** | 0.2977 | 0.2577 | 0.3875 |
| motion_blur | 0.0736 | 0.2977 | 0.2646 | 0.3875 |

**Findings:**
1. In EF, corrupted RGB-only collapses to near-zero at S3 for all types. The TIR safety net (0.2977) provides full compensation — recovery of 61% of dual clean performance even when RGB fails completely.
2. In C2Former, corrupted RGB-only degrades substantially but stays above 0.17 at S3. The TIR safety net is also stronger in absolute terms (0.3875).
3. EF's RGB collapse is more severe: low contrast S3 = 0.000 (total failure) vs C2F = 0.2577. This reflects EF's weaker single-channel RGB path — the concatenated tensor does not train a robust RGB detector.
4. TIR is the correct modality to preserve under RGB corruption in both architectures. The ICA module enables stronger TIR compensation in C2Former (recovery to 55% vs 61% — similar), but C2Former also retains more from the corrupted RGB stream itself.

---

## 4. Under TIR Corruption: Asymmetric Profiles

When TIR is corrupted, rgb_only (constant = clean RGB baseline) serves as the reference.

| Corruption | EF tir_only S3 | EF rgb_only | C2F tir_only S3 | C2F rgb_only |
|---|---|---|---|---|
| blur | **0.2506** | 0.1499 | 0.1522 | 0.4339 |
| intensity_shift | **0.2977** | 0.1499 | 0.3267 | 0.4339 |
| sensor_noise | **0.1663** | 0.1499 | 0.0819 | 0.4339 |

**Findings:**
1. In EF, corrupted TIR-only at S3 is still **above** clean RGB-only in 2 of 3 types (blur: 0.2506 > 0.1499; sensor_noise: 0.1663 > 0.1499). Only complete TIR dropout falls below the RGB fallback. **Even a severely degraded TIR stream in EF is more useful than a perfectly clean RGB stream.**
2. In C2Former, corrupted TIR-only at S3 is **well below** clean RGB-only in all 3 types. RGB provides 2–5× more detection value than severely corrupted TIR. The cross-attention mechanism amplifies TIR degradation through the attention weights.
3. This is the starkest single finding: **the same TIR corruption (blur S3) produces tir_only = 0.2506 in EF but only 0.1522 in C2F** — C2F's cross-attention collapses the TIR stream 40% further than EF's concatenation.
4. EF TIR intensity shift at all severities: tir_only = 0.2977 (identical to clean TIR) — EF is completely robust to TIR brightness variation at the stream level.

---

## 5. The Central Architectural Finding

### EF: TIR Dominant in All Scenarios

In Early Fusion, TIR dominates under every condition:
- When TIR is clean: TIR-only (0.2977) >> RGB-only (0.1499)
- When TIR is corrupted: even corrupted TIR-only (blur S3: 0.2506, noise S3: 0.1663) ≥ clean RGB-only (0.1499)
- Only TIR complete dropout reverses the priority

RGB in Early Fusion is not a meaningful fallback except under total TIR failure. The concatenation architecture concentrates detection capacity in the TIR channel. **Protecting TIR sensor quality is the only operational priority in Early Fusion deployments.**

### C2F: Non-Corrupted Stream Always Wins

In C2Former, the clean stream always outperforms the corrupted stream:
- Under RGB corruption (any type, any severity): TIR-only (0.3875) > corrupted RGB-only (except brightness_shift S1–S2)
- Under TIR corruption (any type, any severity): RGB-only (0.4339) > corrupted TIR-only

C2Former provides bidirectional compensation through its ICA module. **Both sensors are critical to preserve**, but the priority switches based on which is under threat. The architecture resolves the "which modality matters more" question by making it situation-dependent.

---

## 6. Fault Tolerance Comparison

| Failure scenario | EF fallback mAP | EF recovery | C2F fallback mAP | C2F recovery |
|---|---|---|---|---|
| RGB complete dropout | 0.2977 | 61.4% | 0.3875 | 55.0% |
| TIR complete dropout | 0.1499 | 30.9% | 0.4339 | 61.6% |
| TIR blur S3 | 0.2506 | 51.7% | 0.1522 | 21.6% |
| TIR sensor_noise S3 | 0.1663 | 34.3% | 0.0819 | 11.6% |
| RGB gaussian_noise S3 | 0.2977 | 61.4% | 0.1700* | 24.1%* |

*C2F rgb_only under gaussian noise S3; EF uses TIR fallback under RGB corruption (they're not the same scenario).

**Recovery rate asymmetry:**
- EF: TIR failure recovery (30.9%) is half of RGB failure recovery (61.4%)
- C2F: TIR failure recovery (61.6%) exceeds RGB failure recovery (55.0%) — the opposite

The cross-attention architecture inverts the modality priority for complete-failure recovery. Under partial/graded degradation, however, structural TIR degradation (blur, noise) is much more damaging to C2F (11–22% recovery) than to EF (34–52% recovery).

---

## 7. Modality Priority Map for RQ2

| Corruption type | Corrupted modality | EF: preserve | C2F: preserve |
|---|---|---|---|
| All RGB corruptions | RGB | TIR | TIR (from S1–S3) / RGB (bright. S1–S2) |
| TIR blur / noise | TIR | TIR (still dominant) | RGB (critical fallback) |
| TIR intensity shift | TIR | TIR (no degradation) | RGB (minor advantage) |
| TIR complete dropout | TIR | RGB (only fallback) | RGB (only fallback) |
| RGB complete dropout | RGB | TIR (only fallback) | TIR (only fallback) |

**Summary for thesis RQ2:**

> In Early Fusion, TIR is the critical modality to preserve under all conditions — even a severely corrupted TIR stream outperforms a clean RGB stream. This is a direct consequence of the concatenation architecture concentrating detection capacity in the TIR channel during training.
>
> In C2Former, the critical modality to preserve is the one that is currently clean. The cross-attention ICA module provides symmetric compensation: clean TIR compensates for degraded RGB; clean RGB compensates for degraded TIR. However, structural TIR degradation (blur, noise) propagates severely through the attention mechanism, making RGB the critical fallback when TIR faces structural threats. The architecture answers "which modality matters" with: "whichever one is currently intact."

---

## 8. Data Reference

Result files: `results/exp2_modality_removal/` (92 files: 46 per model)  
Clean baselines: `results/exp0_baseline/early_fusion.json`, `results/exp0_baseline/c2former.json`  
Model reports: `docs/exp2_early_fusion_report.md`, `docs/exp2_c2former_report.md`
