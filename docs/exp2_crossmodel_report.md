# Experiment 2: Modality Removal — Cross-Model Comparison (RQ2)

**Date**: 2026-05-18 (updated from 2026-05-08 to include UA-CMDet)  
**Models**: Early Fusion (EF) · C2Former (C2F) · UA-CMDet (UAC)  
**Experiment**: exp2_modality_removal.py  
**Dataset**: DroneVehicle test split  
**Metric**: mAP @ IoU 0.5  
**Status**: 138/138 files valid (46 per model × 3 models)

---

## 1. Overview

Exp2 runs two single-modality inference passes for each of the 23 corruption conditions:

- **rgb_only**: TIR stream zeroed to all zeros; RGB stream receives the corrupted (or dropout) image
- **tir_only**: RGB stream zeroed to all zeros; TIR stream receives the corrupted (or dropout) image

These passes isolate per-stream contribution under real-world degradation. Comparing across all three models directly answers RQ2: *which sensor modality contributes most to model resilience under specific degradation types, and which is most critical to preserve?*

---

## 2. Single-Modality Clean Baselines

The complete_dropout conditions provide clean single-modality references — the corrupted stream is zeroed while the other is clean:

| | RGB-only mAP | % of dual clean | TIR-only mAP | % of dual clean | Dominant |
|---|---|---|---|---|---|
| **Early Fusion** | 0.1499 | 30.9% | 0.2977 | 61.4% | **TIR (1.99×)** |
| **C2Former** | 0.4339 | 61.6% | 0.3875 | 55.0% | **RGB (slight, 1.12×)** |
| **UA-CMDet** | 0.2105 | 98.5% | 0.0019 | 0.9% | **RGB (degenerate, 111×)** |

Three architecturally distinct profiles:

- **EF**: TIR is 2× more informative than RGB in single-modality mode. The concatenation architecture concentrates detection capacity in the single TIR channel.
- **C2F**: Both modalities contribute substantially and comparably. The ICA cross-attention training has developed strong single-stream paths for both modalities.
- **UAC**: RGB carries 98.5% of dual-modality performance in isolation. The uncertainty-guided fusion has collapsed to a single-modality RGB detector. TIR contributes effectively nothing.

---

## 3. Under RGB Corruption

When RGB is corrupted, the non-corrupted stream's single-modality performance (`tir_only`) is the potential safety net. The `rgb_only` value shows how much the corrupted RGB stream still contributes in isolation.

| Corruption | EF rgb_only S3 | EF tir_only | C2F rgb_only S3 | C2F tir_only | UAC rgb_only S3 | UAC tir_only |
|---|---|---|---|---|---|---|
| brightness_shift | 0.0945 | **0.2977** | 0.3560 | **0.3875** | 0.1513 | 0.0019 |
| gaussian_noise | 0.0778 | **0.2977** | 0.1700 | **0.3875** | 0.0097 | 0.0019 |
| low_contrast | **0.000** | **0.2977** | 0.2577 | **0.3875** | 0.1428 | 0.0019 |
| motion_blur | 0.0736 | **0.2977** | 0.2646 | **0.3875** | 0.1252 | 0.0019 |
| complete_dropout | 0.000 | **0.2977** | 0.000 | **0.3875** | 0.0001 | 0.0019 |

**Findings:**

1. **EF and C2F both have a meaningful TIR safety net; UAC has none.** When RGB is corrupted, EF can fall back to 0.2977 (61% of dual clean) and C2F to 0.3875 (55% of dual clean). UA-CMDet's TIR fallback is 0.0019 — effectively zero. There is no recovery path under RGB corruption in UA-CMDet.

2. **EF RGB stream collapses more severely than C2F under corruption.** At low_contrast S3, EF rgb_only = 0.000 vs C2F rgb_only = 0.2577. The 4-channel concatenated encoder does not develop a strong standalone RGB detection path; C2Former's separate RGB branch does.

3. **UAC rgb_only under corruption matches UAC dual-modality exp1 results almost exactly**, confirming that UA-CMDet's dual-modality corruption degradation is driven entirely by the RGB stream with zero TIR compensation.

---

## 4. Under TIR Corruption

When TIR is corrupted, the clean `rgb_only` value is the potential safety net. The `tir_only` value shows how much a corrupted TIR stream still contributes in isolation.

| Corruption | EF tir_only S3 | EF rgb_only | C2F tir_only S3 | C2F rgb_only | UAC tir_only S3 | UAC rgb_only |
|---|---|---|---|---|---|---|
| blur | **0.2506** | 0.1499 | 0.1522 | **0.4339** | 0.0015 | **0.2105** |
| intensity_shift | **0.2977** | 0.1499 | 0.3267 | **0.4339** | 0.0015 | **0.2105** |
| sensor_noise | **0.1663** | 0.1499 | 0.0819 | **0.4339** | ~0.000 | **0.2105** |
| complete_dropout | 0.000 | **0.1499** | 0.000 | **0.4339** | ~0.000 | **0.2105** |

**Findings:**

1. **EF corrupted TIR (blur/sensor_noise S3) remains above clean RGB** — even a severely blurred or noisy TIR stream is more useful than a perfectly clean RGB stream in Early Fusion. Only TIR complete dropout inverts this priority.

2. **C2F corrupted TIR collapses far below clean RGB** at all severities. TIR sensor_noise S3: tir_only = 0.0819, clean RGB = 0.4339 (5.3× difference). The ICA attention mechanism queries TIR for spatial structural features; structural corruption destroys those features and the attention output collapses.

3. **UAC tir_only approaches zero for all TIR corruption types** (sensor_noise S3 ≈ 0.000009; blur S3 = 0.0015). The TIR-only model cannot detect anything under corruption. In UA-CMDet under TIR corruption, rgb_only = 0.2105 is the constant, unaffected baseline — confirming that the dual-modality system completely ignores TIR.

---

## 5. The Three Architectural Profiles

### EF: TIR Dominant in All Scenarios

In Early Fusion, TIR dominates under every condition:
- Clean: TIR-only (0.2977) >> RGB-only (0.1499)
- Corrupted TIR: even corrupted tir_only (blur S3: 0.2506, sensor_noise S3: 0.1663) ≥ clean RGB-only (0.1499)
- Only TIR complete dropout reverses the priority (forces fallback to RGB = 0.1499)

The concatenation architecture concentrates detection capacity in the TIR channel during training. RGB contributes only ~31% of dual-modality performance in isolation. **Protecting TIR quality is the sole operational priority in EF deployments.**

### C2F: Symmetric Compensation — Clean Stream Always Wins

In C2Former, the clean stream always outperforms the corrupted stream:
- Under RGB corruption: TIR-only (0.3875) > corrupted RGB-only at most severities
- Under TIR corruption: RGB-only (0.4339) > corrupted TIR-only at all severities

C2Former provides bidirectional compensation through its ICA module. **Both sensors are critical to preserve**; the priority switches based on which is under threat. The architecture answers "which modality matters" with: whichever one is currently intact. However, structural TIR degradation (blur, sensor_noise) propagates severely through attention, making RGB the critical fallback in those specific conditions.

### UAC: Degenerate Single-Modality — No Fault Tolerance

UA-CMDet's uncertainty-guided fusion has learned to assign all weight to RGB. The system behaves as a single-modality RGB detector:
- RGB-only = 98.5% of dual-modality performance
- TIR-only = 0.9% of dual-modality performance
- Under RGB corruption: no TIR compensation; performance tracks the degraded RGB stream
- Under TIR corruption: rgb_only = 0.2105 (constant); tir_only ≈ 0

**UA-CMDet provides zero fault tolerance despite its nominally dual-modality architecture.** The uncertainty estimator learned a degenerate solution: always assign high confidence to RGB, low confidence to TIR, regardless of sensor state. This likely reflects insufficient diversity in the training signal — DroneVehicle's day/night mix was insufficient to force the TIR branch to develop discriminative features that the fusion mechanism could exploit.

---

## 6. Fault Tolerance Comparison

| Failure scenario | EF fallback mAP | EF recovery | C2F fallback mAP | C2F recovery | UAC fallback mAP | UAC recovery |
|---|---|---|---|---|---|---|
| RGB complete dropout | 0.2977 | **61.4%** | 0.3875 | **55.0%** | 0.0019 | **0.9%** |
| TIR complete dropout | 0.1499 | **30.9%** | 0.4339 | **61.6%** | 0.2105 | **98.5%** |
| TIR blur S3 | 0.2506 | **51.7%** | 0.1522 | **21.6%** | ~0.0015 | **~0.7%** |
| TIR sensor_noise S3 | 0.1663 | **34.3%** | 0.0819 | **11.6%** | ~0.000 | **~0%** |
| RGB gaussian_noise S3 | 0.2977* | **61.4%** | 0.1700† | **24.1%** | 0.0097† | **4.5%** |
| RGB low_contrast S3 | 0.2977* | **61.4%** | 0.2577† | **36.6%** | 0.1428† | **66.8%** |

*EF: TIR compensates fully (TIR-only = 0.2977 regardless of RGB corruption type)  
†Corrupted rgb_only value — the corrupted stream in isolation

**Key observations:**

- UAC's RGB failure recovery (0.9%) is catastrophically below EF (61.4%) and C2F (55.0%)
- UAC's TIR failure recovery (98.5%) is paradoxically the highest — but only because TIR was never used
- C2F's TIR structural corruption recoveries (11.6–21.6%) are lower than EF's (34.3–51.7%) due to attention-based propagation
- EF's RGB failure recovery (61.4%) is consistently strong — the TIR safety net is always available and always dominant

---

## 7. Modality Priority Map for RQ2

The table below answers, for each condition: which modality must be preserved?

| Corruption type | Corrupted modality | EF: preserve | C2F: preserve | UAC: preserve |
|---|---|---|---|---|
| All RGB graded corruptions | RGB | **TIR** (always dominant) | **TIR** (S1+ for noise/blur/contrast; S3 for bright.) | **RGB** (TIR useless) |
| TIR blur / sensor_noise | TIR | **TIR** (still > clean RGB) | **RGB** (critical fallback) | **RGB** (TIR irrelevant) |
| TIR intensity_shift | TIR | **TIR** (no degradation) | **RGB** (minor advantage) | **RGB** (TIR irrelevant) |
| TIR complete dropout | TIR | **RGB** (only fallback) | **RGB** (only fallback) | **RGB** (no change) |
| RGB complete dropout | RGB | **TIR** (only fallback) | **TIR** (only fallback) | Neither (both fail) |

---

## 8. Summary Answer to RQ2

**RQ2 asks**: Which sensor modality contributes most to model resilience under specific degradation types, and which is most critical to preserve?

The answer is architecture-dependent:

> **Early Fusion**: TIR is the critical modality under every realistic condition. Even a severely corrupted TIR stream outperforms a clean RGB stream. RGB provides a meaningful fallback only under total TIR failure. Protect the TIR sensor above all else.

> **C2Former**: The critical modality to preserve is whichever is currently clean. Cross-attention provides symmetric compensation: clean TIR compensates for degraded RGB; clean RGB compensates for degraded TIR. However, structural TIR corruption (blur, sensor_noise) propagates severely through attention, making RGB the critical fallback specifically when TIR faces structural threats. Both sensors must be maintained, with heightened concern for structural TIR integrity.

> **UA-CMDet**: RGB is the only modality with any detection value. TIR preservation has no impact on system performance because the fusion mechanism does not use it. Critically, a failed RGB sensor disables the system entirely with no fallback — despite the nominally dual-modality design. UA-CMDet provides no fault-tolerance benefit over a single-modality RGB detector.

**The cross-architecture finding**: Fusion architecture — not the sensor hardware or dataset — determines which modality becomes load-bearing and whether any fault tolerance exists. A dual-modality architecture does not guarantee dual-modality fault tolerance if the fusion mechanism degenerates.

---

## 9. Data Reference

Per-model exp2 reports:
- `docs/exp2_early_fusion_report.md` (46/46 conditions)
- `docs/exp2_c2former_report.md` (46/46 conditions)
- `docs/exp2_ua_cmddet_report.md` (46/46 conditions)

Result files: `results/exp2_modality_removal/` (138 files: 46 per model)  
Clean baselines: `results/exp0_baseline/*.json`  
Experiment script: `experiments/exp2_modality_removal.py`
