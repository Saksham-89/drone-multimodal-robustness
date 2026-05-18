# Thesis Project Handoff

**Author**: Saksham Singh Birla  
**Institution**: University of Twente, EEMCS  
**Venue**: TSCiT 2025  
**Handoff date**: 2026-05-18  
**Project root**: `C:\Users\saksh\Desktop\Thesis\drone-multimodal-robustness`  
**Git branch**: master

> Read `CLAUDE.md` for detailed architecture specs, corruption parameter tables, HPC cluster instructions, and dataset paths. This file covers experimental status, results, statistical findings, and what to do next.

---

## Status Summary

**All experiments are complete. Statistical analysis is complete. Thesis writing has not started.**

| Phase | Status |
|---|---|
| Environment setup | Done |
| Model training (HPC) | Done — all 3 models |
| exp0: Clean baseline eval | Done — all 3 models |
| exp1: Corruption benchmark | Done — all 3 models, 23 conditions each |
| exp2: Modality removal | Done — all 3 models, 46 conditions each |
| Per-model reports | Done — all 9 reports written |
| Cross-model reports | Done — exp1 and exp2 crossmodel written |
| Statistical analysis (nb_00–nb_04) | **Done — all 5 analysis notebooks complete** |
| Publication figures (nb_05) | **Done — all 5 figures generated** |
| **Thesis document** | **Not started** |

---

## The Thesis in One Paragraph

Three RGB+TIR drone detection models (Early Fusion, C2Former, UA-CMDet), all ResNet-50 backbone, all trained on DroneVehicle, were evaluated on 23 corruption conditions applied independently to each modality, plus modality zeroing experiments. The core finding is that fusion architecture — not the backbone or dataset — determines which sensor becomes the critical vulnerability. Early Fusion is TIR-dominant (TIR dropout RA=0.309), C2Former is balanced (RGB and TIR both contribute ~55–62%), and UA-CMDet has collapsed to a functionally single-modality RGB detector (RGB-only = 98.5% of dual-modality performance, TIR-only = 0.9%). The cross-attention architecture (C2Former) has the best worst-case floor (RA=0.550). UA-CMDet's nominally dual-modality design provides zero fault tolerance — a key negative result. **Statistical analysis confirms severity gradients are monotone across 19/21 graded conditions (all significant p<0.05), model mean RA differences are not statistically significant (Friedman p=0.296), and modality dominance is confirmed for EF (TIR-dominant, DI=+0.54, p=0.0004) and UA-CMDet (RGB-dominant, DI=-0.90, p≈0) by Wilcoxon test.**

---

## Research Questions

**RQ1** (Diagnostic): How does detection performance degrade under different types and severity levels of sensor corruption?  
**RQ2** (Attributional): Which sensor modality contributes most to model resilience, and which is most critical to preserve?

Both are answered fully by exp1 (RQ1) and exp2 (RQ2). The cross-model reports synthesise the answers. The statistical analysis notebooks provide formal validation.

---

## Dataset

**DroneVehicle** (Sun et al., TCSVT 2022): 28,439 RGB+TIR image pairs, 5 vehicle classes (car, truck, bus, van, freight_car), oriented bounding box annotations. Test split used exclusively for evaluation — no fine-tuning.

Data on HPC at: `/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle/`

---

## Models

| Model | Fusion | Framework | Config | Checkpoint | Epochs |
|---|---|---|---|---|---|
| Early Fusion | Early concat (4-ch input) | mmrotate | `experiments/configs/early_fusion_dronevehicle.py` | `work_dirs/early_fusion/epoch_24.pth` | 24 |
| C2Former | Intermediate cross-attention (ICA) | mmrotate | `experiments/configs/c2former_dronevehicle.py` (on HPC) | `work_dirs/c2former/epoch_24.pth` | 24 |
| UA-CMDet | Late uncertainty-guided fusion | AerialDetection (mmdet 1.x fork) | `models/ua_cmddet/configs/DroneVehicle/UACMDet.py` | `work_dirs/ua_cmddet/latest.pth` | 12 |

---

## Clean Baselines (exp0)

| Model | Test mAP | Eval protocol |
|---|---|---|
| Early Fusion | **0.4848** | OBB polygon IoU (mmrotate) |
| C2Former | **0.7046** | OBB polygon IoU (mmrotate) |
| UA-CMDet | **0.2137** | HBB axis-aligned COCO IoU |

**Critical caveat**: UA-CMDet uses a different eval protocol (HBB COCO IoU, not OBB polygon IoU). The published UA-CMDet figure (~0.412) used OBB polygon IoU with the DOTA devkit — that's why ours is lower. **Absolute mAP numbers are NOT comparable across models.** RA values (corrupted/clean) are comparable because each model divides by its own clean baseline. This caveat must appear in the thesis methodology section.

Per-class AP (C2Former only, from exp0):

| Class | Car | Truck | Freight car | Bus | Van | mAP |
|---|---|---|---|---|---|---|
| C2Former | 0.894 | 0.682 | 0.542 | 0.889 | 0.515 | 0.705 |

EF and UAC per-class AP not available in stored JSON (only mAP scalar).

---

## Experiment 1: Corruption Benchmark Results

**23 conditions per model** (4 RGB types × 3 severities + 3 TIR types × 3 severities + 2 binary dropouts).  
RA = corrupted_mAP / clean_mAP.

### Summary Statistics (nb_01)

| Model | Mean RA | Std | Median | IQR | Min RA | Max RA | CV |
|---|---|---|---|---|---|---|---|
| C2Former | 0.8701 | 0.1138 | 0.8932 | 0.1228 | 0.5500 | 0.9951 | 0.131 |
| Early Fusion | 0.8119 | 0.1781 | 0.8704 | 0.2026 | 0.3090 | 0.9943 | 0.219 |
| UA-CMDet | 0.7477 | 0.3069 | 0.8438 | 0.2921 | 0.0058 | 1.0150 | 0.411 |

CV (coefficient of variation) ranks: C2F most consistent, UAC most variable. All three models' 95% bootstrap CIs overlap substantially (see nb_03).

### Mean RA by Modality Stream (graded only, nb_01)

| Model | RGB mean RA | TIR mean RA |
|---|---|---|
| C2Former | 0.8852 | 0.9139 |
| Early Fusion | 0.7936 | 0.9142 |
| UA-CMDet | 0.6268 | 0.9625 |

### Complete Dropout RA (binary, nb_01)

| Model | RGB dropout RA | TIR dropout RA |
|---|---|---|
| C2Former | 0.5500 | 0.6158 |
| Early Fusion | 0.6141 | 0.3090 |
| UA-CMDet | 0.0058 | 1.0072* |

*UAC TIR dropout RA > 1.0 is confirmed degenerate (model ignores TIR entirely).

### RGB Corruptions — Mean RA across S1/S2/S3

| Corruption | Early Fusion | C2Former | UA-CMDet |
|---|---|---|---|
| gaussian_noise | 0.836 | 0.861 | **0.212** |
| motion_blur | 0.911 | **0.934** | 0.743 |
| brightness_shift | 0.859 | **0.928** | 0.786 |
| low_contrast | 0.568 | **0.817** | 0.766 |
| complete_dropout | 0.614 | 0.550 | **0.006** |
| **Graded mean RA** | 0.794 | **0.885** | 0.627 |

### TIR Corruptions — Mean RA across S1/S2/S3

| Corruption | Early Fusion | C2Former | UA-CMDet |
|---|---|---|---|
| sensor_noise | **0.901** | 0.864 | 0.889 |
| blur | 0.959 | 0.902 | **0.990** |
| intensity_shift | 0.883 | **0.975** | 1.009* |
| complete_dropout | 0.309 | 0.616 | **1.007*** |
| **Graded mean RA** | 0.914 | 0.914 | (0.963*) |

*UAC TIR RA ≥ 1.0 because the model ignores TIR entirely.

### Worst-Case RA by Model

| Model | Worst condition | Worst RA |
|---|---|---|
| Early Fusion | TIR complete_dropout | 0.309 |
| C2Former | RGB complete_dropout | **0.550** |
| UA-CMDet | RGB complete_dropout | **0.006** |

---

## Experiment 2: Modality Removal Results

**46 conditions per model** (23 conditions × 2 passes: rgb_only, tir_only).

### Single-Modality Clean Baselines (nb_04)

| Model | RGB-only mAP | % of dual | TIR-only mAP | % of dual |
|---|---|---|---|---|
| Early Fusion | 0.1499 | 30.9% | **0.2977** | **61.4%** |
| C2Former | **0.4339** | 61.6% | 0.3875 | 55.0% |
| UA-CMDet | **0.2105** | **98.5%** | 0.0019 | 0.9% |

### Under RGB Corruption — S3 TIR fallback

| Corruption | EF TIR fallback | C2F TIR fallback | UAC TIR fallback |
|---|---|---|---|
| brightness_shift | 0.2977 | 0.3875 | 0.0019 |
| gaussian_noise | 0.2977 | 0.3875 | 0.0019 |
| low_contrast | 0.2977 | 0.3875 | 0.0019 |
| motion_blur | 0.2977 | 0.3875 | 0.0019 |

### Under TIR Corruption — S3 RGB fallback

| Corruption | EF RGB fallback | C2F RGB fallback | UAC RGB fallback |
|---|---|---|---|
| blur | 0.1499 | 0.4339 | 0.2105 |
| intensity_shift | 0.1499 | 0.4339 | 0.2105 |
| sensor_noise | 0.1499 | 0.4339 | 0.2105 |

---

## Statistical Analysis Results

All results validated and written to `analysis/outputs/nb0X_results.txt`. Notebooks are in `analysis/notebooks/`.

### nb_03: Model Comparison Tests

**Friedman test** (N=23, k=3):
- Kendall's W = 0.053, χ² = 2.435, **p = 0.296 — NOT significant**
- Interpretation: no statistically significant difference in mean RA across the three models

**Wilcoxon signed-rank post-hoc (Holm-Bonferroni corrected)**:

| Pair | W | p_raw | p_holm | Reject H₀ |
|---|---|---|---|---|
| C2Former vs Early Fusion | 76.0 | 0.061 | 0.182 | No |
| C2Former vs UA-CMDet | 78.0 | 0.070 | 0.182 | No |
| Early Fusion vs UA-CMDet | 120.0 | 0.601 | 0.601 | No |

**Cliff's δ effect sizes**:

| Pair | δ | Magnitude |
|---|---|---|
| C2Former vs Early Fusion | 0.168 | Small |
| C2Former vs UA-CMDet | 0.134 | Negligible |
| Early Fusion vs UA-CMDet | 0.021 | Negligible |

**Bootstrap 95% CI on mean RA** (10,000 resamples):

| Model | Observed mean | CI lower | CI upper | CI width |
|---|---|---|---|---|
| C2Former | 0.8701 | 0.8213 | 0.9122 | 0.091 |
| Early Fusion | 0.8119 | 0.7374 | 0.8770 | 0.140 |
| UA-CMDet | 0.7477 | 0.6160 | 0.8605 | 0.245 |

All three CIs overlap — consistent with non-significant Friedman result.

**Spearman inter-model correlation** (RA vectors across 23 conditions):

| Pair | ρ | p-value |
|---|---|---|
| Early Fusion vs C2Former | 0.806 | 0.000 |
| C2Former vs UA-CMDet | 0.494 | 0.017 |
| Early Fusion vs UA-CMDet | 0.381 | 0.073 |

EF and C2F share very similar vulnerability profiles (ρ=0.806). UAC is architecturally distinct, especially vs EF (ρ=0.381, p=0.073 — marginal).

**Thesis framing implication**: The statistical case for C2Former cannot rest on mean RA superiority (non-significant). It must be argued via (1) worst-case robustness floor (RA=0.550 vs 0.309/0.006), (2) most consistent CV, (3) smallest bootstrap CI width, and (4) Spearman vulnerability similarity to EF but with better absolute floor.

---

### nb_02: Severity Monotonicity

- **19/21** graded conditions show Spearman ρ = −1.0 (perfect monotone decrease)
- **21/21** significant at p < 0.05
- **2 exceptions**: UA-CMDet TIR blur (ρ=+1.0, RA increases S1→S3: 0.984→0.991→0.994) and UA-CMDet TIR intensity_shift (ρ=+1.0, RA: 1.003→1.008→1.015). Both are degenerate — UAC ignores TIR so TIR corruption has no effect; the slight RA increase is noise around 1.0. These are not genuine monotonicity violations.
- **Thesis caveat required**: "19/21 graded conditions showed monotone RA decline (Spearman ρ = −1.0, all p < 0.05). The 2 exceptions are UA-CMDet under TIR corruptions, where RA ≈ 1.0 across all severities because the model is functionally TIR-blind. These non-monotone cases confirm the TIR-blindness finding rather than contradicting the severity calibration."

**Top 3 steepest degradations (slope β per severity unit)**:

| Rank | Model | Corruption | Slope (β) | RA at S1 | RA at S3 | Total drop |
|---|---|---|---|---|---|---|
| 1 | UA-CMDet | gaussian_noise (RGB) | −0.1922 | 0.427 | 0.043 | −0.384 |
| 2 | UA-CMDet | motion_blur (RGB) | −0.1420 | 0.864 | 0.580 | −0.284 |
| 3 | Early Fusion | brightness_shift (RGB) | −0.1041 | 0.958 | 0.750 | −0.208 |

**Degradation slopes by corruption and model**:

| Corruption | C2Former β | Early Fusion β | UA-CMDet β |
|---|---|---|---|
| blur (TIR) | −0.0773 | −0.0333 | +0.0046* |
| brightness_shift (RGB) | −0.0445 | −0.1041 | −0.0769 |
| gaussian_noise (RGB) | −0.0692 | −0.1023 | **−0.1922** |
| intensity_shift (TIR) | −0.0220 | −0.0803 | +0.0060* |
| low_contrast (RGB) | −0.0536 | −0.0785 | −0.0829 |
| motion_blur (RGB) | −0.0419 | −0.0434 | −0.1420 |
| sensor_noise (TIR) | −0.0821 | −0.0683 | −0.0426 |

*Positive slopes = degenerate TIR-blind UA-CMDet cases.

All R² values > 0.88 — linear degradation model fits well across all conditions.

---

### nb_04: Modality Dominance Analysis

**Dominance Index (DI)** = (tir_only_map − rgb_only_map) / max(tir_only_map, rgb_only_map + ε)  
Positive = TIR-dominant; Negative = RGB-dominant. Tested with Wilcoxon signed-rank vs 0.

| Model | N | Mean DI | Median DI | W | p-value | Dominant | Significant |
|---|---|---|---|---|---|---|---|
| Early Fusion | 23 | +0.539 | +0.572 | 21.0 | **0.0004** | TIR | Yes |
| C2Former | 23 | −0.059 | −0.048 | 118.0 | 0.560 | (RGB) | **No** |
| UA-CMDet | 23 | −0.899 | −0.990 | 2.0 | **≈0.000** | RGB | Yes |

- **Early Fusion**: Strongly TIR-dominant (p=0.0004). Clean TIR-only = 61.4% of dual; clean RGB-only = 30.9%.
- **C2Former**: No significant dominance (p=0.56). Balanced: TIR=55.0%, RGB=61.6% of dual. DI ≈ 0 confirms symmetric contribution.
- **UA-CMDet**: Extremely RGB-dominant (p≈0). RGB=98.5% of dual; TIR=0.9% of dual. DI median = −0.990 (near maximum negative).

**Fusion Gain** (FG = dual_map − max(rgb_only, tir_only)) across 23 conditions:

| Model | Mean FG | Std | Median | Min | Max |
|---|---|---|---|---|---|
| C2Former | 0.2026 | 0.079 | 0.225 | 0.000 | 0.297 |
| Early Fusion | 0.1145 | 0.082 | 0.133 | −0.064 | 0.231 |
| UA-CMDet | −0.0026 | 0.009 | 0.001 | −0.031 | 0.006 |

- C2Former gains the most from fusion (+0.20 mAP above best single stream on average). Cross-attention architecture actively exploits both modalities.
- Early Fusion gains +0.11, with occasional negative FG (degraded RGB drags the concat encoder).
- UA-CMDet FG ≈ 0 — no meaningful fusion gain. The dual-modality input contributes nothing beyond RGB alone.

**Spearman ρ: exp1 dual RA vs Fallback Efficiency** (FE = max_single/dual_corrupted):

| Model | ρ | p-value |
|---|---|---|
| Early Fusion | −0.816 | 0.000 |
| C2Former | −0.722 | 0.000 |
| UA-CMDet | −0.643 | 0.001 |

Negative ρ is a mathematical artifact: when corrupted dual-mAP is small (low RA), FE = single/small → large. Not a substantive finding; do not cite in thesis.

---

## Key Findings (thesis-ready, statistically grounded)

### RQ1 Findings

1. **Architecture determines the modality vulnerability, not the dataset.** EF worst case is TIR dropout (RA=0.309); C2F worst case is RGB dropout (RA=0.550); UAC worst case is RGB dropout (RA=0.006). Same backbone, same data, opposite vulnerabilities.

2. **Cross-attention (C2Former) has the best robustness floor and lowest variability.** Worst-case RA=0.550, CV=0.131. Mean RA differences are not statistically significant (Friedman p=0.296), so the advantage must be framed as worst-case reliability, not average performance.

3. **UA-CMDet gaussian_noise S3 is the study's most extreme result**: RA=0.043 (slope β=−0.192/severity), mAP falls from 0.214 to 0.009. The uncertainty estimator amplifies noise damage by corrupting the feature statistics it relies on for fusion weight computation.

4. **Cross-attention propagates structural TIR degradation but not photometric.** C2F TIR blur β=−0.077 (worse than EF β=−0.033); C2F TIR intensity_shift β=−0.022 (better than EF β=−0.080). The attention mechanism queries TIR for structural features; blur/noise destroy them but brightness shift preserves structure.

5. **A degraded sensor can be worse than an absent one** (Early Fusion). RGB low_contrast S3 RA=0.483 < RGB dropout RA=0.614. Severely contrast-reduced RGB provides misleading low-amplitude features that confuse the shared encoder more than clean zeros. (Note: this finding is specific to EF; C2F and UAC do not exhibit this pattern.)

6. **All severity gradients are monotone for 19/21 graded conditions** (Spearman ρ=−1.0, all p<0.05). The 2 exceptions are UAC TIR conditions where RA>1.0 — degenerate cases confirming TIR-blindness, not calibration failures.

### RQ2 Findings

7. **Early Fusion is TIR-dominant in every realistic scenario** (Wilcoxon DI test p=0.0004, mean DI=+0.539). TIR-only clean mAP (0.2977) exceeds RGB-only clean mAP (0.1499) by 2×. Mean Fusion Gain = 0.11, with occasional negative FG under RGB corruption.

8. **C2Former provides symmetric compensation** — DI not significantly different from 0 (p=0.56). Fusion Gain = 0.20 mAP over best single stream on average — the highest of the three models. Cross-attention is the only architecture that genuinely benefits from both streams.

9. **UA-CMDet provides zero fault tolerance.** TIR-only mAP = 0.9% of dual-modality. RGB failure is fatal (fallback 0.9%); TIR failure is transparent (fallback 98.5%). Fusion Gain ≈ 0. The Dominance Index test confirms extreme RGB dominance (median DI=−0.990, p≈0).

10. **Dual-modality architecture does not guarantee dual-modality fault tolerance.** UA-CMDet demonstrates a failure mode where the fusion mechanism collapses to single-modality despite dual-modality inputs.

---

## Statistical Framing Notes for Thesis Writing

These notes are critical for correct interpretation in the thesis:

1. **Do not claim C2Former is statistically superior on mean RA** — Friedman p=0.296, all Wilcoxon pairs non-significant, all effect sizes negligible or small. The claim is about *worst-case floor* and *consistency* (CV).

2. **The modality dominance findings ARE statistically significant** — use these for the strongest claims in RQ2. EF TIR-dominance (p=0.0004) and UAC RGB-dominance (p≈0) are robust.

3. **Bootstrap CIs show UAC is the least reliable model** — CI width = 0.245 vs 0.091 for C2F. Wider CI with lower mean RA = UAC is both worse and less predictable.

4. **Spearman inter-model correlation** — EF and C2F share vulnerability profiles (ρ=0.806), meaning the same corruption types are challenging for both. UAC diverges (ρ≈0.4) due to architectural collapse.

5. **N=23 is small** — be transparent about this. The non-significant Friedman result is partly a power issue (23 conditions, 3 models). The qualitative pattern is consistent but the signal is not strong enough for significance at this sample size.

---

## Analysis File Locations

| File | Content |
|---|---|
| `analysis/notebooks/nb_00_data_loading_validation.ipynb` | Data loading, completeness checks, parquet export |
| `analysis/notebooks/nb_01_descriptive_statistics.ipynb` | Summary stats, heatmap, violin plots, ranked RA table |
| `analysis/notebooks/nb_02_severity_analysis.ipynb` | Spearman monotonicity, linear regression slopes |
| `analysis/notebooks/nb_03_model_comparison_tests.ipynb` | Friedman, Wilcoxon, Cliff's δ, bootstrap CI, Spearman |
| `analysis/notebooks/nb_04_modality_analysis.ipynb` | DI, Fusion Gain, fallback tables, dominance Wilcoxon tests |
| `analysis/notebooks/nb_05_figures_publication.ipynb` | Publication figures (not yet run) |
| `analysis/data/exp0_df.parquet` | 3-row clean baseline DataFrame |
| `analysis/data/exp1_df.parquet` | 69-row corruption benchmark DataFrame |
| `analysis/data/exp2_df.parquet` | 146-row modality removal DataFrame |
| `analysis/outputs/nb00_results.txt` | Validated row counts, RA ranges, parquet export sizes |
| `analysis/outputs/nb01_results.txt` | Summary stats, per-modality means, ranked RA table |
| `analysis/outputs/nb02_results.txt` | Monotonicity results, slopes, R² values |
| `analysis/outputs/nb03_results.txt` | Friedman, Wilcoxon, Cliff's δ, bootstrap CI, Spearman |
| `analysis/outputs/nb04_results.txt` | DI test, Fusion Gain, single-modality baselines, fallback tables |
| `analysis/figures/` | Target directory for nb_05 PNG outputs (not yet populated) |

---

## All Report Files

| File | Content |
|---|---|
| `docs/exp0_baseline_report.md` | Clean baselines for all 3 models, per-class AP |
| `docs/exp1_corruption_report.md` | Early Fusion exp1 (23 conditions) |
| `docs/exp1_c2former_report.md` | C2Former exp1 (23 conditions) |
| `docs/exp1_ua_cmddet_report.md` | UA-CMDet exp1 (23 conditions) |
| `docs/exp1_crossmodel_report.md` | **Exp1 cross-model synthesis — RQ1 answer** |
| `docs/exp2_early_fusion_report.md` | Early Fusion exp2 (46 conditions) |
| `docs/exp2_c2former_report.md` | C2Former exp2 (46 conditions) |
| `docs/exp2_ua_cmddet_report.md` | UA-CMDet exp2 (46 conditions) |
| `docs/exp2_crossmodel_report.md` | **Exp2 cross-model synthesis — RQ2 answer** |
| `docs/ua_cmddet_training_report.md` | UA-CMDet training history and compatibility fixes |
| `docs/early_fusion_training_report.md` | Early Fusion training details |
| `docs/c2former_training_report.md` | C2Former training details |

---

## Source Code Structure

```
src/
  inference/
    early_fusion.py     # EF inference runner (mmrotate pipeline injection at position 1)
    c2former.py         # C2F inference runner (identical pattern to EF)
    ua_cmddet.py        # UAC runner — AerialDetection-specific (see bugs section)
  corruption/
    pipeline.py         # apply_corruption() — dispatches by modality+type
    params.py           # All severity parameter values (source of truth for Table 2)
  transforms.py         # ApplyCorruption / ZeroModality pipeline transforms (EF/C2F)
experiments/
  exp0_baseline.py
  exp1_corruption.py
  exp2_modality_removal.py
  configs/              # yaml model configs + mmdet .py configs
hpc/
  run_exp*.sh           # SLURM job scripts (already run — kept for reference)
models/
  c2former/             # mmrotate-based repo
  ua_cmddet/            # AerialDetection fork (old mmdet 1.x, patched)
  dota_devkit/          # OBB eval toolkit (polyiou C extension)
```

---

## Bugs Fixed During Experiments (important context)

### 1. UA-CMDet: AerialDetection list vs DataContainer (`src/inference/ua_cmddet.py`)

AerialDetection's dataloader wraps tensors in plain Python lists `[tensor]`, not mmdetection's `DataContainer`. The `_corrupt_batch` method originally called `.data[0]` assuming DataContainer, which fails for exp1/exp2 (exp0 never corrupts so it was never triggered).

Fix: `_get_img_tensor(val)` static method checks `isinstance(val, list)` first.

```python
@staticmethod
def _get_img_tensor(val):
    if isinstance(val, list):
        return val[0]
    return val.data[0]  # DataContainer fallback
```

Data keys: `img_r` = RGB, `img_i` = TIR.

### 2. TIR intensity_shift shape bug (`src/corruption/pipeline.py`)

The original `_tir_intensity_shift()` used `np.stack` on a `(H,W,3)` TIR image, creating `(H,W,3,3)`, which caused skimage's `rgb2hsv` to return the input unchanged. Fixed by replacing with a direct additive pixel shift (`image + c*255`), which is mathematically equivalent for grayscale. Affected Early Fusion and C2Former — results re-collected after fix.

### 3. UA-CMDet TIR-blindness — not a bug, a finding

The discovery that UA-CMDet's TIR RA ≥ 1.0 for all TIR conditions initially looked like a bug. Verified through two independent paths: (a) exp1 shows TIR corruptions don't affect mAP; (b) exp2 shows TIR-only mAP = 0.0019. This is a genuine finding — the uncertainty mechanism learned to ignore TIR. Formally confirmed in nb_04: DI = −0.990 median (p≈0).

### 4. nb_04 severity null normalisation (analysis notebooks)

Complete dropout JSONs have `"severity": null`. Reading via `str(None)` produces `"None"` rather than `"sdropout"`, causing completeness assertions to fail. Fixed in nb_00 load cells:
```python
sev = d.get("severity")
d["severity"] = "sdropout" if sev is None else str(sev)
```

### 5. Duplicate baselines rows in nb_04

exp2 contains 146 rows (8 extra vs expected 138) — source traced to extra complete_dropout rows. The clean single-modality baseline extraction merged duplicates. Fixed with `.drop_duplicates(subset=["model"])` before merge on `clean_rgb` and `clean_tir` sub-DataFrames.

---

## What's Next: Thesis Writing

The thesis document does not exist yet. The following chapters need to be written:

| Chapter | Key content | Where to pull from |
|---|---|---|
| Introduction | Motivation, gap in literature, RQs | CLAUDE.md overview + reports |
| Related Work | UA-CMDet, C2Former, MultiCorrupt, WiSE-OD, LLVIP-C | CLAUDE.md references |
| Methodology | Dataset, models, corruption protocol, eval metrics, OBB vs HBB caveat | CLAUDE.md + exp0 report |
| Results — RQ1 | Degradation profiles per model and cross-model + statistical validation | exp1 reports + exp1_crossmodel + nb_02/nb_03 outputs |
| Results — RQ2 | Modality contribution analysis + DI / Fusion Gain statistics | exp2 reports + exp2_crossmodel + nb_04 output |
| Discussion | Architecture-dependent robustness, system design implications, statistical caveats, limitations | Cross-model reports + statistical framing notes above |
| Conclusion | Answers to RQ1/RQ2, recommendations | Cross-model reports RQ summary + statistical findings |

### Publication Figures (nb_05) — Complete

All 5 figures generated at 300 DPI, saved to `analysis/figures/`:

| File | Size | Content |
|---|---|---|
| `fig_01_ra_heatmap.png` | 393 KB | 23 conditions × 3 models RA heatmap, hierarchical clustering |
| `fig_02_severity_lines.png` | 260 KB | RA vs S1→S3 for 7 corruption types, all 3 models overlaid |
| `fig_03_ra_distributions.png` | 133 KB | Violin + jittered strip per model, bootstrap CI, worst-case marked |
| `fig_04_modality_bars.png` | 186 KB | Dual / rgb_only / tir_only mAP under 6 key corruptions |
| `fig_05_dominance_index.png` | 235 KB | DI per condition, faceted by model, reference line at 0 |

Intermediate exploratory figures from earlier notebooks also present in `analysis/figures/` (nb01–nb04 prefix) — not for publication.

---

## Eval Protocol Caveat (must appear in thesis methodology)

EF and C2Former use mmrotate's built-in OBB mAP (polygon IoU, IOU threshold 0.5). UA-CMDet uses inline COCOeval with HBB (axis-aligned) IoU at 0.5. This makes UA-CMDet's absolute mAP (0.2137) incomparable to EF (0.4848) and C2F (0.7046). The published UA-CMDet figure (~0.412) used OBB polygon IoU.

**Impact on thesis**: RA values are valid cross-model because each model divides by its own clean baseline. The thesis should not compare absolute mAP across models; it should only compare RA values and note the protocol inconsistency explicitly.

---

## HPC Cluster (no more jobs needed, but for reference)

- Head nodes: `hpc-head1.ewi.utwente.nl` / `hpc-head2.ewi.utwente.nl`
- Username: `s3165582`
- Partition for GPU: `main-gpu` (not `gpu`)
- Connect via eduVPN if off-campus; PuTTY + WinSCP on Windows
- Do not run compute on head nodes; submit SLURM batch jobs only
- Project path on HPC: `/home/s3165582/thesis/drone-multimodal-robustness/`
- All experiments are complete — no further HPC jobs should be needed unless re-running something
