# C2Former — Training Report

**Date**: 2026-05-06  
**Training job ID**: 491914  
**Test job ID**: 492464  
**Status**: Completed (24/24 epochs)  
**Val mAP (epoch 24)**: **0.7145**  
**Test mAP (exp0 baseline)**: **0.7045**

---

## Architecture

| Component | Detail |
|---|---|
| Backbone | ResNet-50, dual-stream (RGB + TIR separately) |
| Fusion | Inter-modality Cross-Attention (ICA) at backbone feature level |
| Neck | FPN, out_channels=256 |
| Head | S2ANet: FAM (RotatedRetinaHead) + ODM (ODMRefineHead) |
| Angle convention | le135 |
| Classes | 5 (car, truck, freight_car, bus, van) |
| Input resolution | 512 × 640 |
| Framework | MMRotate |

**Fusion strategy**: C2Former processes RGB and TIR streams through separate ResNet-50 branches. An Inter-modality Cross-Attention (ICA) module dynamically attends across the two feature maps at the backbone level, allowing each modality to query complementary information from the other before the features are passed to the detection head. This is an intermediate fusion approach: deeper than simple concatenation (Early Fusion), but not deferred all the way to prediction-level fusion like UA-CMDet.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | SGD, lr=0.001, momentum=0.9, weight_decay=1e-4 |
| LR schedule | Step decay at epochs 16, 22; warmup 500 iters (linear, ratio 1/3) |
| Epochs | 24 |
| Batch size | 4 images/GPU |
| Workers | 4 per GPU |
| GPU memory | ~13.8 GB peak |
| GPU | 1× (main-gpu partition) |
| Augmentation | RResize(512×640), RRandomFlip (H/V/diagonal, ratio 0.25 each) |
| Pretrained | resnet50.pth (ImageNet), both branches |
| Gradient clipping | max_norm=35 |

---

## Training History

Training required multiple job restarts due to cluster preemption and iterative compat fixes (NumPy 1.24 deprecations, mmcv API changes). Each restart resumed from the latest saved checkpoint. Final successful run (job 491914) completed all 24 epochs without interruption.

| Log file | End point | Notes |
|---|---|---|
| 20260504_214409.log | Early stop | Preempted |
| 20260504_215637.log | Early stop | Preempted |
| 20260504_225049.log | Early stop | vis_img None crash |
| 20260504_231011.log | Early stop | Preempted |
| 20260504_232946.log | ~epoch 11 | Longest pre-fix run |
| 20260504_233436.log | Early stop | Preempted |
| 20260505_123557.log | Epoch 24 ✓ | **Final completed run** |

---

## Training Dynamics (Final Run, Job 491914)

Loss at end of training (epoch 24, final iters):

| Iter | LR | Total Loss | FAM cls | FAM bbox | ODM cls | ODM bbox | Grad norm |
|---|---|---|---|---|---|---|---|
| 4150/4363 | 1.0e-5 | 0.335 | 0.110 | 0.082 | 0.055 | 0.088 | 3.09 |
| 4200/4363 | 1.0e-5 | 0.301 | 0.096 | 0.076 | 0.050 | 0.079 | 2.94 |
| 4300/4363 | 1.0e-5 | 0.323 | 0.104 | 0.081 | 0.055 | 0.082 | 3.21 |
| 4350/4363 | 1.0e-5 | 0.324 | 0.102 | 0.084 | 0.054 | 0.084 | 3.15 |

Final total loss of ~0.32 is substantially lower than Early Fusion's ~0.57 at the same training point, consistent with the cross-attention mechanism providing a richer feature representation.

---

## Validation Results (Epoch 24, Val Split — from training log)

Evaluated using mmrotate built-in mAP @ IoU 0.5.

| Class | GT boxes | Detections | Recall | AP |
|---|---|---|---|---|
| car | 18,965 | 33,794 | 0.963 | 0.895 |
| truck | 1,336 | 13,404 | 0.870 | 0.646 |
| freight_car | 710 | 11,750 | 0.910 | 0.571 |
| bus | 751 | 4,823 | 0.981 | 0.898 |
| van | 700 | 14,222 | 0.939 | 0.563 |
| **mAP** | | | | **0.7145** |

---

## Test Split Results (Exp 0 Baseline)

**Job ID**: 492464 | **Checkpoint**: `work_dirs/c2former/epoch_24.pth`

| Class | GT boxes | Detections | Recall | AP |
|---|---|---|---|---|
| car | 124,111 | 218,605 | 0.966 | 0.894 |
| truck | 7,102 | 83,147 | 0.910 | 0.682 |
| freight_car | 3,978 | 77,294 | 0.915 | 0.542 |
| bus | 4,161 | 29,148 | 0.960 | 0.889 |
| van | 3,960 | 87,038 | 0.940 | 0.515 |
| **mAP** | | | | **0.705** |

Val → test delta: 0.7145 → 0.7045 (−0.010). Small gap confirms no overfitting.

**Observations**:
- Car and bus APs are very high (0.894, 0.889) — the cross-attention module effectively combines RGB colour/texture with TIR thermal signature for these visually distinct classes
- Truck AP (0.682) is substantially higher than Early Fusion (0.345) — cross-attention appears to resolve truck/van confusion better
- Van and freight_car remain the hardest classes (0.515, 0.542), consistent with Early Fusion; these are small, visually ambiguous objects
- Recall is high across all classes (≥0.91), indicating good detection coverage; the AP gap is driven by precision
- Detection counts are much lower relative to GT compared to Early Fusion, suggesting C2Former produces a tighter, higher-precision set of detections

---

## Comparison Context

| Model | mAP (test) | Delta vs UA-CMDet (pub.) | Source |
|---|---|---|---|
| UA-CMDet | ~0.412 | baseline | Sun et al., TCSVT 2022 |
| **Early Fusion (ours)** | **0.485** | +7.3 pp | This work (job 491917) |
| **C2Former (ours)** | **0.705** | +29.3 pp | This work (job 492464) |

C2Former's cross-attention fusion achieves **+22 pp over Early Fusion** and **+29 pp over UA-CMDet** under clean conditions. This establishes a steep clean-performance hierarchy across the three architectures — the key thesis question is whether this hierarchy is preserved under sensor corruption, or whether simpler fusion strategies are more robust.

Note: The comparison against UA-CMDet's *published* figure should be interpreted cautiously — different evaluation protocols (DOTA devkit vs mmrotate built-in) may contribute to the gap. Clean mAP validation against the published figure will be completed once UA-CMDet test evaluation finishes.

---

## Checkpoint

Saved to: `work_dirs/c2former/epoch_24.pth` (symlinked as `latest.pth`)

All 24 intermediate checkpoints saved at `work_dirs/c2former/epoch_N.pth`.

---

## Next Steps

1. ~~Training~~ **Done** — 24 epochs, job 491914
2. ~~Test-split evaluation~~ **Done** — mAP 0.705, job 492464
3. Submit exp1 and exp2 corruption experiments once GPU slots free up
4. Validate against C2Former published DroneVehicle mAP (Yuan & Wei, TGRS 2024)
