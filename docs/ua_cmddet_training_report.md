# UA-CMDet — Training Report

**Date**: 2026-05-06  
**Final training job ID**: 492010  
**Status**: Completed (12/12 epochs)  
**Test mAP**: TBD (test job 492525 pending)

---

## Architecture

| Component | Detail |
|---|---|
| Backbone | ResNet-50 × 2 (separate RGB and TIR branches) |
| Fusion type | Late / uncertainty-guided |
| Detection framework | Two-stage (Faster RCNN-style), cascade with 2 stages (s0, s1) |
| RPN | Three parallel RPNs: RGB, TIR, and fused |
| Head | Three parallel heads per stage: RGB-only, TIR-only, fused |
| Classes | 5 (car, truck, freight_car, bus, van) |
| Framework | AerialDetection (fork of mmdetection 1.x) |

**Fusion strategy**: UA-CMDet processes RGB and TIR through entirely separate ResNet-50 backbones. At each detection stage, it maintains independent predictions for each modality alongside a fused prediction. Uncertainty scores are computed per-modality, and the final detection is a weighted combination where modalities with lower uncertainty contribute more to the fused output. This is the latest-fusion strategy in the thesis comparison: fusion occurs only at the prediction level, not in feature space.

**Why this is important for the thesis**: Because fusion happens at the prediction level, UA-CMDet can in principle degrade more gracefully under single-modality corruption — the unaffected modality's predictions can dominate. The RQ2 modality removal experiments will test whether this holds in practice.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Config | `models/ua_cmddet/configs/DroneVehicle/UACMDet.py` |
| Optimizer | SGD (standard AerialDetection defaults) |
| LR at epoch 12 | 5.000e-05 (after step decay) |
| Epochs | 12 (UA-CMDet standard schedule) |
| GPU memory | ~3.9 GB peak (most efficient of the three models) |
| GPU | 1× (main-gpu partition) |
| Framework | AerialDetection / mmdetection 1.x |

**Note on epoch count**: UA-CMDet trains for 12 epochs, not 24. This is the schedule used in the original Sun et al. (TCSVT 2022) paper. The AerialDetection framework uses a 1× schedule (12 epochs) as standard, whereas mmrotate-based models (C2Former, Early Fusion) use a 2× schedule (24 epochs). Training time: ~22 hours for 12 epochs on 1 GPU.

---

## Training History and Compatibility Fixes

UA-CMDet required the most engineering effort to get running due to its dependency on an old mmdetection fork (AerialDetection, ~2019) running under Python 3.10 + NumPy 1.24 + modern mmcv. Fixes applied iteratively across multiple job attempts:

| Fix | Symptom | Resolution |
|---|---|---|
| NumPy deprecated aliases | `np.float`, `np.int`, `np.bool`, etc. removed in NumPy 1.24 | Replaced with `np.float64`, `np.int64`, etc. across all UA-CMDet source files |
| `mmcv.cnn.weight_init` moved | `ImportError: cannot import name 'caffe2_xavier_init' from 'mmcv.cnn.weight_init'` | Changed to `from mmcv.cnn import caffe2_xavier_init` |
| `mmcv.impad` keyword-only arg | `impad(mask, pad_shape[:2], ...)` → positional arg rejected | Changed to `impad(mask, shape=pad_shape[:2], ...)` |
| `Runner` logger type | `Runner(logger='INFO')` rejected; newer mmcv requires `logging.Logger` object | Injected `import logging; logger = logging.getLogger("mmdet")` before Runner init |
| Cython bbox extension | `np.float` in `bbox.pyx` broke C extension compilation | Replaced type in pyx, recompiled with setuptools/Cython |
| `--work-dir` vs `--work_dir` | Old train.py uses underscore not hyphen | Fixed SLURM script argument |
| `img_i is None` crash | `mmcv.imread()` returned None for bad image pair, then `mmcv.imcrop(None, ...)` crashed at dataset level | Added None guard in `prepare_train_img` in `two_stream_custom.py` |

All fixes were applied as in-place patches to the `models/ua_cmddet/` directory on the HPC (the models/ directory is not committed to the main repo).

---

## Training Dynamics (Final Run, Job 492010)

Loss structure at epoch 12 (UA-CMDet has many more loss terms than mmrotate models — three parallel heads × two stages):

| Term | Meaning | Value (epoch 12, ~iter 7800) |
|---|---|---|
| `loss_rpn_cls_r/i/f` | RPN classification: RGB / TIR / fused | 0.27–0.48 each |
| `loss_rpn_bbox_r/i/f` | RPN bbox regression: RGB / TIR / fused | 0.01–0.11 each |
| `s0.rbbox_loss_cls_r/i/f` | Stage-0 OBB cls: RGB / TIR / fused | 0.30–0.46 each |
| `s0.rbbox_loss_bbox_r/i/f` | Stage-0 OBB bbox: RGB / TIR / fused | 0.74–4.5 each |
| `s1.rbbox_loss_cls_r/i/f` | Stage-1 OBB cls: RGB / TIR / fused | 0.20–0.31 each |
| `s1.rbbox_loss_bbox_r/i/f` | Stage-1 OBB bbox: RGB / TIR / fused | 0.16–0.53 each |
| **Total loss** | Sum of all above | **~9–14** |

The total loss of 9–14 is not comparable to C2Former's 0.32 or Early Fusion's 0.57 — UA-CMDet has roughly 18 loss terms vs 4. Accuracy metrics (rbbox_acc) are more informative: s1 (stage 1) shows 90–94% accuracy for all three streams at epoch 12, indicating the model has learned effective detections.

Selected iterations at epoch 12 (final epoch):

| Iter | Total Loss | s1 acc (RGB/TIR/fused) | Grad norm |
|---|---|---|---|
| 7690/8987 | 12.66 | 92.1 / 91.8 / 91.1 | 19.5 |
| 7800/8987 | 12.14 | 93.1 / 94.5 / 89.4 | 16.1 |
| 7880/8987 | 12.28 | 93.0 / 93.4 / 91.5 | 20.3 |
| 8980/8987 | 10.89 | 92.2 / 93.1 / 90.3 | 17.7 |

Training completed cleanly: "Saving checkpoint at 12 epochs for ua cmdet."

---

## Test Split Results (Exp 0 Baseline)

**Job ID**: 492525 (pending — queued behind Early Fusion experiment jobs)  
**Expected checkpoint**: `work_dirs/ua_cmddet/latest.pth`

Results will be added here once the test job completes.

**Published reference**: Sun et al. (TCSVT 2022) report ~41.2 mAP for UA-CMDet on DroneVehicle. Our reproduction should be close to this figure if the training and evaluation protocols match. Note: the original paper may use a slightly different eval script (UA-CMDet's own `eval/DroneVehicleEval.py` using polyiou) — we will compare both numbers if they differ.

---

## Comparison Context (updated when test results arrive)

| Model | mAP (test) | Architecture | Source |
|---|---|---|---|
| UA-CMDet (published) | ~0.412 | Late/uncertainty fusion | Sun et al., TCSVT 2022 |
| UA-CMDet (ours) | TBD | Late/uncertainty fusion | This work (job 492525) |
| Early Fusion (ours) | 0.485 | Early concat | This work (job 491917) |
| C2Former (ours) | 0.705 | Cross-attention | This work (job 492464) |

---

## Checkpoint

Saved to: `work_dirs/ua_cmddet/latest.pth` (symlink to final epoch checkpoint)

---

## Next Steps

1. ~~Training~~ **Done** — 12 epochs, job 492010
2. Test-split evaluation — job 492525 (pending)
3. Submit exp1 and exp2 once UA-CMDet test mAP is confirmed
4. Validate against published mAP (~0.412) from Sun et al.
