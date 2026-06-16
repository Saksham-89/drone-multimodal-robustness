# DroneVehicle-C — Corrupted DroneVehicle Test Split

This dataset is a corrupted version of the DroneVehicle test split, generated
for benchmarking multimodal RGB+TIR fusion robustness in drone-based vehicle detection.

## Source

Original dataset: Sun et al., "Drone-based RGB-Infrared Cross-Modality Vehicle Detection
via Uncertainty-Aware Learning," IEEE TCSVT 2022.

Corruptions applied by: Saksham Singh Birla, University of Twente (TSCiT 2025 thesis).

## Structure

```
DroneVehicle_C/
  labels/                    # DOTA-format .txt annotations (same for all conditions)
  clean/rgb/  clean/ir/      # Original unmodified images
  rgb_gaussian_noise_s1/rgb/
  rgb_gaussian_noise_s1/ir/  # TIR unchanged
  ...                        # (24 condition directories in total)
  metadata.json              # Condition descriptions and parameters
  README.md
```

Each condition directory has:
- `rgb/` — RGB images (corrupted for rgb_* conditions, clean for tir_* conditions)
- `ir/`  — TIR images (clean for rgb_* conditions, corrupted for tir_* conditions)

## Conditions (23 total)

### RGB Corruptions (TIR clean)
- rgb_gaussian_noise_s1/s2/s3
- rgb_motion_blur_s1/s2/s3
- rgb_brightness_shift_s1/s2/s3
- rgb_low_contrast_s1/s2/s3
- rgb_complete_dropout

### TIR Corruptions (RGB clean)
- tir_sensor_noise_s1/s2/s3
- tir_blur_s1/s2/s3
- tir_intensity_shift_s1/s2/s3
- tir_complete_dropout

## Annotations

Labels are in DOTA oriented bounding box format:
  `x1 y1 x2 y2 x3 y3 x4 y4 class difficult`

Five vehicle classes: car, truck, freight_car, bus, van.
Image size: 840×712 pixels (100px white border each side, as in original DroneVehicle).

## Citation

If you use this dataset, please cite:
1. Sun et al. (IEEE TCSVT 2022) for the original DroneVehicle dataset.
2. Saksham Singh Birla (TSCiT 2025) for the corruption benchmark.

See metadata.json for full condition parameters.
