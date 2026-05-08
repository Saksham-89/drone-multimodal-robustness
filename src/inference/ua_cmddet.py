"""Inference runner for UA-CMDet (Sun et al., TCSVT 2022).

UA-CMDet is built on AerialDetection (old mmdetection fork). It does NOT
use mmdet's PIPELINES registry — the config has no 'pipeline' list.
Corruptions are applied directly to normalised image tensors in the data
loop (denormalise → corrupt → renormalise), bypassing pipeline injection.

Normalisation: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
Data dict keys (TSDroneVehicleDataset): 'img' = RGB, 'img_i' = TIR.
Images come out of the dataloader as float tensors inside DataContainers.
"""

import sys
import numpy as np
import mmcv
import torch
from pathlib import Path

from .base import BaseInferenceRunner

# Normalisation constants from UACMDet.py img_norm_cfg
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

# Map modality name → data dict key used by TSDroneVehicleDataset
_MODALITY_KEY = {'rgb': 'img', 'tir': 'img_i'}


def _tensor_to_uint8(t):
    """DataContainer float tensor [1,3,H,W] → uint8 numpy [H,W,3] BGR→RGB."""
    arr = t[0].permute(1, 2, 0).cpu().numpy()          # [H,W,3]
    arr = arr * _STD + _MEAN                            # denormalise
    return np.clip(arr, 0, 255).astype(np.uint8)


def _uint8_to_tensor(arr, device):
    """uint8 numpy [H,W,3] → normalised float tensor [1,3,H,W]."""
    f = arr.astype(np.float32)
    f = (f - _MEAN) / _STD                             # normalise
    t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


class UACMDetRunner(BaseInferenceRunner):

    def __init__(self, config_path: str, device_id: int = 0):
        self.config_path = str(config_path)
        self.device_id = device_id
        self.model = None
        self._wrapped = None

    def load_model(self, checkpoint_path: Path) -> None:
        project_root = str(Path(__file__).resolve().parents[2])
        ua_cmddet_root = str(Path(__file__).resolve().parents[2] / 'models' / 'ua_cmddet')
        for p in (project_root, ua_cmddet_root):
            if p not in sys.path:
                sys.path.insert(0, p)

        from mmdet.models import build_detector
        from mmcv.runner import load_checkpoint
        from mmcv.parallel import MMDataParallel

        cfg = mmcv.Config.fromfile(self.config_path)
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, str(checkpoint_path), map_location='cpu')
        model.eval()
        self._wrapped = MMDataParallel(model, device_ids=[self.device_id])

    def run(self, rgb, tir):
        raise NotImplementedError("Use evaluate() for mAP computation.")

    def _corrupt_batch(self, data, corruption_type, modality, severity, zero_modality):
        """Apply corruption / zeroing directly to tensor data in-place."""
        from src.corruption.pipeline import apply_corruption
        from mmcv.parallel import DataContainer

        device = None

        if corruption_type is not None:
            key = _MODALITY_KEY[modality]
            if key in data:
                dc = data[key]
                tensor = dc.data[0]                    # [1,3,H,W] float
                device = tensor.device
                img_uint8 = _tensor_to_uint8(tensor)
                img_corrupted = apply_corruption(img_uint8, modality, corruption_type, severity)
                corrupted_t = _uint8_to_tensor(img_corrupted, device)
                data[key] = DataContainer([corrupted_t], stack=dc.stack,
                                          padding_value=dc.padding_value,
                                          cpu_only=dc.cpu_only)

        if zero_modality is not None:
            key = _MODALITY_KEY[zero_modality]
            if key in data:
                dc = data[key]
                tensor = dc.data[0]
                data[key] = DataContainer([torch.zeros_like(tensor)],
                                          stack=dc.stack,
                                          padding_value=dc.padding_value,
                                          cpu_only=dc.cpu_only)
        return data

    def evaluate(self, corruption_type=None, modality=None, severity=None,
                 zero_modality=None) -> float:
        """Run full test-set evaluation with optional corruption / modality zeroing.

        UA-CMDet (AerialDetection) has no pipeline list in its config, so
        corruptions are applied directly to the loaded image tensors here
        rather than via PIPELINES injection.
        Returns mAP@0.5 as a float.
        """
        from mmdet.datasets import build_dataset, build_dataloader

        cfg = mmcv.Config.fromfile(self.config_path)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=4,
            dist=False, shuffle=False)

        apply_corruption = (corruption_type is not None or zero_modality is not None)

        results = []
        for data in data_loader:
            if apply_corruption:
                data = self._corrupt_batch(
                    data, corruption_type, modality, severity, zero_modality)
            with torch.no_grad():
                result = self._wrapped(return_loss=False, rescale=True, **data)
            results.append(result)

        if hasattr(dataset, 'evaluate'):
            eval_out = dataset.evaluate(results)
            for key in ('mAP', 'bbox_mAP', 'AP50'):
                if key in eval_out:
                    return float(eval_out[key])

        raise RuntimeError(
            "UA-CMDet dataset.evaluate() did not return a recognised mAP key. "
            "Check models/ua_cmddet/eval/DroneVehicleEval.py for the correct "
            "evaluation interface and update this runner accordingly.")
