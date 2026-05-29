"""Inference runner for UA-CMDet (Sun et al., TCSVT 2022).

Built on AerialDetection (old mmdetection 1.x fork), which does not use mmdet's
PIPELINES registry. Corruptions are applied by denormalising tensors in-place,
corrupting as uint8, then renormalising — bypassing pipeline injection entirely.

Data dict keys: 'img_r' = RGB, 'img_i' = TIR.
Eval: inline COCO bbox (HBB, IoU@0.5) against coco_r annotations.
"""

import sys
import numpy as np
import mmcv
import torch
from pathlib import Path

from .base import BaseInferenceRunner

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
_MODALITY_KEY = {'rgb': 'img_r', 'tir': 'img_i'}


def _tensor_to_uint8(t):
    arr = t[0].permute(1, 2, 0).cpu().numpy()
    arr = arr * _STD + _MEAN
    return np.clip(arr, 0, 255).astype(np.uint8)


def _uint8_to_tensor(arr, device):
    f = (arr.astype(np.float32) - _MEAN) / _STD
    return torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).to(device)


def _obb_to_aabb(cx, cy, w, h, angle_deg):
    """Convert oriented box (cx,cy,w,h,angle) to axis-aligned [x1,y1,x2,y2]."""
    import cv2
    box = cv2.boxPoints(((float(cx), float(cy)),
                          (float(w),  float(h)),
                          float(angle_deg)))  # shape (4,2)
    x1 = float(box[:, 0].min())
    y1 = float(box[:, 1].min())
    x2 = float(box[:, 0].max())
    y2 = float(box[:, 1].max())
    return x1, y1, x2, y2


class UACMDetRunner(BaseInferenceRunner):

    def __init__(self, config_path: str, device_id: int = 0):
        self.config_path = str(config_path)
        self.device_id = device_id
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

    @staticmethod
    def _get_img_tensor(val):
        """AerialDetection dataloader returns plain list [tensor], not DataContainer."""
        if isinstance(val, list):
            return val[0]
        return val.data[0]  # DataContainer fallback

    def _corrupt_batch(self, data, corruption_type, modality, severity, zero_modality):
        from src.corruption.pipeline import apply_corruption

        if corruption_type is not None:
            key = _MODALITY_KEY[modality]
            if key in data:
                tensor = self._get_img_tensor(data[key])
                device = tensor.device
                img_uint8 = _tensor_to_uint8(tensor)
                img_corrupted = apply_corruption(img_uint8, modality, corruption_type, severity)
                tensor.copy_(_uint8_to_tensor(img_corrupted, device))

        if zero_modality is not None:
            key = _MODALITY_KEY[zero_modality]
            if key in data:
                self._get_img_tensor(data[key]).zero_()

        return data

    def evaluate(self, corruption_type=None, modality=None, severity=None,
                 zero_modality=None) -> float:
        """Run full test-set evaluation with optional corruption / modality zeroing.

        Uses inline COCO bbox eval (IoU@0.5) against coco_r annotations.
        Returns mAP@IoU=0.5 as a float.
        """
        from mmdet.datasets import get_dataset, build_dataloader
        from pycocotools.cocoeval import COCOeval

        cfg = mmcv.Config.fromfile(self.config_path)
        dataset = get_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=4,
            dist=False, shuffle=False)

        do_corrupt = (corruption_type is not None or zero_modality is not None)

        results = []
        for data in data_loader:
            if do_corrupt:
                data = self._corrupt_batch(
                    data, corruption_type, modality, severity, zero_modality)
            with torch.no_grad():
                result = self._wrapped(return_loss=False, rescale=True, **data)

            # Unwrap multi-stream dict → take fused (last) stream
            if isinstance(result, dict):
                result = list(result.values())[-1]

            # result is either [per_img_result] (batch list, most common in AerialDetection)
            # or per_img_result directly. Detect by checking if first element is a list.
            if isinstance(result, (list, tuple)) and result and isinstance(result[0], list):
                results.extend(result)   # batch list → peel wrapper
            else:
                results.append(result)   # already per-image

        # Build COCO-format detections.
        # Per-image result: list of per-class arrays shaped [N, K]:
        #   K=9 → polygon OBB [x1,y1,x2,y2,x3,y3,x4,y4, score] ← UA-CMDet
        #   K=6 → rotated OBB [cx, cy, w, h, angle_deg, score]
        #   K=5 → HBB [x1, y1, x2, y2, score]
        coco_gt = dataset.coco_r
        cat_ids = dataset.cat_ids_r
        img_ids = dataset.img_ids_r
        dt_anns = []

        for idx, result in enumerate(results):
            if isinstance(result, dict):
                result = list(result.values())[-1]
            img_id = img_ids[idx]
            for cls_idx, dets in enumerate(result):
                if dets is None or len(dets) == 0:
                    continue
                dets = np.asarray(dets)
                if dets.ndim == 1:
                    dets = dets[np.newaxis]  # single detection
                for det in dets:
                    k = len(det)
                    if k == 9:
                        # Polygon OBB: [x1,y1,x2,y2,x3,y3,x4,y4, score]
                        pts = det[:8].reshape(4, 2)
                        x1 = float(pts[:, 0].min())
                        y1 = float(pts[:, 1].min())
                        x2 = float(pts[:, 0].max())
                        y2 = float(pts[:, 1].max())
                        score = float(det[8])
                    elif k == 6:
                        # Rotated OBB: [cx, cy, w, h, angle_deg, score]
                        x1, y1, x2, y2 = _obb_to_aabb(det[0], det[1], det[2], det[3], det[4])
                        score = float(det[5])
                    else:
                        # HBB: [x1, y1, x2, y2, score]
                        x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                        score = float(det[4])
                    dt_anns.append({
                        'image_id': img_id,
                        'category_id': cat_ids[cls_idx],
                        'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        'score': score,
                    })

        if not dt_anns:
            return 0.0

        coco_dt = coco_gt.loadRes(dt_anns)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return float(ev.stats[0])
