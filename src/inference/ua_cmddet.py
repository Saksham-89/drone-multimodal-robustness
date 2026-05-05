"""Inference runner for UA-CMDet (Sun et al., TCSVT 2022).

UA-CMDet is built on AerialDetection (old mmdetection fork). Its mmdet
package must be first on PYTHONPATH when this runner is used:
    export PYTHONPATH=/path/to/drone-multimodal-robustness/models/ua_cmddet:$PYTHONPATH

The corruption/zeroing transforms in src/transforms.py register into
whichever mmdet.datasets.builder.PIPELINES is active at import time —
when UA-CMDet's mmdet is first on PYTHONPATH, they register there.

Test pipeline for UACMDet (from UACMDet.py config):
    [0] LoadPairedImageFromFile  (UA-CMDet's own version)
    [1] ...normalisation/augmentation transforms
Corruption inserts at position 1.
"""

import sys
import mmcv
import torch
from pathlib import Path

from .base import BaseInferenceRunner


class UACMDetRunner(BaseInferenceRunner):

    _INJECT_POS = 1

    def __init__(self, config_path: str, device_id: int = 0):
        self.config_path = str(config_path)
        self.device_id = device_id
        self.model = None
        self._wrapped = None
        self._cfg = None

    def load_model(self, checkpoint_path: Path) -> None:
        project_root = str(Path(__file__).resolve().parents[2])
        ua_cmddet_root = str(Path(__file__).resolve().parents[2] / 'models' / 'ua_cmddet')
        for p in (project_root, ua_cmddet_root):
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch
        from mmdet.models import build_detector
        from mmcv.runner import load_checkpoint
        from mmcv.parallel import MMDataParallel

        self._cfg = mmcv.Config.fromfile(self.config_path)
        model = build_detector(
            self._cfg.model,
            train_cfg=None,
            test_cfg=self._cfg.test_cfg)
        load_checkpoint(model, str(checkpoint_path), map_location='cpu')
        model.eval()
        self.model = model
        self._wrapped = MMDataParallel(model, device_ids=[self.device_id])

    def run(self, rgb, tir):
        raise NotImplementedError("Use evaluate() for mAP computation.")

    def evaluate(self, corruption_type=None, modality=None, severity=None,
                 zero_modality=None) -> float:
        """Run full test-set evaluation with optional corruption / modality zeroing.

        Uses UA-CMDet's own DroneVehicleDataset and eval script (DroneVehicleEval.py).
        Returns mAP@0.5 as a float.
        """
        import src.transforms  # noqa: F401
        from mmdet.datasets import build_dataset, build_dataloader

        cfg = mmcv.Config.fromfile(self.config_path)

        pos = self._INJECT_POS
        if corruption_type is not None:
            cfg.data.test.pipeline.insert(pos, dict(
                type='ApplyCorruption',
                modality=modality,
                corruption_type=corruption_type,
                severity=severity))
            pos += 1
        if zero_modality is not None:
            cfg.data.test.pipeline.insert(pos, dict(
                type='ZeroModality', modality=zero_modality))

        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=4,
            dist=False, shuffle=False)

        results = []
        for data in data_loader:
            with torch.no_grad():
                result = self._wrapped(return_loss=False, rescale=True, **data)
            results.append(result)

        # UA-CMDet's eval returns mAP via DroneVehicleDataset.evaluate() or
        # requires the external DroneVehicleEval.py script. Try built-in first.
        if hasattr(dataset, 'evaluate'):
            eval_out = dataset.evaluate(results)
            # Key name varies by implementation — try common variants
            for key in ('mAP', 'bbox_mAP', 'AP50'):
                if key in eval_out:
                    return float(eval_out[key])

        raise RuntimeError(
            "UA-CMDet dataset.evaluate() did not return a recognised mAP key. "
            "Check models/ua_cmddet/eval/DroneVehicleEval.py for the correct "
            "evaluation interface and update this runner accordingly.")
