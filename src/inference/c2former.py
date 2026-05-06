"""Inference runner for C2Former (Yuan & Wei, IEEE TGRS 2024).

Identical infrastructure to EarlyFusionRunner — both use mmrotate.
Corruption transforms are injected at position 1 (after
LoadPairedImageFromFile, before MultiScaleFlipAug) so they operate on
raw uint8 values before normalisation.
"""

import sys
import mmcv
import torch
from pathlib import Path

from .base import BaseInferenceRunner
from .early_fusion import _patch_mmcv_get_stream


class C2FormerRunner(BaseInferenceRunner):

    # C2Former test pipeline:
    #   [0] LoadPairedImageFromFile
    #   [1] MultiScaleFlipAug   ← insert before this
    _INJECT_POS = 1

    def __init__(self, config_path: str, device_id: int = 0):
        self.config_path = str(config_path)
        self.device_id = device_id
        self.model = None
        self._wrapped = None

    def load_model(self, checkpoint_path: Path) -> None:
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        _patch_mmcv_get_stream()

        from mmrotate.models import build_detector
        from mmcv.runner import load_checkpoint
        from mmcv.parallel import MMDataParallel

        cfg = mmcv.Config.fromfile(self.config_path)
        cfg.model.pretrained = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, str(checkpoint_path), map_location='cpu')
        model.cfg = cfg
        model = model.cuda(self.device_id)
        model.eval()
        self.model = model
        self._wrapped = MMDataParallel(self.model, device_ids=[self.device_id])

    def run(self, rgb, tir):
        raise NotImplementedError(
            "Use evaluate() — per-image inference is not needed for mAP computation.")

    def evaluate(self, corruption_type=None, modality=None, severity=None,
                 zero_modality=None) -> float:
        """Run full test-set evaluation with optional corruption / modality zeroing.

        Returns mAP@0.5 as a float.
        """
        import src.transforms  # noqa: F401
        from mmdet.datasets import build_dataset, build_dataloader

        cfg = mmcv.Config.fromfile(self.config_path)
        cfg.data.test.test_mode = True

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
            dataset, samples_per_gpu=1, workers_per_gpu=4,
            dist=False, shuffle=False)

        results = []
        for data in data_loader:
            with torch.no_grad():
                result = self._wrapped(return_loss=False, rescale=True, **data)
            results.extend(result)

        eval_out = dataset.evaluate(results, metric='mAP')
        return float(eval_out['mAP'])
