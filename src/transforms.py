"""mmdet pipeline transforms for corruption injection and modality dropout.

Insert after LoadPairedImageFromFile, before normalisation, so transforms
operate on raw uint8 pixel values.
"""

import numpy as np

try:
    from mmdet.datasets.builder import PIPELINES          # mmrotate / mmdet 2.x
except ImportError:
    from mmdet.datasets.registry import PIPELINES         # AerialDetection / mmdet 1.x


@PIPELINES.register_module()
class ApplyCorruption:
    """Apply one corruption from the thesis corruption pipeline to one modality."""

    def __init__(self, modality: str, corruption_type: str, severity):
        self.modality = modality
        self.corruption_type = corruption_type
        self.severity = severity

    def __call__(self, results):
        from src.corruption.pipeline import apply_corruption
        if self.modality == 'rgb':
            results['img'] = apply_corruption(
                results['img'], 'rgb', self.corruption_type, self.severity)
        elif self.modality == 'tir':
            results['img_tir'] = apply_corruption(
                results['img_tir'], 'tir', self.corruption_type, self.severity)
        return results


@PIPELINES.register_module()
class ZeroModality:
    """Set one modality stream to all-zeros (modality dropout).

    The channel dimension is preserved — the model architecture stays fixed.
    """

    def __init__(self, modality: str):
        assert modality in ('rgb', 'tir')
        self.modality = modality

    def __call__(self, results):
        if self.modality == 'rgb':
            results['img'] = np.zeros_like(results['img'])
        else:
            results['img_tir'] = np.zeros_like(results['img_tir'])
        return results
