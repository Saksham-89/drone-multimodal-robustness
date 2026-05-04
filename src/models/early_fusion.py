"""Early fusion backbone and pipeline transform for RGB+TIR detection.

Registers two components with the mmdet registry so they can be used
from a mmrotate config via custom_imports:

  EarlyFusionResNet  — ResNet-50 whose first conv accepts 4 channels
                       (3 RGB + 1 TIR grayscale).  Pretrained ImageNet
                       weights fill channels 0-2; channel 3 is initialised
                       as the per-filter mean of channels 0-2.

  ConcatRGBTIR       — Pipeline transform that merges the separate img
                       and img_tir arrays (produced by LoadPairedImageFromFile)
                       into a single H×W×4 array and removes img_tir.
                       Insert it after image loading, before Normalize.
"""

import numpy as np
import torch
import torch.nn as nn

from mmdet.models.builder import BACKBONES
from mmdet.datasets.builder import PIPELINES
from mmdet.models.backbones.resnet import ResNet


@BACKBONES.register_module()
class EarlyFusionResNet(ResNet):
    """ResNet-50 with 4-channel input for early RGB+TIR fusion."""

    def __init__(self, in_channels=4, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(pretrained=None, **kwargs)
        self._pretrained = pretrained
        self._in_channels = in_channels

        old = self.conv1
        self.conv1 = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False)

    def init_weights(self):
        if self._pretrained is None:
            super().init_weights()
            return

        ckpt = torch.load(self._pretrained, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)

        w = state.get('conv1.weight')  # [64, 3, 7, 7]
        if w is not None:
            new_w = torch.zeros(w.shape[0], self._in_channels, *w.shape[2:])
            new_w[:, :3] = w
            new_w[:, 3:] = w.mean(dim=1, keepdim=True)  # TIR = mean of RGB
            state['conv1.weight'] = new_w

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f'[EarlyFusionResNet] missing keys: {missing[:5]}...')


@PIPELINES.register_module()
class ConcatRGBTIR:
    """Concatenate img (H×W×3) and img_tir (H×W×3) into a 4-channel image.

    TIR is reduced to 1 channel (channel mean) before concatenation so the
    output shape is H×W×4.  img_tir is removed from results afterwards.

    Place this transform after LoadPairedImageFromFile and before Normalize.
    """

    def __call__(self, results):
        rgb = results['img']                           # H×W×3
        tir = results['img_tir']                       # H×W×3
        tir_1ch = tir.mean(axis=2, keepdims=True)      # H×W×1
        results['img'] = np.concatenate([rgb, tir_1ch], axis=2)  # H×W×4
        results.pop('img_tir', None)
        results['img_fields'] = ['img']
        return results
