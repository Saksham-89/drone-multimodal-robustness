"""Create resnet50-2stream.pth for C2Former backbone initialisation.

TwoStreamResNet duplicates all ResNet-50 layers with vis_/lwir_ prefixes
and renames the stem BN from bn1 to bnvis/bnlwir.  The standard torchvision
ResNet-50 ImageNet checkpoint is mapped to match those key names so both
streams start from the same pretrained weights.

Run once before training:
    python scripts/create_2stream_pretrain.py
Output: pretrain_weights/resnet50-2stream.pth
"""
import sys
from pathlib import Path

import torch

try:
    from torchvision.models import resnet50, ResNet50_Weights
    src = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
except ImportError:
    print('torchvision not available, falling back to torch.hub')
    src = torch.hub.load('pytorch/vision', 'resnet50',
                         weights='ResNet50_Weights.IMAGENET1K_V1').state_dict()

VIS_MAP = {
    'conv1': 'vis_conv1',
    'bn1':   'bnvis',
    'layer1': 'vis_layer1',
    'layer2': 'vis_layer2',
    'layer3': 'vis_layer3',
    'layer4': 'vis_layer4',
}
LWIR_MAP = {
    'conv1': 'lwir_conv1',
    'bn1':   'bnlwir',
    'layer1': 'lwir_layer1',
    'layer2': 'lwir_layer2',
    'layer3': 'lwir_layer3',
    'layer4': 'lwir_layer4',
}

two_stream = {}
skipped = []
for k, v in src.items():
    prefix = k.split('.')[0]
    if prefix == 'fc':
        continue
    if prefix in VIS_MAP:
        rest = k[len(prefix):]
        two_stream[VIS_MAP[prefix] + rest] = v.clone()
        two_stream[LWIR_MAP[prefix] + rest] = v.clone()
    else:
        skipped.append(k)

out_path = Path('pretrain_weights/resnet50-2stream.pth')
out_path.parent.mkdir(exist_ok=True)
torch.save({'state_dict': two_stream}, str(out_path))
print(f'Saved {len(two_stream)} keys -> {out_path}')
if skipped:
    print(f'Skipped {len(skipped)} unrecognised keys: {skipped[:5]}')
