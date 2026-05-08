"""Debug script: print the raw model output format for UA-CMDet (one image)."""
import sys
import numpy as np
sys.path.insert(0, 'models/ua_cmddet')

import mmcv
import torch
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.datasets import get_dataset, build_dataloader

cfg = mmcv.Config.fromfile('models/ua_cmddet/configs/DroneVehicle/UACMDet.py')
m = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
load_checkpoint(m, 'work_dirs/ua_cmddet/latest.pth', map_location='cpu')
m.eval()
m = MMDataParallel(m, device_ids=[0])
ds = get_dataset(cfg.data.test)
dl = build_dataloader(ds, imgs_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

with torch.no_grad():
    result = m(return_loss=False, rescale=True, **next(iter(dl)))

print('=== RESULT TYPE ===')
print('type:', type(result).__name__)

if isinstance(result, dict):
    print('dict keys:', list(result.keys()))
    for k, v in result.items():
        print(f'  {k}: type={type(v).__name__}, len={len(v)}')
        inner = v[0] if isinstance(v, list) else v
        print(f'    inner type={type(inner).__name__}, len={len(inner)}')
elif isinstance(result, (list, tuple)):
    print('len:', len(result))
    for i, elem in enumerate(result):
        print(f'  [{i}] type={type(elem).__name__}', end='')
        if hasattr(elem, '__len__'):
            print(f' len={len(elem)}', end='')
            if isinstance(elem, (list, tuple)) and elem:
                print(f' [0] type={type(elem[0]).__name__}', end='')
                if hasattr(elem[0], 'shape'):
                    print(f' shape={elem[0].shape}', end='')
        print()

print()
print('=== NAVIGATING TO DETECTIONS ===')

# Try to find detections by traversing structure
def find_dets(obj, depth=0, label='root'):
    prefix = '  ' * depth
    if isinstance(obj, np.ndarray):
        print(f'{prefix}{label}: ndarray shape={obj.shape}')
        if obj.ndim == 2 and obj.shape[0] > 0:
            print(f'{prefix}  first row: {obj[0]}')
        return
    if isinstance(obj, (list, tuple)):
        print(f'{prefix}{label}: {type(obj).__name__} len={len(obj)}')
        if depth < 3:
            for i, v in enumerate(obj[:3]):
                find_dets(v, depth + 1, f'[{i}]')
        return
    if isinstance(obj, dict):
        print(f'{prefix}{label}: dict keys={list(obj.keys())}')
        if depth < 3:
            for k, v in list(obj.items())[:3]:
                find_dets(v, depth + 1, f'[{k!r}]')
        return
    print(f'{prefix}{label}: {type(obj).__name__} = {obj}')

find_dets(result)
