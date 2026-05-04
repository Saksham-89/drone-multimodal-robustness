"""Patch UA-CMDet CUDA ops for PyTorch 2.x compatibility.

PyTorch removed several legacy THC APIs between 1.x and 2.x:
  - AT_CHECK      -> TORCH_CHECK
  - THCudaCheck   -> C10_CUDA_CHECK
  - THC/THC.h     -> ATen/cuda/CUDAContext.h + c10/cuda/CUDAException.h
  - THCState      -> removed (no longer needed)

Run from project root:
    python scripts/patch_ua_cmddet.py
"""
import re
from pathlib import Path

OPS_ROOT = Path('models/ua_cmddet/mmdet/ops')

REPLACEMENTS = [
    # Header replacements (must come before identifier replacements)
    (r'#include\s*[<"]THC/THC\.h[>"]',
     '#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>'),

    # Macro replacements
    (r'\bAT_CHECK\b',     'TORCH_CHECK'),
    (r'\bTHCudaCheck\b',  'C10_CUDA_CHECK'),

    # THCState is no longer needed — remove lines that only declare/obtain it
    (r'^\s*THCState\s*\*state\s*=\s*at::globalContext\(\)\.getTHCState\(\);\s*\n', ''),
    (r'^\s*THCState\s*\*state\s*=\s*at::globalContext\(\)\.lazyInitCUDA\(\);\s*\n', ''),
]

def patch_file(path: Path) -> bool:
    original = path.read_text(encoding='utf-8', errors='replace')
    patched = original
    for pattern, replacement in REPLACEMENTS:
        patched = re.sub(pattern, replacement, patched, flags=re.MULTILINE)
    if patched != original:
        path.write_text(patched, encoding='utf-8')
        return True
    return False

changed = []
for ext in ('*.cpp', '*.cu', '*.cuh', '*.h'):
    for f in OPS_ROOT.rglob(ext):
        if patch_file(f):
            changed.append(f)

if changed:
    print(f'Patched {len(changed)} files:')
    for f in changed:
        print(f'  {f}')
else:
    print('No files needed patching.')
