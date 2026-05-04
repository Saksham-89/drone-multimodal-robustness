"""Patch UA-CMDet CUDA ops for PyTorch 2.x compatibility.

PyTorch removed the entire THC (Torch CUDA) legacy API between 1.x and 2.x.
This script replaces all removed symbols with their modern equivalents.

Removed APIs patched here:
  AT_CHECK          -> TORCH_CHECK
  THCudaCheck       -> C10_CUDA_CHECK
  THCCeilDiv(a,b)   -> inline ceil-div macro
  THCudaMalloc      -> cudaMalloc wrapper
  THCudaFree        -> cudaFree wrapper
  THCState          -> removed (no longer needed)
  THC/THC.h         -> ATen/cuda/CUDAContext.h + c10/cuda/CUDAException.h
  THC/THCAtomics.cuh-> ATen/cuda/Atomic.cuh
  THC/THCDeviceUtils.cuh -> compat macros defined inline

Run from project root:
    python scripts/patch_ua_cmddet.py
"""
import re
from pathlib import Path

OPS_ROOT = Path('models/ua_cmddet/mmdet/ops')

# Block of compatibility macros injected into files that need THC device utils
THC_COMPAT_MACROS = '''\
/* --- PyTorch 2.x THC compatibility --- */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
static inline void* _thc_malloc(size_t n) { void* p; C10_CUDA_CHECK(cudaMalloc(&p, n)); return p; }
static inline void  _thc_free(void* p)    { C10_CUDA_CHECK(cudaFree(p)); }
#define THCCeilDiv(a, b)      (((a) + (b) - 1) / (b))
#define THCudaMalloc(s, n)    _thc_malloc(n)
#define THCudaFree(s, p)      _thc_free(p)
/* --- end THC compatibility --- */
'''

HEADER_REPLACEMENTS = [
    # THC/THC.h + THC/THCDeviceUtils.cuh -> compat block (injected once)
    (r'#include\s*[<"]THC/THC\.h[>"]\s*\n'
     r'(#include\s*[<"]THC/THCDeviceUtils\.cuh[>"]\s*\n)?',
     THC_COMPAT_MACROS),

    # Standalone THCDeviceUtils (in case it wasn't already handled above)
    (r'#include\s*[<"]THC/THCDeviceUtils\.cuh[>"]',
     '/* THCDeviceUtils removed — compat macros already defined */'),

    # THCAtomics -> ATen equivalent
    (r'#include\s*[<"]THC/THCAtomics\.cuh[>"]',
     '#include <ATen/cuda/Atomic.cuh>\n#include <c10/cuda/CUDAException.h>'),
]

IDENTIFIER_REPLACEMENTS = [
    # AT_CHECK -> TORCH_CHECK
    (r'\bAT_CHECK\b', 'TORCH_CHECK'),

    # THCudaCheck -> C10_CUDA_CHECK
    (r'\bTHCudaCheck\b', 'C10_CUDA_CHECK'),

    # Remove THCState declaration lines
    (r'^\s*THCState\s*\*\s*state\s*=\s*at::globalContext\(\)\.[^;]+;\s*\n', ''),
    (r'\bTHCState\b', 'void /* THCState removed */'),
]


def patch_file(path: Path) -> bool:
    original = path.read_text(encoding='utf-8', errors='replace')
    patched = original

    for pattern, replacement in HEADER_REPLACEMENTS:
        patched = re.sub(pattern, replacement, patched, flags=re.MULTILINE)

    # For .cu/.cuh files using C10_CUDA_CHECK but missing the header,
    # ensure the c10 exception header is present.
    if path.suffix in ('.cu', '.cuh') and 'C10_CUDA_CHECK' in patched:
        if 'c10/cuda/CUDAException.h' not in patched:
            patched = '#include <c10/cuda/CUDAException.h>\n' + patched

    for pattern, replacement in IDENTIFIER_REPLACEMENTS:
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
