"""Seed the pinned HUG CPU generation path in a single-process worker.

torch-cluster 1.6.3 CPU FPS calls C rand() when random_start=True. PyTorch's
manual_seed does not seed that generator. This helper retains the upstream
sampling algorithm and seeds both generators. C RNG state is process-global:
use sequential batch-size-one inference, not concurrent generation threads.
Identical samples across operating systems or runtime versions are not promised.
"""
import ctypes

import numpy as np


def seed_cpu_generation(seed, *, torch_module):
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed <= 2**32 - 1):
        raise ValueError("CPU generation seed must be an unsigned 32-bit integer")
    seed = int(seed)
    torch_module.manual_seed(seed)
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.srand.restype = None
    libc.srand(seed)
