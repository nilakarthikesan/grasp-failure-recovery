"""Seeded sampling adapter; geometry and point selection remain upstream-owned."""
from __future__ import annotations
import hashlib
import numpy as np


def seeded_point_cloud(depth_m, rgb, K, *, seed, backproject, sample,
                       n_points=4096, center=None, crop_radius=None):
    """Use upstream geometry and sampler, injecting a fresh request-local PCG64.

    Recreating the RNG for each invocation prevents call order, other requests,
    or global NumPy RNG use from changing a request's point subset.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('preprocessing seed must be a nonnegative integer')
    if n_points <= 0: raise ValueError('n_points must be positive')
    xyz, colors = backproject(depth_m, rgb, K, center=center, crop_radius=crop_radius)
    if len(xyz) == 0: raise ValueError('No points survive depth/crop filtering')
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    xyz, colors = sample(xyz, colors, n_points, rng=rng)
    return np.asarray(xyz, dtype=np.float32), np.asarray(colors, dtype=np.float32)/255.0


def tensor_content_hash(tensors):
    """Hash tensor names, shapes, types and values, not torch.save container bytes."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value=tensors[name]
        if hasattr(value,'detach'): value=value.detach().cpu().numpy()
        array=np.ascontiguousarray(value)
        for part in [name.encode(),array.dtype.str.encode(),str(array.shape).encode(),array.tobytes()]:
            digest.update(len(part).to_bytes(8,'big'));digest.update(part)
    return 'sha256:'+digest.hexdigest()
