"""Depth-derived surface normals."""

from __future__ import annotations

import numpy as np


def normals_from_depth(depth: np.ndarray) -> np.ndarray:
    """Compute Sobel-like unit normals as an ``(H,W,3)`` float32 field."""
    z = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0)
    dz_y, dz_x = np.gradient(z)
    n = np.stack([-dz_x, -dz_y, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-8)
    return n.astype(np.float32)
