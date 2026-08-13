"""Finite-difference curvature descriptors for range images."""

from __future__ import annotations

import numpy as np


def curvature_from_depth(depth: np.ndarray) -> np.ndarray:
    """Return channels ``(mean curvature, Gaussian curvature, shape index)``."""
    z = np.nan_to_num(np.asarray(depth, dtype=np.float64), nan=0.0)
    zy, zx = np.gradient(z)
    zyy, zyx = np.gradient(zy)
    zxy, zxx = np.gradient(zx)
    E = 1.0 + zx * zx
    F = zx * zy
    G = 1.0 + zy * zy
    denom = np.sqrt(1.0 + zx * zx + zy * zy)
    e = zxx / denom
    f = zxy / denom
    g = zyy / denom
    den = (E * G - F * F).clip(min=1e-12)
    mean = (E * g - 2 * F * f + G * e) / (2 * den)
    gauss = (e * g - f * f) / den
    disc = np.sqrt(np.maximum(mean * mean - gauss, 0.0))
    k1, k2 = mean + disc, mean - disc
    shape = (2.0 / np.pi) * np.arctan2(k1 + k2, k1 - k2 + 1e-12)
    return np.stack([mean, gauss, shape], axis=-1).astype(np.float32)
