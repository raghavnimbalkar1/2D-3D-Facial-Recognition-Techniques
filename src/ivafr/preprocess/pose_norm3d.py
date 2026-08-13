"""Correspondence-based pseudo-3D pose normalisation."""

from __future__ import annotations

import numpy as np


def similarity_transform_3d(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve ``target ~= scale * source @ R.T + translation`` by SVD.

    Returns the row-vector rotation matrix, scale, and translation vector.
    Reflections are corrected so the result is a proper rotation.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source and target must both have shape (N,3)")
    if len(src) < 3:
        raise ValueError("at least three points are required")
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    X = src - sc
    Y = dst - dc
    norm_x = np.linalg.norm(X)
    if norm_x == 0:
        raise ValueError("source landmarks are degenerate")
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = float(np.sum((X @ R.T) * Y) / max(np.sum(X * X), 1e-12))
    t = dc - scale * (sc @ R.T)
    return R.astype(np.float32), scale, t.astype(np.float32)


def apply_similarity_3d(points: np.ndarray, R: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    """Apply a transform returned by :func:`similarity_transform_3d`."""
    pts = np.asarray(points, dtype=np.float32)
    return (scale * (pts @ np.asarray(R).T) + np.asarray(t)).astype(np.float32)


def procrustes_normalize(points: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Align corresponding landmarks to a canonical training-only template."""
    R, scale, t = similarity_transform_3d(points, template)
    return apply_similarity_3d(points, R, scale, t)
