"""Distance/similarity primitives.

Convention (hard rule): matchers expose *similarity* (higher = same person).
Distance metrics live here and are converted once, inside the matcher.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def cosine_sim(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(N,D) x (M,D) -> (N,M) cosine similarity in [0, 1] for nonneg vectors."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


def l2_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (N,M)."""
    return cdist(X, Y, metric="euclidean")


def chi2_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Chi-squared distance matrix (N,M), suited to histogram features.

    ``sum((x-y)^2 / (x+y))`` per component pair.
    """
    x = X[:, None, :].astype(np.float64)
    y = Y[None, :, :].astype(np.float64)
    denom = x + y
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(denom > 0, (x - y) ** 2 / denom, 0.0)
    return d.sum(axis=-1)
