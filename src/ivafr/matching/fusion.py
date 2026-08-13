"""Score- and feature-level fusion primitives."""

from __future__ import annotations

import numpy as np

from ivafr.matching.score_norm import normalize_scores


def fuse_scores(
    score_2d: np.ndarray,
    score_3d: np.ndarray,
    weight: float = 0.5,
    norm_2d: dict | None = None,
    norm_3d: dict | None = None,
    rule: str = "weighted_sum",
) -> np.ndarray:
    """Fuse same-shaped similarity matrices using sum, weighted sum, or product."""
    a = normalize_scores(score_2d, norm_2d) if norm_2d else np.asarray(score_2d, dtype=np.float32)
    b = normalize_scores(score_3d, norm_3d) if norm_3d else np.asarray(score_3d, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("fusion score matrices must have the same shape")
    if rule == "sum":
        return a + b
    if rule == "product":
        return a * b
    if rule == "weighted_sum":
        if not 0.0 <= weight <= 1.0:
            raise ValueError("fusion weight must be in [0,1]")
        return float(weight) * a + (1.0 - float(weight)) * b
    raise ValueError(f"Unknown fusion rule {rule!r}")


def weight_sweep(score_2d: np.ndarray, score_3d: np.ndarray, weights=None) -> dict[str, np.ndarray]:
    """Return all weighted score matrices for the configured sweep."""
    ws = np.asarray(list(weights) if weights is not None else np.arange(0, 1.001, 0.05))
    return {f"{w:.2f}": fuse_scores(score_2d, score_3d, float(w)) for w in ws}


def concat_features(X_2d: np.ndarray, X_3d: np.ndarray, whiten: bool = False) -> np.ndarray:
    """L2-normalise each modality and concatenate features."""
    def norm(x):
        x = np.asarray(x, dtype=np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    return np.concatenate([norm(X_2d), norm(X_3d)], axis=1).astype(np.float32)
