"""Score normalisation used by score-level fusion."""

from __future__ import annotations

import numpy as np


def fit_normalizer(scores: np.ndarray, method: str = "zscore") -> dict[str, float | str]:
    """Fit a deterministic normalizer on validation/train scores."""
    s = np.asarray(scores, dtype=np.float64)
    if method == "zscore":
        return {"method": method, "mean": float(s.mean()), "std": float(s.std() or 1.0)}
    if method == "minmax":
        lo, hi = float(s.min()), float(s.max())
        return {"method": method, "min": lo, "max": hi if hi > lo else lo + 1.0}
    if method == "tanh":
        return {"method": method, "mean": float(s.mean()), "std": float(s.std() or 1.0)}
    raise ValueError(f"Unknown score normalizer {method!r}")


def normalize_scores(scores: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    """Apply a fitted normalizer without refitting on probes."""
    s = np.asarray(scores, dtype=np.float32)
    method = params["method"]
    if method == "zscore":
        return (s - float(params["mean"])) / float(params["std"])
    if method == "minmax":
        return (s - float(params["min"])) / (float(params["max"]) - float(params["min"]))
    if method == "tanh":
        return np.tanh((s - float(params["mean"])) / (2.0 * float(params["std"])))
    raise ValueError(f"Unknown score normalizer {method!r}")
