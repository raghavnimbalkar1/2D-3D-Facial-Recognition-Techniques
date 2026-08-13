"""Simple metric helpers used across evaluation."""

from __future__ import annotations

import numpy as np


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std over seeds (empty -> (nan, nan))."""
    if not values:
        return float("nan"), float("nan")
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)
