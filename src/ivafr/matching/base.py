"""Matcher interface.

Every matcher returns a *similarity* matrix via :meth:`Matcher.score_matrix`
— higher = more likely the same person. Distance-based matchers convert once
inside their class (this kills 90% of ROC/EER sign bugs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Matcher(ABC):
    """Base class for all matchers."""

    name: str = ""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Matcher":
        """Learn anything needed from training data (no-op for NN)."""
        return self

    @abstractmethod
    def score_matrix(self, X_probe: np.ndarray, X_gallery: np.ndarray) -> np.ndarray:
        """(Np, Ng) similarity matrix; higher = same identity."""

    def scores_for_pairs(
        self, X: np.ndarray, pairs: list[tuple[int, int]], max_full: int = 1024
    ) -> np.ndarray:
        """Similarity for explicit (i, j) index pairs into ``X``.

        Uses a full ``n x n`` score matrix when ``n <= max_full``, otherwise
        computes the requested pairs only.
        """
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n <= max_full:
            S = self.score_matrix(X, X)
            return S[np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])]
        return np.asarray(
            [self.score_matrix(X[i : i + 1], X[j : j + 1])[0, 0] for i, j in pairs],
            dtype=np.float32,
        )
