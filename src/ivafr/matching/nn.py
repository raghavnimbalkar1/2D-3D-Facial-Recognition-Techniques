"""Nearest-neighbour matchers.

Convention (hard rule): :meth:`score_matrix` returns similarity — higher =
same identity. Distance metrics are converted to similarity once, inside each
matcher class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ivafr.matching import distances
from ivafr.matching.base import Matcher
from ivafr.registry import register_matcher


class NNMatcher(Matcher):
    """Nearest-neighbour matcher: gallery = fit data; score = similarity."""

    name = "nn"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.X_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NNMatcher":
        self.X_train = np.asarray(X_train, dtype=np.float32)
        return self

    def score_matrix(self, X_probe: np.ndarray, X_gallery: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@register_matcher("nn_cosine")
class NNMatcherCosine(NNMatcher):
    """NN in cosine similarity (L2-normalised templates)."""

    name = "nn_cosine"

    def score_matrix(self, X_probe: np.ndarray, X_gallery: np.ndarray) -> np.ndarray:
        return distances.cosine_sim(
            np.asarray(X_probe, dtype=np.float32), np.asarray(X_gallery, dtype=np.float32)
        )


@register_matcher("nn_l2")
class NNMatcherL2(NNMatcher):
    """NN in Euclidean distance, negated to similarity."""

    name = "nn_l2"

    def score_matrix(self, X_probe: np.ndarray, X_gallery: np.ndarray) -> np.ndarray:
        d = distances.l2_dist(
            np.asarray(X_probe, dtype=np.float32), np.asarray(X_gallery, dtype=np.float32)
        )
        return -d


@register_matcher("nn_chi2")
class NNMatcherChi2(NNMatcher):
    """NN in chi-squared distance, negated to similarity (histograms)."""

    name = "nn_chi2"

    def score_matrix(self, X_probe: np.ndarray, X_gallery: np.ndarray) -> np.ndarray:
        d = distances.chi2_dist(
            np.asarray(X_probe, dtype=np.float32), np.asarray(X_gallery, dtype=np.float32)
        )
        return -d
