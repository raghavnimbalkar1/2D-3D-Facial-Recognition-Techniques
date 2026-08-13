"""Sklearn-backed template matchers with the common similarity contract."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ivafr.matching.base import Matcher
from ivafr.matching.distances import cosine_sim, l2_dist
from ivafr.registry import register_matcher


class _TemplateClassifier(Matcher):
    """Classifier for prediction plus template scores for ROC/CMC."""

    kernel = "linear"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "_TemplateClassifier":
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train)
        if self.name.startswith("svm"):
            self.model = SVC(
                kernel=self.kernel,
                C=float(self.params.get("C", 1.0)),
                gamma=self.params.get("gamma", "scale"),
                probability=False,
            ).fit(self.X_train, self.y_train)
        else:
            self.model = KNeighborsClassifier(n_neighbors=int(self.params.get("k", 3))).fit(
                self.X_train, self.y_train
            )
        return self

    def score_matrix(self, X_probe: np.ndarray, X_gallery: np.ndarray) -> np.ndarray:
        # Scores remain template-to-template so the same matrix can be used
        # for identification and verification across all matcher families.
        return cosine_sim(np.asarray(X_probe, dtype=np.float32), np.asarray(X_gallery, dtype=np.float32))


@register_matcher("svm_linear")
class SVMLinearMatcher(_TemplateClassifier):
    name = "svm_linear"
    kernel = "linear"


@register_matcher("svm_rbf")
class SVMRBFMatcher(_TemplateClassifier):
    name = "svm_rbf"
    kernel = "rbf"


@register_matcher("knn3")
class KNN3Matcher(_TemplateClassifier):
    name = "knn3"

