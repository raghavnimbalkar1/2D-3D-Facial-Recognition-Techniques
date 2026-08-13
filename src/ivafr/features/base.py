"""Feature extractor interface.

A feature extractor maps one preprocessed sample (2D image / 3D map / cloud)
to a fixed-length float32 vector (``template_bytes`` for storage cost
benchmarks). Fit happens on the TRAIN split only — this is the hard
anti-leakage rule. Pretrained models (ArcFace) are used frozen and documented.
"""

from __future__ import annotations

import joblib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ivafr.registry import register_feature  # noqa: F401  (re-export)


class FeatureExtractor(ABC):
    """Base class for every feature extractor."""

    name: str = ""
    modality: str = "2d"  # "2d" | "3d" | "pseudo3d" | "multi"
    requires_fit: bool = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self._fit = False

    def fit(self, X: list[np.ndarray], y: np.ndarray | None = None) -> "FeatureExtractor":
        """Fit on training data only (no-op for pretrained extractors)."""
        self._fit = True
        return self

    @abstractmethod
    def transform_one(self, x: np.ndarray) -> np.ndarray:
        """Map a single preprocessed sample to a (D,) float32 vector."""

    def transform(self, X: list[np.ndarray]) -> np.ndarray:
        """Map a list of samples to an (N, D) float32 matrix."""
        return np.stack([self.transform_one(x) for x in X], axis=0).astype(np.float32)

    @property
    def template_bytes(self) -> int:
        """Bytes of a single template (float32 x dimension)."""
        dim = self.feature_dim()
        if dim is None:
            raise NotImplementedError(f"{self.name}: template_bytes needs feature_dim()")
        return int(dim) * 4

    def feature_dim(self) -> int | None:
        """Vector dimension, if statically known (None = dynamic)."""
        return None

    def save(self, path: str | Path) -> None:
        """Persist the fitted extractor (joblib)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureExtractor":
        """Load a fitted extractor."""
        return joblib.load(path)
