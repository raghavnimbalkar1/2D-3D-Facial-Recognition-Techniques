"""Optional frozen deep 2D feature hook."""

from __future__ import annotations

import numpy as np

from ivafr.features.base import FeatureExtractor
from ivafr.registry import register_feature


@register_feature("arcface")
class ArcFaceFeature(FeatureExtractor):
    """Explicit optional hook; the core sprint does not require ArcFace."""

    name = "arcface"
    modality = "2d"

    def fit(self, X, y=None):
        raise RuntimeError("ArcFace is optional; install insightface and onnxruntime to enable it")

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        raise RuntimeError("ArcFace is optional; install insightface and onnxruntime to enable it")
