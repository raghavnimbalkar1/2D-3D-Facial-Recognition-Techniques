"""PCA eigenfaces / eigendepth extractor.

Fit on TRAIN data only, keep 95% variance (cap 150 components), optional
``drop_first_k`` variant (discarding the first k components is the classic
anti-illumination trick). Saves the eigenface montage for the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ivafr.features.base import FeatureExtractor
from ivafr.registry import register_feature


@register_feature("pca")
class PCAFeature(FeatureExtractor):
    """Eigenfaces (2d) or eigendepth (3d) via shared code.

    Config: ``{"variance_keep": 0.95, "max_components": 150,
    "drop_first_k": 0}``. Input is an (H,W) or (H,W,1) array; outputs are
    PCA coefficients, float32.
    """

    name = "pca"
    modality = "2d"
    requires_fit = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.pca: PCA | None = None
        self._dim: int | None = None

    def fit(self, X: list[np.ndarray], y: np.ndarray | None = None) -> "PCAFeature":
        variance_keep = float(self.params.get("variance_keep", 0.95))
        max_components = int(self.params.get("max_components", 150))
        drop_first_k = int(self.params.get("drop_first_k", 0))
        mat = self._to_matrix(X)
        n_available = min(mat.shape[0], mat.shape[1])
        full = PCA(n_components=n_available, random_state=0, svd_solver="full")
        full.fit(mat)
        cum = np.cumsum(full.explained_variance_ratio_)
        n = int(np.searchsorted(cum, variance_keep) + 1)
        n = min(n, max_components, n_available)
        n = max(n - drop_first_k, 1)
        self.pca = PCA(n_components=n, random_state=0, svd_solver="full")
        self.pca.fit(mat)
        self._dim = n
        self._fit = True
        return self

    @property
    def eigenfaces(self) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("fit() before accessing eigenfaces")
        return self.pca.components_

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("fit() before transform_one()")
        vec = np.asarray(x, dtype=np.float32)
        if vec.ndim > 2:
            vec = vec.reshape(-1)
        return self.pca.transform(vec.reshape(1, -1))[0].astype(np.float32)

    def feature_dim(self) -> int | None:
        return self._dim if self._fit else None

    @staticmethod
    def _to_matrix(X: list[np.ndarray]) -> np.ndarray:
        rows = [np.asarray(x, dtype=np.float32).reshape(-1) for x in X]
        return np.stack(rows, axis=0)


@register_feature("depth_pca")
class DepthPCAFeature(PCAFeature):
    """Eigendepth: PCA on 64x64 range images (the clean 2D-vs-3D analogue)."""

    name = "depth_pca"
    modality = "3d"