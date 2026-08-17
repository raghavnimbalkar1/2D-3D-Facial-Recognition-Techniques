"""Dependency-light classical 2D and pseudo-3D feature extractors."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.feature import hog as sk_hog
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from ivafr.features.base import FeatureExtractor
from ivafr.registry import register_feature


def _gray(x: np.ndarray) -> np.ndarray:
    """Return a finite float32 grayscale image."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


class _FixedFeature(FeatureExtractor):
    """Small helper for stateless fixed-dimensional descriptors."""

    requires_fit = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._dim = int(self.params.get("feature_dim", 0)) or None

    def fit(self, X: list[np.ndarray], y: np.ndarray | None = None) -> "_FixedFeature":
        if self._dim is None:
            self._dim = int(self.transform_one(X[0]).size)
        self._fit = True
        return self

    def feature_dim(self) -> int | None:
        return self._dim


@register_feature("lda")
class LDAFeature(FeatureExtractor):
    """Fisherfaces with a train-only supervised projection."""

    name = "lda"
    modality = "2d"
    requires_fit = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.model: LinearDiscriminantAnalysis | None = None
        self._dim: int | None = None

    def fit(self, X: list[np.ndarray], y: np.ndarray | None = None) -> "LDAFeature":
        if y is None:
            raise ValueError("lda.fit requires subject labels")
        mat = np.stack([_gray(x).reshape(-1) for x in X]).astype(np.float32)
        labels = np.asarray(y)
        n_classes = len(np.unique(labels))
        if n_classes < 2:
            self.model = None
            self._mean = mat.mean(axis=0)
            self._dim = 1
        else:
            n = min(n_classes - 1, int(self.params.get("max_components", 50)))
            self.model = LinearDiscriminantAnalysis(solver="svd", n_components=n)
            self.model.fit(mat, labels)
            self._dim = n
        self._fit = True
        return self

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        if not self._fit:
            raise RuntimeError("fit() before transform_one()")
        vec = _gray(x).reshape(1, -1)
        if self.model is None:
            return np.asarray([float((vec - self._mean).mean())], dtype=np.float32)
        return self.model.transform(vec)[0].astype(np.float32)

    def feature_dim(self) -> int | None:
        return self._dim


@register_feature("lbp")
class LBPFeature(_FixedFeature):
    """Uniform local-binary-pattern histograms over image blocks."""

    name = "lbp"
    modality = "2d"

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        a = _gray(x)
        p = int(self.params.get("points", 8))
        radius = float(self.params.get("radius", 1))
        blocks = int(self.params.get("blocks", 8))
        n_bins = p + 2
        hs = []
        for yy in np.array_split(a, blocks, axis=0):
            for xx in np.array_split(yy, blocks, axis=1):
                lbp = local_binary_pattern(xx, p, radius, method="uniform")
                h, _ = np.histogram(lbp, bins=np.arange(n_bins + 1), range=(0, n_bins))
                hs.append(h.astype(np.float32) / max(float(h.sum()), 1.0))
        return np.concatenate(hs).astype(np.float32)


@register_feature("depth_lbp")
class DepthLBPFeature(LBPFeature):
    """LBP on the normalised range image."""

    name = "depth_lbp"
    modality = "pseudo3d"


@register_feature("hog")
class HOGFeature(_FixedFeature):
    """Histogram of oriented gradients for aligned 2D faces."""

    name = "hog"
    modality = "2d"

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        a = _gray(x)
        return sk_hog(
            a,
            orientations=int(self.params.get("orientations", 9)),
            pixels_per_cell=tuple(self.params.get("pixels_per_cell", (8, 8))),
            cells_per_block=tuple(self.params.get("cells_per_block", (2, 2))),
            feature_vector=True,
        ).astype(np.float32)


@register_feature("normal_hog")
class NormalHOGFeature(HOGFeature):
    """HOG descriptor on a normal field's horizontal/vertical energy."""

    name = "normal_hog"
    modality = "pseudo3d"

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 3:
            a = np.linalg.norm(a[..., :2], axis=2)
        return super().transform_one(a)


@register_feature("gabor")
class GaborFeature(FeatureExtractor):
    """Spatial Gabor maps downsampled and projected with train-only PCA."""

    name = "gabor"
    modality = "2d"
    requires_fit = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.pca: PCA | None = None
        self._raw_dim: int | None = None
        self._dim: int | None = None

    def _raw_transform_one(self, x: np.ndarray) -> np.ndarray:
        a = _gray(x)
        frequencies = self.params.get("frequencies", (0.1, 0.2, 0.3, 0.4, 0.5))
        orientations = int(self.params.get("orientations", 8))
        factor = int(self.params.get("downsample_factor", 4))
        if factor < 1:
            raise ValueError("gabor downsample_factor must be positive")
        h, w = a.shape
        out = []
        for freq in frequencies:
            for theta in np.linspace(0, np.pi, orientations, endpoint=False):
                real, imag = gabor(a, frequency=float(freq), theta=float(theta))
                magnitude = np.hypot(real, imag)
                h_crop = (h // factor) * factor
                w_crop = (w // factor) * factor
                small = magnitude[:h_crop, :w_crop].reshape(
                    h_crop // factor,
                    factor,
                    w_crop // factor,
                    factor,
                ).mean(axis=(1, 3))
                out.append(small.reshape(-1))
        return np.concatenate(out).astype(np.float32)

    def fit(self, X: list[np.ndarray], y: np.ndarray | None = None) -> "GaborFeature":
        raw = np.stack([self._raw_transform_one(x) for x in X]).astype(np.float32)
        max_components = int(self.params.get("pca_components", 200))
        n_components = min(max_components, raw.shape[0], raw.shape[1])
        if n_components < 1:
            raise ValueError("Gabor PCA requires at least one component")
        self.pca = PCA(n_components=n_components, random_state=0, svd_solver="full")
        self.pca.fit(raw)
        self._raw_dim = int(raw.shape[1])
        self._dim = n_components
        self._fit = True
        return self

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("fit() before transform_one()")
        return self.pca.transform(self._raw_transform_one(x).reshape(1, -1))[0].astype(np.float32)

    def feature_dim(self) -> int | None:
        return self._dim

    @property
    def raw_feature_dim(self) -> int | None:
        """Dimension before train-only PCA, exposed for validation and reports."""
        return self._raw_dim


@register_feature("curv_hist")
class CurvatureHistogramFeature(_FixedFeature):
    """Compact histograms of H/K/shape-index curvature channels."""

    name = "curv_hist"
    modality = "pseudo3d"

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)
        channels = [a[..., i] for i in range(a.shape[2])] if a.ndim == 3 else [a]
        bins = int(self.params.get("bins", 16))
        out = []
        for ch in channels[:3]:
            ch = np.nan_to_num(ch)
            lo, hi = np.percentile(ch, [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            h, _ = np.histogram(ch, bins=bins, range=(lo, hi))
            out.append(h.astype(np.float32) / max(float(h.sum()), 1.0))
        return np.concatenate(out).astype(np.float32)


@register_feature("lmk3d")
class Landmark3DFeature(_FixedFeature):
    """Scale-normalised pairwise distances for semantic landmarks."""

    name = "lmk3d"
    modality = "pseudo3d"

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        pts = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        if len(pts) < 2:
            return np.zeros(3, dtype=np.float32)
        pts = pts - pts.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(pts[0] - pts[min(1, len(pts) - 1)]) or 1.0
        pts = pts / scale
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        return d[np.triu_indices(len(pts), 1)].astype(np.float32)
