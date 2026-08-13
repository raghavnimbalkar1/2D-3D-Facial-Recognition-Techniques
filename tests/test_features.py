"""Feature extractor contracts via the registry."""

from __future__ import annotations

import numpy as np
import pytest

from ivafr.registry import get_feature, list_features

EXPECTED = {"pca", "depth_pca"}


def test_registry_contents():
    assert EXPECTED <= set(list_features())


def _make_inputs(modality, n=10, h=64):
    rng = np.random.default_rng(0)
    return [rng.normal(0, 1, (h, h)).astype(np.float32) + i * 0.3 for i in range(n)]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fit_transform_contract(name):
    cls = get_feature(name)
    feat = cls({})
    X = _make_inputs("3d" if "depth" in name else "2d")
    feat.fit(X, np.arange(len(X)))
    out = feat.transform(X)
    assert out.shape == (len(X), feat.feature_dim())
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    one = feat.transform_one(X[0])
    assert one.shape == (feat.feature_dim(),)
    assert feat.template_bytes == feat.feature_dim() * 4


def test_fit_on_train_transform_on_test():
    """The anti-leakage contract: fit uses only train data."""
    cls = get_feature("pca")
    feat = cls({})
    rng = np.random.default_rng(1)
    X_train = [rng.normal(0, 1, (64, 64)).astype(np.float32) for _ in range(10)]
    X_test = [rng.normal(0, 1, (64, 64)).astype(np.float32) for _ in range(5)]
    feat.fit(X_train)
    t = feat.transform(X_test)
    assert t.shape[0] == 5
    # No state change from transform: transform again == same result.
    t2 = feat.transform(X_test)
    assert np.allclose(t, t2)


def test_pca_reconstruction_error_decreases_with_components():
    cls = get_feature("pca")
    rng = np.random.default_rng(2)
    X = [rng.normal(0, 1, (64, 64)).astype(np.float32) for _ in range(20)]
    errs = []
    for k in (3, 10, 20):
        feat = cls({"variance_keep": 1.0, "max_components": k, "drop_first_k": 0})
        feat.fit(X)
        components = feat.pca.components_
        mean = feat.pca.mean_
        mat = np.stack([x.ravel() for x in X])
        proj = (mat - mean) @ components.T
        recon = proj @ components + mean
        errs.append(float(np.mean((mat - recon) ** 2)))
    assert errs[0] > errs[1] > errs[2]


def test_save_load_roundtrip(tmp_path):
    cls = get_feature("pca")
    feat = cls({})
    X = _make_inputs("2d", n=8)
    feat.fit(X)
    p = tmp_path / "pca.joblib"
    feat.save(p)
    loaded = cls.load(p)
    assert np.allclose(loaded.transform(X), feat.transform(X))


def test_unknown_feature_raises():
    with pytest.raises(KeyError):
        get_feature("not_a_feature")