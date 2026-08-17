"""2D preprocessing chain: alignment, illumination, shape contracts."""

from __future__ import annotations

import numpy as np

from ivafr.datasets.toy import TOY_CANONICAL_5PT, _render_face, _FaceParams
from ivafr.preprocess.align2d import TEMPLATE_112, align_to_template, similarity_transform
from ivafr.preprocess.illum import normalize_illum


def _params() -> _FaceParams:
    return _FaceParams(head_w=46, head_h=52, nose_scale=11, nose_len=0.7, brow_scale=2.5, cheek_scale=1.2, albedo=0.7)


def test_similarity_transform_exact():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0], [3.0, 7.0]])
    tgt = pts * 2.0 + np.array([5.0, -3.0])
    M = similarity_transform(pts, tgt)
    mapped = np.hstack([pts, np.ones((5, 1))]) @ M.T
    assert np.allclose(mapped, tgt, atol=1e-3)


def test_alignment_output_shape_and_dtype():
    rgb, _, lms = _render_face(160, _params(), 10.0, 0.0, (0.0, 25.0, 1.0), seed=0)
    out = align_to_template(rgb, lms, size=112)
    assert out.shape == (112, 112, 3)
    assert out.dtype == np.uint8


def test_alignment_puts_landmarks_on_template():
    """Landmarks after warping should land near the canonical template."""
    _, _, lms = _render_face(160, _params(), -20.0, 5.0, (20.0, 30.0, 1.0), seed=0)
    M = similarity_transform(lms, TEMPLATE_112)
    mapped = np.hstack([lms, np.ones((5, 1))]) @ M.T
    assert np.allclose(mapped, TEMPLATE_112, atol=0.5)


def test_tantriggs_reduces_illumination_variance():
    """Tan-Triggs should shrink the illumination-driven variance between two
    differently-lit renders of the same identity."""
    a = _render_face(160, _params(), 0.0, 0.0, (0.0, 25.0, 0.35), seed=1)[0][..., 0] / 255.0
    b = _render_face(160, _params(), 0.0, 0.0, (0.0, 55.0, 1.6), seed=1)[0][..., 0] / 255.0
    raw_var = float(np.mean((a - b) ** 2))
    ta = normalize_illum(a, "tantriggs")
    tb = normalize_illum(b, "tantriggs")
    tt_var = float(np.mean((ta - tb) ** 2))
    assert tt_var < raw_var * 0.5


def test_illum_methods_output_range():
    a = _render_face(160, _params(), 0.0, 0.0, (0.0, 25.0, 1.0), seed=1)[0][..., 0]
    for method in ("none", "histeq", "clahe", "tantriggs"):
        out = normalize_illum(a, method)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


def test_illumination_preserves_pixel_spread():
    """Contrast methods must not collapse a non-constant crop to black."""
    a = _render_face(160, _params(), 0.0, 0.0, (0.0, 25.0, 1.0), seed=1)[0][..., 0] / 255.0
    for method in ("histeq", "clahe"):
        out = normalize_illum(a, method)
        assert float(out.std()) > 0.05
        assert float(out.max() - out.min()) > 0.25


def test_cache_mark_and_hit(tmp_path):
    from ivafr.preprocess import cache

    out = tmp_path / "x.npy"
    cfg = {"illum": {"method": "none"}}
    assert not cache.is_cached(out, cfg, "d1")
    cache.mark_cached(out, cfg, "d1")
    assert cache.is_cached(out, cfg, "d1")
    assert not cache.is_cached(out, cfg, "d2")  # input changed
    assert not cache.is_cached(out, {"illum": {"method": "clahe"}}, "d1")  # cfg changed
