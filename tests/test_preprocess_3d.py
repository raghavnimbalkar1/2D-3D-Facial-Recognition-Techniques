"""3D preprocessing: range image contract, NaN policy."""

from __future__ import annotations

import numpy as np

from ivafr.preprocess.range_image import range_image_from_depth


def _toy_depth():
    from ivafr.datasets.toy import _render_face, _FaceParams

    p = _FaceParams(head_w=46, head_h=52, nose_scale=11, nose_len=0.7, brow_scale=2.5, cheek_scale=1.2, albedo=0.7)
    return _render_face(160, p, 0.0, 0.0, (0.0, 25.0, 1.0), seed=0)[1]


def test_range_image_shape_and_no_nan():
    d = _toy_depth()
    rimg, hole = range_image_from_depth(d, size=64)
    assert rimg.shape == (64, 64)
    assert rimg.dtype == np.float32
    assert np.isfinite(rimg).all()
    assert 0.0 < hole < 0.9


def test_z_normalised_units():
    d = _toy_depth()
    rimg, _ = range_image_from_depth(d, size=64, z_norm="std")
    assert abs(float(np.median(rimg))) < 0.15
    assert float(np.std(rimg)) > 0.1


def test_p5p95_norm():
    d = _toy_depth()
    rimg, _ = range_image_from_depth(d, size=64, z_norm="p5p95")
    assert np.percentile(rimg, 5) <= 0.5 and np.percentile(rimg, 95) >= -0.5


def test_all_nan_depth_fails_gracefully():
    d = np.full((32, 32), np.nan, dtype=np.float32)
    rimg, hole = range_image_from_depth(d, size=16)
    assert hole == 1.0
    assert np.isfinite(rimg).all()


def test_downsample_preserves_structure():
    d = _toy_depth()
    r64, _ = range_image_from_depth(d, size=64)
    r128, _ = range_image_from_depth(d, size=128)
    # Nose region (deepest area = closest to camera, highest z) should be
    # near the image centre in both resolutions.
    def nose_val(r):
        c = r.shape[0] // 2
        return float(r[c - 2 : c + 3, c - 2 : c + 3].max())

    assert nose_val(r64) > 1.0 and nose_val(r128) > 1.0