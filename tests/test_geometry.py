import numpy as np
import pytest

from ivafr.preprocess.curvature import curvature_from_depth
from ivafr.preprocess.normals import normals_from_depth
from ivafr.preprocess.mesh_to_depth import mesh_to_depth_map
from ivafr.pipelines.preprocess_run import preprocess_3d_sample


def test_plane_normals_and_curvature_in_normalized_coordinates():
    x = np.linspace(-1, 1, 41)
    xx, yy = np.meshgrid(x, x)
    z = 0.2 * xx + 0.4 * yy
    expected = np.array([-0.2, -0.4, 1])
    expected /= np.linalg.norm(expected)
    n = normals_from_depth(z, spacing=0.05)
    np.testing.assert_allclose(n[2:-2, 2:-2], np.broadcast_to(expected, (37, 37, 3)), atol=1e-5)
    np.testing.assert_allclose(curvature_from_depth(z, spacing=0.05)[2:-2, 2:-2, :2], 0, atol=1e-5)


def test_paraboloid_curvature_at_origin():
    x = np.linspace(-1, 1, 41)
    xx, yy = np.meshgrid(x, x)
    curv = curvature_from_depth((xx**2 + yy**2) / 2, spacing=0.05)
    np.testing.assert_allclose(curv[20, 20, :2], [1, 1], atol=1e-5)


def test_holes_are_measured_before_filling_and_rejected():
    depth = np.full((100, 100), np.nan)
    depth[40:50, 40:50] = np.arange(100).reshape(10, 10)
    _, ok, hole = preprocess_3d_sample(depth, {"quality": {"max_hole_ratio": 0.9}})
    assert hole == pytest.approx(0.99)
    assert not ok


def test_degenerate_mesh_fails():
    with pytest.raises(ValueError):
        mesh_to_depth_map(np.zeros((20, 3)))
