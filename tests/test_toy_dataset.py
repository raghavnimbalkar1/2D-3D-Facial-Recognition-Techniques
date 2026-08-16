"""Toy generator determinism + geometry sanity."""

from __future__ import annotations

import numpy as np

from ivafr.datasets.toy import ToyAdapter, _FaceParams, _render_face, generate_toy


def _params() -> _FaceParams:
    return _FaceParams(
        head_w=46,
        head_h=52,
        nose_scale=11,
        nose_len=0.7,
        brow_scale=2.5,
        cheek_scale=1.2,
        albedo=0.7,
    )


def test_generator_deterministic(tmp_path):
    import cv2

    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_toy(a, n_subjects=2, n_samples=3, size=96, seed=5)
    generate_toy(b, n_subjects=2, n_samples=3, size=96, seed=5)
    for pa in sorted((a / "toy").rglob("*.png")):
        rel = pa.relative_to(a)
        img_b = cv2.imread(str(b / rel))
        img_a = cv2.imread(str(pa))
        assert img_a is not None and img_b is not None
        assert np.array_equal(img_a, img_b), f"render differs: {rel}"


def test_depth_matches_2d_alignment():
    """Depth GT should be consistent with the rendered image (same silhouette)."""
    rgb, depth, lms = _render_face(160, _params(), 0.0, 0.0, (0.0, 25.0, 1.0), seed=1)
    assert depth.shape == (160, 160)
    face_px = ~np.isnan(depth)
    img_face = (rgb[..., 0] > 40)
    overlap = (face_px & img_face).sum() / max(img_face.sum(), 1)
    assert overlap > 0.95


def test_landmarks_inside_image():
    rgb, _, lms = _render_face(160, _params(), 15.0, 5.0, (30.0, 40.0, 1.0), seed=2)
    h, w = rgb.shape[:2]
    assert np.all(lms >= 0) and np.all(lms[:, 0] < w) and np.all(lms[:, 1] < h)


def test_illumination_changes_pixel_stats():
    dark = _render_face(160, _params(), 0.0, 0.0, (0.0, 25.0, 0.35), seed=3)[0]
    strong = _render_face(160, _params(), 0.0, 0.0, (0.0, 55.0, 1.6), seed=3)[0]
    assert dark.mean() < strong.mean()


def test_adapter_discovers_canonical_ids():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        generate_toy(root / "raw", n_subjects=3, n_samples=6, seed=0)
        adapter = ToyAdapter(raw_root=root / "raw")
        samples = adapter.discover()
        assert len(samples) == 18
        assert {s.subject_id for s in samples} == {"S001", "S002", "S003"}
        assert all(s.sample_id.startswith("S") for s in samples)
        assert all(s.path_2d.is_file() and s.path_3d.is_file() for s in samples)
