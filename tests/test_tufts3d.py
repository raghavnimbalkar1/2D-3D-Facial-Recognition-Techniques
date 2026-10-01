"""Tests for Tufts3D dataset adapter and mesh-to-depth preprocessing."""

from pathlib import Path
import numpy as np
import pytest

from ivafr.datasets.base import Sample
from ivafr.datasets.manifest import samples_to_manifest, validate_manifest
from ivafr.datasets.tufts3d import Tufts3DAdapter, parse_ply
from ivafr.preprocess.mesh_to_depth import mesh_to_depth_map
from ivafr.registry import get_dataset


def test_parse_ply(tmp_path: Path):
    ply_content = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
property uchar class
end_header
-10.0 20.0 5.0 0.0 0.0 1.0 255 0 0 0
10.0 20.0 5.0 0.0 0.0 1.0 0 255 0 0
-10.0 -20.0 5.0 0.0 0.0 1.0 0 0 255 0
10.0 -20.0 5.0 0.0 0.0 1.0 255 255 0 0
"""
    ply_file = tmp_path / "test.ply"
    ply_file.write_text(ply_content)

    points, rgb = parse_ply(ply_file)
    assert points.shape == (4, 3)
    assert rgb is not None
    assert rgb.shape == (4, 3)
    assert np.allclose(points[0], [-10.0, 20.0, 5.0])
    assert list(rgb[0]) == [255, 0, 0]


def test_mesh_to_depth_map():
    # Create synthetic hemisphere point cloud
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    tt, pp = np.meshgrid(theta, phi)
    r = 50.0
    x = (r * np.sin(tt) * np.cos(pp)).ravel()
    y = (r * np.sin(tt) * np.sin(pp)).ravel()
    z = (r * np.cos(tt)).ravel()
    pts = np.column_stack([x, y, z])

    depth = mesh_to_depth_map(pts, size=32)
    assert depth.shape == (32, 32)
    assert np.isfinite(depth).all()
    assert depth.dtype == np.float32


def test_tufts_adapter_registered():
    adapter_cls = get_dataset("tufts3d")
    assert adapter_cls is Tufts3DAdapter


def test_photo_only_ingestion_and_duplicate_rejection(tmp_path):
    import cv2

    for subject in (1, 2):
        directory = tmp_path / "TD_RGB_E" / str(subject)
        directory.mkdir(parents=True)
        for expression in (1, 2):
            cv2.imwrite(
                str(directory / f"TD_RGB_E_{expression}.jpg"),
                np.full((12, 12, 3), 50 + subject, np.uint8),
            )
    adapter = Tufts3DAdapter(raw_root=tmp_path)
    samples = adapter.discover()
    assert len(samples) == 4
    assert all(s.path_3d is None for s in samples)
    duplicate = tmp_path / "TD_RGB_E/1/another_1.jpg"
    duplicate.write_bytes(samples[0].path_2d.read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        adapter.discover()


def test_ply_properties_reordered_and_truncation_rejected(tmp_path):
    path = tmp_path / "test.ply"
    header = "ply\nformat ascii 1.0\nelement vertex 2\nproperty float z\nproperty float x\nproperty float y\nend_header\n"
    path.write_text(header + "3 1 2\n6 4 5\n", encoding="utf-8")
    points, _ = parse_ply(path)
    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])
    path.write_text(header + "3 1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        parse_ply(path)


def test_tufts_manifest_validation(tmp_path: Path):
    ply_path = tmp_path / "TD_3D_1.ply"
    ply_path.touch()
    img_path = tmp_path / "TD_RGB_E_1.jpg"
    img_path.touch()

    samples = [
        Sample(
            dataset="tufts3d",
            subject_id="S001",
            sample_id="S001_3d",
            path_3d=ply_path,
            meta={
                "expression": "neutral",
                "pose_yaw": 0.0,
                "pose_pitch": 0.0,
                "illumination": "normal",
            },
        ),
        Sample(
            dataset="tufts3d",
            subject_id="S001",
            sample_id="S001_neutral",
            path_2d=img_path,
            meta={
                "expression": "neutral",
                "pose_yaw": 0.0,
                "pose_pitch": 0.0,
                "illumination": "normal",
            },
        ),
    ]

    manifest = samples_to_manifest(samples)
    validate_manifest(manifest)
    assert len(manifest) == 2
    assert (manifest["data_modality"] == "real").all()
