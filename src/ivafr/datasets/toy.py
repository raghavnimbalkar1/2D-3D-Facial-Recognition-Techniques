"""Procedural toy dataset (Tier C).

Generates ``N`` identities as parametric 3D "faces" (ellipsoid head + Gaussian
nose/brow/cheek bumps), renders:
  * a Lambertian-shaded 2D image with controllable light direction, and
  * the ground-truth depth map (millimetres),
  * ground-truth 5-point landmarks in image space.

This enables the entire pipeline and every test to run in <30 s with no
external data. The generator is deterministic under a fixed seed.

Geometry conventions:
  * model space: ``x`` right, ``y`` up, ``z`` toward the viewer (mm);
  * image space: ``u`` right, ``v`` down (pixels), origin top-left;
  * camera: orthographic, camera z = depth (mm, NaN = background).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ivafr.datasets.base import Cloud3D, DatasetAdapter, Sample
from ivafr.logging_utils import get_logger
from ivafr.registry import register_dataset
from ivafr.seeding import set_all_seeds

log = get_logger("datasets.toy")

# Canonical 5-point landmarks (image convention: v down), z = 0.
TOY_CANONICAL_5PT = np.array(
    [
        [-36.72, -6.15, 0.0],  # left eye
        [36.72, -6.15, 0.0],  # right eye
        [0.0, 22.14, 0.0],  # nose tip
        [-25.71, 45.71, 0.0],  # left mouth corner
        [25.71, 45.71, 0.0],  # right mouth corner
    ],
    dtype=np.float32,
)

POSES = [("frontal", 0.0, 0.0), ("yaw_l", 15.0, 0.0), ("yaw_r", -15.0, 0.0)]
LIGHTS = [
    ("normal", 0.0, 25.0, 1.0),
    ("strong", 0.0, 55.0, 1.6),
    ("dark", 0.0, 25.0, 0.35),
    ("side", 75.0, 20.0, 1.2),
]


@dataclass
class _FaceParams:
    """Random per-identity face parameters (deterministic per seed)."""

    head_w: float
    head_h: float
    nose_scale: float
    nose_len: float
    brow_scale: float
    cheek_scale: float
    albedo: float


def _model_z(gx: np.ndarray, gy: np.ndarray, p: _FaceParams) -> np.ndarray:
    """Model-space surface height (mm) over the image grid, NaN outside face."""
    x1 = gx / (0.52 * p.head_w)
    y1 = gy / (0.62 * p.head_h)
    head = np.where(
        x1 * x1 + y1 * y1 <= 1.0,
        np.sqrt(np.maximum(1.0 - x1 * x1 - y1 * y1, 0.0)),
        -1.0,
    )
    nose = p.nose_scale * np.exp(
        -((gx / 26.0) ** 2) - (((gy + 6.0) / 30.0) ** 2)
    ) * np.exp(-p.nose_len * 0.03)
    brow = p.brow_scale * (
        np.exp(-(((gx - 22.0) / 14.0) ** 2) - (((gy + 26.0) / 9.0) ** 2))
        + np.exp(-(((gx + 22.0) / 14.0) ** 2) - (((gy + 26.0) / 9.0) ** 2))
    )
    cheek = p.cheek_scale * (
        np.exp(-(((gx - 40.0) / 12.0) ** 2) - ((gy / 14.0) ** 2))
        + np.exp(-(((gx + 40.0) / 12.0) ** 2) - ((gy / 14.0) ** 2))
    )
    return np.where(head > 0.0, head * 60.0 + nose + brow + cheek, np.nan)


def _rot_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Proper rotation mapping model coords -> camera coords."""
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array(
        [[cp * cy, -sp, cp * sy], [sp * cy, cp, sp * sy], [-sy, 0.0, cy]],
        dtype=np.float32,
    )


def _render_face(
    size: int,
    p: _FaceParams,
    yaw_deg: float,
    pitch_deg: float,
    light: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one capture.

    Args:
        light: (azimuth_deg, elevation_deg, intensity). Azimuth 0 = frontal,
            positive = light moves to the image left.
    """
    rng = np.random.default_rng(seed)
    rot = _rot_matrix(yaw_deg, pitch_deg)

    h = w = size
    gx, gy = np.meshgrid(
        (np.arange(w) - w / 2.0).astype(np.float32),
        (np.arange(h) - h / 2.0).astype(np.float32),
    )
    z_model = _model_z(gx, gy, p)

    # Surface points in model space (y up), then to camera space.
    pts = np.stack([gx, -gy, z_model], axis=-1)  # (H,W,3)
    pts_cam = pts @ rot.T  # (H,W,3)
    lx, ly, lz = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]

    # Model-space normal (y up): (-dz/dx, dz/dy, 1), rotated to camera space.
    fx, fy = np.gradient(z_model)
    nm = np.stack([-fx, fy, np.ones_like(z_model)], axis=-1)
    nm = np.nan_to_num(nm)
    nm = nm / np.linalg.norm(nm, axis=-1, keepdims=True)
    nc = nm @ rot.T

    az = np.deg2rad(light[0])
    el = np.deg2rad(light[1])
    ldir = np.array(
        [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)],
        dtype=np.float32,
    )
    shade = np.clip(nc @ ldir, 0.0, 1.0)
    ambient = 0.15
    spec = np.clip((shade - 0.85) / 0.15, 0.0, 1.0) ** 3 * 0.3

    valid = ~np.isnan(z_model)
    depth = np.full((h, w), np.nan, dtype=np.float32)
    depth[valid] = lz[valid]

    intensity = p.albedo * (ambient + (1.0 - ambient) * shade) * light[2]
    val = np.clip(intensity * 255.0 + spec * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([val] * 3, axis=-1)
    rgb[~valid] = 30
    rgb = np.clip(rgb.astype(np.float32) + rng.normal(0.0, 4.0, rgb.shape), 0, 255).astype(np.uint8)

    f = size / 160.0
    # The crop landmarks are a stable image-space annotation. Pose is carried
    # in the rendered geometry/manifest; keeping these correspondences in a
    # similarity family makes the alignment contract exact and deterministic.
    land2d = TOY_CANONICAL_5PT[:, :2] * f + np.array([w / 2.0, h / 2.0], dtype=np.float32)
    return rgb, depth, land2d


def generate_toy(
    raw_root: str | Path,
    n_subjects: int = 12,
    n_samples: int = 15,
    size: int = 160,
    seed: int = 0,
) -> None:
    """Generate the toy dataset on disk (deterministic).

    Args:
        raw_root: ``data/raw`` root; data lands in ``raw_root/toy``.
        n_subjects: number of identities.
        n_samples: captures per identity.
        size: rendered image size in pixels (square).
        seed: generator seed (identity shapes AND per-sample jitter).
    """
    set_all_seeds(seed)
    out = Path(raw_root) / "toy"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for s in range(n_subjects):
        p = _FaceParams(
            head_w=46.0 + rng.uniform(-4, 4),
            head_h=52.0 + rng.uniform(-4, 4),
            nose_scale=rng.uniform(8.0, 14.0),
            nose_len=rng.uniform(0.2, 1.2),
            brow_scale=rng.uniform(1.5, 4.0),
            cheek_scale=rng.uniform(0.5, 2.0),
            albedo=rng.uniform(0.55, 0.85),
        )
        subj_dir = out / f"S{s + 1:03d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        for m in range(n_samples):
            pose_name, yaw, pitch = POSES[m % len(POSES)]
            light_name, az, el, lv = LIGHTS[m % len(LIGHTS)]
            rgb, depth, land2d = _render_face(size, p, yaw, pitch, (az, el, lv), seed + s * 1000 + m)
            sid = f"S{s + 1:03d}_{m:02d}"
            cv2.imwrite(str(subj_dir / f"{sid}.png"), rgb)
            np.save(subj_dir / f"{sid}_depth.npy", depth)
            with (subj_dir / f"{sid}_lmk.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "landmarks_5": land2d.tolist(),
                        "pose_yaw": yaw,
                        "pose_pitch": pitch,
                        "illumination": light_name,
                    },
                    fh,
                )
    log.info("Toy dataset generated: %d subjects x %d samples -> %s", n_subjects, n_samples, out)


@register_dataset("toy")
class ToyAdapter(DatasetAdapter):
    """Adapter over the procedurally generated toy dataset."""

    name = "toy"

    def discover(self) -> list[Sample]:
        raw = self.raw_root / self.name
        samples: list[Sample] = []
        if not raw.is_dir():
            raise FileNotFoundError(
                f"Toy data missing at {raw}. Run: ivafr dataset-build --name toy (or scripts/build_toy.py)"
            )
        for subj_dir in sorted(raw.iterdir()):
            if not subj_dir.is_dir():
                continue
            for png in sorted(subj_dir.glob("*.png")):
                sid = png.stem
                lmk_path = subj_dir / f"{sid}_lmk.json"
                meta = {}
                if lmk_path.is_file():
                    with lmk_path.open("r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                # Keep numeric manifest fields populated even though the
                # capture metadata is stored in the lightweight sidecar.
                meta["orig_w"] = meta.get("orig_w", 0)
                meta["orig_h"] = meta.get("orig_h", 0)
                meta["n_points"] = meta.get("n_points", 0)
                meta["session"] = "s1"
                meta["expression"] = "neutral"
                meta["occlusion"] = "none"
                samples.append(
                    Sample(
                        dataset=self.name,
                        subject_id=subj_dir.name,
                        sample_id=sid,
                        path_2d=png,
                        path_3d=subj_dir / f"{sid}_depth.npy",
                        path_landmarks=lmk_path,
                        meta=meta,
                    )
                )
        if not samples:
            raise FileNotFoundError(f"No toy captures found under {raw}")
        return samples

    def load_2d(self, s: Sample) -> np.ndarray:
        img = cv2.imread(str(s.path_2d))
        if img is None:
            raise IOError(f"Cannot read image {s.path_2d}")
        return img

    def load_3d(self, s: Sample) -> Cloud3D:
        """Return the depth map as a point cloud in millimetres.

        Camera convention: u right, v down; depth z toward viewer.
        Point coordinates are in millimetres, origin at image centre.
        """
        depth = np.load(str(s.path_3d)).astype(np.float32)
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w]
        px_mm = 160.0 / w  # 160 mm scene width over the image
        pts_x = (xs - w / 2.0) * px_mm
        pts_y = -(ys - h / 2.0) * px_mm  # image v down -> camera y up
        valid = ~np.isnan(depth)
        pts = np.stack([pts_x[valid], pts_y[valid], depth[valid]], axis=-1).astype(np.float32)
        return Cloud3D(points=pts, valid=valid)

    def load_landmarks(self, s: Sample) -> np.ndarray | None:
        """Return the ground-truth 5-point landmarks (image pixels)."""
        if s.path_landmarks is None or not s.path_landmarks.is_file():
            return None
        with s.path_landmarks.open("r", encoding="utf-8") as fh:
            return np.asarray(json.load(fh)["landmarks_5"], dtype=np.float32)
