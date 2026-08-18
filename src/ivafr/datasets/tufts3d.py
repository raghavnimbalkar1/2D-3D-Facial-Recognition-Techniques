"""Tufts Face Database adapter (TD_3D meshes + TD_RGB_E photos).

TD_3D provides one SfM-reconstructed PLY mesh per participant (ASCII PLY,
~250k-300k vertices: xyz, normals, diffuse RGB, class). Units are arbitrary
SfM scale, NOT millimetres — the 3D preprocessing chain normalises scale and
produces depth range images.

TD_RGB_E pairs each participant with 5 expression photos (neutral, smile,
eyes closed, shocked, sunglasses) captured with a Nikon D3100.

Terms: non-commercial research only, no redistribution; cite Panetta et al.,
IEEE TPAMI 2018. See docs/DATASETS.md and docs/ETHICS.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ivafr.datasets.base import Cloud3D, DatasetAdapter, Sample
from ivafr.logging_utils import get_logger
from ivafr.registry import register_dataset

log = get_logger("datasets.tufts3d")

_PLY_RE = re.compile(r"TD_3D_(\d+)\.ply$", re.IGNORECASE)
_EXPR_MAP = {
    1: "neutral",
    2: "smile",
    3: "eyes_closed",
    4: "shocked",
    5: "sunglasses",
}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}


def parse_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse an ASCII PLY (xyz nx ny nz r g b class) into (points, rgb).

    Args:
        path: PLY file.

    Returns:
        points (N,3) float32, rgb (N,3) uint8 or None.
    """
    p = Path(path)
    n_header = 0
    n_vert = 0
    vertex_props: list[str] = []
    with p.open("rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Malformed PLY: unexpected EOF in header of {p}")
            n_header += 1
            if n_header == 1 and line.strip() != b"ply":
                raise ValueError(f"Not a PLY file: {p}")
            if line.startswith(b"element vertex"):
                n_vert = int(line.split()[-1])
            elif line.startswith(b"property"):
                parts = line.split()
                vertex_props.append(parts[-1].decode())
            elif line.startswith(b"format binary"):
                raise ValueError(
                    f"Binary PLY not supported yet: {p} (format line: {line.decode().strip()})"
                )
            elif line.startswith(b"end_header"):
                break
    data = np.loadtxt(p, skiprows=n_header, max_rows=n_vert, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Malformed vertex block in {p}")
    xyz_cols = [idx for idx, name in enumerate(vertex_props) if name in ("x", "y", "z")]
    rgb_cols = [
        idx
        for idx, name in enumerate(vertex_props)
        if name in ("diffuse_red", "diffuse_green", "diffuse_blue")
    ]
    points = data[:, xyz_cols].astype(np.float32)
    rgb = None
    if len(rgb_cols) == 3 and data.shape[1] > max(xyz_cols + rgb_cols):
        rgb = np.clip(data[:, rgb_cols], 0, 255).astype(np.uint8)
    return points, rgb


@register_dataset("tufts3d")
class Tufts3DAdapter(DatasetAdapter):
    """Discovers TD_3D meshes plus matched TD_RGB_E photos by participant."""

    name = "tufts3d"

    def discover(self) -> list[Sample]:
        samples: list[Sample] = []
        mesh_root = self.raw_root / "TD_3D"
        photo_root = self.raw_root / "TD_RGB_E"

        if not mesh_root.is_dir():
            raise FileNotFoundError(
                f"Tufts TD_3D missing at {mesh_root}. Run scripts/fetch_tufts.sh"
            )

        # Discover 2D photos mapped by subject ID
        photos_by_subject: dict[str, dict[str, Path]] = {}
        if photo_root.is_dir():
            for img in sorted(photo_root.rglob("*")):
                if img.suffix.lower() not in _IMG_EXTS or not img.is_file():
                    continue
                # Folder or filename contains subject number, e.g. TD_RGB_E/1/TD_RGB_E_2.jpg
                parent_stem = img.parent.name
                if parent_stem.isdigit():
                    s_num = int(parent_stem)
                else:
                    m_subj = re.search(r"TD_RGB_E_(\d+)", img.stem)
                    s_num = int(m_subj.group(1)) if m_subj else 0

                if s_num <= 0:
                    continue

                m_expr = re.search(r"(\d+)$", img.stem)
                expr_idx = int(m_expr.group(1)) if m_expr else 1
                expr_name = _EXPR_MAP.get(expr_idx, f"expr_{expr_idx}")
                subject_id = f"S{s_num:03d}"
                photos_by_subject.setdefault(subject_id, {})[expr_name] = img

        for ply in sorted(mesh_root.glob("TD_3D_*.ply")):
            m = _PLY_RE.match(ply.name)
            if not m:
                continue
            s_num = int(m.group(1))
            subject_id = f"S{s_num:03d}"
            photos = photos_by_subject.get(subject_id, {})

            # 1. Add 3D mesh sample
            samples.append(
                Sample(
                    dataset=self.name,
                    subject_id=subject_id,
                    sample_id=f"{subject_id}_3d",
                    path_2d=None,
                    path_3d=ply,
                    meta={
                        "expression": "neutral",
                        "pose_yaw": 0.0,
                        "pose_pitch": 0.0,
                        "illumination": "normal",
                        "occlusion": "none",
                        "session": "s1",
                        "n_points": 250000,
                        "notes": "sfm_mesh",
                    },
                )
            )

            # 2. Add 2D photo samples
            for expr_name, photo_path in sorted(photos.items()):
                occlusion = "sunglasses" if expr_name == "sunglasses" else "none"
                samples.append(
                    Sample(
                        dataset=self.name,
                        subject_id=subject_id,
                        sample_id=f"{subject_id}_{expr_name}",
                        path_2d=photo_path,
                        path_3d=None,
                        meta={
                            "expression": expr_name,
                            "pose_yaw": 0.0,
                            "pose_pitch": 0.0,
                            "illumination": "normal",
                            "occlusion": occlusion,
                            "session": "s1",
                            "notes": "rgb_e_photo",
                        },
                    )
                )

        if not samples:
            raise FileNotFoundError(f"No TD_3D meshes found under {mesh_root}")
        log.info("Tufts3DAdapter discovered %d samples across %d subjects", len(samples), len(set(s.subject_id for s in samples)))
        return samples

    def load_2d(self, s: Sample) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if s.path_2d is None:
            raise ValueError(f"Sample {s.sample_id} has no 2D image")
        img = cv2.imread(str(s.path_2d))
        if img is None:
            raise IOError(f"Cannot read image {s.path_2d}")
        return img

    def load_3d(self, s: Sample) -> Cloud3D:
        if s.path_3d is None:
            raise ValueError(f"Sample {s.sample_id} has no 3D mesh")
        points, rgb = parse_ply(s.path_3d)
        return Cloud3D(points=points, rgb=rgb)