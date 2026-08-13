"""MediaPipe FaceMesh landmark extraction with a stable pseudo-3D contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# MediaPipe's canonical nose-tip landmark. Keep this in one documented place
# so a future version change can be audited and tested visually.
NOSE_TIP_INDEX = 1
N_LANDMARKS = 468


def facemesh_landmarks(
    image: np.ndarray,
    refine: bool = True,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """Return FaceMesh coordinates as ``(468, 3)`` float32 values.

    Coordinates are x/y normalised to [0, 1] and z is MediaPipe's relative
    camera-depth coordinate. The optional dependency is imported lazily.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("MediaPipe is required for Yale pseudo-3D preprocessing") from exc
    rgb = image[..., ::-1] if image.ndim == 3 else image
    if hasattr(mp, "solutions"):
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            raise ValueError("FaceMesh did not detect a face")
        points = result.multi_face_landmarks[0].landmark
    else:
        asset = Path(model_path or os.environ.get("IVAFR_MEDIAPIPE_MODEL", ""))
        if not asset.is_file():
            raise RuntimeError(
                "MediaPipe Tasks requires a Face Landmarker .task model; "
                "set IVAFR_MEDIAPIPE_MODEL or landmarks.model_asset_path"
            )
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(asset),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            raise ValueError("MediaPipe Face Landmarker did not detect a face")
        points = result.face_landmarks[0]
    pts = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    if pts.shape != (N_LANDMARKS, 3):
        raise ValueError(f"Expected {(N_LANDMARKS, 3)} FaceMesh landmarks, got {pts.shape}")
    return pts


def nose_tip(points: np.ndarray, index: int = NOSE_TIP_INDEX) -> np.ndarray:
    """Return and validate the documented nose-tip point."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or index >= len(pts):
        raise ValueError("landmarks must be a (K,3) array containing the nose tip")
    point = pts[index]
    if not np.isfinite(point).all():
        raise ValueError("nose-tip landmark is non-finite")
    return point


def landmark_quality(points: np.ndarray) -> dict[str, Any]:
    """Return manifest-friendly quality flags for a landmark set."""
    pts = np.asarray(points, dtype=np.float32)
    ok = pts.shape == (N_LANDMARKS, 3) and bool(np.isfinite(pts).all())
    return {"n_points": int(len(pts)), "detect_ok": ok, "nosetip_ok": ok}
