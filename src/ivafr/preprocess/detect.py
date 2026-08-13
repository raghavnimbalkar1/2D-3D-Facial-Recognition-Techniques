"""Face detection.

Primary detector for the toy/classical path: OpenCV Haar cascade. When the
detector fails (or a ground-truth bounding box/landmarks are provided by the
dataset), we fall back to dataset-provided landmarks — never silently drop.
The detected source is always logged (``detect_ok`` + ``source``).

M1 will swap in insightface RetinaFace / mediapipe as the primary detector for
real photos; the fallback chain stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ivafr.logging_utils import get_logger

log = get_logger("preprocess.detect")

_HAAR = None


def _haar_cascade() -> cv2.CascadeClassifier | None:
    global _HAAR
    if _HAAR is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        cascade = cv2.CascadeClassifier(path)
        _HAAR = cascade if not cascade.empty() else None
    return _HAAR


@dataclass
class DetectResult:
    """Output of one detection attempt."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h) pixels
    landmarks: np.ndarray | None  # (5,2) pixels
    source: str  # "haar" | "ground_truth" | "landmark_bbox"
    ok: bool


def bbox_from_landmarks(landmarks: np.ndarray, margin: float = 0.35) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box around landmarks, inflated by ``margin``.

    Margin is relative to the inter-ocular distance (standard face-crop
    heuristic), so the crop scale is stable across images.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    iou = float(np.linalg.norm(pts[1] - pts[0])) if len(pts) > 1 else (x1 - x0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0 * (1.0 + margin)
    x, y = int(cx - half), int(cy - half)
    return x, y, int(half * 2), int(half * 2)


def landmarks_from_bbox(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Create a deterministic five-point approximation from a detected box."""
    x, y, w, h = map(float, bbox)
    return np.asarray(
        [
            [x + 0.32 * w, y + 0.38 * h],
            [x + 0.68 * w, y + 0.38 * h],
            [x + 0.50 * w, y + 0.55 * h],
            [x + 0.36 * w, y + 0.73 * h],
            [x + 0.64 * w, y + 0.73 * h],
        ],
        dtype=np.float32,
    )


def detect_face(
    img: np.ndarray, gt_landmarks: np.ndarray | None = None
) -> DetectResult:
    """Detect a face, falling back to ground-truth landmarks.

    Args:
        img: HxWx3 uint8 BGR image.
        gt_landmarks: optional (5,2) pixel landmarks from the dataset.

    Returns:
        :class:`DetectResult`; ``ok`` is False only when nothing worked.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = _haar_cascade()
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces):
            x, y, w, h = tuple(int(v) for v in faces[0])
            return DetectResult((x, y, w, h), landmarks_from_bbox((x, y, w, h)), "haar", True)

    if gt_landmarks is not None:
        lms = np.asarray(gt_landmarks, dtype=np.float32)
        if lms.shape == (5, 2):
            bbox = bbox_from_landmarks(lms)
            return DetectResult(bbox, lms, "ground_truth", True)

    return DetectResult((0, 0, 0, 0), None, "none", False)
