"""Face detection.

The real Yale B path uses OpenCV's ResNet-10 DNN detector with a low-light
enhancement. The toy path can use the Haar cascade and dataset landmarks.
When a detector fails (or ground-truth landmarks are provided by the dataset),
we fall back to dataset-provided landmarks — never silently drop. The detected
source is always logged (``detect_ok`` + ``source``).

M1 will swap in insightface RetinaFace / mediapipe as the primary detector for
real photos; the fallback chain stays identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ivafr.logging_utils import get_logger

log = get_logger("preprocess.detect")

_HAAR = None
_DNN: dict[tuple[str, str], cv2.dnn_Net] = {}


def _haar_cascade() -> cv2.CascadeClassifier | None:
    global _HAAR
    if _HAAR is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        cascade = cv2.CascadeClassifier(path)
        _HAAR = cascade if not cascade.empty() else None
    return _HAAR


def _dnn_detector(cfg: dict[str, object] | None) -> cv2.dnn_Net | None:
    """Load the optional OpenCV ResNet-10 detector from configured paths."""
    params = dict(cfg or {})
    proto = str(params.get("proto") or os.environ.get("IVAFR_FACE_DNN_PROTO", ""))
    model = str(params.get("model") or os.environ.get("IVAFR_FACE_DNN_MODEL", ""))
    if not proto or not model or not Path(proto).is_file() or not Path(model).is_file():
        return None
    key = (proto, model)
    if key not in _DNN:
        _DNN[key] = cv2.dnn.readNetFromCaffe(proto, model)
    return _DNN[key]


def _dnn_image(img: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    """Prepare a low-light image for the DNN without changing saved source data."""
    mode = str(cfg.get("enhance", "gamma"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "gamma":
        gamma = float(cfg.get("gamma", 0.35))
        enhanced = np.power(np.clip(gray.astype(np.float32) / 255.0, 0.0, 1.0), gamma)
        return cv2.cvtColor(np.clip(enhanced * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if mode == "clahe":
        clip = float(cfg.get("clip_limit", 2.0))
        grid = int(cfg.get("grid", 8))
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return img


def _detect_dnn(img: np.ndarray, cfg: dict[str, object]) -> DetectResult | None:
    net = _dnn_detector(cfg)
    if net is None:
        return None
    prepared = _dnn_image(img, cfg)
    h, w = prepared.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(prepared, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()[0, 0]
    confidence = float(cfg.get("confidence", 0.2))
    valid = detections[detections[:, 2] >= confidence]
    if len(valid) == 0:
        return None
    best = valid[int(np.argmax(valid[:, 2]))]
    x0, y0, x1, y1 = best[3:7]
    x = max(0, int(x0 * w))
    y = max(0, int(y0 * h))
    x1_px = min(w - 1, int(x1 * w))
    y1_px = min(h - 1, int(y1 * h))
    bbox = (x, y, max(1, x1_px - x), max(1, y1_px - y))
    return DetectResult(bbox, landmarks_from_bbox(bbox), "dnn_res10", True)


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
    img: np.ndarray,
    gt_landmarks: np.ndarray | None = None,
    detector_cfg: dict[str, object] | None = None,
) -> DetectResult:
    """Detect a face, falling back to ground-truth landmarks.

    Args:
        img: HxWx3 uint8 BGR image.
        gt_landmarks: optional (5,2) pixel landmarks from the dataset.

    Returns:
        :class:`DetectResult`; ``ok`` is False only when nothing worked.
    """
    cfg = dict(detector_cfg or {})
    method = str(cfg.get("method", "haar"))

    if method == "dnn":
        dnn = _detect_dnn(img, cfg)
        if dnn is not None:
            return dnn
        log.warning("OpenCV DNN face detector unavailable or found no face; trying fallback")

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
