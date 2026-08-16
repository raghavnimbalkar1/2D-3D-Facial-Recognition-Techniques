"""Geometric alignment to a canonical 5-point template.

Uses a closed-form similarity transform (Umeyama) from detected landmarks to
the canonical template, then warps with inverse mapping. Outputs two
resolutions: 112x112 colour (ArcFace-compatible) and 64x64 grayscale
(classical methods).
"""

from __future__ import annotations

import numpy as np
import cv2

# Canonical 5-point template in a 112x112 image (image convention: v down):
# left eye, right eye, nose tip, left mouth, right mouth.
TEMPLATE_112 = np.array(
    [
        [30.2946, 51.6963],
        [81.7054, 51.6963],
        [56.0, 71.5],
        [38.0, 88.0],
        [74.0, 88.0],
    ],
    dtype=np.float32,
)

TEMPLATE_64 = TEMPLATE_112 * (64.0 / 112.0)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best-fit similarity (sR, t) mapping ``src`` onto ``dst`` (Umeyama).

    Args:
        src: (K,2) source points.
        dst: (K,2) target points.

    Returns:
        (2,3) affine matrix M such that ``M @ [x, y, 1]`` ~= dst.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c
    u, d, vt = np.linalg.svd(cov)
    sign = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
    u[:, -1] *= sign
    rot = u @ vt
    scale = d.sum() / (np.linalg.norm(src_c) ** 2 + 1e-12)
    t = mu_d - scale * (rot @ mu_s)
    return np.array([[scale * rot[0, 0], scale * rot[0, 1], t[0]],
                     [scale * rot[1, 0], scale * rot[1, 1], t[1]]], dtype=np.float32)


def align_to_template(
    img: np.ndarray, landmarks: np.ndarray, size: int = 112, template: np.ndarray | None = None
) -> np.ndarray:
    """Warp ``img`` so that ``landmarks`` land on the canonical template.

    Args:
        img: HxWx3 uint8 BGR (or HxW grayscale).
        landmarks: (5,2) detected landmarks (same order as template).
        size: output size (square), 112 by default.
        template: canonical landmarks; defaults to the 112 template scaled.

    Returns:
        uint8 image of shape (size, size, 3) or (size, size).
    """
    if template is None:
        template = TEMPLATE_112 * (size / 112.0)
    tpl = np.asarray(template, dtype=np.float32)
    lms = np.asarray(landmarks, dtype=np.float32)
    # ``similarity_transform`` maps source coordinates to output coordinates.
    # OpenCV's default warpAffine convention internally applies the inverse
    # mapping when sampling the source, so passing the inverse here would
    # invert the transform twice and produce mostly empty/constant crops.
    M = similarity_transform(lms, tpl)
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
    return out


def to_gray(img: np.ndarray) -> np.ndarray:
    """BT.601 grayscale conversion."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
