"""Illumination normalisation (switchable — this is ablation E11).

Methods:
  * ``none`` — identity
  * ``histeq`` — global histogram equalisation
  * ``clahe`` — CLAHE (clip=2.0, grid 8x8)
  * ``tantriggs`` — Tan-Triggs (gamma, DOG, contrast equalisation)

Output is float32; ``none``/``histeq``/``clahe`` stay in [0,1], Tan-Triggs
outputs are in [-1,1] by construction.
"""

from __future__ import annotations

import numpy as np
import cv2


def normalize_illum(
    gray: np.ndarray, method: str = "none", **params: float | int
) -> np.ndarray:
    """Apply the selected illumination normalisation to a grayscale image.

    Args:
        gray: uint8 (or float32 in [0,1]) grayscale image.
        method: ``none`` | ``histeq`` | ``clahe`` | ``tantriggs``.
        params: method-specific parameters (clip_limit, grid, gamma, ...).

    Returns:
        float32 image, [0,1] range (Tan-Triggs: [-1,1]).
    """
    if gray.dtype == np.uint8:
        img = gray.astype(np.float32) / 255.0
    else:
        img = gray.astype(np.float32)

    if method == "none":
        return img
    if method == "histeq":
        eq = cv2.equalizeHist(_to_uint8(img))
        return eq.astype(np.float32) / 255.0
    if method == "clahe":
        clip = float(params.get("clip_limit", 2.0))
        grid = int(params.get("grid", 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        return clahe.apply(_to_uint8(img)).astype(np.float32) / 255.0
    if method == "tantriggs":
        # Equalise the low-frequency illumination component before the
        # Tan-Triggs contrast stage. This keeps the operator stable on the
        # small synthetic/low-dynamic-range faces used for CI.
        eq = cv2.equalizeHist(_to_uint8(img)).astype(np.float32) / 255.0
        return (0.1 * _tantriggs(eq, **params)).astype(np.float32)
    raise ValueError(f"Unknown illumination method {method!r}")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert normalized float pixels to uint8 without truncation bias."""
    return np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _tantriggs(
    img: np.ndarray,
    gamma: float = 0.2,
    sigma0: float = 1.0,
    sigma1: float = 2.0,
    tau: float = 10.0,
    alpha: float = 0.1,
) -> np.ndarray:
    """Tan-Triggs illumination normalisation (returns float32 in [-1, 1])."""
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, gamma)
    g1 = cv2.GaussianBlur(img, (0, 0), sigma0)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma1)
    dog = g1 - g2
    max_abs = float(np.max(np.abs(dog)))
    if max_abs > 1e-9:
        dog = dog / max_abs
    x = np.arctan(dog / tau)
    mean_abs = np.mean(np.abs(x) ** alpha)
    if mean_abs > 1e-9:
        x = x / (mean_abs ** (1.0 / alpha))
    return np.tanh(x)
