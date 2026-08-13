"""Probe-only synthetic occlusion operators for the E08 sweep."""

from __future__ import annotations

import numpy as np


def occlude(image: np.ndarray, kind: str = "sunglasses", fraction: float = 0.3, seed: int = 0) -> np.ndarray:
    """Return a copy with a deterministic rectangular occluder."""
    out = np.asarray(image).copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    frac = float(np.clip(fraction, 0.01, 0.95))
    if kind == "sunglasses":
        y0, y1 = int(h * 0.28), int(h * 0.48)
        x0, x1 = int(w * 0.08), int(w * 0.92)
    else:
        area = max(1, int(h * w * frac))
        bh = max(1, int(np.sqrt(area)))
        bw = max(1, int(area / bh))
        y0 = int(rng.integers(0, max(1, h - bh + 1)))
        x0 = int(rng.integers(0, max(1, w - bw + 1)))
        y1, x1 = min(h, y0 + bh), min(w, x0 + bw)
    fill = np.median(out.reshape(-1, out.shape[-1]), axis=0) if out.ndim == 3 else np.median(out)
    out[y0:y1, x0:x1] = fill
    return out
