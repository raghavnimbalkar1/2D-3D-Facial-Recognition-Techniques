"""Probe-only synthetic occlusion operators for the E08 sweep."""

from __future__ import annotations

import numpy as np


def occlude(
    image: np.ndarray, kind: str = "sunglasses", fraction: float = 0.3, seed: int = 0
) -> np.ndarray:
    """Return a copy with a deterministic rectangular occluder."""
    out = np.asarray(image).copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    frac = float(fraction)
    if not 0 < frac < 1 or kind not in {"sunglasses", "block", "random"}:
        raise ValueError("Invalid occlusion kind or fraction")
    if kind == "sunglasses":
        bw = min(w, max(int(round(w * 0.84)), int(np.ceil(w * frac))))
        bh = max(1, min(h, int(round(h * w * frac / bw))))
        x0 = (w - bw) // 2
        y0 = max(0, min(h - bh, int(round(h * 0.38 - bh / 2))))
        y1, x1 = y0 + bh, x0 + bw
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
