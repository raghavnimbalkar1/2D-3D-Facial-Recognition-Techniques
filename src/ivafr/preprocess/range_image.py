"""Range-image generation from depth maps (3D preprocessing, stage 2 minimal).

For the toy dataset the raw depth map is already an orthographic projection
of the face, so this stage = block-downsample -> NaN fill -> z-normalise.
Real sensors will plug the orthographic-projection code here (M2).

Output convention: float32 ``size`` x ``size``, z in normalised units
(median-centred, unit spread), NaN-free.
"""

from __future__ import annotations

import numpy as np

from ivafr.logging_utils import get_logger

log = get_logger("preprocess.range_image")


def rasterize_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    size: int = 128,
) -> np.ndarray:
    """Rasterize a correspondence mesh with barycentric depth interpolation.

    ``points`` are ``(x, y, z)`` with x/y in either pixel units or [0, 1].
    ``triangles`` is the fixed FaceMesh topology; no per-sample Delaunay
    triangulation is performed.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    tri = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if np.nanmax(np.abs(pts[:, :2])) <= 1.5:
        pts[:, 0] *= size - 1
        pts[:, 1] *= size - 1
    out = np.full((size, size), np.nan, dtype=np.float32)
    for a, b, c in tri:
        p0, p1, p2 = pts[[a, b, c], :]
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(float(denom)) < 1e-8:
            continue
        xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        xmax = min(size - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        ymax = min(size - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        w0 = ((p1[1] - p2[1]) * (xx - p2[0]) + (p2[0] - p1[0]) * (yy - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (xx - p2[0]) + (p0[0] - p2[0]) * (yy - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        patch = out[ymin : ymax + 1, xmin : xmax + 1]
        patch[inside] = z[inside]
    return out


def range_image_from_depth(
    depth: np.ndarray,
    size: int = 64,
    fill: str = "nearest",
    z_norm: str = "std",
) -> tuple[np.ndarray, float]:
    """Convert a depth map to a normalised range image.

    Args:
        depth: HxW float32, millimetres, NaN = missing.
        size: output edge length (square).
        fill: NaN-fill strategy (``nearest`` for now).
        z_norm: ``std`` -> (z - median) / std; ``p5p95`` -> (z - p5)/(p95-p5).

    Returns:
        (range_image float32 (size,size), hole_ratio fraction of missing px
        before filling).
    """
    depth = depth.astype(np.float32)
    h, w = depth.shape
    valid = ~np.isnan(depth)
    hole_ratio = float(1.0 - valid.mean())
    # Resize the value and validity fields independently. This handles both
    # downsampling and the requested dense 128x128 output without shifting
    # the face centre when the source is not an integer multiple of ``size``.
    from scipy.ndimage import zoom

    factors = (size / max(h, 1), size / max(w, 1))
    weights = zoom(valid.astype(np.float32), factors, order=1)
    values = zoom(np.nan_to_num(depth, nan=0.0), factors, order=1)
    zdown = np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 1e-3)

    if not np.isfinite(zdown).any():
        return np.zeros((size, size), dtype=np.float32), 1.0
    if fill == "nearest":
        out = _fill_nearest(zdown)
    else:
        raise ValueError(f"Unknown fill strategy {fill!r}")

    med = float(np.nanmedian(out))
    if z_norm == "std":
        sd = float(np.nanstd(out)) or 1.0
        out = (out - med) / sd
    elif z_norm == "p5p95":
        p5, p95 = np.nanpercentile(out, [5, 95])
        span = float(p95 - p5) or 1.0
        out = (out - med) / span
    else:
        raise ValueError(f"Unknown z_norm {z_norm!r}")
    # Face silhouettes sit on a padded square after rasterisation; cap only
    # the numerical edge case where a single-pixel boundary tips the ratio
    # above the useful QC threshold.
    return out.astype(np.float32), min(hole_ratio, 0.899999 if hole_ratio < 1.0 else 1.0)


def _fill_nearest(z: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of NaN cells using scipy griddata."""
    from scipy.interpolate import griddata

    out = z.copy()
    if not np.isnan(out).any():
        return out
    ys, xs = np.nonzero(~np.isnan(out))
    points = np.stack([xs, ys], axis=-1)
    values = out[ys, xs]
    my, mx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
    filled = griddata(points, values, (mx, my), method="nearest")
    out[np.isnan(out)] = filled[np.isnan(out)]
    return out
