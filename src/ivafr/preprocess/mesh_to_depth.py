"""Mesh / Point Cloud to Depth Map projection (3D preprocessing for reconstructed meshes).

Converts an unorganized (N,3) point cloud (e.g. from Tufts SfM PLY) into a
canonical orthographic depth map suitable for range-image and surface-feature
extraction.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from ivafr.logging_utils import get_logger

log = get_logger("preprocess.mesh_to_depth")


def mesh_to_depth_map(
    points: np.ndarray,
    size: int = 64,
    crop_radius_ratio: float = 0.85,
) -> np.ndarray:
    """Project an unorganized (N,3) facial point cloud to an orthographic depth map.

    Args:
        points: (N,3) float32 array in arbitrary SfM coordinates.
        size: output grid dimension (size x size).
        crop_radius_ratio: fraction of face spread to retain.

    Returns:
        depth: (size, size) float32 array, with frontal depth values.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")

    # 1. Center the point cloud horizontally and vertically
    med_x = float(np.median(pts[:, 0]))
    med_y = float(np.median(pts[:, 1]))
    centered_x = pts[:, 0] - med_x
    centered_y = pts[:, 1] - med_y
    z = pts[:, 2]

    # 2. Filter outlier points beyond central face region (e.g. 5th-95th percentile)
    p5_x, p95_x = np.percentile(centered_x, [5, 95])
    p5_y, p95_y = np.percentile(centered_y, [5, 95])
    span_x = max(float(p95_x - p5_x), 1e-4)
    span_y = max(float(p95_y - p5_y), 1e-4)
    span = max(span_x, span_y) * crop_radius_ratio

    mask = (
        (centered_x >= -span)
        & (centered_x <= span)
        & (centered_y >= -span)
        & (centered_y <= span)
    )
    if mask.sum() < 100:
        mask = np.ones(len(pts), dtype=bool)

    valid_pts = np.column_stack([centered_x[mask], centered_y[mask]])
    valid_z = z[mask]

    # 3. Create regular grid in normalized coordinate range [-span, span]
    grid_y, grid_x = np.mgrid[-span : span : complex(0, size), -span : span : complex(0, size)]

    # 4. Interpolate depth using linear griddata, fallback to nearest for border NaNs
    depth_linear = griddata(valid_pts, valid_z, (grid_x, grid_y), method="linear")
    if np.isnan(depth_linear).any():
        depth_nearest = griddata(valid_pts, valid_z, (grid_x, grid_y), method="nearest")
        depth = np.where(np.isnan(depth_linear), depth_nearest, depth_linear)
    else:
        depth = depth_linear

    return depth.astype(np.float32)
