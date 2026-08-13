"""Montage / grid of images for QC and report figures."""

from __future__ import annotations

import math

import numpy as np

from ivafr.viz.style import save_fig, setup_style


def montage(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    cols: int | None = None,
    name: str = "fig_montage",
    figsize_scale: float = 2.2,
    cmap: str | None = None,
) -> None:
    """Arrange images in a grid and save as PNG+PDF.

    Args:
        images: list of 2D (grayscale) or 3D (H,W,3) arrays.
        titles: optional per-image titles.
        cols: grid columns (defaults to ceil(sqrt(n))).
        name: output basename.
        figsize_scale: inches per cell edge.
        cmap: forced colormap (e.g. ``viridis`` for depth).
    """
    setup_style()
    import matplotlib.pyplot as plt

    n = len(images)
    if n == 0:
        raise ValueError("montage() needs at least one image")
    cols = cols or max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale))
    axes = np.atleast_1d(axes).ravel()
    for k, img in enumerate(images):
        ax = axes[k]
        arr = np.asarray(img)
        if arr.ndim == 2:
            ax.imshow(arr, cmap=cmap or "gray")
        else:
            ax.imshow(arr)
        if titles is not None:
            ax.set_title(titles[k], fontsize=8)
        ax.axis("off")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    save_fig(fig, name)
