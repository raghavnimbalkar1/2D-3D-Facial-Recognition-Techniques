"""Confusion matrix figure + CSV export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ivafr.viz.style import save_fig, setup_style


def plot_cm(cm: np.ndarray, labels: list[str], name: str, normalize: bool = False) -> Path:
    """Plot a confusion matrix; returns the PNG path.

    Args:
        cm: (C, C) raw counts.
        labels: class (subject) names.
        name: figure basename.
        normalize: row-normalise before plotting.
    """
    setup_style()
    import matplotlib.pyplot as plt

    mat = cm.astype(np.float64)
    if normalize:
        rows = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.45)))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    if len(labels) <= 20:
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({'normalized' if normalize else 'raw'})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return save_fig(fig, name)


def cm_csv(cm: np.ndarray, labels: list[str], path: str | Path) -> None:
    """Write the raw confusion matrix as CSV for report tables."""
    df = pd.DataFrame(cm, index=labels, columns=labels)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)