"""Shared matplotlib style + figure saving.

All figures are 300 dpi PNG + PDF with a consistent look. Every save writes
into the active run directory (set via :func:`set_output_dir`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_OUTPUT_DIR: Path = Path("results/figures")
_DPI = 300

# Method-palette for 2D vs 3D vs fusion overlays.
COLOR_2D = "#1f77b4"
COLOR_3D = "#d62728"
COLOR_FUSION = "#2ca02c"
COLOR_TRAIN = "#ff7f0e"

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def setup_style() -> None:
    """Apply the ivafr global style to matplotlib."""
    plt.rcParams.update(
        {
            "figure.dpi": _DPI,
            "savefig.dpi": _DPI,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.constrained_layout.use": True,
        }
    )


def set_output_dir(path: str | Path) -> None:
    """Redirect all :func:`save_fig` output to ``path``."""
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def out_dir() -> Path:
    return _OUTPUT_DIR


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` as PNG + PDF in the output dir; returns the PNG path."""
    png = _OUTPUT_DIR / f"{name}.png"
    pdf = _OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png
