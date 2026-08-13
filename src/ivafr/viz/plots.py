"""Standard report plots: CMC, ROC/DET, FAR-FRR, score histograms, PR."""

from __future__ import annotations

import math

import numpy as np

from ivafr.viz.style import save_fig, setup_style


def plot_cmc(arms: dict[str, list[float]], title: str = "Cumulative Match Characteristic", name: str = "fig_cmc_all") -> None:
    """CMC curves for several arms (cmc lists of equal length)."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, cmc in arms.items():
        ax.plot(np.arange(1, len(cmc) + 1), cmc, marker="o", ms=3, label=label)
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Identification rate")
    ax.set_title(title)
    ax.set_xlim(1, max(len(v) for v in arms.values()) if arms else 20)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    save_fig(fig, name)


def plot_roc(arms: dict[str, tuple[np.ndarray, np.ndarray, float]], name: str = "fig_roc_all") -> None:
    """ROC curves: label -> (fpr, tpr, auc)."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (fpr, tpr, auc) in arms.items():
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FAR (false accept rate)")
    ax.set_ylabel("TAR (true accept rate)")
    ax.set_title("ROC curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save_fig(fig, name)


def plot_det(arms: dict[str, tuple[np.ndarray, np.ndarray]], name: str = "fig_det_all") -> None:
    """DET curves on normal-deviate axes: label -> (far, frr)."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, (far, frr) in arms.items():
        with np.errstate(divide="ignore"):
            x = _norm_ppf(np.clip(far, 1e-5, 1 - 1e-5))
            y = _norm_ppf(np.clip(frr, 1e-5, 1 - 1e-5))
        ax.plot(x, y, label=label)
    ax.set_xlabel("FAR (%, normal deviate)")
    ax.set_ylabel("FRR (%, normal deviate)")
    ax.set_title("DET curves")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ticks = [0.1, 1, 5, 10, 20, 40, 60, 80, 95]
    ax.set_xticks([_norm_ppf(t / 100) for t in ticks])
    ax.set_yticks([_norm_ppf(t / 100) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    save_fig(fig, name)


def _norm_ppf(p: float | np.ndarray) -> float | np.ndarray:
    from scipy.stats import norm

    return norm.ppf(p)


def plot_far_frr(arms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], name: str = "fig_far_frr") -> None:
    """FAR and FRR vs threshold: label -> (thresholds, far, frr)."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (thr, far, frr) in arms.items():
        ax.plot(thr, far, linestyle="--", label=f"{label} FAR")
        ax.plot(thr, frr, linestyle="-", label=f"{label} FRR")
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Rate")
    ax.set_title("FAR / FRR vs threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", ncol=2)
    save_fig(fig, name)


def plot_score_hists(
    arms: dict[str, tuple[np.ndarray, np.ndarray]], name: str = "fig_score_hists"
) -> None:
    """Genuine vs impostor score histograms: label -> (genuine, impostor)."""
    setup_style()
    import matplotlib.pyplot as plt

    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * max(n, 1), 4), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (label, (g, i)) in zip(axes, arms.items()):
        ax.hist(g, bins=40, alpha=0.6, color="#1f77b4", label="genuine")
        ax.hist(i, bins=40, alpha=0.6, color="#d62728", label="impostor")
        ax.set_title(label)
        ax.set_xlabel("similarity")
        ax.legend()
    save_fig(fig, name)


def plot_pr(arms: dict[str, tuple[np.ndarray, np.ndarray, float]], name: str = "fig_pr_curves") -> None:
    """Precision-recall curves: label -> (precision, recall, ap)."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (prec, rec, ap) in arms.items():
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    save_fig(fig, name)


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Indices of the Pareto frontier (maximize x=accuracy, minimize y=time).

    Args:
        points: (N, 2) array of (accuracy, time_ms).

    Returns:
        Indices sorted by accuracy.
    """
    pts = np.asarray(points, dtype=np.float64)
    order = np.argsort(-pts[:, 0])
    frontier = []
    best_y = math.inf
    for idx in order:
        if pts[idx, 1] < best_y:
            frontier.append(int(idx))
            best_y = pts[idx, 1]
    return np.asarray(sorted(frontier))
