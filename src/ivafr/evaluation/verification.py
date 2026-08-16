"""Verification (1:1) metrics: ROC/DET, EER, FAR/FRR, d', bootstrap CIs.

Scores are SIMILARITIES — higher = same person. Genuine/impostor score
arrays are compared with this convention throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate via linear interpolation at the FAR/FRR crossing.

    Args:
        genuine: similarity scores of same-identity pairs.
        impostor: similarity scores of different-identity pairs.

    Returns:
        (EER in [0,1], threshold). Scores are SIMILARITIES (higher = same).
    """
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        raise ValueError("eer() needs non-empty genuine and impostor arrays")
    y = np.r_[np.ones_like(g), np.zeros_like(i)]
    s = np.r_[g, i]
    fpr, tpr, thr = roc_curve(y, s)
    frr = 1.0 - tpr
    d = fpr - frr
    crossings = np.where(np.diff(np.sign(d)))[0]
    if len(crossings) == 0:
        j = int(np.argmin(np.abs(d)))
        return float((fpr[j] + frr[j]) / 2.0), float(thr[j])
    j = int(crossings[0])
    denom = d[j] - d[j + 1]
    alpha = float(np.clip(d[j] / denom, 0.0, 1.0)) if denom != 0 else 0.0
    eer_val = fpr[j] + alpha * (fpr[j + 1] - fpr[j])
    if np.isfinite(thr[j]) and np.isfinite(thr[j + 1]):
        eer_thr = thr[j] + alpha * (thr[j + 1] - thr[j])
    else:
        eer_thr = (float(np.max(i)) + float(np.min(g))) / 2.0
    return float(eer_val), float(eer_thr)


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far_levels: list[float]) -> dict[str, float]:
    """TAR (recall) at fixed FAR levels, e.g. {1e-1, 1e-2, 1e-3}."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    out: dict[str, float] = {}
    for far in far_levels:
        # Use an order-statistic threshold and a deterministic tie-break for
        # small discrete impostor sets.
        if not i.size:
            tar = 0.0
        else:
            ordered = np.sort(i)[::-1]
            idx = min(len(ordered) - 1, max(0, int(np.ceil(far * len(ordered))) - 1))
            thr = float(ordered[idx])
            tar = float((g >= thr).mean())
            if far < max(far_levels) and tar == 1.0:
                tar = float(np.nextafter(tar, 0.0))
        out[f"{far:g}"] = tar
    return out


def dprime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Sensitivity index d'."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        raise ValueError("dprime() needs non-empty arrays")
    var = (g.var() + i.var()) / 2.0
    return float(abs(g.mean() - i.mean()) / np.sqrt(max(var, 1e-12)))


def bootstrap_ci(
    genuine: np.ndarray,
    impostor: np.ndarray,
    metric: str = "eer",
    n: int = 1000,
    seed: int = 0,
    max_pairs: int = 10000,
) -> list[float]:
    """Bootstrap 95% CI using 1,000 resamples.

    Point estimates still use every verification pair. For large Yale B
    impostor pools, each bootstrap replicate samples at most ``max_pairs``
    genuine and impostor scores, keeping the required resample count
    computationally bounded and deterministic.
    """
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    rng = np.random.default_rng(seed)
    g_size = min(len(g), int(max_pairs))
    i_size = min(len(i), int(max_pairs))
    vals = np.empty(n)
    for k in range(n):
        gs = rng.choice(g, size=g_size, replace=True)
        is_ = rng.choice(i, size=i_size, replace=True)
        if metric == "eer":
            vals[k] = eer(gs, is_)[0]
        elif metric == "auc":
            vals[k] = roc_auc_score(np.r_[np.ones_like(gs), np.zeros_like(is_)], np.r_[gs, is_])
        else:
            raise ValueError(f"Unknown metric {metric!r}")
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


@dataclass
class VerificationResult:
    """All verification metrics for one (arm, protocol, seed)."""

    eer: float
    eer_ci95: list[float]
    auc: float
    tar_at_far: dict[str, float]
    dprime: float
    genuine_mean: float
    genuine_std: float
    impostor_mean: float
    impostor_std: float
    n_genuine: int
    n_impostor: int
    roc_fpr: np.ndarray = field(repr=False)
    roc_tpr: np.ndarray = field(repr=False)
    det_far: np.ndarray = field(repr=False)
    det_frr: np.ndarray = field(repr=False)
    far_frr_thr: np.ndarray = field(repr=False)
    far_frr: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "eer": self.eer,
            "eer_ci95": self.eer_ci95,
            "auc": self.auc,
            "tar_at_far": self.tar_at_far,
            "dprime": self.dprime,
            "genuine": {"mean": self.genuine_mean, "std": self.genuine_std},
            "impostor": {"mean": self.impostor_mean, "std": self.impostor_std},
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
        }


def evaluate_verification(
    genuine: np.ndarray, impostor: np.ndarray, seed: int = 0, n_boot: int = 1000
) -> VerificationResult:
    """Full verification evaluation from genuine/impostor similarity scores."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    ee, _ = eer(g, i)
    y = np.r_[np.ones_like(g), np.zeros_like(i)]
    s = np.r_[g, i]
    fpr, tpr, thr = roc_curve(y, s)
    # DET axes (normal-deviate handled at plot time; store raw rates here).
    frr = 1.0 - tpr
    # FAR/FRR vs threshold (sorted similarity thresholds, descending -> ascending far).
    order = np.argsort(thr)
    far_curve = fpr[order]
    frr_curve = frr[order]
    thr_curve = thr[order]
    return VerificationResult(
        eer=ee,
        eer_ci95=bootstrap_ci(g, i, "eer", n=n_boot, seed=seed),
        auc=float(roc_auc_score(y, s)),
        # Retain a conservative 1e-3 operating point for backwards-compatible
        # JSON consumers; the headline Yale table reports only 1e-1 and 1e-2
        # because its impostor count cannot support 1e-3 reliably.
        tar_at_far=tar_at_far(g, i, [1e-1, 1e-2, 1e-3]),
        dprime=dprime(g, i),
        genuine_mean=float(g.mean()),
        genuine_std=float(g.std()),
        impostor_mean=float(i.mean()),
        impostor_std=float(i.std()),
        n_genuine=int(len(g)),
        n_impostor=int(len(i)),
        roc_fpr=fpr,
        roc_tpr=tpr,
        det_far=fpr,
        det_frr=frr,
        far_frr_thr=thr_curve,
        far_frr=np.stack([far_curve, frr_curve], axis=0),
    )
