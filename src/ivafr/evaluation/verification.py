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


def eer(genuine: np.ndarray, impostor: np.ndarray, weights=None) -> tuple[float, float]:
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
    if not np.isfinite(s).all():
        raise ValueError("Verification scores must be finite")
    fpr, tpr, thr = roc_curve(y, s, sample_weight=weights)
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


def tar_at_far(
    genuine: np.ndarray, impostor: np.ndarray, far_levels: list[float]
) -> dict[str, float]:
    """TAR (recall) at fixed FAR levels, e.g. {1e-1, 1e-2, 1e-3}."""
    return {
        key: point["tar"] for key, point in operating_points(genuine, impostor, far_levels).items()
    }


def operating_points(genuine, impostor, far_levels) -> dict:
    """Descriptive evaluation-set operating points, not calibrated deployment thresholds.

    Accept similarity >= threshold. A tied block is accepted only when its
    complete false-accept count fits the requested empirical FAR budget.
    """
    g, i = np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)
    if not len(g) or not len(i) or not np.isfinite(np.r_[g, i]).all():
        raise ValueError("Operating points require finite, nonempty scores")
    values = np.unique(np.r_[g, i])
    accepted = len(i) - np.searchsorted(np.sort(i), values, side="left")
    out = {}
    for far in far_levels:
        if not 0 <= far <= 1:
            raise ValueError("FAR must be in [0,1]")
        allowed = int(np.floor(float(far) * len(i)))
        feasible = np.flatnonzero(accepted <= allowed)
        threshold = (
            float(values[feasible[0]]) if len(feasible) else float(np.nextafter(values[-1], np.inf))
        )
        out[f"{far:g}"] = {
            "tar": float(np.mean(g >= threshold)),
            "achieved_far": float(np.mean(i >= threshold)),
            "threshold": threshold,
            "far_resolution": 1.0 / len(i),
            "below_resolution": 0 < far < 1.0 / len(i),
            "threshold_source": "evaluation_roc_not_deployment_calibration",
        }
    return out


def subject_bootstrap_ci(genuine, impostor, genuine_subjects, impostor_subjects, n=1000, seed=0):
    """Identity-cluster bootstrap with shared subject weights for both pair sets."""
    gp = np.asarray(genuine_subjects)
    ip = np.asarray(impostor_subjects)
    subjects = np.unique(np.r_[gp.ravel(), ip.ravel()])
    if len(subjects) < 2 or n < 2 or len(gp) != len(genuine) or len(ip) != len(impostor):
        raise ValueError("Cluster CI requires matching pair identities, >=2 subjects and resamples")
    index = {s: k for k, s in enumerate(subjects)}
    gi = np.array([index[a] for a, _ in gp])
    ia = np.array([index[a] for a, _ in ip])
    ib = np.array([index[b] for _, b in ip])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n * 20):
        weights = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)))
        gw, iw = weights[gi], weights[ia] * weights[ib]
        if not gw.any() or not iw.any():
            continue
        values.append(eer(genuine, impostor, np.r_[gw, iw])[0])
        if len(values) == n:
            break
    if len(values) != n:
        raise ValueError("Insufficient valid identity bootstrap replicates")
    return np.percentile(values, [2.5, 97.5]).astype(float).tolist()


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
    operating_points: dict = field(default_factory=dict)
    ci_method: str = "pair_bootstrap"

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
            "operating_points": self.operating_points,
            "ci_method": self.ci_method,
        }


def evaluate_verification(
    genuine: np.ndarray,
    impostor: np.ndarray,
    seed: int = 0,
    n_boot: int = 1000,
    genuine_subjects=None,
    impostor_subjects=None,
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
    clustered = genuine_subjects is not None and impostor_subjects is not None
    ci = (
        subject_bootstrap_ci(g, i, genuine_subjects, impostor_subjects, n_boot, seed)
        if clustered
        else bootstrap_ci(g, i, "eer", n=n_boot, seed=seed)
    )
    return VerificationResult(
        eer=ee,
        eer_ci95=ci,
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
        operating_points=operating_points(g, i, [1e-1, 1e-2, 1e-3]),
        ci_method="subject_cluster_bootstrap" if clustered else "pair_bootstrap",
    )
