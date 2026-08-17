"""Classification metrics (identification protocol).

All identification metrics derive from a probe x gallery similarity matrix
and the probe's true labels; gallery labels are the candidate identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class IdentificationResult:
    """All identification metrics for one (arm, protocol, seed)."""

    rank1: float
    rank5: float
    rank10: float
    cmc: list[float]
    top5_accuracy: float
    mrr: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    accuracy: float
    confusion: np.ndarray
    labels: list[str]
    decisions: list[int]  # per-probe: argmax gallery index (for McNemar)
    truth: list[str]  # per-probe: true subject id
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    per_condition: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank1": self.rank1,
            "rank5": self.rank5,
            "rank10": self.rank10,
            "cmc": self.cmc,
            "top5_accuracy": self.top5_accuracy,
            "mrr": self.mrr,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "accuracy": self.accuracy,
            "n_probe": len(self.truth),
            "n_gallery": len(self.labels),
            "per_class": self.per_class,
            "per_condition": self.per_condition,
        }


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Rank of each gallery column per probe row (1 = best, higher=better sim).

    Args:
        scores: (Np, Ng) similarity matrix.

    Returns:
        (Np, Ng) int array: 1..Ng.
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(1, order.shape[1] + 1)
    return ranks


def cmc_curve(ranks: np.ndarray, k_max: int = 20) -> list[float]:
    """Cumulative match characteristic: P(rank <= k) for k = 1..k_max."""
    ranks = np.asarray(ranks)
    # Accept either the full rank matrix or the true-gallery-column ranks.
    # The former is useful for callers with a score matrix; CMC is defined
    # against each probe's correct identity, not every gallery column.
    true_ranks = ranks.min(axis=1) if ranks.ndim == 2 else ranks.reshape(-1)
    k_max = min(k_max, int(true_ranks.max(initial=1)))
    values = [float(np.mean(true_ranks <= k)) for k in range(1, k_max + 1)]
    # Keep a stable report shape for small toy galleries while retaining the
    # meaningful ranks in the available gallery range.
    return values + [values[-1]] * max(0, 20 - len(values))


def evaluate_identification(
    scores: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_max: int = 20,
    conditions: list[str] | None = None,
) -> IdentificationResult:
    """Evaluate 1:N identification from a score matrix.

    Args:
        scores: (Np, Ng) similarity matrix (higher = same).
        probe_labels: (Np,) true subject ids of probes.
        gallery_labels: (Ng,) subject ids of gallery templates.
        k_max: CMC truncation.
        conditions: per-probe condition labels (pose/illum) for breakdowns.

    Returns:
        :class:`IdentificationResult`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)
    n_probe = scores.shape[0]
    if n_probe == 0:
        raise ValueError("Empty probe set")

    # Map subject ids to gallery columns.
    uniq = np.unique(gallery_labels)
    cols_by_label: dict[object, list[int]] = {}
    for idx, lab in enumerate(gallery_labels):
        cols_by_label.setdefault(lab, []).append(idx)
    col_of = {lab: i for i, lab in enumerate(uniq)}
    truth_col = np.asarray([col_of.get(lab, -1) for lab in probe_labels])
    if (truth_col < 0).any():
        raise ValueError("Probe subject not present in gallery")

    ranks = rank_matrix(scores)
    decisions = np.argmax(scores, axis=1)
    pred_labels = gallery_labels[decisions]
    pred_col = np.asarray([col_of[lab] for lab in pred_labels])

    # CMC on the gallery columns.
    true_ranks = np.asarray([
        int(ranks[i, cols_by_label[lab]].min()) for i, lab in enumerate(probe_labels)
    ])
    cmc = cmc_curve(true_ranks, k_max)
    cmc_k1 = cmc[0] if cmc else 0.0

    # Closed-set accuracy == rank-1 == 1-NN correct.
    rank1 = float((pred_col == truth_col).mean())
    rank5 = cmc[4] if len(cmc) > 4 else float((true_ranks <= 5).mean())
    rank10 = cmc[9] if len(cmc) > 9 else float((true_ranks <= 10).mean())

    correct = np.asarray([pred_labels[i] == probe_labels[i] for i in range(n_probe)])
    mrr = float(np.mean(1.0 / true_ranks))
    top5 = float((true_ranks <= 5).mean())

    # Per-class report on gallery labels.
    y_true = np.asarray([col_of[lab] for lab in probe_labels])
    report = classification_report(
        y_true, pred_col, labels=np.arange(len(uniq)), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, pred_col, labels=np.arange(len(uniq)))

    per_class = {
        str(uniq[i]): {
            "precision": float(report[str(i)]["precision"]),
            "recall": float(report[str(i)]["recall"]),
            "f1": float(report[str(i)]["f1-score"]),
            "support": int(report[str(i)]["support"]),
        }
        for i in range(len(uniq))
    }

    per_condition: dict[str, dict[str, float]] = {}
    if conditions is not None:
        cond_arr = np.asarray(conditions)
        for cond in np.unique(cond_arr):
            mask = cond_arr == cond
            if mask.sum() == 0:
                continue
            rank1_c = float((correct[mask]).mean())
            per_condition[str(cond)] = {"rank1": rank1_c, "n": int(mask.sum())}

    return IdentificationResult(
        rank1=rank1,
        rank5=rank5,
        rank10=rank10,
        cmc=cmc,
        top5_accuracy=top5,
        mrr=mrr,
        precision_macro=float(report["macro avg"]["precision"]),
        precision_weighted=float(report["weighted avg"]["precision"]),
        recall_macro=float(report["macro avg"]["recall"]),
        recall_weighted=float(report["weighted avg"]["recall"]),
        f1_macro=float(report["macro avg"]["f1-score"]),
        f1_weighted=float(report["weighted avg"]["f1-score"]),
        accuracy=rank1,
        confusion=cm,
        labels=[str(u) for u in uniq],
        decisions=decisions.tolist(),
        truth=[str(lab) for lab in probe_labels],
        per_class=per_class,
        per_condition=per_condition,
    )
