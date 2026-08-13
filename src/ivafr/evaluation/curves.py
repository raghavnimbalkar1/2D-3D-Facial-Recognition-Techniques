"""Curve data helpers: CMC, PR curves."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from ivafr.evaluation.identification import rank_matrix


def pr_curve_from_scores(
    scores: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    subject_of_gallery: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Micro-averaged precision-recall from a score matrix.

    Each (probe, gallery) pair is a binary decision: same identity or not.
    The PR curve is over all pairs sorted by similarity.

    Returns:
        (precision, recall, average_precision) in sklearn order.
    """
    gcol = np.asarray([subject_of_gallery.get(g, "") for g in gallery_labels])
    y = (probe_labels[:, None] == gcol[None, :]).astype(int).ravel()
    s = np.asarray(scores).ravel()
    prec, rec, _ = precision_recall_curve(y, s)
    ap = float(average_precision_score(y, s))
    return prec, rec, ap
