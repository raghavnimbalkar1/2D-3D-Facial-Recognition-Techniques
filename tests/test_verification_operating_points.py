import numpy as np
import pytest

from ivafr.evaluation.verification import operating_points, subject_bootstrap_ci


def test_ties_never_exceed_false_accept_budget():
    g, i = np.array([0.8, 0.7, 0.5]), np.array([0.8, 0.8, 0.6, 0.5])
    points = operating_points(g, i, [0, 0.001, 0.25, 0.5, 1])
    for far, point in points.items():
        assert point["achieved_far"] <= float(far)
        assert np.mean(i >= point["threshold"]) == point["achieved_far"]
        assert np.mean(g >= point["threshold"]) == point["tar"]
    assert points["0.25"]["achieved_far"] == 0
    assert points["0.001"]["below_resolution"]
    assert points["0.5"]["tar"] == pytest.approx(2 / 3)


def test_perfect_separation_preserves_all_genuine_accepts():
    point = operating_points([0.61, 0.8], [0.5, 0.4], [0])["0"]
    assert point["tar"] == 1
    assert point["achieved_far"] == 0


def test_bootstrap_uses_shared_subject_clusters_and_is_deterministic():
    g, i = np.array([0.9, 0.6, 0.4]), np.array([0.2, 0.5, 0.7])
    gp = [("a", "a"), ("b", "b"), ("c", "c")]
    ip = [("a", "b"), ("a", "c"), ("b", "c")]
    interval = subject_bootstrap_ci(g, i, gp, ip, n=40, seed=7)
    assert interval == subject_bootstrap_ci(g, i, gp, ip, n=40, seed=7)
    assert 0 <= interval[0] < interval[1] <= 1
    with pytest.raises(ValueError):
        subject_bootstrap_ci(g, i, gp[:-1], ip, n=40)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(ValueError):
        operating_points([bad], [0], [0.1])
