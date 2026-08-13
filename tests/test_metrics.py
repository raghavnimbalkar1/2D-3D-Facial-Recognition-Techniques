"""Verification + identification metric correctness on synthetic scores."""

from __future__ import annotations

import numpy as np

from ivafr.evaluation.identification import cmc_curve, evaluate_identification, rank_matrix
from ivafr.evaluation.verification import (
    bootstrap_ci,
    dprime,
    eer,
    evaluate_verification,
    tar_at_far,
)


def _scores(gen_mu: float, imp_mu: float, seed: int = 0, n: int = 1000):
    rng = np.random.default_rng(seed)
    g = rng.normal(gen_mu, 0.1, n)
    i = rng.normal(imp_mu, 0.1, n)
    return g, i


def test_eer_perfectly_separable():
    g, i = _scores(1.0, 0.0)
    ee, thr = eer(g, i)
    assert ee < 0.01
    assert 0.0 < thr < 1.0


def test_eer_identical_distributions():
    g, i = _scores(0.5, 0.5)
    ee, _ = eer(g, i)
    assert 0.45 < ee < 0.55


def test_auc_identical_distributions_about_half():
    ver = evaluate_verification(*_scores(0.5, 0.5))
    assert 0.45 < ver.auc < 0.55
    ver2 = evaluate_verification(*_scores(1.0, 0.0))
    assert ver2.auc > 0.99


def test_tar_at_far():
    g, i = _scores(1.0, 0.0)
    out = tar_at_far(g, i, [1e-1, 1e-2])
    assert out["0.1"] > out["0.01"] > 0.9


def test_dprime():
    g, i = _scores(1.0, 0.0)
    assert dprime(g, i) > 5.0
    g2, i2 = _scores(0.0, 0.0)
    assert dprime(g2, i2) < 1.0


def test_bootstrap_ci_covers_point_estimate():
    g, i = _scores(1.0, 0.0)
    est = eer(g, i)[0]
    lo, hi = bootstrap_ci(g, i, n=200, seed=1)
    assert lo <= est <= hi


def _perfect_rank_matrix():
    # 3 probes, 5 gallery: probe i matches gallery column i.
    rng = np.random.default_rng(0)
    S = rng.uniform(0.2, 0.8, (3, 5))
    S[0, 1] = 0.95
    S[1, 3] = 0.9
    S[2, 0] = 0.85
    return S


def test_cmc_monotone_and_ends_at_one():
    S = _perfect_rank_matrix()
    cmc = cmc_curve(rank_matrix(S), k_max=5)
    assert all(cmc[i] <= cmc[i + 1] for i in range(len(cmc) - 1))
    assert cmc[-1] == 1.0
    assert cmc[0] == 1.0  # all three matching at rank 1


def test_rank1_accuracy_closed_set():
    S = _perfect_rank_matrix()
    probe_labels = np.asarray(["S1", "S2", "S3"])
    gallery_labels = np.asarray(["S1", "S2", "S3", "S4", "S5"])
    res = evaluate_identification(S, probe_labels, gallery_labels)
    assert res.rank1 == 1.0
    assert res.accuracy == 1.0
    assert res.mrr == 1.0
    assert res.f1_macro == pytest_approx(1.0)


def pytest_approx(v):
    import pytest

    return pytest.approx(v)


def test_identification_breaks():
    S = _perfect_rank_matrix()
    S2 = S.copy()
    S2[0, 0] = 0.1  # probe 0 now fails
    probe_labels = np.asarray(["S1", "S2", "S3"])
    gallery_labels = np.asarray(["S1", "S2", "S3", "S4", "S5"])
    res = evaluate_identification(S2, probe_labels, gallery_labels)
    assert res.rank1 == pytest_approx(2 / 3)
    assert res.cmc[1] == 1.0  # rank-2 catches all


def test_verification_contract_arrays():
    g, i = _scores(0.9, 0.1, n=300)
    ver = evaluate_verification(g, i, seed=2)
    assert ver.n_genuine == 300 and ver.n_impostor == 300
    assert ver.roc_fpr.shape == ver.roc_tpr.shape
    assert 0.5 < ver.auc <= 1.0
    assert len(ver.eer_ci95) == 2