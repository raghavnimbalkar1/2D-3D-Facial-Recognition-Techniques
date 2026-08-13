"""Matcher contracts: similarity convention (higher = same person)."""

from __future__ import annotations

import numpy as np
import pytest

from ivafr.registry import get_matcher, list_matchers

EXPECTED = {"nn_cosine", "nn_l2", "nn_chi2"}


def test_registry_contents():
    assert EXPECTED <= set(list_matchers())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_identical_inputs_max_similarity(name):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 16)).astype(np.float32)
    m = get_matcher(name)({})
    S = m.score_matrix(X, X)
    assert S.shape == (5, 5)
    # Diagonal (self-matches) must be the max of each row.
    assert np.allclose(S.diagonal(), S.max(axis=1), atol=1e-5)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_similarity_ordering(name):
    """A probe matching its own gallery item scores higher than other items."""
    rng = np.random.default_rng(1)
    gallery = rng.normal(0, 1, (4, 16)).astype(np.float32)
    probe = gallery[2:3].copy()
    m = get_matcher(name)({})
    S = m.score_matrix(probe, gallery)
    assert int(np.argmax(S[0])) == 2


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_scores_for_pairs_matches_full_matrix(name):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (10, 8)).astype(np.float32)
    m = get_matcher(name)({})
    S = m.score_matrix(X, X)
    pairs = [(0, 3), (7, 2), (5, 5)]
    direct = m.scores_for_pairs(X, pairs, max_full=10)
    expected = np.asarray([S[i, j] for i, j in pairs])
    assert np.allclose(direct, expected, atol=1e-5)


def test_cosine_self_similarity_is_one():
    m = get_matcher("nn_cosine")({})
    X = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert m.score_matrix(X, X)[0, 0] == pytest.approx(1.0)


def test_l2_distance_negated():
    m = get_matcher("nn_l2")({})
    X = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    S = m.score_matrix(X[:1], X)
    # closer point -> higher similarity
    assert S[0, 0] > S[0, 1]