"""Plain-assert unit tests for src/fuzzy/fit_weights.py (toy numpy inputs, in-process).

    python tests/test_fit_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import fis_score, fit_weights, rules  # noqa: E402


def _toy_problem():
    """A toy train set where genre/era fit should separate positives from negatives."""
    n_users, n_items = 6, 8
    rng = np.random.default_rng(7)
    mainstream = rng.random(n_items)
    concepts = {
        "mainstream_appeal": mainstream,
        "niche_factor": 1.0 - mainstream,
        "era_fit": rng.random((n_users, n_items)),
        "genre_match": rng.random((n_users, n_items)),
    }
    breakpoints = {name: (0.25, 0.5, 0.75) for name in rules.CONCEPTS}
    # Positive = the item with the highest genre_match for each user; negative = lowest.
    gm = concepts["genre_match"]
    pairs = []
    for u in range(1, n_users):  # skip PAD user 0
        pos = int(np.argmax(gm[u, 1:]) + 1)
        neg = int(np.argmin(gm[u, 1:]) + 1)
        if pos != neg:
            pairs.append((u, pos, neg))
    return concepts, breakpoints, np.array(pairs)


def test_fit_returns_finite_vector():
    concepts, bp, targets = _toy_problem()
    w = fit_weights.fit_weights(concepts, bp, targets, max_iter=100)
    assert isinstance(w, np.ndarray)
    assert w.shape == (rules.N_RULES,), w.shape
    assert np.all(np.isfinite(w)), w


def test_fitted_loss_not_worse_than_a0():
    concepts, bp, targets = _toy_problem()
    w = fit_weights.fit_weights(concepts, bp, targets, max_iter=100)

    def loss(weights):
        m = fis_score.score_matrix(concepts, bp, weights=weights)
        return fit_weights._pairwise_bpr_loss(m, targets)

    a0_loss = loss(np.ones(rules.N_RULES))
    a1_loss = loss(w)
    # Fitted weights should not be worse than A0 (allow a tiny numerical slack).
    assert a1_loss <= a0_loss + 1e-6, (a1_loss, a0_loss)


def test_non_convergence_raises():
    concepts, bp, targets = _toy_problem()
    raised = False
    try:
        # tol effectively unreachable + 1 iter -> cannot converge.
        fit_weights.fit_weights(concepts, bp, targets, max_iter=1, tol=-1.0)
    except RuntimeError:
        raised = True
    assert raised, "expected RuntimeError on non-convergence past max_iter"


def main() -> int:
    test_fit_returns_finite_vector()
    test_fitted_loss_not_worse_than_a0()
    test_non_convergence_raises()
    print("test_fit_weights PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
