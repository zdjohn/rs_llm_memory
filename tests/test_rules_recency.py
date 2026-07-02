"""Plain-assert unit tests for src/fuzzy/rules_recency.py (toy concepts, in-process).

    python tests/test_rules_recency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import rules_recency as rr  # noqa: E402

BP = {c: (0.25, 0.5, 0.75) for c in rr.CONCEPTS_RECENCY}


def test_rule_base_validates_and_outputs_aligned():
    assert rr.N_RULES == len(rr.RULE_BASE_RECENCY)
    assert rr.OUTPUTS == ("suppress", "neutral", "surface")
    # fire_recency returns 3 output terms.
    memb = {c: np.ones((2, 3)) / 3.0 for c in rr.CONCEPTS_RECENCY}
    firing = rr.fire_recency(memb)
    assert firing.shape == (2, 3)


def test_r2_vs_r3_stable_survives_volatile_sinks():
    # One user, two items. Both: relevance HIGH, recency LOW. Item0 volatility LOW (stable) ->
    # surface (R3); Item1 volatility HIGH (volatile) -> suppress (R2).
    con = {
        "relevance": np.array([[0.9, 0.9]]),
        "recency": np.array([[0.1, 0.1]]),
        "volatility": np.array([0.1, 0.9]),
        "frequency": np.array([0.1, 0.1]),
    }
    M = rr.score_matrix_recency(con, BP, weights=None)
    assert M.shape == (1, 2)
    assert M[0, 0] > 0.8, ("stable stale relevant should surface", M[0, 0])
    assert M[0, 1] < 0.2, ("volatile stale relevant should suppress", M[0, 1])


def test_relevance_low_suppresses():
    con = {
        "relevance": np.array([[0.05]]),
        "recency": np.array([[0.9]]),
        "volatility": np.array([0.5]),
        "frequency": np.array([0.9]),
    }
    M = rr.score_matrix_recency(con, BP, weights=None)
    assert M[0, 0] < 0.2, M[0, 0]  # R5


def test_weights_change_scores():
    # Split memberships (values between breakpoints) so BOTH surface and neutral fire; only then
    # can re-weighting a single output term shift the centroid defuzz (track-a-findings.md §5:
    # scaling one output term's rules is inert when it is the ONLY term firing).
    con = {
        "relevance": np.array([[0.6]]),
        "recency": np.array([[0.6]]),
        "volatility": np.array([0.5]),
        "frequency": np.array([0.6]),
    }
    base = rr.score_matrix_recency(con, BP, weights=None)
    w = np.ones(rr.N_RULES)
    w[0] = 5.0  # up-weight R1 (surface); neutral fillers also fire -> defuzz moves
    bumped = rr.score_matrix_recency(con, BP, weights=w)
    assert not np.allclose(base, bumped), (base, bumped)


def test_range_within_centroids():
    rng = np.random.default_rng(0)
    con = {
        "relevance": rng.random((5, 7)),
        "recency": rng.random((5, 7)),
        "volatility": rng.random(7),
        "frequency": rng.random(7),
    }
    M = rr.score_matrix_recency(con, BP, weights=None)
    assert M.min() >= 0.1 - 1e-9 and M.max() <= 0.9 + 1e-9, (M.min(), M.max())


def main() -> int:
    test_rule_base_validates_and_outputs_aligned()
    test_r2_vs_r3_stable_survives_volatile_sinks()
    test_relevance_low_suppresses()
    test_weights_change_scores()
    test_range_within_centroids()
    print("test_rules_recency PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
