"""Plain-assert unit tests for src/fuzzy/rules.py + src/fuzzy/fis_score.py (toy inputs).

    python tests/test_fis_score.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import fis_score, rules  # noqa: E402


def _onehot_memberships(term_per_concept: dict[str, str]):
    """Build crisp memberships activating exactly one term per concept (scalar cell)."""
    m = {}
    for c in rules.CONCEPTS:
        vec = np.zeros(3, dtype=float)
        vec[rules.TERMS.index(term_per_concept[c])] = 1.0
        m[c] = vec
    return m


def test_single_rule_activation_dominates():
    # Fully activate the first recommend rule: genre_match=high, era_fit=high.
    # Other concepts set to medium so no avoid/neutral rule fully fires.
    m = _onehot_memberships({
        "genre_match": "high", "era_fit": "high",
        "mainstream_appeal": "medium", "niche_factor": "medium",
    })
    firing = rules.fire(m)
    rec = rules.OUTPUTS.index("recommend")
    assert firing.shape == (3,), firing.shape
    assert firing[rec] == 1.0, firing
    assert np.argmax(firing) == rec, firing


def test_fire_none_equals_ones():
    rng = np.random.default_rng(0)
    m = {c: rng.random((4, 3)) for c in rules.CONCEPTS}
    a = rules.fire(m)
    b = rules.fire(m, np.ones(rules.N_RULES))
    assert np.allclose(a, b), "fire(m) must equal fire(m, ones)"


def test_wrong_weight_length_raises():
    m = {c: np.ones((2, 3)) for c in rules.CONCEPTS}
    raised = False
    try:
        rules.fire(m, np.ones(rules.N_RULES + 1))
    except ValueError:
        raised = True
    assert raised, "expected ValueError on weight-length mismatch"


def test_defuzz_centroids():
    # one-hot recommend -> 0.9, avoid -> 0.1, uniform -> 0.5
    assert np.isclose(fis_score.defuzz(np.array([0.0, 0.0, 1.0])), 0.9)
    assert np.isclose(fis_score.defuzz(np.array([1.0, 0.0, 0.0])), 0.1)
    assert np.isclose(fis_score.defuzz(np.array([1.0, 1.0, 1.0])), 0.5)


def test_defuzz_zero_firing_no_div():
    out = fis_score.defuzz(np.array([0.0, 0.0, 0.0]))
    assert np.isfinite(out) and np.isclose(out, 0.5), out


def _toy_concepts():
    n_users, n_items = 4, 5
    rng = np.random.default_rng(1)
    concepts = {
        "mainstream_appeal": rng.random(n_items),
        "niche_factor": None,  # filled below
        "era_fit": rng.random((n_users, n_items)),
        "genre_match": rng.random((n_users, n_items)),
    }
    concepts["niche_factor"] = 1.0 - concepts["mainstream_appeal"]
    breakpoints = {name: (0.25, 0.5, 0.75) for name in rules.CONCEPTS}
    return concepts, breakpoints, n_users, n_items


def test_score_matrix_shape_range_and_broadcast():
    concepts, bp, nu, ni = _toy_concepts()
    s = fis_score.score_matrix(concepts, bp)
    assert s.shape == (nu, ni), s.shape
    assert np.all(s >= 0.0) and np.all(s <= 1.0), (s.min(), s.max())


def test_score_matrix_a0_equals_ones():
    concepts, bp, _, _ = _toy_concepts()
    a0 = fis_score.score_matrix(concepts, bp, weights=None)
    ones = fis_score.score_matrix(concepts, bp, weights=np.ones(rules.N_RULES))
    assert np.allclose(a0, ones), "A0 (weights=None) must equal ones-weights"


def test_score_matrix_nan_raises():
    concepts, bp, _, _ = _toy_concepts()
    concepts = dict(concepts)
    concepts["era_fit"] = concepts["era_fit"].copy()
    concepts["era_fit"][0, 0] = np.nan
    raised = False
    try:
        fis_score.score_matrix(concepts, bp)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on NaN in score_matrix"


def main() -> int:
    test_single_rule_activation_dominates()
    test_fire_none_equals_ones()
    test_wrong_weight_length_raises()
    test_defuzz_centroids()
    test_defuzz_zero_firing_no_div()
    test_score_matrix_shape_range_and_broadcast()
    test_score_matrix_a0_equals_ones()
    test_score_matrix_nan_raises()
    print("test_fis_score PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
