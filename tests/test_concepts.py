"""Plain-assert unit tests for src/fuzzy/concepts.py (toy numpy inputs, in-process).

    python tests/test_concepts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import concepts  # noqa: E402


def _toy():
    # 3 users (incl PAD 0), 3 items (incl PAD 0), 2 genres.
    n_users, n_items, n_genres = 3, 3, 2
    train_inter = np.array([[1, 1], [2, 1], [1, 2]])  # item 1 popular (2), item 2 (1)
    item_genres = np.array([[0.0, 0.0],   # PAD
                            [1.0, 0.0],   # item 1: genre 0
                            [0.0, 1.0]])  # item 2: genre 1
    item_decades = np.array([0.0, 1990.0, 2000.0])
    user_pref_decade = np.array([0.0, 1990.0, 2000.0])
    user_genre_profile = np.array([[0.0, 0.0],   # PAD
                                   [1.0, 0.0],   # user 1 likes genre 0
                                   [0.0, 1.0]])  # user 2 likes genre 1
    return (train_inter, item_genres, item_decades, user_pref_decade,
            user_genre_profile, n_users, n_items)


def test_keys_and_niche_relation():
    c = concepts.build_concepts(*_toy())
    assert set(c) == {"mainstream_appeal", "niche_factor", "era_fit", "genre_match"}, set(c)
    assert np.allclose(c["niche_factor"], 1.0 - c["mainstream_appeal"]), c["niche_factor"]


def test_shapes():
    train_inter, ig, idc, upd, ugp, nu, ni = _toy()
    c = concepts.build_concepts(train_inter, ig, idc, upd, ugp, nu, ni)
    assert c["mainstream_appeal"].shape == (ni,), c["mainstream_appeal"].shape
    assert c["niche_factor"].shape == (ni,)
    assert c["era_fit"].shape == (nu, ni), c["era_fit"].shape
    assert c["genre_match"].shape == (nu, ni)


def test_genre_match_cosine():
    c = concepts.build_concepts(*_toy())
    # user 1 (profile [1,0]) vs item 1 (genre [1,0]) -> cosine 1.0
    assert np.isclose(c["genre_match"][1, 1], 1.0), c["genre_match"][1, 1]
    # user 1 (profile [1,0]) vs item 2 (genre [0,1]) -> cosine 0.0
    assert np.isclose(c["genre_match"][1, 2], 0.0), c["genre_match"][1, 2]
    # user 2 (profile [0,1]) vs item 2 (genre [0,1]) -> cosine 1.0
    assert np.isclose(c["genre_match"][2, 2], 1.0), c["genre_match"][2, 2]


def test_pad_never_scored():
    c = concepts.build_concepts(*_toy())
    assert c["mainstream_appeal"][0] == 0.0
    assert np.all(c["era_fit"][0, :] == 0.0) and np.all(c["era_fit"][:, 0] == 0.0)
    assert np.all(c["genre_match"][0, :] == 0.0) and np.all(c["genre_match"][:, 0] == 0.0)


def test_empty_profile_fallback_no_nan():
    # user 2 has an all-zero genre profile -> documented uniform fallback, no NaN.
    train_inter, ig, idc, upd, ugp, nu, ni = _toy()
    ugp = ugp.copy()
    ugp[2] = 0.0
    c = concepts.build_concepts(train_inter, ig, idc, upd, ugp, nu, ni)
    assert np.all(np.isfinite(c["genre_match"])), c["genre_match"]


def test_nan_input_raises():
    train_inter, ig, idc, upd, ugp, nu, ni = _toy()
    # NaN decades are tolerated (treated as "unknown"); NaN in genres must raise.
    ig_bad = ig.copy()
    ig_bad[1, 0] = np.nan
    raised = False
    try:
        concepts.build_concepts(train_inter, ig_bad, idc, upd, ugp, nu, ni)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on NaN-producing genre input"


def test_shape_mismatch_raises():
    train_inter, ig, idc, upd, ugp, nu, ni = _toy()
    raised = False
    try:
        concepts.build_concepts(train_inter, ig, idc, upd, ugp, nu, ni + 1)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on n_items mismatch"


def main() -> int:
    test_keys_and_niche_relation()
    test_shapes()
    test_genre_match_cosine()
    test_pad_never_scored()
    test_empty_profile_fallback_no_nan()
    test_nan_input_raises()
    test_shape_mismatch_raises()
    print("test_concepts PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
