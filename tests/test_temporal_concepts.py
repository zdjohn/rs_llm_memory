"""Plain-assert unit tests for src/fuzzy/temporal_concepts.py (hand-built history).

    python tests/test_temporal_concepts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import temporal_concepts as tc  # noqa: E402

DAY = tc.SECONDS_PER_DAY


def _scenario():
    """u1=1, u2=2 (+PAD 0); items i1=1,i2=2,i3=3 (+PAD 0).

    u1 history: i1 at t=0 (old), i2 at t=1e7s (recent).  u2 history: i1, i3 (links i1~i3).
    So for u1, i3 is similar only to the OLD item i1 -> recency of i3 must fall as lam grows.
    """
    hp = np.array([[1, 1], [1, 2], [2, 1], [2, 3]], dtype=np.int64)
    ts = np.array([0.0, 1e7, 50.0, 60.0], dtype=np.float64)
    active = np.array([1, 2])
    return hp, ts, active, 3, 4  # n_users(+PAD)=3, n_items(+PAD)=4


def test_shapes_ranges_and_pad():
    hp, ts, active, nu, ni = _scenario()
    con = tc.build_temporal_concepts(hp, ts, active, nu, ni, lam=0.0)
    assert con["relevance"].shape == (nu, ni)
    assert con["recency"].shape == (nu, ni)
    assert con["volatility"].shape == (ni,)
    assert con["frequency"].shape == (ni,)
    for name, arr in con.items():
        assert np.all(arr >= -1e-12) and np.all(arr <= 1 + 1e-9), (name, arr.min(), arr.max())
        assert np.all(np.isfinite(arr))
    # PAD zeroed.
    assert con["relevance"][0].sum() == 0 and con["relevance"][:, 0].sum() == 0
    assert con["recency"][0].sum() == 0 and con["recency"][:, 0].sum() == 0
    assert con["volatility"][0] == 0 and con["frequency"][0] == 0


def test_recency_zero_lambda_is_max_similarity():
    hp, ts, active, nu, ni = _scenario()
    con = tc.build_temporal_concepts(hp, ts, active, nu, ni, lam=0.0)
    # lam=0 -> decay==1 -> recency = max_j sim >= mean_j sim = relevance (elementwise).
    assert np.all(con["recency"] + 1e-12 >= con["relevance"]), \
        (con["recency"] - con["relevance"]).min()


def test_recency_decays_with_lambda():
    hp, ts, active, nu, ni = _scenario()
    r0 = tc.build_temporal_concepts(hp, ts, active, nu, ni, lam=0.0)["recency"][1, 3]
    r_hi = tc.build_temporal_concepts(hp, ts, active, nu, ni, lam=0.05)["recency"][1, 3]
    assert r0 > 0.0, r0
    assert r_hi < r0, (r0, r_hi)  # i3 is similar only to the OLD item -> larger lam suppresses it


def test_volatility_burst_gt_evergreen():
    # i1 bursty (3 interactions within a day); i2 evergreen (spread over ~100 days).
    hp = np.array([[1, 1], [2, 1], [3, 1], [1, 2], [2, 2], [3, 2]], dtype=np.int64)
    ts = np.array([0.0, 0.5 * DAY, 1.0 * DAY,  0.0, 50 * DAY, 100 * DAY], dtype=np.float64)
    active = np.array([1, 2, 3])
    con = tc.build_temporal_concepts(hp, ts, active, 4, 3, lam=0.0)
    assert con["volatility"][1] > con["volatility"][2], con["volatility"]


def main() -> int:
    test_shapes_ranges_and_pad()
    test_recency_zero_lambda_is_max_similarity()
    test_recency_decays_with_lambda()
    test_volatility_burst_gt_evergreen()
    print("test_temporal_concepts PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
