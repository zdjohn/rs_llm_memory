"""Unit tests for the expected-rank / Hit@k tie math in src/run_recency.py, plus a guarded
end-to-end on ml100k (set RUN_INTEGRATION=1).

    python tests/test_run_recency.py
    RUN_INTEGRATION=1 python tests/test_run_recency.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_recency as rrun  # noqa: E402


def test_expected_rank_no_ties():
    # distinct scores; target at index 2 (value 0.5), one item strictly above.
    row = np.array([0.9, 0.6, 0.5, 0.3, 0.1])
    exp_rank, hits = rrun._expected_rank_and_hits(row, target=2, ks=(1, 3, 10))
    assert exp_rank == 3.0, exp_rank            # greater=2 -> 2 + (1+1)/2 = 3
    assert hits[1] == 0.0 and hits[3] == 1.0 and hits[10] == 1.0, hits


def test_expected_rank_with_ties():
    # target tied with two others; one strictly above.
    row = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    exp_rank, hits = rrun._expected_rank_and_hits(row, target=2, ks=(1, 3, 10))
    # greater=1, equal=3 -> exp_rank = 1 + (3+1)/2 = 3.0
    assert exp_rank == 3.0, exp_rank
    assert abs(hits[1] - 0.0) < 1e-9              # clip((1-1)/3)
    assert abs(hits[3] - (2 / 3)) < 1e-9          # clip((3-1)/3)
    assert abs(hits[10] - 1.0) < 1e-9


def test_masked_row_excludes_history_but_keeps_target():
    row = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    out = rrun._masked_row(row, u=1, target=2, history={2, 3})  # target 2 stays, 3 masked
    assert out[0] == -np.inf                       # PAD always masked
    assert np.isfinite(out[2])                      # target un-masked even though in history
    assert out[3] == -np.inf                        # non-target history masked


def test_integration_ml100k():
    if os.environ.get("RUN_INTEGRATION") != "1":
        print("  [skip] set RUN_INTEGRATION=1 for the ml100k end-to-end")
        return
    inter = REPO_ROOT / "data" / "ml100k" / "ml100k.inter"
    rows = rrun.run(str(inter), n_active=500, seed=2020, ppr_engine="power", fit_a1=False)
    test_rows = {r["model"]: r for r in rows if r["cohort"] == "test"}
    assert set(test_rows) == {"PPR", "FIS-recency-A0", "FIS-recency-A1"}
    for r in test_rows.values():
        assert 0.0 <= r["mrr"] <= 1.0
        assert r["n_users"] == 250


def main() -> int:
    test_expected_rank_no_ties()
    test_expected_rank_with_ties()
    test_masked_row_excludes_history_but_keeps_target()
    test_integration_ml100k()
    print("test_run_recency PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
