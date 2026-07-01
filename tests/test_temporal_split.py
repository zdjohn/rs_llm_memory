"""Plain-assert unit tests for src/temporal_split.py (synthetic .inter, in-process).

    python tests/test_temporal_split.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import temporal_split as ts  # noqa: E402


def _synthetic_inter(n_users=12, per_user=8) -> str:
    """Write a tiny atomic .inter: each user has `per_user` positives at increasing ts."""
    tmp = Path(tempfile.mkdtemp()) / "syn.inter"
    lines = ["user_id:token\titem_id:token\trating:float\ttimestamp:float"]
    for u in range(1, n_users + 1):
        for j in range(per_user):
            item = (u + j) % 20 + 1
            ts_val = 1000 + u * 100 + j * 10  # strictly increasing within user
            lines.append(f"u{u}\ti{item}\t5.0\t{ts_val}")
    tmp.write_text("\n".join(lines) + "\n")
    return str(tmp)


def test_build_split_sizes_and_pad():
    path = _synthetic_inter(n_users=12, per_user=8)
    sp = ts.build_split(path, n_active=10, seed=2020)
    assert sp.user_id2token[0] == ts.PAD_TOKEN and sp.item_id2token[0] == ts.PAD_TOKEN
    assert sp.n_users == 11  # 10 active + PAD
    assert len(sp.active_users) == 10
    assert len(sp.val_users) == 5 and len(sp.test_users) == 5
    # val/test partition the active set with no overlap.
    assert set(sp.val_users.tolist()) | set(sp.test_users.tolist()) == set(sp.active_users.tolist())
    assert set(sp.val_users.tolist()) & set(sp.test_users.tolist()) == set()


def test_leave_last_one_temporal_invariant():
    path = _synthetic_inter(n_users=12, per_user=8)
    sp = ts.build_split(path, n_active=10, seed=2020)
    # Build per-user max history ts and assert target ts >= it (temporal LLO).
    max_hist = {}
    for (u, _i), t in zip(sp.history_pairs, sp.history_ts):
        max_hist[int(u)] = max(max_hist.get(int(u), -np.inf), float(t))
    # every active user has exactly one target and it is not None.
    assert set(sp.targets) == set(sp.active_users.tolist())
    # each user contributed per_user-1 history rows.
    counts = {}
    for u, _ in sp.history_pairs:
        counts[int(u)] = counts.get(int(u), 0) + 1
    assert all(c == 7 for c in counts.values()), counts


def test_cohort_split_deterministic():
    users = [f"u{i}" for i in range(10)]
    a1 = ts.cohort_split(users, seed=2020)
    a2 = ts.cohort_split(users, seed=2020)
    a3 = ts.cohort_split(users, seed=7)
    assert a1 == a2
    assert a1 != a3  # different seed -> different partition (overwhelmingly likely)


def test_top_active_tiebreak_deterministic():
    import pandas as pd
    # users a,b,c,d all tie at 2; e has 3. Expect e first, then a,b,c,d by user asc.
    rows = [("e", "i", 5.0, 1.0)] * 3
    for u in ["d", "c", "b", "a"]:
        rows += [(u, "i", 5.0, 1.0)] * 2
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    top = ts.top_active_users(df, n_active=3)
    assert top[0] == "e", top
    assert top[1:3] == ["a", "b"], top  # tie-break user asc


def main() -> int:
    test_build_split_sizes_and_pad()
    test_leave_last_one_temporal_invariant()
    test_cohort_split_deterministic()
    test_top_active_tiebreak_deterministic()
    print("test_temporal_split PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
