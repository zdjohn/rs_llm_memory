"""Plain-assert unit tests for src/ppr.py (tiny universe, in-process).

    python tests/test_ppr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ppr  # noqa: E402


# 2 users (+PAD => 3 slots), 3 items (+PAD => 4 slots).
# u1 -> {i1, i2}; u2 -> {i2, i3}.
HP = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.int64)
NU, NI = 3, 4
TU = np.array([1, 2])


def test_engines_agree():
    if not ppr._HAVE_NX:
        print("  [skip] networkx not installed; power-only")
        return
    a = ppr.ppr_score_matrix(HP, NU, NI, TU, engine="networkx")
    b = ppr.ppr_score_matrix(HP, NU, NI, TU, engine="power")
    assert np.abs(a - b).max() < 1e-4, np.abs(a - b).max()


def test_pad_masked_and_shapes():
    m = ppr.ppr_score_matrix(HP, NU, NI, TU, engine="power")
    assert m.shape == (NU, NI)
    assert m[:, 0].sum() == 0.0  # PAD col masked
    assert m[0].sum() == 0.0     # PAD user row never targeted
    assert (m[TU].sum(axis=1) > 0).all()  # targeted users get mass


def test_history_items_personalization_prefers_neighbors():
    # u1 personalized on {i1,i2}. i2 is shared with u2, so diffusion should give i3 (u2's other
    # item) positive mass -> a nonzero score even though u1 never touched i3.
    m = ppr.ppr_score_matrix(HP, NU, NI, TU, engine="power")
    assert m[1, 3] > 0.0, m[1]


def test_user_node_mode_runs():
    m = ppr.ppr_score_matrix(HP, NU, NI, TU, engine="power", personalization_mode="user_node")
    assert m.shape == (NU, NI) and (m[TU].sum(axis=1) > 0).all()


def main() -> int:
    test_engines_agree()
    test_pad_masked_and_shapes()
    test_history_items_personalization_prefers_neighbors()
    test_user_node_mode_runs()
    print("test_ppr PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
