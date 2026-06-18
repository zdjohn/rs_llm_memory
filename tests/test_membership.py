"""Plain-assert unit tests for src/fuzzy/membership.py (toy numpy inputs, in-process).

    python tests/test_membership.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import membership  # noqa: E402


def test_compute_breakpoints_matches_percentile():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    bp = membership.compute_breakpoints(x)
    expected = tuple(float(v) for v in np.percentile(x, [25, 50, 75]))
    assert np.allclose(bp, expected), (bp, expected)


def test_fuzzify_trailing_dim_and_interior_sum():
    x = np.linspace(0, 10, 50)
    bp = membership.compute_breakpoints(x)
    m = membership.fuzzify(x, bp)
    assert m.shape == (50, 3), m.shape
    # Interior rows (strictly between q25 and q75) sum to ~1.0 for triangular MFs.
    q25, q50, q75 = bp
    interior = (x > q25) & (x < q75)
    sums = m[interior].sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-9), sums


def test_peaks_at_breakpoints():
    bp = (2.0, 5.0, 8.0)
    # value == q50 -> medium ~1
    m_mid = membership.fuzzify(np.array([5.0]), bp)[0]
    assert np.isclose(m_mid[1], 1.0), m_mid
    # value <= q25 -> low ~1
    m_lo = membership.fuzzify(np.array([1.0]), bp)[0]
    assert np.isclose(m_lo[0], 1.0), m_lo
    # value >= q75 -> high ~1
    m_hi = membership.fuzzify(np.array([9.0]), bp)[0]
    assert np.isclose(m_hi[2], 1.0), m_hi


def test_degenerate_constant_input_no_nan():
    x = np.full(20, 3.0)
    bp = membership.compute_breakpoints(x)  # all equal -> (3,3,3)
    m = membership.fuzzify(x, bp)
    assert np.all(np.isfinite(m)), m
    assert m.shape == (20, 3)


def test_non_monotone_breakpoints_raise():
    raised = False
    try:
        membership.fuzzify(np.array([1.0]), (5.0, 2.0, 8.0))
    except ValueError:
        raised = True
    assert raised, "expected ValueError on non-monotone breakpoints"


def main() -> int:
    test_compute_breakpoints_matches_percentile()
    test_fuzzify_trailing_dim_and_interior_sum()
    test_peaks_at_breakpoints()
    test_degenerate_constant_input_no_nan()
    test_non_monotone_breakpoints_raise()
    print("test_membership PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
