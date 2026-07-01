"""Cross-check: the numpy Mamdani and the simpful reference must AGREE IN RANKING.

The two engines are not bit-identical (numpy = Mamdani MAX aggregation + centroid defuzz;
simpful = Sugeno firing-weighted sum), so we assert a high Spearman rank correlation on sampled
cells rather than absolute equality. Skipped if simpful is not installed.

    python tests/test_fis_recency_spec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fuzzy import rules_recency as rr  # noqa: E402
from fuzzy.membership import compute_breakpoints  # noqa: E402

BP = {c: (0.25, 0.5, 0.75) for c in rr.CONCEPTS_RECENCY}


def test_simpful_matches_numpy_ranking():
    try:
        from fuzzy.fis_recency_spec import build_simpful_fis, simpful_score
        FS = build_simpful_fis(BP)
    except ImportError:
        print("  [skip] simpful not installed")
        return

    rng = np.random.default_rng(0)
    n = 150
    rel = rng.random(n)
    rec = rng.random(n)
    vol = rng.random(n)
    freq = rng.random(n)
    con = {"relevance": rel[None, :], "recency": rec[None, :],
           "volatility": vol, "frequency": freq}
    numpy_scores = rr.score_matrix_recency(con, BP, weights=None)[0]

    simpful_scores = np.array([
        simpful_score(FS, {"relevance": rel[i], "recency": rec[i],
                           "volatility": vol[i], "frequency": freq[i]})
        for i in range(n)])

    # Spearman via rank correlation (no scipy dependency needed for the assert).
    def _rank(a):
        order = np.argsort(np.argsort(a))
        return order.astype(float)
    rho = np.corrcoef(_rank(numpy_scores), _rank(simpful_scores))[0, 1]
    assert rho > 0.95, f"numpy vs simpful rank correlation too low: {rho:.4f}"
    print(f"  numpy vs simpful Spearman rho = {rho:.4f}")


def main() -> int:
    test_simpful_matches_numpy_ranking()
    print("test_fis_recency_spec PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
