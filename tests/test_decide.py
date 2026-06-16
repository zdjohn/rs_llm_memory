"""Plain-assert unit tests for src/decide.py (toy in-memory metric dicts, in-process).

    python tests/test_decide.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import decide  # noqa: E402

N = decide.COLD_NDCG
H = decide.COLD_HIT


def _roles(floor, a0, a1):
    return {
        "floor": {N: floor[0], H: floor[1]},
        "A0": {N: a0[0], H: a0[1]},
        "A1": {N: a1[0], H: a1[1]},
    }


def test_pass_when_a0_strictly_beats_on_both():
    v = decide.decide(_roles(floor=(0.10, 0.20), a0=(0.15, 0.25), a1=(0.05, 0.10)))
    assert v == "PASS", v


def test_pass_when_a1_beats_even_if_a0_fails():
    v = decide.decide(_roles(floor=(0.10, 0.20), a0=(0.05, 0.10), a1=(0.12, 0.21)))
    assert v == "PASS", v


def test_pass_when_ge_on_both_one_strictly_better():
    # NDCG ties, Hit strictly better -> still PASS (>= on both, strictly better on one).
    v = decide.decide(_roles(floor=(0.10, 0.20), a0=(0.10, 0.25), a1=(0.0, 0.0)))
    assert v == "PASS", v


def test_weak_pass_on_exact_tie():
    v = decide.decide(_roles(floor=(0.10, 0.20), a0=(0.10, 0.20), a1=(0.05, 0.05)))
    assert v == "WEAK PASS", v


def test_fail_when_underperforms_on_either():
    # A0 beats NDCG but loses Hit; A1 loses both -> FAIL.
    v = decide.decide(_roles(floor=(0.10, 0.20), a0=(0.15, 0.10), a1=(0.05, 0.05)))
    assert v == "FAIL", v


def test_inconclusive_when_floor_missing():
    roles = {"floor": {N: None, H: None}, "A0": {N: 0.1, H: 0.1}, "A1": {N: 0.1, H: 0.1}}
    assert decide.decide(roles) == "INCONCLUSIVE"


def test_comparison_table_includes_ceil_ref():
    # Smoke the end-to-end main()-style path on a tiny DataFrame including DeepFM.
    import pandas as pd
    df = pd.DataFrame({
        "model": ["BPR", "FIS-A0", "FIS-A1", "DeepFM"],
        N: [0.10, 0.15, 0.05, 0.30],
        H: [0.20, 0.25, 0.10, 0.40],
    })
    metrics_by_role = {
        role: {N: decide._latest_metric(df, model, N), H: decide._latest_metric(df, model, H)}
        for role, model in decide.ROLE_MODELS.items()
    }
    assert metrics_by_role["ceil-ref"][N] == 0.30, metrics_by_role["ceil-ref"]
    assert decide.decide(metrics_by_role) == "PASS"


def main() -> int:
    test_pass_when_a0_strictly_beats_on_both()
    test_pass_when_a1_beats_even_if_a0_fails()
    test_pass_when_ge_on_both_one_strictly_better()
    test_weak_pass_on_exact_tie()
    test_fail_when_underperforms_on_either()
    test_inconclusive_when_floor_missing()
    test_comparison_table_includes_ceil_ref()
    print("test_decide PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
