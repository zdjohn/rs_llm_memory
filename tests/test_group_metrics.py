"""Plain-assert unit tests for the pure helpers in src/group_metrics.py (toy inputs).

Covers hit_at_k, cold-slice construction, load_item_groups, slice_metrics /
flatten_slice_metrics. Does NOT test collect_per_user_ndcg (requires a trained model —
covered by TASK-013's [INTEGRATION] test).

    python tests/test_group_metrics.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import group_metrics as gm  # noqa: E402


def test_hit_at_k():
    assert gm.hit_at_k([10, 20, 30], {20}, 3) == 1.0
    assert gm.hit_at_k([10, 20, 30], {99}, 3) == 0.0
    # relevant item present but beyond top-k
    assert gm.hit_at_k([10, 20, 30], {30}, 2) == 0.0
    assert gm.hit_at_k([10, 20, 30], {10}, 1) == 1.0


def test_cold_entities():
    counts = {"a": 1, "b": 5, "c": 6, "d": 100}
    cold = gm.cold_entities(counts, threshold=5)
    assert cold == {"a", "b"}, cold


def test_load_item_groups():
    content = (
        "item_id:token\tgenre:token_seq\trelease_year:token\n"
        "1\tComedy Action\t1990\n"
        "2\tDrama\t2000\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".item", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    groups = gm.load_item_groups(path)
    assert set(groups) == {"1", "2"}, set(groups)
    assert groups["1"]["genre"] == "Comedy Action", groups["1"]
    assert groups["2"]["release_year"] == "2000", groups["2"]


def test_slice_and_flatten_metrics():
    per_user_ndcg = {"u1": 1.0, "u2": 0.5, "u3": 0.0}
    per_user_hit = {"u1": 1.0, "u2": 1.0, "u3": 0.0}
    # u2 is a cold user (<=5 train); u1 has a cold test item; u3 neither.
    per_user_test_items = {"u1": {"i_cold"}, "u2": {"i_warm"}, "u3": {"i_warm"}}
    user_train_counts = {"u1": 50, "u2": 3, "u3": 80}
    item_train_counts = {"i_cold": 2, "i_warm": 40}

    sliced = gm.slice_metrics(per_user_ndcg, per_user_hit, per_user_test_items,
                              user_train_counts, item_train_counts, cold_threshold=5)
    assert set(sliced) == {"all", "cold_user", "cold_item"}, set(sliced)
    # all: mean ndcg = (1+0.5+0)/3
    assert abs(sliced["all"]["ndcg"] - 0.5) < 1e-9, sliced["all"]
    # cold_user: only u2 -> ndcg 0.5, hit 1.0
    assert sliced["cold_user"]["ndcg"] == 0.5 and sliced["cold_user"]["hit"] == 1.0
    # cold_item: only u1 (test item i_cold is cold) -> ndcg 1.0, hit 1.0
    assert sliced["cold_item"]["ndcg"] == 1.0 and sliced["cold_item"]["hit"] == 1.0

    # sanitize_key collapses `__` -> `_`, so the emitted keys are single-underscore.
    flat = gm.flatten_slice_metrics(sliced, k=3)
    for sl in ("all", "cold_user", "cold_item"):
        assert f"ndcg_at_3_slice_{sl}" in flat, flat
        assert f"hit_at_3_slice_{sl}" in flat, flat
    # all keys MLflow-safe (no '@')
    assert all("@" not in key for key in flat), flat


def test_slice_metrics_empty_slice_zero():
    per_user_ndcg = {"u1": 1.0}
    per_user_hit = {"u1": 1.0}
    per_user_test_items = {"u1": {"i_warm"}}
    sliced = gm.slice_metrics(per_user_ndcg, per_user_hit, per_user_test_items,
                              {"u1": 100}, {"i_warm": 100}, cold_threshold=5)
    assert sliced["cold_user"]["ndcg"] == 0.0
    assert sliced["cold_item"]["hit"] == 0.0


def main() -> int:
    test_hit_at_k()
    test_cold_entities()
    test_load_item_groups()
    test_slice_and_flatten_metrics()
    test_slice_metrics_empty_slice_zero()
    print("test_group_metrics PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
