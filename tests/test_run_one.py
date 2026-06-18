"""Plain-assert tests for the ml100k harness wiring (TASK-002).

Unit asserts (run anywhere — no training, no MLflow):
  - run_one.run_one default dataset == "ml100k"; argparse --dataset default == "ml100k".
  - run_all.DEFAULT_DATASETS == ["ml100k"].
  - group_metrics.atomic_path resolves ml100k by default, ml-100k when asked, and the
    ml100k paths point at the files materialized by TASK-001 (data on disk).

Integration (guarded — needs the `recsys` conda env: torch + recbole):
  - run_one("BPR", quick) trains against real ml100k and returns ndcg@3. [INTEGRATION]
Run the integration block with:  RUN_INTEGRATION=1 conda run -n recsys python tests/test_run_one.py

    python tests/test_run_one.py
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import group_metrics as gm  # noqa: E402
import run_all  # noqa: E402
import run_one  # noqa: E402


def test_run_one_default_dataset():
    sig = inspect.signature(run_one.run_one)
    assert sig.parameters["dataset"].default == "ml100k", sig.parameters["dataset"].default


def test_argparse_dataset_default_ml100k():
    # Parse with only the required --model; --dataset should default to ml100k.
    src = inspect.getsource(run_one.main)
    assert 'p.add_argument("--dataset", default="ml100k")' in src, "argparse default not ml100k"


def test_run_all_default_datasets():
    assert run_all.DEFAULT_DATASETS == ["ml100k"], run_all.DEFAULT_DATASETS


def test_atomic_path_resolution():
    p_user = gm.atomic_path("user")
    assert p_user == REPO_ROOT / "data" / "ml100k" / "ml100k.user", p_user
    p_item_legacy = gm.atomic_path("item", dataset="ml-100k")
    assert p_item_legacy == REPO_ROOT / "data" / "ml-100k" / "ml-100k.item", p_item_legacy


def test_atomic_path_points_at_materialized_files():
    # Migration check: the ml100k paths must exist (created by TASK-001).
    for suffix in ("inter", "user", "item"):
        assert gm.atomic_path(suffix).exists(), gm.atomic_path(suffix)
    # Regression: legacy ml-100k files still resolve.
    for suffix in ("inter", "user", "item"):
        assert gm.atomic_path(suffix, dataset="ml-100k").exists(), suffix


def integration_run_one_bpr_quick():  # [INTEGRATION] needs recsys env
    result = run_one.run_one("BPR", "ml100k", user_feats=False, item_feats=False, quick=True)
    assert "ndcg@3" in result["test_result"], list(result["test_result"])
    print(f"[integration] BPR ml100k quick ndcg@3={result['test_result']['ndcg@3']:.4f}")


def main() -> int:
    test_run_one_default_dataset()
    test_argparse_dataset_default_ml100k()
    test_run_all_default_datasets()
    test_atomic_path_resolution()
    test_atomic_path_points_at_materialized_files()
    if os.environ.get("RUN_INTEGRATION"):
        integration_run_one_bpr_quick()
    print("test_run_one PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
