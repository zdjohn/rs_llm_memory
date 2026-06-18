"""Plain-assert tests for the BPR floor under ml100k (TASK-004). [INTEGRATION]

These require the `recsys` conda env (torch + recbole) AND the MLflow stack on :5002.
They are guarded behind RUN_INTEGRATION so the file imports cleanly anywhere.

    RUN_INTEGRATION=1 conda run -n recsys python tests/test_bpr_floor.py

What it does:
  - Train BPR on real ml100k at the frozen budget (user_feats=off, item_feats=off),
    log via log_to_mlflow, read the run back and assert params + finite metrics.ndcg_at_3.
  - Regression: a BPR run under the legacy ml-100k name still completes + persists ndcg_at_3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_one  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002")
EXPERIMENT = "track-a-floor-test"


def _run_and_readback(dataset: str):
    import mlflow

    result = run_one.run_one("BPR", dataset, user_feats=False, item_feats=False)
    run_id = run_one.log_to_mlflow(result, tracking_uri=TRACKING_URI,
                                   experiment=EXPERIMENT, run_name=f"BPR-{dataset}-floor")
    mlflow.set_tracking_uri(TRACKING_URI)
    fetched = mlflow.tracking.MlflowClient().get_run(run_id)
    params, metrics = fetched.data.params, fetched.data.metrics
    assert params["model"] == "BPR", params
    assert params["dataset"] == dataset, params
    assert params["user_feats"] == "off" and params["item_feats"] == "off", params
    ndcg3 = metrics.get("ndcg_at_3")
    assert ndcg3 is not None and ndcg3 == ndcg3, metrics  # finite (not NaN)
    return float(ndcg3)


def integration_floor_ml100k():  # [INTEGRATION]
    ndcg3 = _run_and_readback("ml100k")
    print(f"[integration] BPR ml100k floor ndcg_at_3={ndcg3:.6f} (record as the floor reference)")


def integration_floor_ml100k_legacy_regression():  # [INTEGRATION] regression
    ndcg3 = _run_and_readback("ml-100k")
    print(f"[integration] BPR ml-100k (legacy) floor ndcg_at_3={ndcg3:.6f}")


def main() -> int:
    if not os.environ.get("RUN_INTEGRATION"):
        print("test_bpr_floor SKIPPED (set RUN_INTEGRATION=1 + recsys env + MLflow on :5002)")
        return 0
    integration_floor_ml100k()
    integration_floor_ml100k_legacy_regression()
    print("test_bpr_floor PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
