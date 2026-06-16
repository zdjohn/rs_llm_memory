"""End-to-end smoke test: data -> train -> eval -> MLflow -> read-back verification.

Confirms the whole pipeline runs and that metrics actually persist in the MLflow
Postgres backend (not just that the client call returned). Exits non-zero on any failure.

    python src/smoke_test.py                       # BPR quick run, verify it lands
    python src/smoke_test.py --model DeepFM        # smoke the DeepFM path too
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_one import log_to_mlflow, run_one  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="BPR")
    p.add_argument("--dataset", default="ml100k")
    p.add_argument("--user_feats", choices=["on", "off"], default="off")
    p.add_argument("--item_feats", choices=["on", "off"], default="off")
    p.add_argument("--tracking_uri", default="http://localhost:5002")
    p.add_argument("--experiment", default="smoke-test")
    args = p.parse_args()

    uf, itf = args.user_feats == "on", args.item_feats == "on"

    # 1. atomic files present?
    for suffix in ("inter", "user", "item"):
        f = REPO_ROOT / "data" / args.dataset / f"{args.dataset}.{suffix}"
        if not f.exists():
            print(f"[FAIL] missing atomic file {f}. Run: python src/materialize_ml100k.py")
            return 1
    print("[ok] atomic files present")

    # 2. MLflow reachable?
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        client.search_experiments()
    except Exception as e:
        print(f"[FAIL] MLflow not reachable at {args.tracking_uri}: {e}\n"
              f"       Start it: cd mlflow-docker && docker compose up -d")
        return 1
    print(f"[ok] MLflow reachable at {args.tracking_uri}")

    # 3. train + eval (quick)
    print(f"[..] training {args.model} (quick: epochs=1) ...")
    result = run_one(args.model, args.dataset, user_feats=uf, item_feats=itf, quick=True)
    test = result["test_result"]
    print(f"[ok] trained. test metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in test.items()))
    assert "ndcg@3" in test, f"expected ndcg@3 in results, got {list(test)}"

    # 4. log to MLflow
    run_name = f"smoke-{args.model}-{args.dataset}-u{args.user_feats}-i{args.item_feats}"
    run_id = log_to_mlflow(result, tracking_uri=args.tracking_uri,
                           experiment=args.experiment, run_name=run_name)
    print(f"[ok] logged run {run_id}")

    # 5. READ BACK from the backend store and verify it persisted
    fetched = client.get_run(run_id)
    metrics = fetched.data.metrics
    assert fetched.info.status == "FINISHED", f"run status {fetched.info.status}"
    for required in ("ndcg_at_3", "hit_at_3", "precision_at_3", "train_time_sec"):
        assert required in metrics, f"metric {required} not persisted (have {sorted(metrics)})"
    assert fetched.data.tags.get("git_commit"), "git_commit tag not persisted"
    print(f"[ok] read back from backend: ndcg_at_3={metrics['ndcg_at_3']}  "
          f"hit_at_3={metrics['hit_at_3']}  precision_at_3={metrics['precision_at_3']}")
    print(f"[ok] params={len(fetched.data.params)} metrics={len(metrics)} "
          f"git_commit={fetched.data.tags['git_commit'][:8]}")

    print("\nSMOKE TEST PASSED — train -> eval -> MLflow persisted & verified.")
    print(f"View: {args.tracking_uri}/#/experiments/{fetched.info.experiment_id}/runs/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
