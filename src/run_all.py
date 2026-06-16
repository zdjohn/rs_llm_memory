"""Experiment-matrix loop (plan §8/§9). Each cell = one run + one MLflow run.

Two modes:

  Baseline matrix (default): models × datasets × seeds × side-info (BPR + DeepFM).

      python src/run_all.py                       # full matrix, real training budget
      python src/run_all.py --quick               # smoke: epochs=1 across the matrix
      python src/run_all.py --models BPR --seeds 2020 --quick

  Track-A matrix (--track_a): floor=BPR-MF, ceil-ref=DeepFM, A0 (FIS frozen weights),
  A1 (FIS fitted weights). Calls the BPR-reproduction sanity gate FIRST (HARD BLOCKER):
  if it raises, the process aborts non-zero and NO FIS run is attempted. Every scheduled
  run also logs uniform slice metrics (all / cold_user / cold_item × NDCG@3, Hit@3).

      python src/run_all.py --track_a --dry_run   # list floor/ceil-ref/A0/A1 and exit
      python src/run_all.py --track_a             # run the full Track-A matrix
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_one import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, log_to_mlflow, run_one

# Per-model side-info configurations: list of (user_feats, item_feats).
SIDEINFO = {
    "BPR": [(False, False)],
    "DeepFM": [(False, False), (True, True)],
}
DEFAULT_MODELS = ["BPR", "DeepFM"]
DEFAULT_DATASETS = ["ml100k"]
DEFAULT_SEEDS = [2020]

# Track-A roles: (role label, kind, model/mode, user_feats, item_feats).
#   kind "baseline" -> run_baseline_with_handles(model)
#   kind "fis"      -> run_fuzzy(experiment, mode)
TRACK_A_ROLES = [
    ("floor", "baseline", "BPR", False, False),
    ("ceil-ref", "baseline", "DeepFM", True, True),
    ("A0", "fis", "A0", False, True),
    ("A1", "fis", "A1", False, True),
]


def iter_matrix(models, datasets, seeds):
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                for user_feats, item_feats in SIDEINFO.get(model, [(False, False)]):
                    yield model, dataset, seed, user_feats, item_feats


def _slice_flat_keys(result: dict, config, k: int = 3) -> dict:
    """Compute the all/cold_user/cold_item slice metrics for a completed run's handles."""
    import group_metrics as gm

    handles = result.get("_handles")
    if handles is None:
        return {}
    model = handles["model"]
    train_data = handles["train_data"]
    test_data = handles["test_data"]

    per_ndcg, per_hit, per_items = gm.collect_per_user_metrics(model, test_data, config, k=k)
    user_counts, item_counts = gm.train_interaction_counts(train_data, config)
    sliced = gm.slice_metrics(per_ndcg, per_hit, per_items, user_counts, item_counts)
    return gm.flatten_slice_metrics(sliced, k=k)


def _log_with_slices(result: dict, *, tracking_uri, experiment, run_name) -> str:
    """Merge slice metrics into result['test_result'] (sanitize_key-idempotent) then log."""
    flat = _slice_flat_keys(result, result["config"])
    # sanitize_key is idempotent on these already-flat keys, so log_to_mlflow stays unchanged.
    result["test_result"] = {**result["test_result"], **flat}
    return log_to_mlflow(result, tracking_uri=tracking_uri, experiment=experiment,
                         run_name=run_name)


def run_baseline_matrix(args) -> None:
    cells = list(iter_matrix(args.models, args.datasets, args.seeds))
    print(f"matrix: {len(cells)} run(s)")
    for model, dataset, seed, uf, itf in cells:
        print(f"  {model:7s} {dataset:8s} seed={seed} "
              f"user={'on' if uf else 'off':3s} item={'on' if itf else 'off':3s}")
    if args.dry_run:
        return

    ok, failed = 0, 0
    for model, dataset, seed, uf, itf in cells:
        tag = f"{model}-{dataset}-s{seed}-u{'on' if uf else 'off'}-i{'on' if itf else 'off'}"
        print(f"\n>>> {tag}")
        try:
            result = run_one(model, dataset, user_feats=uf, item_feats=itf,
                             seed=seed, quick=args.quick)
            print("    test: " + "  ".join(f"{k}={v:.4f}" for k, v in result["test_result"].items()))
            if not args.no_mlflow:
                run_id = log_to_mlflow(result, tracking_uri=args.tracking_uri,
                                       experiment=args.experiment, run_name=tag)
                print(f"    mlflow run {run_id}")
            ok += 1
        except Exception:
            failed += 1
            print(f"    FAILED:\n{traceback.format_exc()}")

    print(f"\ndone: {ok} ok, {failed} failed")
    if failed:
        raise SystemExit(1)


def run_track_a(args) -> None:
    print(f"track-a matrix: {len(TRACK_A_ROLES)} run(s)")
    for role, kind, target, uf, itf in TRACK_A_ROLES:
        print(f"  {role:9s} [{kind:8s}] {target:7s} "
              f"user={'on' if uf else 'off':3s} item={'on' if itf else 'off':3s}")
    if args.dry_run:
        return

    from fuzzy.run_fuzzy import (bpr_floor_ndcg3_from_mlflow, bpr_sanity_check,
                                 run_baseline_with_handles, run_fuzzy)

    dataset = args.datasets[0]

    # --- HARD BLOCKER: BPR-reproduction sanity gate BEFORE any FIS run. ---
    print("\n>>> sanity gate (BPR through the FISRecommender adapter)")
    floor = bpr_floor_ndcg3_from_mlflow(tracking_uri=args.tracking_uri, dataset_name=dataset)
    bpr_sanity_check(floor, dataset_name=dataset)  # raises on red -> aborts the process

    ok, failed = 0, 0
    for role, kind, target, uf, itf in TRACK_A_ROLES:
        tag = f"{role}-{target}-{dataset}"
        print(f"\n>>> {tag}")
        try:
            if kind == "baseline":
                result = run_baseline_with_handles(target, dataset_name=dataset,
                                                   user_feats=uf, item_feats=itf,
                                                   quick=args.quick)
            else:  # fis
                result = run_fuzzy(args.experiment, target, dataset_name=dataset)
            print("    test: " + "  ".join(
                f"{k}={v:.4f}" for k, v in result["test_result"].items()))
            if not args.no_mlflow:
                run_id = _log_with_slices(result, tracking_uri=args.tracking_uri,
                                          experiment=args.experiment, run_name=tag)
                print(f"    mlflow run {run_id}")
            ok += 1
        except Exception:
            failed += 1
            print(f"    FAILED:\n{traceback.format_exc()}")

    print(f"\ndone: {ok} ok, {failed} failed")
    if failed:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--quick", action="store_true", help="Smoke mode: epochs=1, stopping_step=1")
    p.add_argument("--track_a", action="store_true",
                   help="Run the Track-A matrix (floor/ceil-ref/A0/A1) with the sanity gate")
    p.add_argument("--tracking_uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--no_mlflow", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print the matrix and exit")
    args = p.parse_args()

    if args.track_a:
        run_track_a(args)
    else:
        run_baseline_matrix(args)


if __name__ == "__main__":
    main()
