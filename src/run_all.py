"""Experiment-matrix loop (plan §8/§9). Each cell = one run_one + one MLflow run.

    models   = [BPR, DeepFM]
    datasets = [ml-100k]            # lastfm added once its atomic files exist
    seeds    = [2020]               # expand to [2020..2024] once the pipeline is trusted
    sideinfo = BPR    -> [(off, off)]               # ID-only floor
               DeepFM -> [(off, off), (on, on)]     # no side info vs full

Usage
-----
    python src/run_all.py                       # full matrix, real training budget
    python src/run_all.py --quick               # smoke: epochs=1 across the matrix
    python src/run_all.py --models BPR --seeds 2020 --quick
"""
from __future__ import annotations

import argparse
import traceback

from run_one import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, log_to_mlflow, run_one

# Per-model side-info configurations: list of (user_feats, item_feats).
SIDEINFO = {
    "BPR": [(False, False)],
    "DeepFM": [(False, False), (True, True)],
}
DEFAULT_MODELS = ["BPR", "DeepFM"]
DEFAULT_DATASETS = ["ml-100k"]
DEFAULT_SEEDS = [2020]


def iter_matrix(models, datasets, seeds):
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                for user_feats, item_feats in SIDEINFO.get(model, [(False, False)]):
                    yield model, dataset, seed, user_feats, item_feats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--quick", action="store_true", help="Smoke mode: epochs=1, stopping_step=1")
    p.add_argument("--tracking_uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--no_mlflow", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print the matrix and exit")
    args = p.parse_args()

    cells = list(iter_matrix(args.models, args.datasets, args.seeds))
    print(f"matrix: {len(cells)} run(s)")
    for model, dataset, seed, uf, itf in cells:
        print(f"  {model:7s} {dataset:8s} seed={seed} user={'on' if uf else 'off':3s} item={'on' if itf else 'off':3s}")
    if args.dry_run:
        return

    ok, failed = 0, 0
    for model, dataset, seed, uf, itf in cells:
        tag = f"{model}-{dataset}-s{seed}-u{'on' if uf else 'off'}-i{'on' if itf else 'off'}"
        print(f"\n>>> {tag}")
        try:
            result = run_one(model, dataset, user_feats=uf, item_feats=itf,
                             seed=seed, quick=args.quick)
            print(f"    test: " + "  ".join(f"{k}={v:.4f}" for k, v in result["test_result"].items()))
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


if __name__ == "__main__":
    main()
