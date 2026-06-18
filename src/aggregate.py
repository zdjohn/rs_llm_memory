"""Aggregate MLflow runs into the canonical CSV (plan §6.8, §10).

The MLflow UI is the exploration layer; THIS CSV is the source of truth for paper
numbers. Regenerate it from scratch any time:

    python src/aggregate.py                       # -> results/baseline_summary.csv
    python src/aggregate.py --experiment smoke-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

GROUP_KEYS = ["model", "dataset", "user_feats", "item_feats"]
PARAM_COLS = ["model", "dataset", "seed", "user_feats", "item_feats",
              "embedding_size", "learning_rate", "epochs"]
# Track-A slice metrics (group_metrics.flatten_slice_metrics -> sanitize_key). The literal
# keys below MUST match group_metrics' emitted keys and decide.py's lookups verbatim.
SLICE_METRIC_COLS = [
    "ndcg_at_3_slice_all", "hit_at_3_slice_all",
    "ndcg_at_3_slice_cold_user", "hit_at_3_slice_cold_user",
    "ndcg_at_3_slice_cold_item", "hit_at_3_slice_cold_item",
]
METRIC_COLS = ["ndcg_at_3", "hit_at_3", "precision_at_3",
               "ndcg_at_1", "hit_at_1", "precision_at_1", "train_time_sec",
               *SLICE_METRIC_COLS]
SUMMARY_METRICS = ["ndcg_at_3", "hit_at_3", "precision_at_3", "train_time_sec",
                   "ndcg_at_3_slice_cold_item", "hit_at_3_slice_cold_item"]


def tidy(runs: pd.DataFrame) -> pd.DataFrame:
    """Flatten MLflow's params.*/metrics.* columns into clean named columns."""
    out = pd.DataFrame()
    for col in PARAM_COLS:
        out[col] = runs.get(f"params.{col}")
    for col in METRIC_COLS:
        out[col] = pd.to_numeric(runs.get(f"metrics.{col}"), errors="coerce")
    out["git_commit"] = runs.get("tags.git_commit")
    out["run_id"] = runs.get("run_id")
    # numeric param coercion for tidy sorting/grouping
    for col in ("seed", "embedding_size", "epochs"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds per (model, dataset, user_feats, item_feats)."""
    rows = []
    for keys, grp in tidy_df.groupby(GROUP_KEYS, dropna=False):
        row = dict(zip(GROUP_KEYS, keys))
        row["n_seeds"] = grp["seed"].nunique()
        for m in SUMMARY_METRICS:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_std"] = grp[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_KEYS).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracking_uri", default="http://localhost:5002")
    p.add_argument("--experiment", default="openba-recsys-baselines")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "baseline_summary.csv"))
    p.add_argument("--runs_out", default=str(REPO_ROOT / "results" / "all_runs.csv"))
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    runs = mlflow.search_runs(experiment_names=[args.experiment])
    if runs is None or runs.empty:
        print(f"no runs found in experiment '{args.experiment}' at {args.tracking_uri}")
        return

    tidy_df = tidy(runs)
    summary = summarize(tidy_df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tidy_df.to_csv(args.runs_out, index=False)
    summary.to_csv(args.out, index=False)

    print(f"{len(tidy_df)} runs -> {args.runs_out}")
    print(f"{len(summary)} groups -> {args.out}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        cols = GROUP_KEYS + ["n_seeds", "ndcg_at_3_mean", "hit_at_3_mean", "precision_at_3_mean"]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
