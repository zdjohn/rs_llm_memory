"""Track-A PASS / WEAK PASS / FAIL decision on the cold-item slice (plan TASK-017).

Reads the aggregated results (results/all_runs.csv by default, or via mlflow.search_runs)
and applies the LOCKED decision rule on the cold-item slice for NDCG@3 AND Hit@3, gated on
a green adapter (the sanity check having passed upstream):

    PASS       = (A0 OR A1) >= BPR floor on cold-item for BOTH NDCG@3 and Hit@3
    WEAK PASS  = A0/A1 merely TIES the floor on the cold slice (>= holds via equality only)
    FAIL       = underperforms the floor on EITHER metric (with a green adapter)

Prints exactly one verdict plus the comparison table (floor vs A0 vs A1 vs DeepFM ceil-ref
on cold-item). Read-only.

    python src/decide.py
    python src/decide.py --runs_csv results/all_runs.csv
    python src/decide.py --from_mlflow --experiment openba-recsys-baselines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical sanitized slice keys (must match group_metrics.flatten_slice_metrics + aggregate.py).
COLD_NDCG = "ndcg_at_3_slice_cold_item"
COLD_HIT = "hit_at_3_slice_cold_item"

# Role -> the `model` param value written by run_all / run_fuzzy.
ROLE_MODELS = {"floor": "BPR", "A0": "FIS-A0", "A1": "FIS-A1", "ceil-ref": "DeepFM"}

TOL = 1e-9  # equality tolerance for the tie / >= comparisons.


def _latest_metric(df: pd.DataFrame, model: str, metric: str) -> float | None:
    """Most-recent finite value of `metric` for rows whose `model` matches."""
    rows = df[df["model"] == model]
    if rows.empty or metric not in rows.columns:
        return None
    vals = pd.to_numeric(rows[metric], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


def decide(metrics_by_role: dict[str, dict[str, float | None]]) -> str:
    """Apply the locked rule. `metrics_by_role[role]` = {COLD_NDCG: v, COLD_HIT: v}.

    Returns one of {PASS, WEAK PASS, FAIL, INCONCLUSIVE}. INCONCLUSIVE if the floor or both
    FIS variants are missing the cold-item metrics.
    """
    floor = metrics_by_role.get("floor", {})
    f_ndcg, f_hit = floor.get(COLD_NDCG), floor.get(COLD_HIT)
    if f_ndcg is None or f_hit is None:
        return "INCONCLUSIVE"

    best_verdict = "FAIL"
    saw_candidate = False
    for role in ("A0", "A1"):
        m = metrics_by_role.get(role, {})
        c_ndcg, c_hit = m.get(COLD_NDCG), m.get(COLD_HIT)
        if c_ndcg is None or c_hit is None:
            continue
        saw_candidate = True
        ndcg_ge = c_ndcg >= f_ndcg - TOL
        hit_ge = c_hit >= f_hit - TOL
        if ndcg_ge and hit_ge:
            # Exact tie on BOTH -> WEAK PASS; strictly better on at least one -> PASS.
            strictly_better = (c_ndcg > f_ndcg + TOL) or (c_hit > f_hit + TOL)
            verdict = "PASS" if strictly_better else "WEAK PASS"
            if verdict == "PASS":
                return "PASS"  # best possible; short-circuit.
            best_verdict = "WEAK PASS"
    if not saw_candidate:
        return "INCONCLUSIVE"
    return best_verdict


def _load_runs(args) -> pd.DataFrame:
    if args.from_mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.tracking_uri)
        runs = mlflow.search_runs(experiment_names=[args.experiment])
        if runs is None or runs.empty:
            raise SystemExit(f"no runs in experiment '{args.experiment}'")
        # Flatten params.model + metrics.* to bare columns matching the CSV layout.
        df = pd.DataFrame({"model": runs.get("params.model")})
        for metric in (COLD_NDCG, COLD_HIT):
            df[metric] = pd.to_numeric(runs.get(f"metrics.{metric}"), errors="coerce")
        return df
    path = Path(args.runs_csv)
    if not path.exists():
        raise SystemExit(f"runs CSV not found: {path}. Run src/aggregate.py first.")
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_csv", default=str(REPO_ROOT / "results" / "all_runs.csv"))
    p.add_argument("--from_mlflow", action="store_true", help="Read from MLflow instead of the CSV")
    p.add_argument("--tracking_uri", default="http://localhost:5002")
    p.add_argument("--experiment", default="openba-recsys-baselines")
    args = p.parse_args()

    df = _load_runs(args)

    metrics_by_role = {}
    for role, model in ROLE_MODELS.items():
        metrics_by_role[role] = {
            COLD_NDCG: _latest_metric(df, model, COLD_NDCG),
            COLD_HIT: _latest_metric(df, model, COLD_HIT),
        }

    verdict = decide(metrics_by_role)

    print("Cold-item slice comparison (NDCG@3, Hit@3):")
    print(f"  {'role':9s} {'model':8s} {'NDCG@3':>10s} {'Hit@3':>10s}")
    for role in ("floor", "A0", "A1", "ceil-ref"):
        m = metrics_by_role[role]
        nd = m[COLD_NDCG]
        hi = m[COLD_HIT]
        nd_s = f"{nd:.4f}" if nd is not None else "    n/a"
        hi_s = f"{hi:.4f}" if hi is not None else "    n/a"
        print(f"  {role:9s} {ROLE_MODELS[role]:8s} {nd_s:>10s} {hi_s:>10s}")

    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
