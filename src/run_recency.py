"""Orchestrate the PPR-vs-memory-decay comparison on the 500 most-active users.

Pipeline (plan §"Tuning / evaluation flow"), all self-contained (numpy/scipy/pandas/networkx;
RecBole only for the optional adapter sanity gate):

  1. (optional) RecBole adapter sanity gate — proves the FIS matrix -> RecBole eval path is
     faithful. Skipped with a clear message if RecBole is not importable.
  2. Build the two-level split (500 users -> VAL/TEST cohorts; per-user leave-last-1).
  3. lambda sweep on the VAL cohort -> lambda* (max mean MRR on VAL targets).
  4. A1 rule-weight fit on VAL history pairs at lambda* (pairwise BPR via fit_weights).
  5. Build three full-catalog score matrices: PPR, FIS-A0 (weights=None), FIS-A1 (fitted).
  6. Report on the TEST cohort (and VAL for reference): MRR + Hit@{1,3,10}.

Metrics use EXPECTED values under uniform random tie-breaking, because the Mamdani defuzz
produces many tied scores; the same convention is applied to PPR so the comparison is fair.
Results are written to results/recency_summary.csv (and MLflow if reachable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_split import build_split
from ppr import ppr_score_matrix
from fuzzy.temporal_concepts import build_temporal_concepts
from fuzzy.membership import compute_breakpoints
from fuzzy import rules_recency as rr
from fuzzy import fit_weights as fw

DEFAULT_LAMBDAS = (0.0, 0.001, 0.005, 0.02, 0.1)  # per day; 0 == time-agnostic
KS = (1, 3, 10)


# ---------------------------------------------------------------------------------------
# Expected ranking metrics under uniform random tie-breaking (fair for the discrete FIS).
# ---------------------------------------------------------------------------------------
def _masked_row(score_row: np.ndarray, u: int, target: int, history: set) -> np.ndarray:
    """Copy of a score row with PAD + the user's history (except the target) set to -inf."""
    row = score_row.astype(np.float64, copy=True)
    row[0] = -np.inf
    mask = history - {target}
    if mask:
        row[list(mask)] = -np.inf
    return row


def _expected_rank_and_hits(row: np.ndarray, target: int, ks=KS) -> tuple[float, dict]:
    """Expected rank of `target` and expected Hit@k under uniform tie-breaking.

    greater = #items strictly above target; equal = #items tied with target (incl. target).
    Expected rank = greater + (equal + 1) / 2. Expected Hit@k = clip((k - greater)/equal, 0, 1).
    Masked (-inf) items are never above a finite target and are excluded from `equal`.
    """
    ts = row[target]
    finite = np.isfinite(row)
    greater = int(np.sum(finite & (row > ts)))
    equal = int(np.sum(finite & (row == ts)))
    equal = max(equal, 1)
    exp_rank = greater + (equal + 1) / 2.0
    hits = {k: float(np.clip((k - greater) / equal, 0.0, 1.0)) for k in ks}
    return exp_rank, hits


def evaluate(score: np.ndarray, users: np.ndarray, targets: dict[int, int],
             history_items: dict[int, set], ks=KS) -> dict:
    """Mean MRR + Hit@k over `users` for a (n_users, n_items) score matrix.

    Returns {"mrr": float, "hit_at_{k}": float, "n": n_users} plus per-user "_rr" array.
    """
    rrs = []
    hit_acc = {k: [] for k in ks}
    for u in users.tolist():
        tgt = targets[u]
        row = _masked_row(score[u], u, tgt, history_items.get(u, set()))
        exp_rank, hits = _expected_rank_and_hits(row, tgt, ks)
        rrs.append(1.0 / exp_rank)
        for k in ks:
            hit_acc[k].append(hits[k])
    out = {"mrr": float(np.mean(rrs)), "n": int(len(users)), "_rr": np.asarray(rrs)}
    for k in ks:
        out[f"hit_at_{k}"] = float(np.mean(hit_acc[k]))
    return out


# ---------------------------------------------------------------------------------------
# FIS concept/matrix builders.
# ---------------------------------------------------------------------------------------
def _fis_matrix(split, lam: float, weights=None):
    """Build concepts at `lam`, data-driven breakpoints, and the recency score matrix."""
    con = build_temporal_concepts(split.history_pairs, split.history_ts, split.active_users,
                                  split.n_users, split.n_items, lam=lam)
    bp = {k: compute_breakpoints(v) for k, v in con.items()}
    matrix = rr.score_matrix_recency(con, bp, weights=weights)
    return matrix, con, bp


def _val_pairs(split, *, max_pairs: int = 20000, seed: int = 2020) -> np.ndarray:
    """(user, pos_item, neg_item) pairs from VAL-cohort HISTORY for the A1 weight fit."""
    rng = np.random.default_rng(seed)
    val_set = set(split.val_users.tolist())
    mask = np.array([int(u) in val_set for u in split.history_pairs[:, 0]])
    hp = split.history_pairs[mask]
    if len(hp) > max_pairs:
        sel = rng.choice(len(hp), size=max_pairs, replace=False)
        hp = hp[sel]
    negs = rng.integers(1, split.n_items, size=len(hp))
    return np.stack([hp[:, 0], hp[:, 1], negs], axis=1)


def tune_lambda(split, lambdas=DEFAULT_LAMBDAS) -> tuple[float, dict]:
    """Pick lambda* maximizing mean MRR on the VAL cohort (A0 weights). Returns (lam*, log)."""
    log = {}
    best_lam, best_mrr = lambdas[0], -1.0
    for lam in lambdas:
        matrix, _, _ = _fis_matrix(split, lam, weights=None)
        res = evaluate(matrix, split.val_users, split.targets, split.user_history_items)
        log[lam] = res["mrr"]
        print(f"  lambda={lam:<7} VAL MRR={res['mrr']:.4f}")
        if res["mrr"] > best_mrr:
            best_lam, best_mrr = lam, res["mrr"]
    print(f"  -> lambda* = {best_lam} (VAL MRR {best_mrr:.4f})")
    return best_lam, log


# ---------------------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------------------
def run(inter_path: str, *, n_active: int = 500, seed: int = 2020,
        lambdas=DEFAULT_LAMBDAS, sanity_gate: bool = False,
        ppr_engine: str = "networkx", fit_a1: bool = True) -> list[dict]:
    """Run the full comparison; return a list of result rows (one per model x cohort)."""
    if sanity_gate:
        _maybe_sanity_gate()

    print(f"[1] building split from {inter_path}")
    split = build_split(inter_path, n_active=n_active, seed=seed)
    print(f"    users(+PAD)={split.n_users} items(+PAD)={split.n_items} "
          f"history={len(split.history_pairs)} val={len(split.val_users)} test={len(split.test_users)}")

    print("[2] tuning lambda on VAL")
    lam_star, lam_log = tune_lambda(split, lambdas)

    print("[3] building score matrices (PPR, FIS-A0, FIS-A1) at lambda*")
    ppr = ppr_score_matrix(split.history_pairs, split.n_users, split.n_items,
                           split.active_users, engine=ppr_engine)
    fis_a0, con, bp = _fis_matrix(split, lam_star, weights=None)

    weights = None
    if fit_a1:
        try:
            pairs = _val_pairs(split, seed=seed)
            weights = fw.fit_weights(con, bp, pairs, score_fn=rr.score_matrix_recency,
                                     n_rules=rr.N_RULES, max_iter=200)
            print(f"    A1 weights fitted: {np.round(weights, 3)}")
        except Exception as exc:  # noqa: BLE001 - A1<A0 or non-convergence is not fatal
            print(f"    [warn] A1 fit failed ({type(exc).__name__}: {exc}); falling back to A0 weights")
            weights = None
    fis_a1 = rr.score_matrix_recency(con, bp, weights=weights)

    print("[4] evaluating on VAL + TEST")
    models = {"PPR": ppr, "FIS-recency-A0": fis_a0, "FIS-recency-A1": fis_a1}
    rows = []
    for cohort, users in (("val", split.val_users), ("test", split.test_users)):
        for name, mat in models.items():
            res = evaluate(mat, users, split.targets, split.user_history_items)
            row = {"cohort": cohort, "model": name, "lambda": lam_star,
                   "n_active": n_active, "n_users": res["n"], "mrr": res["mrr"]}
            row.update({f"hit_at_{k}": res[f"hit_at_{k}"] for k in KS})
            rows.append(row)
            if cohort == "test":
                print(f"    {name:16s} TEST  MRR={res['mrr']:.4f}  "
                      + "  ".join(f"H@{k}={res[f'hit_at_{k}']:.4f}" for k in KS))
    return rows


def _maybe_sanity_gate() -> None:
    """Run the RecBole adapter sanity gate if RecBole is importable; else skip with a note."""
    try:
        from fuzzy.run_fuzzy import bpr_sanity_check
    except Exception as exc:  # noqa: BLE001
        print(f"[0] sanity gate SKIPPED (RecBole unavailable: {type(exc).__name__}). "
              "The FIS->RecBole adapter was already validated in Track A; Option-E1 eval below "
              "does not depend on it.")
        return
    print("[0] sanity gate (BPR through the FISRecommender adapter)")
    bpr_sanity_check()  # raises on red


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cohort", "model", "lambda", "n_active", "n_users", "mrr",
              *(f"hit_at_{k}" for k in KS)]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"[5] wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inter", default=str(REPO_ROOT / "data/ml100k/ml100k.inter"),
                   help="Atomic .inter path (Amazon category or ml100k for a dry run).")
    p.add_argument("--n_active", type=int, default=500)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--ppr_engine", choices=["networkx", "power"], default="networkx")
    p.add_argument("--no_a1", action="store_true", help="Skip the A1 weight fit.")
    p.add_argument("--sanity_gate", action="store_true", help="Run the RecBole adapter gate first.")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "recency_summary.csv"))
    p.add_argument("--no_mlflow", action="store_true")
    args = p.parse_args()

    rows = run(args.inter, n_active=args.n_active, seed=args.seed, sanity_gate=args.sanity_gate,
               ppr_engine=args.ppr_engine, fit_a1=not args.no_a1)
    _write_csv(rows, Path(args.out))
    if not args.no_mlflow:
        _maybe_log_mlflow(rows)


def _maybe_log_mlflow(rows: list[dict]) -> None:
    """Best-effort MLflow logging (one run per model, TEST metrics); silent skip if unreachable."""
    try:
        import mlflow
        from run_one import DEFAULT_TRACKING_URI, DEFAULT_EXPERIMENT, sanitize_key
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        mlflow.set_experiment(DEFAULT_EXPERIMENT)
    except Exception as exc:  # noqa: BLE001
        print(f"    [info] MLflow logging skipped ({type(exc).__name__}).")
        return
    test_rows = [r for r in rows if r["cohort"] == "test"]
    for r in test_rows:
        with mlflow.start_run(run_name=f"recency-{r['model']}"):
            mlflow.log_params({"model": r["model"], "lambda": r["lambda"],
                               "n_active": r["n_active"], "cohort": "test"})
            mlflow.log_metrics({sanitize_key("mrr"): r["mrr"],
                                **{sanitize_key(f"hit_at_{k}"): r[f"hit_at_{k}"] for k in KS}})
    print("    [info] logged TEST rows to MLflow.")


if __name__ == "__main__":
    main()
