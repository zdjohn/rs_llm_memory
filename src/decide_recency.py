"""PPR vs memory-decay FIS verdict on the TEST cohort (read-only; decide.py-style).

Reads results/recency_summary.csv (written by run_recency.py) and prints the comparison table
for PPR / FIS-recency-A0 / FIS-recency-A1 over the 250 TEST-cohort users at MRR + Hit@{1,3,10}.

Verdict (locked): the memory-decay FIS "wins" if EITHER A0 or A1 beats PPR on BOTH the headline
metrics — MRR and Hit@10 — on the TEST cohort. Ties or single-metric wins are WEAK.

    python src/decide_recency.py
    python src/decide_recency.py --runs_csv results/recency_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 10)
MODELS = ("PPR", "FIS-recency-A0", "FIS-recency-A1")
HEADLINE = ("mrr", "hit_at_10")
TOL = 1e-9


def decide(test: pd.DataFrame) -> str:
    """Apply the locked rule over the TEST rows. Returns PASS / WEAK PASS / FAIL / INCONCLUSIVE."""
    def val(model, metric):
        rows = test[test["model"] == model]
        if rows.empty or metric not in rows.columns:
            return None
        v = pd.to_numeric(rows[metric], errors="coerce").dropna()
        return float(v.iloc[-1]) if not v.empty else None

    ppr = {m: val("PPR", m) for m in HEADLINE}
    if any(v is None for v in ppr.values()):
        return "INCONCLUSIVE"

    best = "FAIL"
    saw = False
    for cand in ("FIS-recency-A0", "FIS-recency-A1"):
        cm = {m: val(cand, m) for m in HEADLINE}
        if any(v is None for v in cm.values()):
            continue
        saw = True
        ge = all(cm[m] >= ppr[m] - TOL for m in HEADLINE)
        if ge:
            strictly = any(cm[m] > ppr[m] + TOL for m in HEADLINE)
            if strictly:
                return "PASS"
            best = "WEAK PASS"
    return best if saw else "INCONCLUSIVE"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_csv", default=str(REPO_ROOT / "results" / "recency_summary.csv"))
    args = p.parse_args()

    path = Path(args.runs_csv)
    if not path.exists():
        raise SystemExit(f"summary CSV not found: {path}. Run src/run_recency.py first.")
    df = pd.read_csv(path)
    test = df[df["cohort"] == "test"]

    cols = ["mrr", *(f"hit_at_{k}" for k in KS)]
    print("TEST-cohort comparison (250 users, expected metrics under tie-breaking):")
    header = f"  {'model':16s}" + "".join(f"{c:>10s}" for c in cols)
    print(header)
    for model in MODELS:
        rows = test[test["model"] == model]
        if rows.empty:
            print(f"  {model:16s}" + "".join(f"{'n/a':>10s}" for _ in cols))
            continue
        r = rows.iloc[-1]
        print(f"  {model:16s}" + "".join(f"{float(r[c]):>10.4f}" for c in cols))

    print(f"\nHeadline gate: FIS beats PPR on BOTH {HEADLINE}")
    print(f"VERDICT: {decide(test)}")


if __name__ == "__main__":
    main()
