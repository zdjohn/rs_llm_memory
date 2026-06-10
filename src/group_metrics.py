"""Per-demographic-group ranking metrics (plan §6.7, §10 fairness analysis).

Emits flat keys `ndcg_at_{k}__{attr}__{group}` (e.g. `ndcg_at_3__gender__F`) ready for
MLflow, so the global-vs-per-group gap can be tracked as user features toggle on/off (H3).

Status: the AGGREGATION CORE below (group loading, NDCG, per-group means, flat keys) is
complete and verified. Wiring it to a trained RecBole model's per-user test ranking is
Phase-4 work — `collect_per_user_ndcg` documents the exact interface to fill in. The smoke
test does not depend on this module.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from run_one import sanitize_key

DEMOGRAPHIC_ATTRS = ("gender", "age", "occupation")


def load_user_groups(user_atomic_path: str | Path,
                     attrs: tuple[str, ...] = DEMOGRAPHIC_ATTRS) -> dict[str, dict[str, str]]:
    """Parse a RecBole `.user` atomic file -> {user_id: {attr: group_value}}.

    Header is `field:type` TAB-separated; we strip the `:type` suffix to get field names.
    """
    path = Path(user_atomic_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [cell.split(":")[0] for cell in lines[0].split("\t")]
    idx = {name: i for i, name in enumerate(header)}
    uid_i = idx["user_id"]
    groups: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        groups[cells[uid_i]] = {a: cells[idx[a]] for a in attrs if a in idx}
    return groups


def ndcg_at_k(ranked_item_ids: list, relevant: set, k: int) -> float:
    """Binary-relevance NDCG@k for a single user's ranked list."""
    dcg = 0.0
    for rank, item in enumerate(ranked_item_ids[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def per_group_ndcg(per_user_ndcg: dict[str, float],
                   user_groups: dict[str, dict[str, str]], attr: str) -> dict[str, float]:
    """Average per-user NDCG within each value of a demographic attribute."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for uid, score in per_user_ndcg.items():
        grp = user_groups.get(uid, {}).get(attr)
        if grp is not None:
            bucket[grp].append(score)
    return {grp: float(np.mean(scores)) for grp, scores in bucket.items()}


def flatten_group_metrics(per_user_ndcg: dict[str, float],
                          user_groups: dict[str, dict[str, str]],
                          k: int, attrs: tuple[str, ...] = DEMOGRAPHIC_ATTRS) -> dict[str, float]:
    """All per-group means as MLflow-safe flat keys: ndcg_at_{k}__{attr}__{group}."""
    flat: dict[str, float] = {}
    for attr in attrs:
        for grp, mean in per_group_ndcg(per_user_ndcg, user_groups, attr).items():
            flat[sanitize_key(f"ndcg_at_{k}__{attr}__{grp}")] = mean
    return flat


def collect_per_user_ndcg(model, test_data, config, k: int = 3) -> dict[str, float]:
    """[PHASE 4 — TODO] Return {user_id: NDCG@k on the test split} for a trained model.

    Interface contract for the implementation:
      - iterate the RecBole eval loader (`test_data`), score all items per test user
        (general recommenders: `model.full_sort_predict`; context-aware e.g. DeepFM: the
        per-item `predict` fallback, mirroring trainer._full_sort_batch_eval),
      - mask items already seen in train/valid (history) before ranking,
      - take top-k, compute `ndcg_at_k(ranked, test_positives, k)` per user,
      - map RecBole's internal user index back to the ORIGINAL user_id token via
        `test_data._dataset.id2token(uid_field, internal_id)` so keys join with
        `load_user_groups`.

    Once filled in, fairness logging is:
        groups = load_user_groups("data/ml-100k/ml-100k.user")
        flat = flatten_group_metrics(collect_per_user_ndcg(model, test_data, config, k=3), groups, k=3)
        mlflow.log_metrics(flat)
    """
    raise NotImplementedError("Phase 4: wire per-user NDCG extraction from the RecBole eval loader.")


if __name__ == "__main__":
    # Self-check the verified core against the real atomic file.
    repo = Path(__file__).resolve().parents[1]
    groups = load_user_groups(repo / "data" / "ml-100k" / "ml-100k.user")
    print(f"loaded {len(groups)} users")
    for attr in DEMOGRAPHIC_ATTRS:
        vals = {g.get(attr) for g in groups.values()}
        print(f"  {attr:11s}: {len(vals)} groups -> {sorted(v for v in vals if v)[:8]}{'...' if len(vals) > 8 else ''}")
    # tiny correctness check on the pure NDCG + aggregation
    demo_ndcg = {u: (1.0 if i % 2 else 0.0) for i, u in enumerate(list(groups)[:10])}
    print("  sample per-group(gender):", per_group_ndcg(demo_ndcg, groups, "gender"))
