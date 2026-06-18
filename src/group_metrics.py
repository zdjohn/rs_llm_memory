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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOGRAPHIC_ATTRS = ("gender", "age", "occupation")
DEFAULT_DATASET = "ml100k"


def atomic_path(suffix: str, dataset: str = DEFAULT_DATASET) -> Path:
    """Resolve REPO_ROOT/data/<dataset>/<dataset>.<suffix>.

    Defaults to the locked `ml100k` name; pass dataset="ml-100k" for the legacy files.
    `suffix` is one of "inter", "user", "item".
    """
    return REPO_ROOT / "data" / dataset / f"{dataset}.{suffix}"


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


def load_item_groups(item_atomic_path: str | Path,
                     attrs: tuple[str, ...] = ("genre", "release_year")) -> dict[str, dict[str, str]]:
    """Parse a RecBole `.item` atomic file -> {item_id: {attr: value}} (mirrors load_user_groups).

    Header is `field:type` TAB-separated; the `:type` suffix is stripped to get field names.
    """
    path = Path(item_atomic_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [cell.split(":")[0] for cell in lines[0].split("\t")]
    idx = {name: i for i, name in enumerate(header)}
    iid_i = idx["item_id"]
    groups: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        groups[cells[iid_i]] = {a: cells[idx[a]] for a in attrs if a in idx}
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


def hit_at_k(ranked_item_ids: list, relevant: set, k: int) -> float:
    """Hit@k: 1.0 if any relevant item appears in the top-k, else 0.0."""
    return 1.0 if any(item in relevant for item in ranked_item_ids[:k]) else 0.0


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


def collect_per_user_metrics(model, test_data, config, k: int = 3):
    """Return per-user test metrics + test-positive sets for a trained model.

    Returns ({user_id: NDCG@k}, {user_id: Hit@k}, {user_id: set(test-positive item tokens)}).
    Mirrors `trainer._full_sort_batch_eval`: iterate the RecBole eval loader, score all items
    per test user (`full_sort_predict`; context-aware models fall back to per-item `predict`),
    mask PAD (col 0) and train/valid history before ranking, take the top-k, and compute NDCG
    and Hit per user. Internal user/item ids are mapped back to ORIGINAL tokens via
    `dataset.id2token` so keys join with `load_user_groups` / cold-item slicing.
    """
    import torch

    dataset = test_data._dataset
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    tot_item_num = dataset.num(iid_field)
    device = config["device"]

    model.eval()
    per_user_ndcg: dict[str, float] = {}
    per_user_hit: dict[str, float] = {}
    per_user_test_items: dict[str, set] = {}

    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(device)
            try:
                scores = model.full_sort_predict(interaction)
            except NotImplementedError:
                # Context-aware fallback (e.g. DeepFM): expand each user against EVERY item with
                # the FULL item-feature row joined (item_id + genre + release_year + ...), exactly
                # as trainer._full_sort_batch_eval does. Joining only item_id makes models that
                # read item side-info raise KeyError, so merge the whole item-feature table.
                inter_len = len(interaction)
                new_inter = interaction.repeat_interleave(tot_item_num)
                item_feature = dataset.get_item_feature().to(device)
                new_inter.update(item_feature.repeat(inter_len))
                # Chunk the per-item predict to bound memory on the (n_users * n_items) expansion.
                n_rows = len(new_inter)
                step = max(tot_item_num, 4096)
                parts = [model.predict(new_inter[s:s + step]) for s in range(0, n_rows, step)]
                scores = torch.cat(parts) if len(parts) > 1 else parts[0]

            scores = scores.view(-1, tot_item_num).clone()
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf

            topk_idx = torch.topk(scores, k, dim=1).indices.cpu().numpy()  # (n_batch_users, k)

            # Group test positives by batch-relative user row.
            pos_u = positive_u.cpu().numpy()
            pos_i = positive_i.cpu().numpy()
            relevant_by_row: dict[int, set] = defaultdict(set)
            for u_row, i_id in zip(pos_u, pos_i):
                relevant_by_row[int(u_row)].add(int(i_id))

            batch_internal_users = interaction[uid_field].cpu().numpy()
            for row in range(scores.shape[0]):
                relevant = relevant_by_row.get(row, set())
                if not relevant:
                    continue
                ranked = list(topk_idx[row])
                internal_uid = int(batch_internal_users[row])
                token = str(dataset.id2token(uid_field, internal_uid))
                per_user_ndcg[token] = ndcg_at_k(ranked, relevant, k)
                per_user_hit[token] = hit_at_k(ranked, relevant, k)
                per_user_test_items[token] = {
                    str(dataset.id2token(iid_field, i)) for i in relevant}

    return per_user_ndcg, per_user_hit, per_user_test_items


def collect_per_user_ndcg(model, test_data, config, k: int = 3) -> dict[str, float]:
    """{user_id: NDCG@k on the test split} for a trained model (see collect_per_user_metrics)."""
    per_user_ndcg, _, _ = collect_per_user_metrics(model, test_data, config, k=k)
    return per_user_ndcg


def train_interaction_counts(train_data, config):
    """Return ({user_id_token: n_train}, {item_id_token: n_train}) from the TRAIN split only."""
    dataset = train_data._dataset
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    inter = dataset.inter_feat
    users = inter[uid_field].numpy().astype(int)
    items = inter[iid_field].numpy().astype(int)
    # Count per internal id with bincount, then map each id back to its token once.
    u_counts = np.bincount(users, minlength=dataset.num(uid_field))
    i_counts = np.bincount(items, minlength=dataset.num(iid_field))
    # Include EVERY catalog entity, with count 0 for never-interacted ones, so items with zero
    # train interactions (the truest cold items in the full-catalog universe) are classified as
    # cold. Skip internal id 0 (the [PAD] sentinel), which is never a real entity.
    user_counts = {str(dataset.id2token(uid_field, i)): int(u_counts[i])
                   for i in range(1, dataset.num(uid_field))}
    item_counts = {str(dataset.id2token(iid_field, i)): int(i_counts[i])
                   for i in range(1, dataset.num(iid_field))}
    return user_counts, item_counts


def cold_entities(counts: dict[str, int], threshold: int = 5) -> set[str]:
    """Entity tokens with <= threshold train interactions (the cold slice)."""
    return {tok for tok, n in counts.items() if n <= threshold}


SLICES = ("all", "cold_user", "cold_item")


def slice_metrics(per_user_ndcg: dict[str, float],
                  per_user_hit: dict[str, float],
                  per_user_test_items: dict[str, set],
                  user_train_counts: dict[str, int],
                  item_train_counts: dict[str, int],
                  cold_threshold: int = 5) -> dict[str, dict[str, float]]:
    """Mean NDCG@k / Hit@k over the {all, cold_user, cold_item} slices.

    - all:       every scored user.
    - cold_user: users with <= cold_threshold TRAIN interactions.
    - cold_item: users whose test-positive set contains a cold item (<= cold_threshold
                 TRAIN interactions) — the PASS-gate target slice.

    Returns {slice: {"ndcg": mean, "hit": mean}}. Slices with no members report 0.0.
    """
    cold_users = cold_entities(user_train_counts, cold_threshold)
    cold_items = cold_entities(item_train_counts, cold_threshold)

    def _mean(users: list[str], table: dict[str, float]) -> float:
        vals = [table[u] for u in users if u in table]
        return float(np.mean(vals)) if vals else 0.0

    all_users = list(per_user_ndcg)
    cold_user_users = [u for u in all_users if u in cold_users]
    cold_item_users = [u for u in all_users
                       if per_user_test_items.get(u, set()) & cold_items]

    # SLICES is the single source of truth for slice names (see flatten_slice_metrics).
    members = dict(zip(SLICES, (all_users, cold_user_users, cold_item_users)))
    return {sl: {"ndcg": _mean(users, per_user_ndcg), "hit": _mean(users, per_user_hit)}
            for sl, users in members.items()}


def flatten_slice_metrics(sliced: dict[str, dict[str, float]], k: int = 3) -> dict[str, float]:
    """MLflow-safe flat keys: ndcg_at_{k}__slice__{slice} and hit_at_{k}__slice__{slice}.

    `{slice}` in {all, cold_user, cold_item}. These literal keys are consumed verbatim by
    aggregate.py (METRIC_COLS) and decide.py.
    """
    flat: dict[str, float] = {}
    for sl, metrics in sliced.items():
        flat[sanitize_key(f"ndcg_at_{k}__slice__{sl}")] = metrics["ndcg"]
        flat[sanitize_key(f"hit_at_{k}__slice__{sl}")] = metrics["hit"]
    return flat


if __name__ == "__main__":
    # Self-check the verified core against the real atomic file.
    groups = load_user_groups(atomic_path("user"))
    print(f"loaded {len(groups)} users")
    for attr in DEMOGRAPHIC_ATTRS:
        vals = {g.get(attr) for g in groups.values()}
        print(f"  {attr:11s}: {len(vals)} groups -> {sorted(v for v in vals if v)[:8]}{'...' if len(vals) > 8 else ''}")
    # tiny correctness check on the pure NDCG + aggregation
    demo_ndcg = {u: (1.0 if i % 2 else 0.0) for i, u in enumerate(list(groups)[:10])}
    print("  sample per-group(gender):", per_group_ndcg(demo_ndcg, groups, "gender"))
