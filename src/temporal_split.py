"""Cohort split + per-user temporal leave-last-1 for the PPR-vs-memory-decay experiment.

Self-contained (numpy/pandas only — NO RecBole): the recency comparison uses its own
contiguous id maps built directly from a RecBole atomic `.inter` file, so it can express
the two things RecBole's global splitter cannot — a *user cohort* split (whole users held
out for reporting) and a *per-user temporal* leave-last-1 target. This is Option E1 in the
plan; RecBole/`FISRecommender` are only used by the optional adapter sanity gate.

Id convention (matches `src/fuzzy/concepts.py`): internal id 0 = `[PAD]`, real entities are
1-based. `n_users`/`n_items` include the PAD slot. Every array/matrix downstream is indexed
by these internal ids and its PAD row/col is zeroed.

The two-level split (plan §"Two-level split"):
  - Level 1: the top-`n_active` users by #positives -> random 50/50 VAL / TEST cohorts.
  - Level 2: per user, order positives by timestamp; the single most-recent positive is the
    held-out TARGET; all earlier positives are that user's HISTORY.
Global statistics (breakpoints, volatility, popularity, the co-occurrence graph) are built
from HISTORY only, so no target ever leaks into concept construction or PPR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAD_TOKEN = "[PAD]"


@dataclass
class RecencySplit:
    """Everything the recency pipeline needs, all indexed by internal id (0 = PAD)."""

    n_users: int
    n_items: int
    user_id2token: list[str]
    item_id2token: list[str]
    user_token2id: dict[str, int]
    item_token2id: dict[str, int]
    history_pairs: np.ndarray            # (n_hist, 2) int internal (user, item)
    history_ts: np.ndarray               # (n_hist,) float timestamps aligned to history_pairs
    targets: dict[int, int]              # {internal_user: internal_target_item} (leave-last-1)
    active_users: np.ndarray             # (n_active,) internal user ids, sorted
    val_users: np.ndarray                # (n_active//2,) internal user ids (tuning cohort)
    test_users: np.ndarray               # (n_active//2,) internal user ids (reporting cohort)
    user_history_items: dict[int, set] = field(default_factory=dict)  # internal u -> set(internal i)


def load_positives(inter_path: str | Path, *, pos_threshold: float = 4.0) -> pd.DataFrame:
    """Read a RecBole atomic `.inter` file -> DataFrame of positives (rating >= threshold).

    Columns are `field:type` TAB-separated; we strip the `:type` suffix. Requires the four
    fields user_id, item_id, rating, timestamp. Returns columns [user, item, rating, ts]
    with native dtypes (user/item as strings/tokens, rating/ts as float).
    """
    path = Path(inter_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [cell.split(":")[0] for cell in lines[0].split("\t")]
    idx = {name: i for i, name in enumerate(header)}
    for req in ("user_id", "item_id", "rating", "timestamp"):
        if req not in idx:
            raise ValueError(f"{path} missing required field '{req}' (has {header})")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        c = line.split("\t")
        rating = float(c[idx["rating"]])
        if rating < pos_threshold:
            continue
        rows.append((c[idx["user_id"]], c[idx["item_id"]], rating, float(c[idx["timestamp"]])))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    if df.empty:
        raise ValueError(f"no positives (rating >= {pos_threshold}) in {path}")
    return df


def top_active_users(df: pd.DataFrame, n_active: int = 500) -> list[str]:
    """Return the `n_active` most-active user tokens by #positives.

    Deterministic tie-break `count desc, user_id asc` (ties exist at the cutoff, so a stable
    order matters for reproducibility). Raises if fewer than `n_active` users are available.
    """
    counts = df.groupby("user").size()
    n_users = int(counts.shape[0])
    if n_users < n_active:
        raise ValueError(f"only {n_users} users with positives; need n_active={n_active}. "
                         "Pick a denser Amazon category or lower n_active.")
    ranked = counts.reset_index(name="n")
    ranked = ranked.sort_values(["n", "user"], ascending=[False, True], kind="mergesort")
    return ranked["user"].tolist()[:n_active]


def cohort_split(active_users: list[str], *, seed: int = 2020) -> tuple[list[str], list[str]]:
    """Randomly partition the active users 50/50 into (val_cohort, test_cohort).

    The odd user (if `len` is odd) goes to the test cohort. Deterministic under `seed`.
    """
    rng = np.random.default_rng(seed)
    order = np.array(active_users, dtype=object)
    perm = rng.permutation(len(order))
    shuffled = order[perm]
    half = len(shuffled) // 2
    val = sorted(shuffled[:half].tolist())
    test = sorted(shuffled[half:].tolist())
    return val, test


def build_split(inter_path: str | Path, *, n_active: int = 500, seed: int = 2020,
                pos_threshold: float = 4.0) -> RecencySplit:
    """Build the full `RecencySplit` from an atomic `.inter` file.

    Steps: load positives -> pick top-`n_active` users -> per-user temporal leave-last-1
    (target = most-recent positive; history = earlier positives) -> contiguous internal id
    maps over the users and the items appearing in HISTORY ∪ TARGETS (so every target is at
    least rankable) -> cohort split. Enforces the leakage invariant `target_ts > max(history_ts)`.

    A user with only ONE positive has no history and is dropped from the active set with a
    top-up from the next-most-active user, so exactly `n_active` users with >=1 history item
    are returned.
    """
    df = load_positives(inter_path, pos_threshold=pos_threshold)

    # Rank ALL users once; walk down the list taking users that have >=2 positives (need >=1
    # history + 1 target) until we have n_active of them (handles the single-positive dropouts).
    counts = df.groupby("user").size()
    ranked = (counts.reset_index(name="n")
              .sort_values(["n", "user"], ascending=[False, True], kind="mergesort"))
    eligible = ranked[ranked["n"] >= 2]["user"].tolist()
    if len(eligible) < n_active:
        raise ValueError(f"only {len(eligible)} users have >=2 positives; need {n_active}.")
    active_tokens = eligible[:n_active]
    active_set = set(active_tokens)

    sub = df[df["user"].isin(active_set)].copy()
    # Stable sort by (user, ts) so the last row per user is the most-recent positive. A
    # secondary key on item keeps ties deterministic.
    sub = sub.sort_values(["user", "ts", "item"], kind="mergesort")

    # Per-user leave-last-1.
    history_rows: list[tuple[str, str, float]] = []
    target_tok: dict[str, str] = {}
    for user, g in sub.groupby("user", sort=False):
        items = g["item"].tolist()
        tss = g["ts"].tolist()
        # target = last (most-recent). history = all earlier.
        target_tok[user] = items[-1]
        max_hist_ts = tss[-2]
        if not (tss[-1] >= max_hist_ts):  # temporal invariant (>= because equal ts allowed)
            raise AssertionError(f"user {user}: target ts {tss[-1]} < max history ts {max_hist_ts}")
        for it, ts in zip(items[:-1], tss[:-1]):
            history_rows.append((user, it, ts))

    # Contiguous internal id maps (0 = PAD). Users: the active tokens. Items: HISTORY ∪ TARGETS.
    user_id2token = [PAD_TOKEN] + sorted(active_tokens)
    hist_items = {it for _, it, _ in history_rows}
    tgt_items = set(target_tok.values())
    item_id2token = [PAD_TOKEN] + sorted(hist_items | tgt_items)
    user_token2id = {t: i for i, t in enumerate(user_id2token)}
    item_token2id = {t: i for i, t in enumerate(item_id2token)}

    n_users = len(user_id2token)
    n_items = len(item_id2token)

    history_pairs = np.array(
        [(user_token2id[u], item_token2id[it]) for u, it, _ in history_rows], dtype=np.int64)
    history_ts = np.array([ts for _, _, ts in history_rows], dtype=np.float64)

    targets = {user_token2id[u]: item_token2id[it] for u, it in target_tok.items()}

    user_history_items: dict[int, set] = {}
    for u_int, i_int in history_pairs:
        user_history_items.setdefault(int(u_int), set()).add(int(i_int))

    active_users = np.array(sorted(user_token2id[t] for t in active_tokens), dtype=np.int64)
    val_tok, test_tok = cohort_split(active_tokens, seed=seed)
    val_users = np.array(sorted(user_token2id[t] for t in val_tok), dtype=np.int64)
    test_users = np.array(sorted(user_token2id[t] for t in test_tok), dtype=np.int64)

    return RecencySplit(
        n_users=n_users, n_items=n_items,
        user_id2token=user_id2token, item_id2token=item_id2token,
        user_token2id=user_token2id, item_token2id=item_token2id,
        history_pairs=history_pairs, history_ts=history_ts, targets=targets,
        active_users=active_users, val_users=val_users, test_users=test_users,
        user_history_items=user_history_items,
    )


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "data/ml100k/ml100k.inter")
    split = build_split(p, n_active=500)
    print(f"users(+PAD)={split.n_users} items(+PAD)={split.n_items} "
          f"history={len(split.history_pairs)} targets={len(split.targets)} "
          f"val={len(split.val_users)} test={len(split.test_users)}")
