"""Memory-decay concept builder — the temporal analog of `concepts.py` (numpy/scipy only).

Four HISTORY-ONLY linguistic variables for the recency FIS, all indexed by internal id
(0 = PAD, zeroed). User-conditioned concepts are (n_users, n_items); item-only concepts are
(n_items,). Nothing here reads a leave-last-1 target, so no label leaks into normalization.

  - relevance  r(u,i)  (n_users, n_items) — mean item-item co-occurrence similarity of i to
                       u's history items.  r(u,i) = mean_{j in hist(u)} cos(i, j).
  - recency    rho(u,i)(n_users, n_items) — freshest similar activity.
                       rho(u,i) = max_{j in hist(u)} cos(i, j) * exp(-lam * dt_j),
                       dt_j = (t_ref(u) - t_j) in days, t_ref(u) = u's last history ts. `lam`
                       is the tuned decay (per day); lam=0 -> time-agnostic.
  - volatility v(i)    (n_items,) — burstiness: short interaction lifespan -> volatile/faddish
                       (HIGH), lifespan spread over the window -> evergreen (LOW).
                       v(i) = 1 - lifespan(i) / max_lifespan.
  - frequency  f(i)    (n_items,) — history popularity count, min-max normalized (== Track-A
                       `mainstream_appeal`).

relevance and recency share one per-user similarity block, so the cosine work is done once.
The co-occurrence cosine is over the user-incidence of items (collaborative, no metadata).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

SECONDS_PER_DAY = 86400.0


def _validate_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"concept {name} contains NaN/inf")


def _incidence(history_pairs: np.ndarray, n_users: int, n_items: int) -> sparse.csr_matrix:
    """Binary item x user incidence (n_items, n_users) from history (deduplicated)."""
    u = history_pairs[:, 0].astype(np.int64)
    i = history_pairs[:, 1].astype(np.int64)
    data = np.ones(len(u), dtype=np.float64)
    B = sparse.coo_matrix((data, (i, u)), shape=(n_items, n_users)).tocsr()
    B.data[:] = 1.0  # dedup repeats -> presence
    return B


def _row_normalize(B: sparse.csr_matrix) -> sparse.csr_matrix:
    """L2-normalize each row; zero rows stay zero (so cosine with them is 0, not NaN)."""
    norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
    inv = np.zeros_like(norms)
    nz = norms > 0
    inv[nz] = 1.0 / norms[nz]
    return sparse.diags(inv) @ B


def build_temporal_concepts(history_pairs: np.ndarray,
                            history_ts: np.ndarray,
                            active_users: np.ndarray,
                            n_users: int,
                            n_items: int,
                            *,
                            lam: float = 0.0) -> dict[str, np.ndarray]:
    """Compute {relevance, recency, volatility, frequency} from history-only inputs.

    Parameters (internal ids, 0 = PAD):
      history_pairs : (n_hist, 2) int (user, item).
      history_ts    : (n_hist,) float unix timestamps aligned to history_pairs.
      active_users  : (n_active,) internal user ids to populate rows for (others stay 0).
      n_users, n_items : id counts including PAD.
      lam           : recency decay per DAY (>=0). 0 => recency ignores time.

    Returns the four arrays with PAD row/col zeroed. Raises ValueError on NaN/inf or shape
    problems. Vectorized per active user over a shared cosine block.
    """
    history_pairs = np.asarray(history_pairs, dtype=np.int64)
    history_ts = np.asarray(history_ts, dtype=np.float64)
    if history_pairs.ndim != 2 or history_pairs.shape[1] != 2:
        raise ValueError(f"history_pairs must be (n_hist, 2), got {history_pairs.shape}")
    if history_ts.shape[0] != history_pairs.shape[0]:
        raise ValueError("history_ts length != history_pairs rows")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")

    B = _incidence(history_pairs, n_users, n_items)      # (n_items, n_users) presence
    Bhat = _row_normalize(B)                             # cosine-ready rows

    # Per-user history rows: item ids + timestamps (keep repeats; recency's max picks freshest).
    hist_items_by_user: dict[int, list[int]] = {}
    hist_ts_by_user: dict[int, list[float]] = {}
    for (u, i), ts in zip(history_pairs, history_ts):
        hist_items_by_user.setdefault(int(u), []).append(int(i))
        hist_ts_by_user.setdefault(int(u), []).append(float(ts))

    relevance = np.zeros((n_users, n_items), dtype=np.float64)
    recency = np.zeros((n_users, n_items), dtype=np.float64)

    for u in active_users.tolist():
        items = np.asarray(hist_items_by_user.get(u, []), dtype=np.int64)
        if items.size == 0:
            continue
        tss = np.asarray(hist_ts_by_user[u], dtype=np.float64)
        # block[k, i] = cos(item_k, i) for k over u's history rows.
        block = np.asarray((Bhat[items] @ Bhat.T).todense())  # (hist_len, n_items)
        # relevance: mean similarity of each catalog item to u's history.
        relevance[u] = block.mean(axis=0)
        # recency: freshest similar activity. dt in days from u's last history timestamp.
        t_ref = tss.max()
        decay = np.exp(-lam * (t_ref - tss) / SECONDS_PER_DAY)  # (hist_len,)
        recency[u] = (block * decay[:, None]).max(axis=0)

    relevance[0, :] = 0.0
    relevance[:, 0] = 0.0
    recency[0, :] = 0.0
    recency[:, 0] = 0.0

    # --- volatility: 1 - normalized item lifespan (burst -> high, evergreen -> low). ---
    item_min = np.full(n_items, np.nan, dtype=np.float64)
    item_max = np.full(n_items, np.nan, dtype=np.float64)
    order = np.argsort(history_pairs[:, 1], kind="mergesort")
    items_sorted = history_pairs[order, 1]
    ts_sorted = history_ts[order]
    # reduceat over contiguous item groups.
    uniq, starts = np.unique(items_sorted, return_index=True)
    grp_min = np.minimum.reduceat(ts_sorted, starts)
    grp_max = np.maximum.reduceat(ts_sorted, starts)
    item_min[uniq] = grp_min
    item_max[uniq] = grp_max
    lifespan = np.where(np.isfinite(item_max), (item_max - item_min) / SECONDS_PER_DAY, np.nan)
    max_life = np.nanmax(lifespan) if np.any(np.isfinite(lifespan)) else 0.0
    if max_life > 0:
        volatility = 1.0 - np.where(np.isfinite(lifespan), lifespan / max_life, 1.0)
    else:
        volatility = np.zeros(n_items, dtype=np.float64)
    # Items with no history (target-only cold items) get 0 volatility (no signal, low/evergreen).
    volatility[~np.isfinite(lifespan)] = 0.0
    volatility[0] = 0.0

    # --- frequency: history popularity count, min-max normalized. ---
    counts = np.bincount(history_pairs[:, 1].astype(np.int64), minlength=n_items).astype(np.float64)
    fmax = counts.max() if counts.size else 0.0
    frequency = counts / fmax if fmax > 0 else np.zeros(n_items, dtype=np.float64)
    frequency[0] = 0.0

    concepts = {
        "relevance": relevance,
        "recency": recency,
        "volatility": volatility,
        "frequency": frequency,
    }
    for name, arr in concepts.items():
        _validate_finite(name, arr)
    return concepts
