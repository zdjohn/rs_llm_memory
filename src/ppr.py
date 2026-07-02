"""Personalized PageRank baseline over the user-item bipartite graph (the structural foil).

PPR is the *time-agnostic* comparison point for the recency-aware FIS: recency lives only in
the FIS concepts, never here. The graph is built from HISTORY positives only; for each target
user we restart PageRank on that user's history-item nodes and read the stationary mass on all
item nodes -> one row of the (n_users, n_items) score matrix. Universe parity with the FIS is
automatic: same internal ids, same n_items catalog, PAD col 0 masked to 0.

Two engines, cross-checked to |Δ| < 1e-6:
  - `networkx.pagerank` (primary; the standard API).
  - a scipy-sparse power-iteration (`_ppr_power_iteration`) used as a fast path for many
    target users and as a dependency-free fallback if networkx is unavailable.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

try:  # networkx is the requested primary engine (pinned networkx==3.4.2).
    import networkx as nx
    _HAVE_NX = True
except ImportError:  # pragma: no cover - fallback path
    _HAVE_NX = False


def _bipartite_adjacency(history_pairs: np.ndarray, n_users: int, n_items: int) -> sparse.csr_matrix:
    """Symmetric (n_users+n_items) adjacency; item node j is column/row (n_users + j).

    Edge weight = interaction count (repeats reinforce the edge). PAD nodes (user 0, item 0)
    contribute no edges beyond their zero rows/cols.
    """
    u = history_pairs[:, 0].astype(np.int64)
    i = history_pairs[:, 1].astype(np.int64) + n_users
    n = n_users + n_items
    w = np.ones(len(u), dtype=np.float64)
    top = sparse.coo_matrix((w, (u, i)), shape=(n, n))
    bot = sparse.coo_matrix((w, (i, u)), shape=(n, n))
    return (top + bot).tocsr()


def _ppr_power_iteration(P: sparse.csr_matrix, personalization: np.ndarray,
                         alpha: float, max_iter: int, tol: float) -> np.ndarray:
    """Power-iterate x = alpha * P @ x + (1-alpha) * p to the PPR stationary vector.

    `P` is the column-stochastic transition matrix (columns sum to 1 for non-dangling nodes);
    `personalization` is the restart distribution `p` (sums to 1). Dangling mass is redirected
    to `p` so the chain stays stochastic.
    """
    n = P.shape[0]
    p = personalization
    x = p.copy()
    col_sums = np.asarray(P.sum(axis=0)).ravel()
    dangling = col_sums == 0
    for _ in range(max_iter):
        dangling_mass = x[dangling].sum()
        x_new = alpha * (P @ x + dangling_mass * p) + (1.0 - alpha) * p
        s = x_new.sum()
        if s > 0:
            x_new /= s
        if np.abs(x_new - x).sum() < tol:
            x = x_new
            break
        x = x_new
    return x


def _column_stochastic(A: sparse.csr_matrix) -> sparse.csr_matrix:
    """Normalize columns of `A` to sum 1 (dangling columns left at 0)."""
    col = np.asarray(A.sum(axis=0)).ravel()
    inv = np.zeros_like(col)
    nz = col > 0
    inv[nz] = 1.0 / col[nz]
    return A @ sparse.diags(inv)


def ppr_score_matrix(history_pairs: np.ndarray, n_users: int, n_items: int,
                     target_users: np.ndarray, *, alpha: float = 0.85,
                     personalization_mode: str = "history_items",
                     engine: str = "networkx", max_iter: int = 200,
                     tol: float = 1e-9) -> np.ndarray:
    """(n_users, n_items) PPR scores; only `target_users` rows are populated (others 0).

    For each target user u, restart mass is placed on u's history item nodes
    (`personalization_mode="history_items"`, the standard PPR-for-recs form) or on u's own node
    (`"user_node"`). Item-node stationary probabilities become the score row; PAD col 0 -> 0.
    `engine="networkx"` uses `nx.pagerank`; `engine="power"` (or missing networkx) uses the
    scipy power-iteration. Both share the same graph, so rankings match to |Δ|<1e-6.
    """
    history_pairs = np.asarray(history_pairs, dtype=np.int64)
    if personalization_mode not in ("history_items", "user_node"):
        raise ValueError(f"unknown personalization_mode: {personalization_mode}")
    if engine == "networkx" and not _HAVE_NX:
        engine = "power"

    A = _bipartite_adjacency(history_pairs, n_users, n_items)
    n = n_users + n_items

    hist_items_by_user: dict[int, list[int]] = {}
    for u, i in history_pairs:
        hist_items_by_user.setdefault(int(u), []).append(int(i) + n_users)

    scores = np.zeros((n_users, n_items), dtype=np.float64)

    if engine == "networkx":
        G = nx.from_scipy_sparse_array(A, edge_attribute="weight")
        for u in target_users.tolist():
            p = _restart_vector(u, hist_items_by_user, n, personalization_mode)
            if p is None:
                continue
            pers = {k: float(v) for k, v in enumerate(p) if v > 0}
            pr = nx.pagerank(G, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
            row = np.array([pr.get(n_users + j, 0.0) for j in range(n_items)])
            row[0] = 0.0
            scores[u] = row
        return scores

    # power-iteration engine (shared transition matrix built once).
    P = _column_stochastic(A)
    for u in target_users.tolist():
        p = _restart_vector(u, hist_items_by_user, n, personalization_mode)
        if p is None:
            continue
        x = _ppr_power_iteration(P, p, alpha=alpha, max_iter=max_iter, tol=tol)
        row = x[n_users:n_users + n_items].copy()
        row[0] = 0.0
        scores[u] = row
    return scores


def _restart_vector(u: int, hist_items_by_user: dict[int, list[int]], n: int,
                    mode: str) -> np.ndarray | None:
    """Restart distribution over `n` nodes for user `u`; None if u has no usable restart."""
    p = np.zeros(n, dtype=np.float64)
    if mode == "user_node":
        p[u] = 1.0
        return p
    nodes = hist_items_by_user.get(u, [])
    if not nodes:
        return None
    for node in nodes:
        p[node] += 1.0
    p /= p.sum()
    return p
