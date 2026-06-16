"""Centroid defuzzification + full score_matrix orchestration (numpy-only).

Type-1 centroid defuzzification over FIXED consequent centroids
`CENTROIDS = (avoid=0.1, neutral=0.5, recommend=0.9)`. `score_matrix` chains
membership.fuzzify -> rules.fire -> defuzz into the single (n_users, n_items) suitability
matrix the evaluator consumes. Item-only concepts broadcast across all users; user-
conditioned concepts are indexed per (user, item). A0 (`weights=None`) and A1 (fitted
weights) flow through the SAME `rules.fire`.
"""
from __future__ import annotations

import numpy as np

from fuzzy import membership, rules

# Fixed consequent centroids, aligned to rules.OUTPUTS = (avoid, neutral, recommend).
CENTROIDS = np.array([0.1, 0.5, 0.9], dtype=float)


def defuzz(firing: np.ndarray) -> np.ndarray:
    """Centroid defuzzification over CENTROIDS along the last axis.

    `sum(firing * centroids) / sum(firing)`. Where the total firing is 0 (no rule fired),
    falls back to the neutral centroid (0.5) instead of dividing by zero.
    """
    firing = np.asarray(firing, dtype=float)
    if firing.shape[-1] != CENTROIDS.shape[0]:
        raise ValueError(f"firing last axis {firing.shape[-1]} != n_centroids {CENTROIDS.shape[0]}")
    numer = np.sum(firing * CENTROIDS, axis=-1)
    denom = np.sum(firing, axis=-1)
    out = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), 0.5)
    return out


def score_matrix(concepts: dict[str, np.ndarray],
                 breakpoints: dict[str, tuple[float, float, float]],
                 weights: np.ndarray | None = None) -> np.ndarray:
    """Build the (n_users, n_items) suitability matrix in [0, 1].

    `concepts`: {name: array}. Item-only concepts are 1-D `(n_items,)`; user-conditioned
    concepts are 2-D `(n_users, n_items)`. `breakpoints`: {name: (q25,q50,q75)} from the
    train split. `weights` None/ones -> A0; fitted -> A1, both via `rules.fire`.

    Raises ValueError on any NaN in the resulting matrix.
    """
    # Infer (n_users, n_items) from a user-conditioned (2-D) concept; fall back to item-only.
    n_users = n_items = None
    for arr in concepts.values():
        a = np.asarray(arr)
        if a.ndim == 2:
            n_users, n_items = a.shape
            break
    if n_users is None:
        # All concepts item-only: a single pseudo-user row.
        n_items = int(np.asarray(next(iter(concepts.values()))).shape[-1])
        n_users = 1

    fuzzified: dict[str, np.ndarray] = {}
    for name in rules.CONCEPTS:
        if name not in concepts:
            raise ValueError(f"missing concept: {name}")
        values = np.asarray(concepts[name], dtype=float)
        memb = membership.fuzzify(values, breakpoints[name])  # (...,) + (3,)
        if values.ndim == 1:
            # Item-only (n_items, 3) -> broadcast across users -> (n_users, n_items, 3).
            memb = np.broadcast_to(memb[None, :, :], (n_users, n_items, 3))
        fuzzified[name] = memb

    firing = rules.fire(fuzzified, weights=weights)  # (n_users, n_items, 3)
    matrix = defuzz(firing)  # (n_users, n_items)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("score_matrix produced NaN/inf")
    return matrix
