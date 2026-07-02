"""Memory-decay Mamdani rule base — the temporal analog of `rules.py` (numpy-only).

`rules.py` hardcodes `CONCEPTS/OUTPUTS/TERMS` as module constants and `fis_score.score_matrix`
iterates `rules.CONCEPTS`, so a second rule base cannot be hosted by editing the frozen Track-A
code without rippling into 5 tests. This parallel module carries its OWN constants, a
`fire_recency` that is a faithful copy of `rules.fire` parameterized over them, and a
`score_matrix_recency` mirroring `fis_score.score_matrix`. It reuses `membership.fuzzify` and
`fis_score.defuzz`/`CENTROIDS` verbatim, so the Mamdani semantics (triangular MFs, PRODUCT
t-norm, MAX aggregation, centroid defuzz over 0.1/0.5/0.9) are identical to Track A.

Linguistic variables: relevance, recency, volatility, frequency (each low/medium/high).
Outputs: suppress(0.1), neutral(0.5), surface(0.9) — same centroids as {avoid,neutral,recommend}.
"""
from __future__ import annotations

import numpy as np

from fuzzy import fis_score, membership

CONCEPTS_RECENCY = ("relevance", "recency", "volatility", "frequency")
TERMS = ("low", "medium", "high")
OUTPUTS = ("suppress", "neutral", "surface")  # aligned to fis_score.CENTROIDS = [0.1, 0.5, 0.9]

_TERM_IDX = {t: i for i, t in enumerate(TERMS)}
_OUT_IDX = {o: i for i, o in enumerate(OUTPUTS)}

# The memory-decay rule base (plan §"Rule base"). Each rule: (antecedent {concept: term},
# consequent output term). An omitted concept is "don't care" (factor 1.0), exactly as in
# rules.fire. R2 vs R3 is the whole point: an old-but-volatile relevant item sinks while an
# old-but-stable (evergreen) relevant item survives — something no single linear (α,β,γ) can do.
RULE_BASE_RECENCY: tuple[tuple[dict[str, str], str], ...] = (
    # R1: relevant AND fresh -> surface.
    ({"relevance": "high", "recency": "high"}, "surface"),
    # R2: relevant but stale AND volatile -> suppress (faded, time-sensitive).
    ({"relevance": "high", "recency": "low", "volatility": "high"}, "suppress"),
    # R3: relevant but stale AND stable -> surface (evergreen survives).
    ({"relevance": "high", "recency": "low", "volatility": "low"}, "surface"),
    # R4: moderately relevant, fresh, frequently accessed -> surface.
    ({"relevance": "medium", "recency": "high", "frequency": "high"}, "surface"),
    # R5: not relevant -> suppress.
    ({"relevance": "low"}, "suppress"),
    # Neutral filler (avoid all-zero firing on the mid manifold).
    ({"relevance": "medium", "recency": "medium"}, "neutral"),
    ({"relevance": "high", "recency": "medium", "volatility": "medium"}, "neutral"),
    ({"relevance": "medium", "recency": "low", "frequency": "low"}, "neutral"),
)

N_RULES = len(RULE_BASE_RECENCY)


def _validate_rule_base() -> None:
    for antecedent, consequent in RULE_BASE_RECENCY:
        if consequent not in _OUT_IDX:
            raise ValueError(f"undefined consequent term: {consequent}")
        for concept, term in antecedent.items():
            if concept not in CONCEPTS_RECENCY:
                raise ValueError(f"undefined concept in recency rule base: {concept}")
            if term not in _TERM_IDX:
                raise ValueError(f"undefined antecedent term: {term}")


_validate_rule_base()


def fire_recency(memberships: dict[str, np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """Fire `RULE_BASE_RECENCY` over `memberships` -> firing strengths per output term.

    Faithful copy of `rules.fire` over the local constants: `memberships` = {concept: (...,3)}
    over {low,medium,high}; returns `S + (3,)` over {suppress,neutral,surface}. PRODUCT t-norm
    for the antecedent, MAX aggregation across rules mapping to the same output. `weights`
    (length N_RULES) scales each rule; None/ones -> A0.
    """
    for concept in CONCEPTS_RECENCY:
        if concept not in memberships:
            raise ValueError(f"missing concept membership: {concept}")
        last = np.asarray(memberships[concept]).shape[-1]
        if last != len(TERMS):
            raise ValueError(f"concept {concept} membership last axis {last} != {len(TERMS)}")

    if weights is None:
        w = np.ones(N_RULES, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (N_RULES,):
            raise ValueError(f"weights length {w.shape} != n_rules ({N_RULES},)")

    lead_shape = np.broadcast_shapes(*(memberships[c][..., 0].shape for c in CONCEPTS_RECENCY))
    firing = np.zeros(lead_shape + (len(OUTPUTS),), dtype=float)

    for rule_idx, (antecedent, consequent) in enumerate(RULE_BASE_RECENCY):
        strength = np.ones(lead_shape, dtype=float)
        for concept, term in antecedent.items():
            strength = strength * memberships[concept][..., _TERM_IDX[term]]
        strength = strength * w[rule_idx]
        out_idx = _OUT_IDX[consequent]
        firing[..., out_idx] = np.maximum(firing[..., out_idx], strength)

    if not np.all(np.isfinite(firing)):
        raise ValueError("fire_recency produced non-finite firing strengths")
    return firing


def score_matrix_recency(concepts: dict[str, np.ndarray],
                         breakpoints: dict[str, tuple[float, float, float]],
                         weights: np.ndarray | None = None) -> np.ndarray:
    """Build the (n_users, n_items) suitability matrix in [0, 1] for the recency FIS.

    Mirrors `fis_score.score_matrix` but iterates `CONCEPTS_RECENCY` and fires the recency
    rule base. Item-only concepts (1-D) broadcast across users; user-conditioned concepts are
    2-D. Reuses `membership.fuzzify` and `fis_score.defuzz` verbatim. `weights` None/ones -> A0.
    """
    n_users = n_items = None
    for arr in concepts.values():
        a = np.asarray(arr)
        if a.ndim == 2:
            n_users, n_items = a.shape
            break
    if n_users is None:
        n_items = int(np.asarray(next(iter(concepts.values()))).shape[-1])
        n_users = 1

    fuzzified: dict[str, np.ndarray] = {}
    for name in CONCEPTS_RECENCY:
        if name not in concepts:
            raise ValueError(f"missing concept: {name}")
        values = np.asarray(concepts[name], dtype=float)
        memb = membership.fuzzify(values, breakpoints[name])  # (...,) + (3,)
        if values.ndim == 1:
            memb = np.broadcast_to(memb[None, :, :], (n_users, n_items, 3))
        fuzzified[name] = memb

    firing = fire_recency(fuzzified, weights=weights)  # (n_users, n_items, 3)
    matrix = fis_score.defuzz(firing)  # (n_users, n_items)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("score_matrix_recency produced NaN/inf")
    return matrix
