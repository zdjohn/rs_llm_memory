"""Mamdani rule firing: PRODUCT t-norm antecedent, MAX aggregation (numpy-only).

The locked rule base (`RULE_BASE`) maps antecedent term-combinations over the 4 concepts
to one of the output terms {avoid, neutral, recommend}. `fire` is the single shared seam:

    weights = None / ones  -> A0 (frozen)
    weights = fitted vector -> A1

Both modes run the IDENTICAL arithmetic; `weights` only scales each rule's firing strength
in the same code path (no shadow / duplicate firing logic). A1 is therefore a faithful
rehearsal of A0 with learned weights.

Concept order (fixed): mainstream_appeal, niche_factor, era_fit, genre_match.
Term order (fixed):    low(0), medium(1), high(2).
Output order (fixed):  avoid(0), neutral(1), recommend(2).
"""
from __future__ import annotations

import numpy as np

CONCEPTS = ("mainstream_appeal", "niche_factor", "era_fit", "genre_match")
TERMS = ("low", "medium", "high")
OUTPUTS = ("avoid", "neutral", "recommend")

_TERM_IDX = {t: i for i, t in enumerate(TERMS)}
_OUT_IDX = {o: i for i, o in enumerate(OUTPUTS)}

# Locked rule base. Each rule: (antecedent dict {concept: term}, consequent output term).
# An antecedent that omits a concept treats that concept as "don't care" (factor 1.0).
# Designed headphones-style: strong genre/era fit pushes "recommend"; weak fit + low
# mainstream appeal pushes "avoid"; the rest land "neutral".
RULE_BASE: tuple[tuple[dict[str, str], str], ...] = (
    # Strong personal fit -> recommend.
    ({"genre_match": "high", "era_fit": "high"}, "recommend"),
    ({"genre_match": "high", "mainstream_appeal": "high"}, "recommend"),
    ({"genre_match": "high", "era_fit": "medium"}, "recommend"),
    ({"era_fit": "high", "niche_factor": "high"}, "recommend"),
    # Ambiguous / partial fit -> neutral.
    ({"genre_match": "medium"}, "neutral"),
    ({"era_fit": "medium", "genre_match": "medium"}, "neutral"),
    ({"mainstream_appeal": "high", "genre_match": "low"}, "neutral"),
    # Poor fit -> avoid.
    ({"genre_match": "low", "era_fit": "low"}, "avoid"),
    ({"genre_match": "low", "mainstream_appeal": "low"}, "avoid"),
    ({"niche_factor": "high", "genre_match": "low"}, "avoid"),
)

N_RULES = len(RULE_BASE)


def _validate_rule_base() -> None:
    for antecedent, consequent in RULE_BASE:
        if consequent not in _OUT_IDX:
            raise ValueError(f"undefined consequent term: {consequent}")
        for concept, term in antecedent.items():
            if concept not in CONCEPTS:
                raise ValueError(f"undefined concept in rule base: {concept}")
            if term not in _TERM_IDX:
                raise ValueError(f"undefined antecedent term: {term}")


_validate_rule_base()


def fire(memberships: dict[str, np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """Fire the rule base over `memberships` -> firing strengths per output term.

    `memberships`: {concept_name: array(..., 3)} where the last axis is {low, medium, high}.
    All concept arrays must broadcast to a common leading shape `S`.
    Returns array of shape `S + (3,)` over {avoid, neutral, recommend}.

    PRODUCT t-norm for the antecedent conjunction; MAX aggregation across all rules that
    map to the same output term. `weights` (length N_RULES) scales each rule's firing
    strength; None / ones reproduce the unweighted (A0) firing exactly.

    Raises ValueError on weight-length mismatch or an undefined concept/term.
    """
    for concept in CONCEPTS:
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

    # Common broadcast shape across all concept membership arrays (drop the term axis).
    lead_shape = np.broadcast_shapes(*(memberships[c][..., 0].shape for c in CONCEPTS))
    firing = np.zeros(lead_shape + (len(OUTPUTS),), dtype=float)

    for rule_idx, (antecedent, consequent) in enumerate(RULE_BASE):
        strength = np.ones(lead_shape, dtype=float)
        for concept, term in antecedent.items():
            strength = strength * memberships[concept][..., _TERM_IDX[term]]
        strength = strength * w[rule_idx]
        out_idx = _OUT_IDX[consequent]
        firing[..., out_idx] = np.maximum(firing[..., out_idx], strength)

    if not np.all(np.isfinite(firing)):
        raise ValueError("fire produced non-finite firing strengths")
    return firing
