"""Human-readable simpful spec for the memory-decay FIS (REFERENCE / CROSS-CHECK ONLY).

This is the literal `simpful.FuzzySystem` the user asked for. It is NOT the production scorer:
simpful's per-sample inference API would need ~840k Mamdani evaluations per lambda candidate,
so the harness scores the (n_users x n_items) matrix with the vectorized numpy Mamdani
(`rules_recency.score_matrix_recency`) instead. This module exists so the rule base is
readable as a standard FIS and so a test can prove the numpy path agrees with a canonical
fuzzy engine on a small sample.

Fidelity to `membership.py` + `fis_score.py`:
  - Triangular medium at (q25, q50, q75); trapezoidal shoulders (low: 0,0,q25,q50 ; high:
    q50,q75,1,1) reproduce `membership.fuzzify`'s left/right shoulders exactly.
  - Fixed crisp consequents 0.1 / 0.5 / 0.9 mirror the centroid defuzz over `fis_score.CENTROIDS`
    (a weighted average of fixed points is a Sugeno-0 output).

The two engines are NOT bit-identical: the numpy path is a true Mamdani (MAX aggregation across
rules sharing an output term, then centroid defuzz), while simpful's Sugeno does a firing-
weighted SUM over all rules. They therefore differ by up to ~0.1 in absolute score but agree in
RANKING (Spearman rho ~ 0.99 on sampled cells) — and ranking is all the retrieval metric uses.
The cross-check test asserts rank agreement, not absolute equality.

`build_simpful_fis` raises ImportError if simpful is absent; callers guard on that.
"""
from __future__ import annotations

# The exact rule strings, kept next to the numpy RULE_BASE_RECENCY they mirror.
RULE_STRINGS = (
    "IF (relevance IS high) AND (recency IS high) THEN (score IS surface)",
    "IF (relevance IS high) AND (recency IS low) AND (volatility IS high) THEN (score IS suppress)",
    "IF (relevance IS high) AND (recency IS low) AND (volatility IS low) THEN (score IS surface)",
    "IF (relevance IS medium) AND (recency IS high) AND (frequency IS high) THEN (score IS surface)",
    "IF (relevance IS low) THEN (score IS suppress)",
    "IF (relevance IS medium) AND (recency IS medium) THEN (score IS neutral)",
    "IF (relevance IS high) AND (recency IS medium) AND (volatility IS medium) THEN (score IS neutral)",
    "IF (relevance IS medium) AND (recency IS low) AND (frequency IS low) THEN (score IS neutral)",
)

_VARS = ("relevance", "recency", "volatility", "frequency")


def build_simpful_fis(breakpoints: dict[str, tuple[float, float, float]]):
    """Return a `simpful.FuzzySystem` for the memory-decay FIS given per-variable breakpoints.

    `breakpoints[var] = (q25, q50, q75)` (data-driven, from history values). Raises ImportError
    if simpful is not installed.
    """
    from simpful import (FuzzySystem, FuzzySet, LinguisticVariable,
                         Triangular_MF, Trapezoidal_MF)

    FS = FuzzySystem(show_banner=False)
    for var in _VARS:
        q25, q50, q75 = breakpoints[var]
        # Clamp to keep MFs well-formed if breakpoints are (near-)degenerate.
        q25, q50, q75 = float(q25), float(max(q50, q25)), float(max(q75, q50, q25))
        low = FuzzySet(function=Trapezoidal_MF(0.0, 0.0, q25, q50), term="low")
        med = FuzzySet(function=Triangular_MF(q25, q50, q75), term="medium")
        high = FuzzySet(function=Trapezoidal_MF(q50, q75, 1.0, 1.0), term="high")
        FS.add_linguistic_variable(
            var, LinguisticVariable([low, med, high], universe_of_discourse=[0.0, 1.0]))

    # Fixed consequents == numpy centroid defuzz over CENTROIDS = [0.1, 0.5, 0.9].
    FS.set_crisp_output_value("suppress", 0.1)
    FS.set_crisp_output_value("neutral", 0.5)
    FS.set_crisp_output_value("surface", 0.9)
    FS.add_rules(list(RULE_STRINGS))
    return FS


def simpful_score(FS, cell: dict[str, float]) -> float:
    """Score one (u,i) cell {var: value} through the simpful FIS (Sugeno inference)."""
    for var in _VARS:
        FS.set_variable(var, float(cell[var]))
    out = FS.Sugeno_inference(["score"])
    return float(out["score"])
