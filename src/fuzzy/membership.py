"""Vectorized triangular membership functions + data-driven breakpoints (numpy-only).

The ONLY place membership-function breakpoints are set is `compute_breakpoints`, which
reads the 25/50/75 percentiles off the supplied (TRAIN-SPLIT-ONLY) values. Breakpoints are
never hand-set. `fuzzify` is a pure stateless function — A0 (frozen weights) and A1 (fitted
weights) fuzzify identically; only the downstream rule firing differs.

Terms (last axis, fixed order): {low, medium, high}.
  - low  : left shoulder — 1 at/below q25, ramps to 0 at q50.
  - medium: triangle — 0 at q25, 1 at q50, 0 at q75.
  - high : right shoulder — 0 at q50, ramps to 1 at/above q75.
"""
from __future__ import annotations

import numpy as np

N_TERMS = 3  # low, medium, high


def compute_breakpoints(values: np.ndarray) -> tuple[float, float, float]:
    """Return (q25, q50, q75) percentiles of `values`. The sole breakpoint source."""
    arr = np.asarray(values, dtype=float).ravel()
    q25, q50, q75 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    return q25, q50, q75


def _ramp_up(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 at/below lo, 1 at/above hi, linear between. Degenerate lo==hi -> step at lo."""
    if hi <= lo:
        return (x >= lo).astype(float)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _ramp_down(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1 at/below lo, 0 at/above hi, linear between. Degenerate lo==hi -> step at lo."""
    if hi <= lo:
        return (x < lo).astype(float)
    return np.clip((hi - x) / (hi - lo), 0.0, 1.0)


def fuzzify(values: np.ndarray, breakpoints: tuple[float, float, float]) -> np.ndarray:
    """Triangular fuzzification of `values` -> array with a trailing axis of size 3.

    Output[..., 0]=low, [..., 1]=medium, [..., 2]=high. Vectorized over any input shape.
    Raises ValueError on non-monotone breakpoints (q25 > q50 or q50 > q75).
    """
    q25, q50, q75 = breakpoints
    if not (q25 <= q50 <= q75):
        raise ValueError(f"non-monotone breakpoints: {breakpoints}")

    x = np.asarray(values, dtype=float)
    low = _ramp_down(x, q25, q50)
    high = _ramp_up(x, q50, q75)
    # medium triangle: rising q25->q50, falling q50->q75.
    medium = np.minimum(_ramp_up(x, q25, q50), _ramp_down(x, q50, q75))
    out = np.stack([low, medium, high], axis=-1)
    if not np.all(np.isfinite(out)):
        raise ValueError("fuzzify produced non-finite membership values")
    return out
