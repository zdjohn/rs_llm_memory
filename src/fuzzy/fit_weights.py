"""Numpy-only per-rule weight fitting via a pairwise BPR-style loss (A1).

`fit_weights` optimizes the rule weights `w_j` (length rules.N_RULES) by minimizing a
pairwise BPR loss over TRAIN positive/negative item pairs. The forward score in the loss
calls the SAME `fis_score.score_matrix(..., weights=w)` path that A0/eval use — no shadow
firing. A finite-difference gradient descent keeps it numpy-only (no autodiff, no scipy).

A1 is therefore a faithful rehearsal of A0 with learned weights; at this tiny capacity it
may or may not beat A0 on real data (expected, not a failure).
"""
from __future__ import annotations

import numpy as np

from fuzzy import fis_score, rules


def _pairwise_bpr_loss(matrix: np.ndarray, pairs: np.ndarray) -> float:
    """Mean -log(sigmoid(s_pos - s_neg)) over (user, pos_item, neg_item) train pairs."""
    u = pairs[:, 0].astype(int)
    pos = pairs[:, 1].astype(int)
    neg = pairs[:, 2].astype(int)
    diff = matrix[u, pos] - matrix[u, neg]
    # numerically stable -log(sigmoid(diff)) = softplus(-diff)
    return float(np.mean(np.logaddexp(0.0, -diff)))


def fit_weights(concepts: dict[str, np.ndarray],
                breakpoints: dict[str, tuple[float, float, float]],
                train_targets: np.ndarray,
                *,
                lr: float = 1.0,
                max_iter: int = 400,
                tol: float = 1e-4,
                eps: float = 1e-3,
                score_fn=fis_score.score_matrix,
                n_rules: int | None = None) -> np.ndarray:
    """Fit rule weights minimizing pairwise BPR loss; return ndarray length `n_rules`.

    `train_targets`: (n_pairs, 3) int array of (user, pos_item, neg_item) from TRAIN only.
    Optimizes via finite-difference gradient descent through `score_fn` (default the Track-A
    `fis_score.score_matrix`; pass `rules_recency.score_matrix_recency` to fit the memory-decay
    rule base). `n_rules` defaults to `rules.N_RULES` — set it to the length of the rule base
    `score_fn` fires (e.g. `rules_recency.N_RULES`). Raises RuntimeError if not converged.
    """
    targets = np.asarray(train_targets)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"train_targets must be (n_pairs, 3), got {targets.shape}")

    n = rules.N_RULES if n_rules is None else int(n_rules)
    w = np.ones(n, dtype=float)

    def loss_of(weights: np.ndarray) -> float:
        matrix = score_fn(concepts, breakpoints, weights=weights)
        return _pairwise_bpr_loss(matrix, targets)

    prev_loss = loss_of(w)
    converged = False
    for _ in range(max_iter):
        grad = np.zeros(n, dtype=float)
        for j in range(n):
            bumped = w.copy()
            bumped[j] += eps
            grad[j] = (loss_of(bumped) - prev_loss) / eps
        w = np.clip(w - lr * grad, 1e-6, None)  # weights stay positive
        cur_loss = loss_of(w)
        if abs(prev_loss - cur_loss) < tol:
            converged = True
            prev_loss = cur_loss
            break
        prev_loss = cur_loss

    if not converged:
        raise RuntimeError(
            f"fit_weights did not converge within max_iter={max_iter} (last loss {prev_loss:.6f})")
    if not np.all(np.isfinite(w)):
        raise RuntimeError("fit_weights produced non-finite weights")
    return w
