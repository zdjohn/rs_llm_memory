"""Concept builder — 4 faked headphones-shaped concepts, TRAIN-SPLIT-ONLY (numpy-only).

All statistics derive solely from TRAIN interactions; nothing leaks from valid/test. Arrays
are indexed by RecBole INTERNAL contiguous id (id 0 = [PAD], defined but never scored).

Concepts:
  - mainstream_appeal : item-only (n_items,) — train popularity, min-max normalized to [0,1].
  - niche_factor      : item-only (n_items,) — exactly 1 - mainstream_appeal.
  - era_fit           : user-conditioned (n_users, n_items) — closeness of the item's decade
                        to the user's train-preferred decade.
  - genre_match       : user-conditioned (n_users, n_items) — cosine of the item genre
                        multi-hot vs the user's train-derived genre profile.
"""
from __future__ import annotations

import numpy as np


def _validate_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"concept {name} contains NaN/inf")


def build_concepts(train_inter: np.ndarray,
                   item_genres: np.ndarray,
                   item_decades: np.ndarray,
                   user_pref_decade: np.ndarray,
                   user_genre_profile: np.ndarray,
                   n_users: int,
                   n_items: int) -> dict[str, np.ndarray]:
    """Compute the four locked concepts from train-split-only inputs.

    Parameters (all indexed by RecBole internal id, id 0 = PAD):
      train_inter        : (n_train, 2) int array of (user_internal_id, item_internal_id).
      item_genres        : (n_items, n_genres) float multi-hot of item genres.
      item_decades       : (n_items,) float decade per item (e.g. 1990.0); NaN/0 = unknown.
      user_pref_decade   : (n_users,) float train-preferred decade per user.
      user_genre_profile : (n_users, n_genres) float train-derived genre affinity per user.
      n_users, n_items   : internal id counts (include the PAD slot 0).

    Returns {mainstream_appeal, niche_factor, era_fit, genre_match}. Raises ValueError on
    shape/n_items mismatch, NaN, or an empty user genre profile (an explicit documented
    uniform fallback is applied for the empty-profile case rather than producing NaN).
    """
    train_inter = np.asarray(train_inter)
    item_genres = np.asarray(item_genres, dtype=float)
    item_decades = np.asarray(item_decades, dtype=float)
    user_pref_decade = np.asarray(user_pref_decade, dtype=float)
    user_genre_profile = np.asarray(user_genre_profile, dtype=float)

    if item_genres.shape[0] != n_items:
        raise ValueError(f"item_genres rows {item_genres.shape[0]} != n_items {n_items}")
    if item_decades.shape[0] != n_items:
        raise ValueError(f"item_decades len {item_decades.shape[0]} != n_items {n_items}")
    if user_pref_decade.shape[0] != n_users:
        raise ValueError(f"user_pref_decade len {user_pref_decade.shape[0]} != n_users {n_users}")
    if user_genre_profile.shape != (n_users, item_genres.shape[1]):
        raise ValueError(
            f"user_genre_profile shape {user_genre_profile.shape} != "
            f"(n_users={n_users}, n_genres={item_genres.shape[1]})")
    if train_inter.ndim != 2 or train_inter.shape[1] != 2:
        raise ValueError(f"train_inter must be (n_train, 2), got {train_inter.shape}")

    # Reject NaN in the genre/profile inputs up front (decades may legitimately be NaN =
    # "unknown" and are handled explicitly below, so they are exempt from this check).
    if not np.all(np.isfinite(item_genres)):
        raise ValueError("item_genres contains NaN/inf")
    if not np.all(np.isfinite(user_genre_profile)):
        raise ValueError("user_genre_profile contains NaN/inf")

    # --- mainstream_appeal: train popularity, min-max normalized over real items. ---
    pop = np.zeros(n_items, dtype=float)
    if train_inter.shape[0] > 0:
        item_ids = train_inter[:, 1].astype(int)
        counts = np.bincount(item_ids, minlength=n_items).astype(float)
        pop = counts[:n_items]
    pmax = pop.max() if pop.size else 0.0
    mainstream_appeal = pop / pmax if pmax > 0 else np.zeros(n_items, dtype=float)
    mainstream_appeal[0] = 0.0  # PAD, never scored.
    niche_factor = 1.0 - mainstream_appeal

    # --- era_fit: 1 / (1 + |decade_gap|/10), broadcast user vs item decades. ---
    safe_item_dec = np.where(np.isfinite(item_decades), item_decades, 0.0)
    safe_user_dec = np.where(np.isfinite(user_pref_decade), user_pref_decade, 0.0)
    gap = np.abs(safe_user_dec[:, None] - safe_item_dec[None, :]) / 10.0  # (n_users, n_items)
    era_fit = 1.0 / (1.0 + gap)
    era_fit[0, :] = 0.0
    era_fit[:, 0] = 0.0

    # --- genre_match: cosine(user_profile, item_genre_multihot). ---
    item_norm = np.linalg.norm(item_genres, axis=1)  # (n_items,)
    profile = user_genre_profile.copy()
    user_norm = np.linalg.norm(profile, axis=1)  # (n_users,)
    # Empty-profile fallback (documented): a user with no train genre signal gets a uniform
    # profile so cosine is well-defined (no NaN) rather than 0/0.
    empty = user_norm == 0
    if empty.any():
        profile[empty] = 1.0
        user_norm = np.linalg.norm(profile, axis=1)
    denom = np.outer(user_norm, item_norm)  # (n_users, n_items)
    dots = profile @ item_genres.T  # (n_users, n_items)
    genre_match = np.where(denom > 0, dots / np.where(denom > 0, denom, 1.0), 0.0)
    genre_match[0, :] = 0.0
    genre_match[:, 0] = 0.0

    concepts = {
        "mainstream_appeal": mainstream_appeal,
        "niche_factor": niche_factor,
        "era_fit": era_fit,
        "genre_match": genre_match,
    }
    for name, arr in concepts.items():
        _validate_finite(name, arr)
    return concepts
