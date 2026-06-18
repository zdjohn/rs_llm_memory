---
task_id: '1ac2045f-7ac7-4244-8fd8-1166a7ecba87'
tracked_project_id: '4f11db28-835d-4294-95f7-7099c88ba9f4'
generated_at: '2026-06-16T10:08:34Z'
prd_source: 'solar-cat task 1ac2045f-7ac7-4244-8fd8-1166a7ecba87'
---

# Implementation Plan: Track A — Mamdani FIS Plumbing Validation (RecBole Rig)
_Generated 2026-06-16 from PRD_

> ✅ IMPLEMENTED — all 22 tasks landed. Pure-numpy unit tests GREEN; RecBole/MLflow integration paths written import-clean and guarded behind `RUN_INTEGRATION` (run live by the user — see PR Test Checklist). `configs/base.yaml` and `data/ml-100k/` untouched.

## Constraints & Decisions

**Tech Stack:** Python 3.10 (conda env `recsys`), numpy 1.26.4, RecBole 1.2.0, PyTorch 2.5.1, MLflow (server on :5002), pandas. CPU/Apple-Silicon. No web/JS stack. FIS is numpy-only.

**Locked Decisions:**

| Topic | Decision |
|-------|----------|
| dataset name | Use `ml100k` (NOT `ml-100k`, which triggers RecBole 1.2.0's bundled-example override). Materialize `data/ml100k/ml100k.{inter,user,item}` and re-establish the BPR floor under `ml100k`. |
| migration source | `data/ml-100k/raw/` does not exist; re-derive `data/ml100k/*` by **copying** the existing byte-identical `data/ml-100k/ml-100k.*` atomic files (additive; do not delete `data/ml-100k/`). |
| eval protocol | `configs/base.yaml` FROZEN verbatim — mode full, RS [0.8,0.1,0.1], order RO, group_by user, rating≥4 positive, seed 2020, metrics NDCG/Hit/Precision@[1,3], valid_metric NDCG@3. **Do NOT touch eval semantics.** |
| experiments | Run BOTH A0 (FIS frozen weights w_j=1) and A1 (FIS fitted weights). A0 is the hard gate; A1 follows. Plus floor=BPR-MF and ceil-ref=DeepFM, each on all / cold-user / cold-item slices. |
| FIS math | Mamdani Type-1; PRODUCT t-norm; max aggregation; centroid defuzz over FIXED consequent centroids avoid=0.1, neutral=0.5, recommend=0.9; 3 terms per variable {low,medium,high}; output suitability {avoid,neutral,recommend}. |
| MF breakpoints | From DATA (25/50/75 percentiles over the train split), never hand-set. |
| concepts | 4 faked headphones-shaped concepts from TRAIN SPLIT ONLY: `mainstream_appeal` (item-only), `niche_factor`=1−mainstream_appeal (item-only), `era_fit` (user-conditioned), `genre_match` (user-conditioned). |
| deps | numpy-only for the FIS. NO scikit-fuzzy. No new heavy deps. Stay in `recsys`. |
| eval reuse | Reuse the repo's NDCG via a minimal RecBole `GeneralRecommender` wrapper whose `full_sort_predict(user)` returns the FIS row (Adapter A, preferred), OR raw-matrix injection (Adapter B, contingency). **Do NOT reimplement NDCG.** |
| masking | Use the `GeneralRecommender` wrapper so RecBole's full-ranking eval masks train-positive items automatically; replicate masking only on the raw-injection path. |
| sanity gate | BPR-through-the-adapter must reproduce the `ml100k` BPR NDCG@3 to ~3 decimals (`\|Δ\| < 1e-3`). HARD BLOCKER — do not proceed past a red sanity check. |
| pass criteria | PASS = A0 or A1 ≥ BPR on cold-item slice for NDCG@3 AND Hit@3 (adapter green). WEAK PASS = ties cold. FAIL = underperforms cold with a green adapter. |
| task granularity | Single task (FIS core + tests + concepts + adapter + sanity check + experiments together). |

**Out of Scope:** Track B entirely (real guide corpus, LLM rule/concept extraction, rule survival, the Stage-3 verification loop, Amazon atomic files, temporal split); scikit-fuzzy / any new heavy dependency; touching `configs/base.yaml` eval semantics; beating DeepFM on the warm slice (explicitly NOT a goal — warm loss is acceptable); renaming or deleting the existing `data/ml-100k/` files.

**Migration Required:** yes (data-materialization: regenerate atomic files under `ml100k` + re-establish the BPR floor; must precede any FIS run).

**Flags:** migration_required: true · has_ui_component: false · has_db_component: false · has_test_requirement: true

## Architecture

### Data Model

This is a Python ML/research pipeline, not a database-backed web app. "Data Model" means the on-disk RecBole atomic files plus the in-memory numpy artifacts (concept arrays, membership tensors, the users×items score matrix) that the FIS modules and the RecBole evaluator exchange.

#### New Tables / Schema Changes

No SQL. Two classes of artifact change:

1. New atomic-file materialization under the locked `ml100k` name (additive; existing `data/ml-100k/*` untouched):
   - `data/ml100k/ml100k.inter` — header `user_id:token\titem_id:token\trating:float\ttimestamp:float`.
   - `data/ml100k/ml100k.user` — `user_id:token\tage:token\tgender:token\toccupation:token`.
   - `data/ml100k/ml100k.item` — `item_id:token\tgenre:token_seq\trelease_year:token`.

   No format change is needed — the encoding is identical to the existing files; only the directory + name differ. **Resolved (orchestrator):** `data/ml-100k/raw/` does NOT exist in this worktree, so the migration re-derives `data/ml100k/*` by **copying** the already-materialized `data/ml-100k/*` atomic files (byte-identical; atomic files embed no internal dataset name), NOT by re-running `prepare_ml100k.py` from raw.

2. New in-memory numpy artifacts (no new on-disk schema; transient, produced inside `run_fuzzy.py`):
   - `concepts`: dict `{name: np.ndarray}` — item-only vector `(n_items,)` (mainstream_appeal, niche_factor) or user×item matrix `(n_users, n_items)` (era_fit, genre_match). Indexed by RecBole **internal** contiguous id (aligns 1:1 with `full_sort_predict` row/column order); id 0 is RecBole's `[PAD]` sentinel.
   - `breakpoints`: dict `{concept_name: (q25, q50, q75)}` from the TRAIN split only.
   - `membership`: per-concept fuzzification `(..., 3)` last-axis = {low, medium, high}.
   - `score_matrix`: `np.ndarray (n_users, n_items)` float in [0,1] = defuzzified suitability; the single artifact handed to the evaluator seam.

#### RLS Policies

_No RLS — not a database-backed web app._

#### Migration Strategy (migration_required: true)

Must be runnable and green before any FIS scoring code is trusted. Pure data-materialization + floor re-establishment; no FIS module is on its critical path.

- **Step 1 — Materialize under the locked name (RESOLVED: copy, not re-run from raw).** Copy `data/ml-100k/ml-100k.{inter,user,item}` → `data/ml100k/ml100k.{inter,user,item}`. Additive; `data/ml-100k/` untouched.
- **Step 2 — Register `ml100k` in the harness (additive).** Extend `run_one.py` (`run_one` default + `--dataset` default), `run_all.py` (`DEFAULT_DATASETS`), and `group_metrics.py` (atomic-file path helper) to use `ml100k` while keeping `ml-100k` reachable.
- **Step 3 — Re-establish the BPR floor under `ml100k`.** `python src/run_one.py --model BPR --dataset ml100k --user_feats off --item_feats off`. The recorded NDCG@3 is BOTH the PASS-gate floor and the reference the sanity gate reproduces.
- **Step 4 — Verify with the smoke path.** `python src/smoke_test.py --dataset ml100k`.

#### Seed Data

No seed rows. The "seed" of the FIS is the four faked concept definitions and the fixed consequent centroids `{avoid:0.1, neutral:0.5, recommend:0.9}` — code constants, not data files. MF breakpoints are derived from the train split at runtime, never seeded.

### Hook / Service Layer

Five new pure-Python modules under `src/fuzzy/` plus the evaluator seam, ordered by the data-flow dependency chain.

**`build_concepts` — `src/fuzzy/concepts.py`**
- **Interface:** `build_concepts(train_inter, item_genres, item_decades, user_pref_decade, user_genre_profile, n_users, n_items) -> dict[str, np.ndarray]`. Computes the four locked concepts from TRAIN-SPLIT-ONLY inputs, numpy arrays indexed by RecBole internal id. Invariants: (1) statistics derived solely from train interactions; (2) `mainstream_appeal`/`niche_factor` item-only `(n_items,)` with `niche_factor == 1 - mainstream_appeal` exactly; (3) `era_fit`/`genre_match` user-conditioned `(n_users, n_items)`; (4) `genre_match` = cosine of item genre multi-hot vs user's train-genre profile; (5) finite, normalized to [0,1]; id 0 (PAD) defined but never scored. Error modes: raise on shape/`n_items` mismatch, on NaN, or on empty train profile (explicit uniform fallback). Config: none.
- **Seam:** exported `build_concepts -> dict[str, np.ndarray]`.
- **Dependency category:** in-process.
- **Deletion test:** If deleted, complexity vanishes — leakage-discipline logic + concept definitions have one reason to exist and one consumer (the fuzzifier).

**`membership` — `src/fuzzy/membership.py`**
- **Interface:** `compute_breakpoints(concept_values) -> (q25,q50,q75)` (the ONLY place breakpoints are set, from data). `fuzzify(values, breakpoints) -> np.ndarray (..., 3)` over {low, medium, high}, vectorized triangular. Invariants: breakpoints only from `compute_breakpoints`; pure function (A0 and A1 fuzzify identically); explicit degenerate-percentile handling. Error modes: raise on non-monotone breakpoints. Config: 3 terms, fixed.
- **Seam:** exported `fuzzify -> np.ndarray (..., 3)`, `compute_breakpoints -> (q25,q50,q75)`.
- **Dependency category:** in-process.
- **Deletion test:** If deleted, complexity vanishes — vectorized triangular MF is the one thing it owns; inlining would couple breakpoint logic to firing and break the A0/A1 shared path.

**`rules` — `src/fuzzy/rules.py`**
- **Interface:** `fire(memberships, weights=None) -> np.ndarray (..., 3)` over {avoid, neutral, recommend}. PRODUCT t-norm antecedent, MAX aggregation across rules per output term. `weights` None/ones ⇒ A0; fitted ⇒ A1 — **identical arithmetic, weights only scale rule contributions.** Invariants: A0 and A1 share this exact function; rule base is a locked module constant; output finite, non-negative. Error modes: raise on weight-length mismatch / undefined concept term. Config: rule base constant; t-norm/aggregation locked.
- **Seam:** exported `fire(memberships, weights) -> np.ndarray (..., 3)`; also expose the rule-base constant.
- **Dependency category:** in-process.
- **Deletion test:** If deleted, complexity vanishes — heart of the Mamdani engine and the shared A0/A1 seam; deletion would duplicate firing into `fis_score` and `fit_weights` and de-synchronize them.

**`fis_score` — `src/fuzzy/fis_score.py`**
- **Interface:** `defuzz(firing_strengths) -> np.ndarray` centroid defuzz over FIXED centroids `(0.1, 0.5, 0.9)`. `score_matrix(concepts, breakpoints, weights=None) -> np.ndarray (n_users, n_items)` orchestrates fuzzify→fire→defuzz to the full suitability matrix — the single artifact the evaluator consumes. Invariants: centroids locked; item-only concepts broadcast across users; user-conditioned indexed per (u,i); no NaN; PAD row/col present but never ranked; `weights=None`/ones ⇒ A0, fitted ⇒ A1 via the same `rules.fire`. Error modes: raise on centroid/term-count mismatch, on NaN in the matrix. Config: centroids fixed.
- **Seam:** exported `score_matrix -> np.ndarray (n_users, n_items)` — the upstream half of the FIS↔evaluator seam.
- **Dependency category:** in-process.
- **Deletion test:** If deleted, complexity concentrates at the single caller (`run_fuzzy`); it earns its place as the stable `(n_users, n_items)` contract that keeps the wrapper ~30 lines.

**`fit_weights` — `src/fuzzy/fit_weights.py`**
- **Interface:** `fit_weights(concepts, breakpoints, train_targets) -> np.ndarray` returns the fitted `w_j` vector. Invariants: MUST call the same `rules.fire` path (no shadow firing) so A1 is a faithful rehearsal of A0 with learned weights; fits only against TRAIN targets; numpy-only optimizer; returns locked-length vector. Error modes: raise on non-convergence past max-iter; raise if targets reference non-train interactions. Config: optimizer hyperparameters defaulted.
- **Seam:** exported `fit_weights -> np.ndarray (n_concepts,)`.
- **Dependency category:** in-process (depends on `rules`, `membership`).
- **Deletion test:** If deleted, complexity vanishes for the A0-only world and A1 disappears — A0 still runs with `weights=None`. Clean optional extension of the shared firing path.

**`FISRecommender` + `run_fuzzy` (evaluator seam + orchestrator) — `src/fuzzy/run_fuzzy.py`**
The one place a real port + adapters is justified: the locked decision names two candidate adapters. The seam is "given a `(n_users, n_items)` score matrix and RecBole `test_data`, produce NDCG/Hit/Precision@[1,3] with train-positive masking."
- **Interface (the seam):** `class FISRecommender(GeneralRecommender)` with `__init__(self, config, dataset)` and `full_sort_predict(self, interaction) -> torch.Tensor` returning the precomputed FIS row(s); declares `MODEL_TYPE = GENERAL` so RecBole masks train-positives automatically. Matrix injected at construction (wrapper does no FIS math; thin torch view, internal-id aligned). `run_fuzzy(experiment, weights_mode) -> dict` builds `Config`/`create_dataset`/`data_preparation` exactly as `run_one` (reusing frozen `base.yaml`), extracts the train split, calls `build_concepts → compute_breakpoints → score_matrix` (A0 `weights=None`; A1 `fit_weights(...)`), wraps the matrix, runs the same evaluator, and returns a `run_one`-shaped result dict so `log_to_mlflow` and `aggregate.py` consume it unchanged.
- **Seam:** `FISRecommender` (`full_sort_predict -> torch.Tensor`) and `run_fuzzy -> dict` (same shape as `run_one`'s return).
- **Dependency category:** remote-owned (RecBole owns the `GeneralRecommender` contract, the eval-loader objects, and masking semantics).
- **Deletion test:** If deleted, complexity concentrates — the FIS pipeline would have no path to the NDCG numbers and the "do not reimplement NDCG" decision would be violated. It is the integration boundary.
- **Adapters (≥2 justified):** Adapter A (production/preferred) = `FISRecommender` wrapper, masking free via RecBole's full-ranking path; Adapter B (contingency) = raw-matrix injection into the repo's metric functions, which MUST replicate masking. Implement Adapter A first; treat B as a fallback.

**`sanity_gate` (BPR-through-the-adapter) — in `src/fuzzy/run_fuzzy.py`**
- **Interface:** `bpr_sanity_check(bpr_reference_ndcg3, tol=1e-3) -> bool`. Trains/loads BPR as `run_one(model="BPR", dataset="ml100k", ...)`, extracts BPR's full score matrix, pushes it through the SAME `FISRecommender` path (Adapter A), asserts the resulting NDCG@3 reproduces the floor within `|Δ| < 1e-3`. Invariants: exercises the identical adapter the FIS uses; HARD BLOCKER — on red it raises and the pipeline must not proceed. Error modes: raise (not warn) on `|Δ| ≥ tol`.
- **Seam:** exported `bpr_sanity_check -> bool` (raises on failure).
- **Dependency category:** in-process orchestration over the remote-owned evaluator seam.
- **Deletion test:** If deleted, complexity vanishes but so does all trust in FIS numbers — one job (validate the seam), one consumer (the pipeline gate).

#### Replaced or Extended Functions

- `src/run_one.py` — `run_one`, `build_load_col`, `log_to_mlflow`, `sanitize_key` reused unchanged for BPR/DeepFM, the sanity gate's BPR run, and logging FIS runs. Extension is additive (FIS path lives in `run_fuzzy`).
- `src/run_all.py` — `DEFAULT_DATASETS` extended to include `ml100k`; scheduling extended to add A0, A1, floor=BPR, ceil-ref=DeepFM across all/cold-user/cold-item slices. Additive.
- `src/group_metrics.py` — implement `collect_per_user_ndcg` (currently a Phase-4 `NotImplementedError` stub) per its documented contract; add `hit_at_k`, a `load_item_groups` counterpart, and cold-user/cold-item (≤5 train interactions) slicing. The aggregation core (`load_user_groups`, `ndcg_at_k`, `per_group_ndcg`, `flatten_group_metrics`) is reused.
- `src/aggregate.py` — reused unchanged; FIS runs land in the same MLflow experiment and flatten into the same CSV.

#### State Management

No persistent store. FIS state is transient numpy arrays passed explicitly down the call chain (no module-level globals — so A0 and A1 share the firing path without contamination). Durable state: MLflow runs (via existing `log_to_mlflow`) and the materialized `data/ml100k/` atomic files. Breakpoints/concepts are recomputed per run from the train split (cheap, leakage-safe, deterministic under seed 2020).

### Component Boundaries

#### New Components
_No React components — Python research pipeline._ (New modules are covered under Hook/Service Layer.)

#### Modified Components
_No React components — Python research pipeline._

#### Data Flow

```
[migration]
  data/ml-100k/ml-100k.{inter,user,item}  ──copy──▶  data/ml100k/ml100k.{inter,user,item}
       │
       └──run_one(BPR, ml100k)──▶ MLflow: BPR floor NDCG@3  (the reference number)

[per FIS run: run_fuzzy(experiment, weights_mode)]
  base.yaml (frozen) ──▶ RecBole Config ──create_dataset/data_preparation──▶ (train, valid, test, Dataset)
       │  train split only                                                          │ (internal-id maps,
       ▼                                                                            ▼  train-positive history)
  concepts.py: build_concepts(...)  ──▶ concepts {name: ndarray, internal-id indexed}
       ▼
  membership.py: compute_breakpoints ──▶ (q25,q50,q75) ; fuzzify(values,bp) ──▶ (...,3)
       ▼
  rules.py: fire(memberships, weights)   ◀── A0: weights=None/ones
       ▲                                  ◀── A1: weights = fit_weights(...)  [SAME fire() path]
       ▼
  fis_score.py: defuzz over centroids(0.1,0.5,0.9) ; score_matrix(...) ──▶ S: (n_users, n_items) in [0,1]
       │   (the FIS↔evaluator seam — numpy matrix handed across)
       ▼
  run_fuzzy.py: FISRecommender(config, dataset) wraps S ; full_sort_predict(user) ──▶ torch row of S[user]
       │                                                   (RecBole masks train-positives automatically — mode:full)
       ▼
  RecBole evaluator ──▶ test_result {NDCG/Hit/Precision@[1,3]}
       ├─ group_metrics.py: load_user_groups / load_item_groups ──▶ all / cold-user / cold-item NDCG@3, Hit@3
       ▼
  run_one-shaped dict ──log_to_mlflow──▶ MLflow ──aggregate.py──▶ results/*.csv ──▶ PASS / WEAK PASS / FAIL

[sanity gate — BEFORE trusting any FIS result, HARD BLOCKER]
  BPR score matrix ──▶ SAME FISRecommender path ──▶ NDCG@3  ⟺  BPR floor NDCG@3  (|Δ| < 1e-3 else RAISE)
```

Key contract: every concept/membership/score array is indexed by RecBole **internal** contiguous id (from the `Dataset` after `data_preparation`), so `S[user_internal_id]` aligns row-for-row with `full_sort_predict`. Mapping back to original `user_id` tokens (for `load_user_groups` joins) uses `dataset.id2token`.

### Deepening Opportunities

- The A0/A1 shared firing path is the deepest leverage: `rules.fire(memberships, weights)` with `weights=None` ⇒ A0 and fitted ⇒ A1 collapses two experiments into one implementation. Resist giving A1 its own firing/defuzz copy.
- `FISRecommender` and `sanity_gate` share one adapter — implement it once and have the gate exercise it with BPR's matrix (two callers over one implementation).
- Do not introduce ports over the four pure-numpy modules; they have a single in-process consumer each. The only justified port is the FIS↔RecBole evaluator seam.

### Open Architectural Questions

Both raised by the architect were **resolved in-context by the orchestrator** (see Open Questions section): (1) migration source → copy from existing atomic files (raw archive absent); (2) cold-slice extraction → in-scope and required (the PASS gate is defined on the cold-item slice).

## Task List

### [x] TASK-001: Materialize ml100k atomic files by copying ml-100k
**Type:** migration
**Layer:** infra
**Files:** data/ml100k/ml100k.inter, data/ml100k/ml100k.user, data/ml100k/ml100k.item, src/materialize_ml100k.py
**Depends on:** —
**Blocked by:** —
**Estimate:** S
**Description:** Create directory `data/ml100k/` and populate it by byte-copying the three existing atomic files: `data/ml-100k/ml-100k.inter -> data/ml100k/ml100k.inter`, `.user`, `.item`. Atomic files embed no internal dataset name, so copy+rename is correct. ADDITIVE: do NOT delete/modify `data/ml-100k/`. Provide runnable `src/materialize_ml100k.py` performing the copy idempotently (skip if target byte-identical else copy) using `shutil.copyfile`, REPO_ROOT via `Path(__file__).resolve().parents[1]`. Raw archive `data/ml-100k/raw/` does not exist; do NOT re-run `prepare_ml100k.py`.
**Acceptance:** After `python src/materialize_ml100k.py`, all three `data/ml100k/ml100k.{inter,user,item}` exist and are byte-identical to their `data/ml-100k/ml-100k.*` counterparts (`filecmp.cmp(..., shallow=False)`); `data/ml-100k/` unchanged.
**Tests:**
- Unit: assert `materialize_ml100k.materialize()` creates the three target paths and each `filecmp.cmp(src, dst, shallow=False)` is True; assert idempotency (second call does not raise, reports byte-identical skip); assert sources under `data/ml-100k/` still present.
- Migration: run the materialize entry point against the real `data/ml-100k/` files and assert all three `data/ml100k/ml100k.*` exist and are byte-identical (migration path works on real data). [INTEGRATION]
- Regression: after materialize, assert the original `data/ml-100k/ml-100k.*` are byte-unchanged so existing ml-100k-keyed code still resolves the old files.
**Test file:** tests/test_materialize_ml100k.py

### [x] TASK-002: Wire ml100k into run_one / run_all / group_metrics paths additively
**Type:** migration
**Layer:** infra
**Files:** src/run_one.py, src/run_all.py, src/group_metrics.py
**Depends on:** TASK-001
**Blocked by:** TASK-001
**Estimate:** M
**Description:** Add `ml100k` support without breaking `ml-100k`. In `run_one.py`: `run_one` default `dataset` and argparse `--dataset` default → `ml100k`; leave `data_path` default and `build_load_col` unchanged. In `run_all.py`: `DEFAULT_DATASETS = ["ml100k"]`. In `group_metrics.py`: add a module-level helper resolving `REPO_ROOT/data/<dataset>/<dataset>.user|.item` defaulting to `ml100k`, and parameterize the `__main__` self-check path. Keep `ml-100k` runnable by leaving these as defaults only.
**Acceptance:** `python src/run_one.py --model BPR --quick --no_mlflow` (no `--dataset`) runs against `ml100k` and prints `ndcg@3`; `python src/run_all.py --dry_run` lists `ml100k`; passing `--dataset ml-100k` still resolves the old files.
**Tests:**
- Unit: assert `run_one.run_one` default `dataset` is `"ml100k"` and argparse `--dataset` default is `"ml100k"`; assert `run_all.DEFAULT_DATASETS == ["ml100k"]`; assert the new group_metrics path helper resolves `data/ml100k/ml100k.user|.item` by default and `data/ml-100k/ml-100k.*` when passed `dataset="ml-100k"` (path-string assertions, no training).
- Integration: `python src/run_one.py --model BPR --quick --no_mlflow` trains BPR (epochs=1) against the real ml100k atomic files and prints `ndcg@3`; assert returned `test_result` contains `ndcg@3`. [INTEGRATION]
- Migration: run the group_metrics path helper for `dataset="ml100k"` and assert it points at the files created by TASK-001 (new dataset path works).
- Regression: run the path helper and a `run_one`-config build with `dataset="ml-100k"` and assert it still resolves the original `data/ml-100k/ml-100k.*` (old name unbroken).
**Test file:** tests/test_run_one.py

### [x] TASK-003: Verify ml100k end-to-end via smoke_test
**Type:** test
**Layer:** test
**Files:** src/smoke_test.py
**Depends on:** TASK-002
**Blocked by:** TASK-002
**Estimate:** S
**Description:** Update `smoke_test.py` argparse `--dataset` default `ml-100k` → `ml100k` (the existence check already derives paths from `args.dataset`). Confirm the data→train→eval→MLflow→read-back path runs under the new name.
**Acceptance:** `python src/smoke_test.py --dataset ml100k` prints atomic-files-present, trains BPR quick, logs+reads back `ndcg_at_3`/`hit_at_3`/`precision_at_3`, exits 0 with `SMOKE TEST PASSED`.
**Tests:** N/A — this task is itself a test task.
**Test file:** src/smoke_test.py

### [x] TASK-004: Re-establish BPR floor under ml100k (records sanity-gate reference + floor)
**Type:** migration
**Layer:** infra
**Files:** src/run_all.py
**Depends on:** TASK-002, TASK-003
**Blocked by:** TASK-002
**Estimate:** S
**Description:** Run the BPR-MF floor under `ml100k` at the frozen full-epoch budget via `python src/run_one.py --model BPR --dataset ml100k --user_feats off --item_feats off`, logging via unchanged `log_to_mlflow`. Record the resulting NDCG@3 — this single number is BOTH the PASS-gate floor and the reference the sanity gate (TASK-012) reproduces. Persist it durably (e.g. a documented constant/results note) for TASK-012/017. Do NOT modify `configs/base.yaml` or `configs/bpr.yaml`.
**Acceptance:** An MLflow run with `params.model=BPR`, `params.dataset=ml100k`, `user_feats=off`, `item_feats=off` exists with a persisted `metrics.ndcg_at_3`; that value is captured as the documented floor reference.
**Tests:**
- Integration: run BPR at the frozen budget on real ml100k, `log_to_mlflow`, read the run back (`client.get_run`) and assert `params.dataset=="ml100k"`, `params.model=="BPR"`, `user_feats=="off"`, `item_feats=="off"`, and `metrics.ndcg_at_3` is persisted and finite; capture the value. [INTEGRATION]
- Migration: same run under the new `ml100k` name proves the floor is re-establishable on migrated data.
- Regression: assert a BPR `--dataset ml-100k --user_feats off --item_feats off` run still completes and persists `metrics.ndcg_at_3` (legacy floor still reproducible). [INTEGRATION]
**Test file:** tests/test_bpr_floor.py

### [x] TASK-005: Implement vectorized triangular membership functions and data-driven breakpoints
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/__init__.py, src/fuzzy/membership.py
**Depends on:** —
**Blocked by:** —
**Estimate:** M
**Description:** Create `src/fuzzy/__init__.py` (empty package marker) and `src/fuzzy/membership.py` (pure numpy, no scikit-fuzzy). `compute_breakpoints(values) -> (q25,q50,q75)` via `np.percentile` over the supplied (train-split-only) values — the ONLY place breakpoints are set; never hand-set. `fuzzify(values, breakpoints) -> np.ndarray (..., 3)` over {low, medium, high} using vectorized triangular membership (shoulders for low/high, triangle for medium) anchored on (q25,q50,q75). Pure stateless (A0 & A1 identical). Handle degenerate percentiles (e.g. constant concept) explicitly (no div-by-zero). Raise `ValueError` on non-monotone breakpoints.
**Acceptance:** `fuzzify` returns last dim 3 with interior rows summing ~1.0; value==q50→medium≈1, ≤q25→low≈1, ≥q75→high≈1; `compute_breakpoints` matches `np.percentile(x,[25,50,75])`; non-monotone breakpoints raise `ValueError`.
**Tests:**
- Unit: assert `compute_breakpoints` equals `np.percentile(values,[25,50,75])`; `fuzzify` trailing dim 3 and interior rows sum ~1.0; peaks at q25/q50/q75; degenerate constant input produces no NaN/div-by-zero; non-monotone breakpoints raise `ValueError`. (Toy numpy inputs, in-process — NOT integration.)
**Test file:** tests/test_membership.py

### [x] TASK-006: Implement Mamdani rule firing (PRODUCT t-norm, MAX aggregation)
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/rules.py
**Depends on:** TASK-005
**Blocked by:** TASK-005
**Estimate:** M
**Description:** Create `src/fuzzy/rules.py` (pure numpy). Expose the locked rule base as module constant `RULE_BASE` mapping antecedent term-combinations over the 4 concepts ({low,medium,high}) to one of {avoid, neutral, recommend}. `fire(memberships, weights=None) -> np.ndarray (..., 3)` over outputs; PRODUCT t-norm for antecedent conjunction, MAX aggregation across rules mapping to the same output term. `weights=None`/ones ⇒ A0; fitted vector ⇒ A1 — IDENTICAL arithmetic, weights only scale per-rule firing in the same code path (no shadow firing). Raise `ValueError` on weight-length mismatch / undefined term.
**Acceptance:** a membership tensor fully activating one known rule yields MAX-aggregated firing dominated by that rule's output term; `fire(m)` == `fire(m, ones)`; mismatched weight length raises `ValueError`.
**Tests:**
- Unit: single-rule full activation → dominant output term matches that rule's `RULE_BASE` consequent (PRODUCT-then-MAX); `fire(m)` allclose `fire(m, np.ones(n))`; wrong-length weights raise `ValueError`; undefined antecedent term raises `ValueError`. (Toy numpy inputs, in-process — NOT integration.)
**Test file:** tests/test_fis_score.py

### [x] TASK-007: Implement concept builder (train-split-only, 2 item-only / 2 user-conditioned)
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/concepts.py
**Depends on:** TASK-006
**Blocked by:** TASK-006
**Estimate:** L
**Description:** Create `src/fuzzy/concepts.py` (pure numpy). `build_concepts(train_inter, item_genres, item_decades, user_pref_decade, user_genre_profile, n_users, n_items) -> dict[str, np.ndarray]` computing exactly 4 concepts from TRAIN-SPLIT-ONLY data, indexed by RecBole internal id (id 0 = PAD, never scored): `mainstream_appeal` (item-only `(n_items,)`, train popularity normalized), `niche_factor = 1 - mainstream_appeal` (item-only, exactly), `era_fit` (user-conditioned `(n_users, n_items)`, from `user_pref_decade` vs `item_decades`), `genre_match` (user-conditioned `(n_users, n_items)`, cosine of item genre multi-hot vs user's train-derived genre profile). Raise on NaN, shape mismatch, and empty user profile unless an explicit documented uniform fallback is applied. Genre/decade from the `.item` atomic columns; user profiles from train interactions only.
**Acceptance:** keys `{mainstream_appeal, niche_factor, era_fit, genre_match}`; `np.allclose(niche_factor, 1 - mainstream_appeal)`; item-only `(n_items,)`, user-conditioned `(n_users, n_items)`; PAD index 0 never scored; NaN/wrong-shape input raises.
**Tests:**
- Unit: keys exactly `{mainstream_appeal, niche_factor, era_fit, genre_match}`; `np.allclose(niche, 1-mainstream)`; shapes correct; `genre_match` equals hand-computed cosine on a tiny hand-built genre multi-hot vs profile; NaN and shape-mismatch raise; empty profile uses documented uniform fallback (no NaN); PAD index 0 never assigned a real score. (Toy numpy inputs, in-process — NOT integration.)
**Test file:** tests/test_concepts.py

### [x] TASK-008: Implement centroid defuzzification and full score_matrix orchestration
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/fis_score.py
**Depends on:** TASK-005, TASK-006, TASK-007
**Blocked by:** TASK-007
**Estimate:** L
**Description:** Create `src/fuzzy/fis_score.py` (pure numpy). `defuzz(firing) -> np.ndarray` Type-1 centroid over FIXED `CENTROIDS = (0.1, 0.5, 0.9)` (module constant): `sum(firing*centroids)/sum(firing)` along the last axis, with explicit `sum(firing)==0` handling. `score_matrix(concepts, breakpoints, weights=None) -> np.ndarray (n_users, n_items)` in [0,1] orchestrating `membership.fuzzify` → `rules.fire` → `defuzz`, broadcasting item-only concepts across all users and combining with user-conditioned concepts. `weights=None` ⇒ A0; fitted ⇒ A1, both through the SAME `rules.fire`. Raise on any NaN in the matrix.
**Acceptance:** `defuzz` one-hot recommend≈0.9, avoid≈0.1, uniform≈0.5; `score_matrix` returns `(n_users, n_items)` in [0,1]; A0 (`weights=None`) and ones-weights matrices allclose; NaN-producing input raises.
**Tests:**
- Unit: `defuzz` one-hot recommend≈0.9, avoid≈0.1, uniform≈0.5 over CENTROIDS; handles sum==0 without div-by-zero; `score_matrix` shape `(n_users, n_items)` all in [0,1] and broadcasts item-only across users; `score_matrix(...,weights=None)` allclose `score_matrix(...,weights=ones)`; NaN in matrix raises. (Toy numpy inputs, in-process — NOT integration.)
**Test file:** tests/test_fis_score.py

### [x] TASK-009: Implement FISRecommender GeneralRecommender wrapper
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/run_fuzzy.py
**Depends on:** TASK-008
**Blocked by:** TASK-008
**Estimate:** L
**Description:** Create `src/fuzzy/run_fuzzy.py`. Adapter A: `class FISRecommender(GeneralRecommender)` (subclass of `recbole.model.abstract_recommender.GeneralRecommender`) with `MODEL_TYPE = ModelType.GENERAL` so RecBole masks train-positives during full-sort eval. `__init__(self, config, dataset)` accepts the precomputed `(n_users, n_items)` FIS score matrix injected at construction (store as a tensor on `config["device"]`); the wrapper performs NO FIS math. `full_sort_predict(self, interaction) -> torch.Tensor` returns the precomputed row(s) for the batch's users (RecBole internal user id), shaped to the full-sort contract. Provide a minimal `calculate_loss`/`forward` stub (no training). Matrix injected via constructor/`config` since we instantiate `FISRecommender` directly (bypass `get_model`).
**Acceptance:** constructing `FISRecommender` with a known matrix and calling `full_sort_predict` on an interaction batch returns a `torch.Tensor` whose rows equal the injected matrix rows for those users; `MODEL_TYPE` is `GENERAL`.
**Tests:**
- Unit: construct `FISRecommender` with a known small injected matrix + minimal config/dataset stub, call `full_sort_predict` on an interaction carrying a few internal user ids, assert returned `torch.Tensor` rows equal the injected matrix rows; assert `MODEL_TYPE` is GENERAL. (Direct construction with injected matrix; no real evaluator/training — NOT integration; requires torch + RecBole class import.)
- Integration: instantiate `FISRecommender` and run it through the real RecBole evaluator over ml100k `test_data` (the remote-owned seam); assert finite `NDCG@3`/`Hit@3`. [INTEGRATION]
**Test file:** tests/test_run_fuzzy.py

### [x] TASK-010: Implement run_fuzzy orchestrator producing run_one-shaped dict
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/run_fuzzy.py
**Depends on:** TASK-009
**Blocked by:** TASK-009
**Estimate:** L
**Description:** In `src/fuzzy/run_fuzzy.py`, implement `run_fuzzy(experiment, weights_mode) -> dict` mirroring `run_one`'s RecBole setup: build `Config` from `configs/base.yaml` (reuse; do not touch eval semantics) for dataset `ml100k`, `create_dataset` + `data_preparation`, extract the TRAIN split, derive item genres/decades (via dataset id2token) and user profiles. Call `build_concepts` → `compute_breakpoints` (per concept) → `score_matrix`: `weights_mode=="A0"` ⇒ `weights=None`; `"A1"` ⇒ weights from `fit_weights.fit_weights` (TASK-011). Wrap the matrix in `FISRecommender`, run the SAME RecBole evaluator over `test_data`, and return a dict shaped exactly like `run_one`'s return (`config`, `test_result`, `best_valid_score`, `best_valid_result`, `train_time_sec`, `peak_mem_mb`, `stats`, `user_feats`, `item_feats`) so `log_to_mlflow`/`sanitize_key` and `aggregate.py` consume it unchanged. Set `user_feats`/`item_feats` to fixed flags (e.g. off/off) so MLflow params populate.
**Acceptance:** `run_fuzzy("smoke-test", "A0")` returns a dict whose `test_result` contains `NDCG@3`/`Hit@3` and that passes unchanged into `log_to_mlflow` producing a persisted MLflow run; no edits to `configs/base.yaml`.
**Tests:**
- Unit: assert the returned dict has exactly the run_one-shaped keys so it is shape-compatible with `log_to_mlflow` (key-presence assertion on a small fixture dict if the orchestrator is structured to allow it).
- Integration: `run_fuzzy("smoke-test", "A0")` runs the full real RecBole setup over ml100k (create_dataset + data_preparation + FISRecommender evaluator over test_data), asserts `test_result` has finite `NDCG@3`/`Hit@3`, passes the dict into `log_to_mlflow`, reads the run back asserting `metrics.ndcg_at_3` persisted; assert `configs/base.yaml` byte-unchanged. [INTEGRATION]
**Test file:** tests/test_run_fuzzy.py

### [x] TASK-011: Implement numpy-only weight fitting (pairwise BPR-style loss)
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/fit_weights.py
**Depends on:** TASK-008
**Blocked by:** TASK-008
**Estimate:** L
**Description:** Create `src/fuzzy/fit_weights.py` (pure numpy, no new heavy deps). `fit_weights(concepts, breakpoints, train_targets) -> np.ndarray (n_concepts,)`. Optimize a pairwise BPR-style loss with a global temperature using a numpy-only optimizer (manual or finite-difference gradient descent); TRAIN-only targets (positive/negative item pairs from train interactions). The forward score in the loss MUST call the SAME `rules.fire` path (via `fis_score.score_matrix` with candidate weights) — no shadow/duplicate firing logic. Raise `RuntimeError` on non-convergence past a defined `max_iter`.
**Acceptance:** returns a finite `np.ndarray` length `n_concepts`; on a toy train set the fitted weights yield a `score_matrix` whose pairwise ranking loss is ≤ the all-ones (A0) loss; exceeding `max_iter` without convergence raises `RuntimeError`.
**Tests:**
- Unit: `fit_weights` returns finite `ndarray` length `n_concepts`; on a toy train set the fitted-weight `score_matrix` pairwise loss ≤ the all-ones (A0) loss (through the same `fis_score.score_matrix`, no shadow firing); a config that cannot converge before `max_iter` raises `RuntimeError`. (Toy numpy inputs, in-process — NOT integration.)
**Test file:** tests/test_fit_weights.py

### [x] TASK-012: Implement BPR-reproduction sanity gate (hard blocker)
**Type:** feature
**Layer:** hook
**Files:** src/fuzzy/run_fuzzy.py
**Depends on:** TASK-009, TASK-004
**Blocked by:** TASK-009, TASK-004
**Estimate:** M
**Description:** In `src/fuzzy/run_fuzzy.py`, implement `bpr_sanity_check(bpr_reference_ndcg3, tol=1e-3) -> bool`. Obtain the trained BPR model's full score matrix under `ml100k` and push that SAME matrix through the identical `FISRecommender` + RecBole evaluator path used by `run_fuzzy`. Assert the resulting `NDCG@3` reproduces the recorded BPR floor (TASK-004) within `|Δ| < tol`. RAISE on red — HARD BLOCKER; downstream FIS runs must not proceed when this fails. Return `True` only on green.
**Acceptance:** given a correct adapter, returns `True` with `|NDCG@3_adapter - NDCG@3_floor| < 1e-3`; artificially perturbing the matrix flips the result to a raised error; the gate raising halts the pipeline (called before scheduling in TASK-015).
**Tests:**
- Unit: given `bpr_reference_ndcg3` and a reproduced NDCG@3 differing < 1e-3 it returns `True`; given a perturbed value (|Δ| > tol) it raises (decision/tolerance branch exercised with an injected reproduced-NDCG value, no retraining).
- Integration: train BPR under ml100k, extract its full score matrix, push through the SAME `FISRecommender` + real evaluator path, assert `bpr_sanity_check` reproduces the TASK-004 floor within |Δ| < 1e-3 (returns True); then perturb the matrix and assert it raises (HARD BLOCKER). [INTEGRATION]
**Test file:** tests/test_run_fuzzy.py

### [x] TASK-013: Implement collect_per_user_ndcg from the RecBole eval loader
**Type:** feature
**Layer:** hook
**Files:** src/group_metrics.py
**Depends on:** TASK-002
**Blocked by:** TASK-002
**Estimate:** L
**Description:** Replace the `collect_per_user_ndcg(model, test_data, config, k=3)` stub (currently `raise NotImplementedError("Phase 4")`) with a real implementation per its documented contract: iterate the RecBole eval loader (`test_data`), score all items per test user via `model.full_sort_predict` (works for `FISRecommender` and BPR; for context-aware DeepFM use the per-item `predict` fallback mirroring `trainer._full_sort_batch_eval`), mask items seen in train/valid history before ranking, take top-k, compute `ndcg_at_k(ranked, test_positives, k)` per user, and map RecBole internal user index back to the original token via `test_data._dataset.id2token(uid_field, internal_id)` so keys join with `load_user_groups`. Return `{user_id: NDCG@k}`.
**Acceptance:** for a trained model, returns a dict keyed by original user_id tokens with finite NDCG; the population mean approximately matches RecBole's reported global `NDCG@3` for the same run (within small tolerance); no longer raises `NotImplementedError`.
**Tests:**
- Integration: train a real RecBole model on ml100k, call `collect_per_user_ndcg(model, test_data, config, k=3)`, assert it no longer raises, returns a dict keyed by original `user_id` tokens (joins with `load_user_groups`) with finite NDCG, and the population mean ≈ RecBole's global NDCG@3 within a small tolerance. Requires trained model + real eval loader. [INTEGRATION]
**Test file:** tests/test_group_metrics.py

### [x] TASK-014: Add hit_at_k and cold-user / cold-item slice helpers to group_metrics
**Type:** feature
**Layer:** hook
**Files:** src/group_metrics.py
**Depends on:** TASK-013
**Blocked by:** TASK-013
**Estimate:** L
**Description:** Extend `src/group_metrics.py`. Add `hit_at_k(ranked_item_ids, relevant, k) -> float` (1.0 if any relevant item in top-k else 0.0), alongside `ndcg_at_k`. Add a `load_item_groups`-style helper mirroring `load_user_groups` that parses the `.item` atomic file. Add cold-slice construction from TRAIN interactions: `cold_users` = users with ≤5 train interactions, `cold_items` = items with ≤5 train interactions, plus helpers to restrict per-user/per-item metrics to those slices. Provide `slice_metrics(...)` / `flatten_slice_metrics(...)` that, given per-user NDCG/Hit data and train-derived counts, emits `NDCG@3` AND `Hit@3` for `all`, `cold-user`, and `cold-item` slices using MLflow-safe flat keys (reusing `sanitize_key`). Cold-item is the PASS-gate target; computed uniformly for BPR/DeepFM/A0/A1.
**Acceptance:** `hit_at_k` returns 1.0 iff a relevant item is in top-k; cold-user/cold-item slice sets contain only entities with ≤5 train interactions on a toy count map; `slice_metrics` returns flat keys for `{all, cold-user, cold-item} × {NDCG@3, Hit@3}`.
**Tests:**
- Unit: `hit_at_k` 1.0 iff a relevant item in top-k else 0.0; `cold_users`/`cold_items` select only ≤5-interaction entities on a hand-built count map; `load_item_groups` parses a tiny `.item`-format header correctly; `slice_metrics`/`flatten_slice_metrics` emit MLflow-safe flat keys (`sanitize_key`) for {all, cold-user, cold-item} × {NDCG@3, Hit@3}. (Toy inputs, in-process pure helpers — NOT integration.)
**Test file:** tests/test_group_metrics.py

### [x] TASK-015: Schedule A0/A1/floor/ceil-ref runs across all/cold-user/cold-item slices in run_all
**Type:** feature
**Layer:** infra
**Files:** src/run_all.py
**Depends on:** TASK-010, TASK-011, TASK-012, TASK-014
**Blocked by:** TASK-010, TASK-011, TASK-012, TASK-014
**Estimate:** L
**Description:** Extend `src/run_all.py` to schedule the full Track-A matrix and call the sanity gate FIRST. Add scheduling for `floor=BPR-MF` (existing `run_one`), `ceil-ref=DeepFM` (existing `run_one`), `A0` (`run_fuzzy(experiment, "A0")`), `A1` (`run_fuzzy(experiment, "A1")`). Before any FIS run, invoke `bpr_sanity_check(bpr_reference_ndcg3)` (TASK-012); if it raises, abort non-zero and do NOT run A0/A1 (hard block on red). For every scheduled model, compute uniform slice metrics (`all`, `cold-user`, `cold-item` × `NDCG@3`,`Hit@3`) via `group_metrics.slice_metrics` (TASK-014) and merge them into the metrics logged through the unchanged `log_to_mlflow`. Keep the existing `iter_matrix`/`SIDEINFO` BPR+DeepFM behavior reachable; add FIS rows additively.
**Acceptance:** `python src/run_all.py --dry_run` lists rows for BPR(floor), DeepFM(ceil-ref), A0, A1; a live run calls `bpr_sanity_check` before FIS runs and aborts non-zero if it raises; each completed run logs MLflow metrics including `ndcg_at_3` and `hit_at_3` for `all`, `cold-user`, and `cold-item` slices.
**Tests:**
- Unit: assert the scheduler enumerates the four roles BPR(floor), DeepFM(ceil-ref), A0, A1 (via `--dry_run`/matrix iterator) and existing BPR+DeepFM cells remain reachable; assert ordering — `bpr_sanity_check` invoked before any FIS run, and a raised gate causes a non-zero `SystemExit` with no A0/A1 attempted (gate stubbed to raise).
- Integration: live `run_all` over ml100k: `bpr_sanity_check` called before FIS; on green each of floor/ceil-ref/A0/A1 logs `ndcg_at_3`+`hit_at_3` for all/cold-user/cold-item (read back from MLflow); on a forced-red gate the process aborts non-zero without scheduling A0/A1. [INTEGRATION]
**Test file:** tests/test_run_all.py

### [x] TASK-016: Aggregate Track-A runs to canonical CSV (reuse aggregate.py, verify slice columns)
**Type:** test
**Layer:** infra
**Files:** src/aggregate.py
**Depends on:** TASK-015
**Blocked by:** TASK-015
**Estimate:** S
**Description:** `src/aggregate.py` is reused UNCHANGED to flatten MLflow runs into `results/baseline_summary.csv` and `results/all_runs.csv`. Verify the slice metric keys logged in TASK-015 are picked up. If the cold-slice flat keys are NOT already covered by the existing `METRIC_COLS`/`SUMMARY_METRICS` lists, add the new column names ADDITIVELY (no removals, no grouping change). Prefer no edit if the existing dynamic `runs.get` already captures them.
**Acceptance:** `python src/aggregate.py --experiment <track-a-exp>` writes `results/all_runs.csv` and `results/baseline_summary.csv` containing per-run `ndcg_at_3`/`hit_at_3` for floor/ceil-ref/A0/A1, including the cold-item slice columns needed for the PASS decision.
**Tests:** N/A — this task is itself a test task.
**Test file:** src/aggregate.py

### [x] TASK-017: Implement PASS / WEAK PASS / FAIL decision step
**Type:** feature
**Layer:** infra
**Files:** src/decide.py
**Depends on:** TASK-016
**Blocked by:** TASK-016
**Estimate:** M
**Description:** Create `src/decide.py` reading the aggregated results (from `results/all_runs.csv` / `results/baseline_summary.csv` or via `mlflow.search_runs`). Implement the locked decision rule on the cold-item slice for `NDCG@3` AND `Hit@3`, gated on a green adapter (sanity check passed): PASS = (A0 OR A1) ≥ BPR floor on cold-item for BOTH metrics; WEAK PASS = A0/A1 merely ties BPR on the cold slice; FAIL = underperforms cold-item with a green adapter. Print the verdict and the comparison table (floor vs A0 vs A1 vs DeepFM ceil-ref on cold-item). Read-only on data.
**Acceptance:** given aggregated cold-item `NDCG@3`/`Hit@3` for floor/A0/A1, `python src/decide.py` prints exactly one verdict in `{PASS, WEAK PASS, FAIL}` consistent with the locked rule, plus the comparison table.
**Tests:**
- Unit: feed hand-built cold-item NDCG@3/Hit@3 for floor/A0/A1 and assert exactly one verdict in {PASS, WEAK PASS, FAIL} matching the locked rule — A0-or-A1 ≥ floor on BOTH → PASS; exact ties → WEAK PASS; underperform on either → FAIL; assert the comparison table includes floor vs A0 vs A1 vs DeepFM on cold-item. (Toy in-memory metric dicts / tiny CSV fixture — no MLflow needed; in-process.)
**Test file:** tests/test_decide.py

### [x] TASK-018: Unit tests for membership functions (toy inputs)
**Type:** test
**Layer:** test
**Files:** tests/test_membership.py
**Depends on:** TASK-005
**Blocked by:** TASK-005
**Estimate:** S
**Description:** Create `tests/test_membership.py` with toy-input asserts for `src/fuzzy/membership.py`: `compute_breakpoints` equals `np.percentile(x,[25,50,75])`; `fuzzify` trailing dim 3; peaks at q25/q50/q75; interior rows sum ≈1; degenerate constant input handled (no NaN/div-by-zero); non-monotone breakpoints raise `ValueError`. Repo convention: plain `assert` + runnable `__main__` (NO pytest); add `tests/__init__.py` if needed; prepend `src/` to `sys.path` like `smoke_test.py`.
**Acceptance:** `python tests/test_membership.py` exits 0 with all asserts passing.
**Tests:** N/A — this task is itself a test task.
**Test file:** tests/test_membership.py

### [x] TASK-019: Unit tests for concept builder (toy inputs)
**Type:** test
**Layer:** test
**Files:** tests/test_concepts.py
**Depends on:** TASK-007
**Blocked by:** TASK-007
**Estimate:** S
**Description:** Create `tests/test_concepts.py` with toy-input asserts for `src/fuzzy/concepts.py`: 4 expected keys; `niche_factor == 1 - mainstream_appeal` (`np.allclose`); item-only `(n_items,)`, user-conditioned `(n_users, n_items)`; `genre_match` equals expected cosine for a hand-built tiny case; NaN and wrong-shape inputs raise; empty-profile triggers the documented fallback (no NaN). Plain-assert/`__main__` convention.
**Acceptance:** `python tests/test_concepts.py` exits 0 with all asserts passing.
**Tests:** N/A — this task is itself a test task.
**Test file:** tests/test_concepts.py

### [x] TASK-020: Unit tests for firing + defuzz + score_matrix (toy inputs)
**Type:** test
**Layer:** test
**Files:** tests/test_fis_score.py
**Depends on:** TASK-006, TASK-008
**Blocked by:** TASK-008
**Estimate:** M
**Description:** Create `tests/test_fis_score.py` covering BOTH `src/fuzzy/rules.py` and `src/fuzzy/fis_score.py` with toy inputs: `rules.fire` PRODUCT-then-MAX correctness on a single-rule activation; `fire(m)==fire(m, ones)`; weight-length mismatch raises; `defuzz` one-hot recommend≈0.9 / avoid≈0.1 / uniform≈0.5 over fixed centroids `(0.1,0.5,0.9)`; `score_matrix` `(n_users,n_items)` in [0,1]; A0 (`weights=None`) and ones-weights allclose; NaN-producing input raises. Plain-assert/`__main__` convention.
**Acceptance:** `python tests/test_fis_score.py` exits 0 with all asserts passing, including firing and defuzz correctness.
**Tests:** N/A — this task is itself a test task.
**Test file:** tests/test_fis_score.py

### [x] TASK-021: Unit tests for weight fitting (toy inputs)
**Type:** test
**Layer:** test
**Files:** tests/test_fit_weights.py
**Depends on:** TASK-011
**Blocked by:** TASK-011
**Estimate:** S
**Description:** Create `tests/test_fit_weights.py` with toy-input asserts for `src/fuzzy/fit_weights.py`: fitted weights finite, length `n_concepts`; on a constructed toy train set the fitted weights produce a `score_matrix` pairwise loss ≤ the A0/all-ones loss (confirms it uses the shared `rules.fire` path and actually learns); non-convergence past `max_iter` raises `RuntimeError`. Plain-assert/`__main__` convention.
**Acceptance:** `python tests/test_fit_weights.py` exits 0 with all asserts passing.
**Tests:** N/A — this task is itself a test task.
**Test file:** tests/test_fit_weights.py

### [x] TASK-022: Unit tests for group_metrics hit_at_k and cold slices (toy inputs)
**Type:** test
**Layer:** test
**Files:** tests/test_group_metrics.py
**Depends on:** TASK-014
**Blocked by:** TASK-014
**Estimate:** S
**Description:** Create `tests/test_group_metrics.py` with toy-input asserts for the new `src/group_metrics.py` helpers: `hit_at_k` 1.0 iff a relevant item is in top-k; cold-user/cold-item slice construction selects only entities with ≤5 train interactions from a toy count map; `slice_metrics`/`flatten_slice_metrics` emit MLflow-safe flat keys for `{all, cold-user, cold-item} × {NDCG@3, Hit@3}`. Do NOT test `collect_per_user_ndcg` here (requires a trained model — covered by TASK-013's integration test). Plain-assert/`__main__` convention.
**Acceptance:** `python tests/test_group_metrics.py` exits 0 with all asserts passing.
**Tests:** N/A — this task is itself a test task.
**Test file:** tests/test_group_metrics.py

## Open Questions

All open questions surfaced by the prd-parser and architect were **resolved in-context during planning** (none were high-risk/low-confidence; the dataset-name decision was confirmed by the user). Recorded here for traceability:

- **A0 vs A1 / "frozen-weight" wording** → Resolved: PASS = A0 *or* A1 ≥ BPR on cold-item (NDCG@3 *and* Hit@3); A0 is the hard gate, both are run and reported (PRD §6.1 + Q&A).
- **MF breakpoints "never hand-set"** → Resolved: hard requirement — 25/50/75 percentiles from train-split data only.
- **scikit-fuzzy** → Resolved: hard — numpy-only, no new heavy deps.
- **Sanity-check tolerance "~3 decimals"** → Resolved: `|ΔNDCG@3| < 1e-3`.
- **Train-positive masking** → Resolved: use the `GeneralRecommender` wrapper (Adapter A) so RecBole's full-ranking eval masks automatically; replicate masking only on the raw-injection fallback (Adapter B).
- **Consequent centroids** → Resolved: fixed module constants avoid=0.1, neutral=0.5, recommend=0.9.
- **w_j semantics** → Resolved: A0 freezes w_j=1; A1 optimizes {w_j} + a global temperature with MF breakpoints frozen, via the SAME `rules.fire` path.
- **Migration source (architect)** → Resolved: `data/ml-100k/raw/` absent → copy the existing byte-identical `data/ml-100k/*` atomic files into `data/ml100k/` (do NOT re-run `prepare_ml100k.py` from raw).
- **Cold-slice scope (architect)** → Resolved: in-scope and REQUIRED — the PASS gate is defined on the cold-item slice, so `collect_per_user_ndcg` + `hit_at_k` + cold-user/cold-item slicing (TASK-013/014) must be built even though the repo had deferred them to "Phase 4".

## Verifier Report

**Verdict:** PASS_WITH_WARNINGS

### Coverage Checklist

- [x] Each bullet in Desired Behaviour maps to ≥1 task (FIS build → 005–008; matrix → 008; wrapper/eval → 009/010; masking → 009; sanity gate → 012; A0/A1/floor/ceil-ref across slices → 004/014/015; MLflow → 010/015; CSV → 016; decision → 017; A1 fit → 011; cold-slice extraction → 013/014; unit tests → 018–022).
- [x] Each Locked Decision is reflected in the plan.
- [x] Each [AUTHOR_SUGGESTED] hint has a corresponding task (hints 1–7 → tasks; hint 8 "single task" is satisfied by keeping this one plan).
- [x] Migration tasks (001, 002, 004) sequenced before dependent feature tasks (012/013/015).
- [x] Each migration task has a migration-path test AND a regression test.
- [x] No out-of-scope items were planned (DeepFM is scheduled only as ceil-ref reference, not as a beat-target).
- [x] No task has "TBD", "TODO", or "implement later" in any field.
- [x] Every task has a verifiable Acceptance criterion; no unresolved [NEEDS_REVIEW] tags.
- [x] Every new module has a Deletion Test line; none read "merely moves complexity".
- [x] Every new module has a Dependency Category (concepts/membership/rules/fis_score/fit_weights = in-process; run_fuzzy/FISRecommender = remote-owned).
- [x] The one remote-owned port (FIS↔RecBole evaluator seam) names two adapters (A: GeneralRecommender wrapper; B: raw-matrix injection).
- [x] No test asserts on internal state or on a helper extracted purely for testability (TASK-009 tests at the `full_sort_predict` interface).
- [x] Necessity audit — no harmful redundancy (see Warnings re: dedicated test tasks vs. annotated Tests blocks — representational, not scope-bloating).
- [x] Wiki write-guard — no task targets `.solar-cat/wiki/**` or `/.solar-cat-app/wiki/`.

### Gaps Found

None.

### Contradictions Found

None. (The "frozen-weight" wording vs. the A1 fitted variant is reconciled by PRD §6.1 + Q&A: A0 is the frozen hard gate; A1 is an additional fitted run; PASS uses A0 *or* A1.)

### Warnings

- **Cold-slice scope:** TASK-013/014 implement cold-user/cold-item slicing + `hit_at_k` + per-user NDCG extraction that the repo had deferred to "Phase 4". Required here because the PASS gate is defined on the cold-item slice. Larger than the PRD's "reuse the repo's H1 logic" wording implies — the H1 machinery is only partially built.
- **Live infrastructure required:** TASK-002/004/010/012/013/015 integration tests (and the actual experiment runs) need the `recsys` conda env active and the MLflow Docker stack up on :5002 (`cd mlflow-docker && docker compose up -d`). Pure-numpy unit tests (005–008, 011, 014, 017) need neither.
- **Test-file representation overlap:** TASK-018–022 are the actionable test-authoring tasks; the matching Tests-block annotations on TASK-005/006/007/008/011/014 reference the same files (`tests/test_membership.py`, `tests/test_fis_score.py`, `tests/test_concepts.py`, `tests/test_fit_weights.py`, `tests/test_group_metrics.py`). Author each test file once; do not duplicate.
- **TASK-004 floor handoff:** TASK-004 records the BPR floor in MLflow; ensure the numeric value is also captured durably (constant or results note) so TASK-012 (`bpr_reference_ndcg3`) and TASK-017 can consume it without ambiguity.
- **A1-vs-A0 acceptance (TASK-011):** the "fitted loss ≤ A0 loss" acceptance is scoped to a toy train set and should hold there; on real data A1 may not beat A0 — the PRD explicitly anticipates this ("shows whether fitting helps at this tiny capacity"), so a non-improving A1 on real ml100k is NOT a failure.
- **Test convention:** the repo has no pytest dependency; tests use plain `assert` + runnable `__main__` (per `smoke_test.py`, `group_metrics.py`). All test tasks follow this; `[INTEGRATION]` tests are guarded/separate runnable scripts, not pytest markers.

## Deviations

Implementation notes vs. the plan (none alter locked decisions or eval semantics):

- **BPR floor single source of truth (review refinement #3):** implemented as `run_fuzzy.bpr_floor_ndcg3_from_mlflow` reading the BPR `ml100k` off/off run's `metrics.ndcg_at_3` back from MLflow, with a documented `BPR_FLOOR_NDCG3_FALLBACK = None` constant fallback. `bpr_sanity_check` and `decide.py` consume this.
- **Slice flat-key template pinned (refinements #1/#2):** `group_metrics.flatten_slice_metrics` emits `ndcg_at_{k}__slice__{slice}` / `hit_at_{k}__slice__{slice}` (slice ∈ {all, cold_user, cold_item}); after `sanitize_key` these become e.g. `ndcg_at_3_slice_cold_item`. `aggregate.SLICE_METRIC_COLS` was added to `METRIC_COLS`/`SUMMARY_METRICS` additively (required — `tidy()` is hardcoded, not dynamic), and `decide.py` references the identical literal keys.
- **Helper naming:** `collect_per_user_ndcg` delegates to a new `collect_per_user_metrics` (NDCG + Hit); added `atomic_path`, `hit_at_k`, `load_item_groups`, `train_interaction_counts`, `cold_entities`, `slice_metrics`/`flatten_slice_metrics` (group_metrics.py); `FISRecommender`, `run_fuzzy`, `bpr_sanity_check`, `assert_within_tolerance`, `_setup_recbole`, `_evaluate_matrix`, `_build_fis_model`, `run_baseline_with_handles`, `run_track_a`/`TRACK_A_ROLES` (run_fuzzy.py / run_all.py).
- **Post-review cleanups applied:** removed an unreachable `return True` in `bpr_sanity_check`; vectorized the genre multi-hot scatter in `_build_concept_inputs` (equivalence-verified); `np.bincount` in `train_interaction_counts`; wired the `SLICES` constant as the single source of slice names; added cross-reference comments on the three shared RecBole-setup sites (`run_one.run_one`, `_setup_recbole`, `run_baseline_with_handles`).
- **TASK-013 integration assertion** (per-user mean ≈ global NDCG@3) has no standalone runnable block in `tests/test_group_metrics.py` (kept pure-helper-only per the plan); it is exercised implicitly by the live `run_all --track_a` slice logging (PR Test Checklist).
