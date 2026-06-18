# Track A — Findings: Mamdani FIS Plumbing Validation

_Task `1ac2045f-7ac7-4244-8fd8-1166a7ecba87` · PR #1 · run 2026-06-17 (`recsys` env, MLflow :5002, CPU, seed 2020)._

## TL;DR

The end-to-end Mamdani FIS pipeline is **built and validated** (adapter sanity gate green, non-degenerate score matrix). On the pre-registered cold-item gate the faked FIS **loses to the BPR floor → literal verdict FAIL**, but this is **weak faked concepts, not a mechanism/weighting/skew bug** — proven by three diagnostic probes. The cold-item question is genuinely gated on **Track B** (real guide-derived content rules), not on any further tuning of the Track-A faked structure.

> Per the PRD: _"Track A's real output is a green pipeline, not a headline number."_ That deliverable is met.

---

## 1. Setup

- **Dataset:** `ml100k` — 943 users, **full 1683-item catalog** (item file loaded for concept extraction), 55,375 rating≥4 positives.
- **Protocol:** `configs/base.yaml` frozen verbatim — `mode: full`, `RS [0.8,0.1,0.1]`, `group_by: user`, `rating ≥ 4`, seed 2020, metrics NDCG/Hit/Precision@[1,3].
- **Cold slices (H1, ≤5 train interactions):** 767 cold items (incl. ~235 with *zero* train interactions), cold users similarly.
- **Roles:** `floor` = BPR-MF, `ceil-ref` = DeepFM (warm reference), `A0` = FIS frozen weights, `A1` = FIS fitted weights.

## 2. The item-universe finding (caught by the sanity gate)

The first matrix run **hard-blocked at the sanity gate** (working as designed). Root cause: the BPR floor ranked over the inter-only universe (**1448 items**) while the FIS loads the `.item` file to build concepts → the **full catalog (1683 items)**. Different candidate sets ⇒ the cold-item comparison was **invalid**.

**Fix (commit `24f2a81`):**
- All four roles now load the `.item` file → **shared 1683-item universe** (BPR stays ID-only; universe parity only).
- `bpr_sanity_check` is **self-contained**: trains BPR on the FIS universe, reloads the **best** checkpoint (mirrors `run_one`'s `load_best_model`), and reproduces RecBole's *native* NDCG@3 through the adapter. **Now reproduces exactly (|Δ| = 0.00).**
- `train_interaction_counts` counts zero-interaction catalog items as cold (the truest cold items).
- DeepFM full-ranking for per-user slices (`commit 906d214`): join the full item-feature table per RecBole's `_full_sort_batch_eval` (chunked predict). DeepFM's per-user all-slice (0.1420) reproduces its RecBole-native global NDCG@3 exactly — validating the extraction.

## 3. Results (full 1683-item universe)

| slice | metric | BPR floor | DeepFM (ceil-ref) | A0 (frozen) | A1 (fitted) |
|---|---|---|---|---|---|
| all | NDCG@3 / Hit@3 | 0.152 / 0.321 | 0.142 / 0.280 | 0.018 / 0.049 | 0.019 / 0.049 |
| cold-user | NDCG@3 / Hit@3 | 0.136 / 0.136 | 0.136 / 0.182 | 0.023 / 0.045 | 0.023 / 0.045 |
| **cold-item** (gate) | **NDCG@3 / Hit@3** | **0.213 / 0.432** | 0.187 / 0.368 | 0.032 / 0.088 | 0.034 / 0.088 |

**`decide.py` → VERDICT: FAIL** (A0/A1 below the BPR floor on cold-item for both metrics).

**Structural confirmation of the thesis premise:** DeepFM (0.187) **loses to BPR (0.213) on cold-item** — a high-capacity learner underperforming in the cold regime, exactly the gap a high-prior model is meant to fill. The faked FIS is simply not that model.

## 4. Diagnostic probes — why FAIL, precisely

### 4.1 The matrix is **not** degenerate (rules out the PRD's "degenerate matrix" FAIL cause)
A0 score matrix (943×1683): per-user row std median **0.23**, **0%** near-constant rows, full **[0.1, 0.9]** range, 7,988 unique values. Real ranking signal exists; the adapter gate is green. So the low number is **not** a firing/defuzz/scoring bug or a degenerate matrix — the two causes the PRD's FAIL clause names.

### 4.2 Un-skewing `mainstream_appeal` → no effect (popularity is the wrong lever)
`mainstream_appeal` is min-max normalized count (heavily skewed: q25/q50/q75 = 0.002/0.017/0.076). Replacing it with a **rank-percentile** (uniform; breakpoints → 0.21/0.50/0.75, mean 0.063→0.500) leaves cold-item **unchanged at 0.032 / 0.088**.
- **Why:** cold items are unpopular *by definition* — mean `mainstream_appeal` 0.0036 (skewed) / 0.228 (un-skewed): bottom of the scale either way. The cold-item slice rewards ranking **cold** test-positives *high*; any popularity concept ranks them *low*. Re-scaling popularity changes how unpopular they look, not that they're unpopular.

### 4.3 Up-weighting `genre_match`/`era_fit` → no effect; pure-content ceiling is ~5× short
The structurally-correct lever for cold items is the **content / user-conditioned** concepts (independent of popularity). But:

| variant | cold-item NDCG@3 / Hit@3 |
|---|---|
| A0 (weights = 1) | 0.032 / 0.088 |
| content rules ×5 / ×20 / ×100 | 0.032 / 0.088 (inert) |
| pure `genre_match` | 0.023 / 0.064 (cold-**user** = 0.000) |
| **pure `era_fit`** (best content signal) | **0.044 / 0.128** |
| pure (`genre`+`era`)/2 | 0.025 / 0.080 |
| pure `genre`×`era` | 0.023 / 0.072 |

- Up-weighting the recommend rules is **inert** — uniformly scaling recommend-firing rescales every item identically, so the centroid-defuzz ranking order is invariant.
- Even the **absolute ceiling** (ranking purely by the best content concept, `era_fit`) lands at **0.044 vs the floor's 0.213 — ~5× short**. No weighting/fitting can close a 5× gap that isn't in the signal.

**Conclusion:** not the mechanism, not the weights, not the skew — **the faked content itself.** MovieLens genre/decade over hand-written rules is too weak a content prior to beat collaborative filtering on cold items. This is precisely the PRD's prediction and the reason Track B exists.

## 5. Side-findings (carry into Track B)

- **`era_fit` > `genre_match`** as a cold-item signal here (dense decade proximity vs sparse genre-cosine; the faked genre profiles rank *zero* cold-user test-positives into top-3).
- **FIS ranking is invariant to uniform recommend-rule scaling** — a property of max-aggregation + centroid defuzz. Track-B weight tuning must vary weights *differentially across output terms* (or use the A1 pairwise-BPR fit), not scale a single output's rules.
- **DeepFM** has no `full_sort_predict`; per-user slice extraction uses the item-feature-joined `predict` fallback (validated against RecBole's native global metric).

## 6. Verdict

| Dimension | Outcome |
|---|---|
| Plumbing (the Track-A deliverable) | **VALIDATED** — gate green (|Δ|=0), matrix non-degenerate, end-to-end path proven for all 4 roles |
| Pre-registered cold-item gate | **FAIL by the literal rule** (A0/A1 < BPR floor) |
| Cause of FAIL | **Weak faked concepts** — *not* a bug/degeneracy/weighting/skew issue (proven) |
| Go/No-Go for Track B | **GO** — collect the headphones guide corpus; the plumbing is the only gate that's now closed |

---

## 7. Follow-up action items

**Track B (the real test — unblocked by this validated plumbing):**
- [ ] Collect the **headphones guide corpus** (the only remaining gate per the PRD).
- [ ] Swap faked `concepts.py` for **LLM extractors over guide text** (real content priors with `source_span` provenance).
- [ ] Swap hand-written `rules.py` for **LLM-extracted rules-with-provenance**; add the Stage-3 verification loop.
- [ ] Re-point the harness at Amazon atomic files with a **temporal split** (the one protocol change from ML's random split). The FIS → fuzzify → fire → defuzz → RecBole-eval plumbing underneath is unchanged.

**Carry-over engineering notes for Track B:**
- [ ] When tuning rule weights, vary them **differentially across output terms** (avoid/neutral/recommend) or use the A1 pairwise-BPR fit — uniform single-output scaling is inert (§5).
- [ ] Prefer a **per-(u,i) scorer** over building the full 943×1683 matrix if the candidate set grows (Amazon catalog is larger); the current `score_matrix` is dense.
- [ ] `fit_weights` uses finite-difference gradient descent (slow: ~3 min for 943×1683). Consider an analytic/autodiff gradient for the larger Track-B corpus.

**Optional Track-A cleanups (do not affect the verdict; defer unless useful):**
- [ ] Align `mainstream_appeal` to the PRD's literal "popularity **percentile**" wording (rank-percentile). Probed: **zero effect** on results — cosmetic/spec-fidelity only.
- [ ] Add a dedicated `RUN_INTEGRATION`-guarded test asserting `collect_per_user_metrics` mean ≈ RecBole global NDCG@3 for both a general (BPR) and a context-aware (DeepFM) model.

## 8. Reproduction

```bash
conda activate recsys                       # numpy 1.26.4, torch 2.5.1, recbole 1.2.0
cd mlflow-docker && docker compose up -d && cd ..   # MLflow on :5002

python src/materialize_ml100k.py            # data/ml100k/ml100k.* (idempotent copy from ml-100k)
python src/run_one.py --model BPR --dataset ml100k --user_feats off --item_feats off   # records the BPR floor
python src/run_all.py --track_a             # sanity gate (hard blocker) -> floor/ceil-ref/A0/A1 + slices -> MLflow
python src/aggregate.py --experiment openba-recsys-baselines    # -> results/*.csv (incl. cold-item slice columns)
python src/decide.py                        # PASS / WEAK PASS / FAIL on the cold-item slice
```

MLflow runs for the canonical matrix (experiment `openba-recsys-baselines`): `floor-BPR-ml100k`, `ceil-ref-DeepFM-ml100k`, `A0-A0-ml100k`, `A1-A1-ml100k`. Probe scripts (un-skew, content-ceiling) were exploratory and intentionally **not** committed — their findings are captured in §4.
