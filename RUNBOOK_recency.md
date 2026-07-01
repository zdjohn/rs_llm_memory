# Run book — PPR vs. Memory-Decay FIS (local)

Step-by-step instructions to run the recency experiment on your own machine
(macOS/Apple Silicon or Linux). Every command is copy-paste and is run from the repo root
inside the `recsys` conda environment.

## 1. What this runs

Compares two rankers on the **500 most-active users** of a dataset:

- **PPR** — Personalized PageRank over the user–item graph (time-agnostic structural baseline).
- **Memory-decay FIS** — a Mamdani fuzzy inference over `relevance / recency / volatility /
  frequency` (`FIS-recency-A0` = uniform rule weights, `FIS-recency-A1` = BPR-fitted weights).

The 500 users are split **50/50** into a VAL (tuning) cohort and a TEST (reporting) cohort;
within each user the single most-recent positive is held out (**leave-last-1**) as the target.
Metrics: **MRR + Hit@{1,3,10}** (leave-last-1 makes Precision and NDCG@1 redundant). The protocol
is documented in [`configs/recency.yaml`](configs/recency.yaml).

## 2. Prerequisites

- `conda` (Miniconda/Anaconda). Python **3.10**.
- **Docker Desktop** — only for the optional MLflow UI (§7).
- Internet access to the Amazon Reviews 2023 host `mcauleylab.ucsd.edu` (only for the real
  dataset in §6; the ml100k dry run in §5 needs no network).

## 3. Environment setup

```bash
conda create -n recsys python=3.10 && conda activate recsys
pip install -r requirements.txt          # includes networkx==3.4.2 and simpful==2.12.0
```

Load-bearing pins (see the README "Environment notes" — do not bump these):

- `numpy==1.26.4` (**<2**) — RecBole 1.2.0 monkeypatches `np.float_`; numpy ≥2 crashes it.
- `torch==2.5.1` (**<2.6**) — torch ≥2.6 breaks RecBole's best-checkpoint reload.
- `kmeans-pytorch` — undeclared-but-mandatory for RecBole's model registry.

**RecBole is only needed for the optional adapter sanity gate (§8).** The PPR-vs-FIS comparison
itself imports only numpy / scipy / pandas / networkx (+ simpful for the reference cross-check),
so §4–§6 work even without a full RecBole install.

## 4. Fast sanity check (no dataset, no RecBole)

Confirm the install before downloading anything — the six new modules are pure numpy/scipy:

```bash
for t in temporal_split ppr temporal_concepts rules_recency fis_recency_spec run_recency; do
  python tests/test_$t.py
done
RUN_INTEGRATION=1 python tests/test_run_recency.py     # tiny end-to-end on ml100k
```

Each prints `test_<name> PASSED`. (`test_fis_recency_spec` self-skips if simpful is absent.)

## 5. Dry run on ml100k (no download, ~35 s)

Proves the whole path locally against the in-repo MovieLens data:

```bash
python src/materialize_ml100k.py                       # ensures data/ml100k/ml100k.inter exists
python src/run_recency.py --ppr_engine power --no_mlflow
python src/decide_recency.py
```

Expected: a VAL λ sweep, then a TEST table for PPR / FIS-recency-A0 / FIS-recency-A1.
**PPR is expected to win on ml100k** (a short 7-month window with weak co-occurrence relevance) —
this step is a plumbing check, not the scientific result. The real signal needs §6.

## 6. Real run on a mid-size Amazon category

The actual experiment — a category with multi-year timestamps (so `volatility` is meaningful):

```bash
python src/prepare_amazon.py --category Video_Games     # -> data/amazon_video_games/amazon_video_games.inter
python src/run_recency.py --inter data/amazon_video_games/amazon_video_games.inter
python src/decide_recency.py
```

Before running `run_recency`, check the line `prepare_amazon.py` prints:
`users with >=50 positives: <N>`.

- **Want N ≥ 500.** If lower, either pick a denser category (`Video_Games`, `CDs_and_Vinyl`,
  `Musical_Instruments`) or lower `--n_active` on `run_recency.py`
  (e.g. `--n_active 300` → 150/150 cohorts).
- **Catalog size:** the scorer builds a dense `(n_active × n_items)` matrix. Mid-size categories
  (tens of thousands of items) are fine. If a category's catalog is huge and memory is tight, drop
  to a smaller/denser category (a top-k candidate prefilter is the documented next step, not yet
  wired).
- **PPR engine:** the default `--ppr_engine networkx` matches the plan; use `--ppr_engine power`
  (scipy power-iteration, identical rankings) if the per-user networkx calls feel slow.

## 7. Optional — MLflow tracking

Log runs to the existing MLflow stack (UI on **http://localhost:5002**, experiment
`openba-recsys-baselines`):

```bash
cd mlflow-docker && docker compose up -d --build && cd ..
python src/run_recency.py --inter data/amazon_video_games/amazon_video_games.inter   # (no --no_mlflow)
# open http://localhost:5002 -> runs: recency-PPR, recency-FIS-recency-A0, recency-FIS-recency-A1
```

Without Docker/MLflow, always pass `--no_mlflow`; results still land in
`results/recency_summary.csv`.

## 8. Optional — adapter sanity gate (needs RecBole)

```bash
python src/run_recency.py --inter <amazon.inter> --sanity_gate
```

Trains BPR and asserts the FIS→RecBole evaluator adapter is faithful before comparing. If RecBole
is not installed, it prints a skip message and continues (the adapter was already validated in
Track A; the §6 evaluation does not depend on it).

## 9. Reading the output

`decide_recency.py` prints the TEST-cohort table and a verdict; `results/recency_summary.csv` has
both cohorts:

| column | meaning |
|---|---|
| `cohort` | `val` (tuning) or `test` (reporting) |
| `model` | `PPR`, `FIS-recency-A0`, `FIS-recency-A1` |
| `lambda` | the recency decay λ* chosen on VAL |
| `mrr` | mean reciprocal rank of the held-out target (the headline) |
| `hit_at_1/3/10` | fraction of users whose target lands in the top-1/3/10 |

Metrics are **expected values under uniform tie-breaking** (fair for the discrete FIS scores).
**Verdict:** the FIS "wins" if either A0 or A1 beats PPR on **both** MRR and Hit@10 on TEST.
Reminder: with leave-last-1 there is one relevant item per user, so Precision@k and NDCG@1 carry no
extra information beyond Hit@k.

## 10. Tuning knobs

| flag / constant | where | effect |
|---|---|---|
| `--inter <path>` | `run_recency.py` | dataset atomic `.inter` (default `data/ml100k/ml100k.inter`) |
| `--n_active` | `run_recency.py` | number of most-active users (default 500 → 250/250 cohorts) |
| `--seed` | `run_recency.py` | cohort-split + sampling seed (default 2020) |
| `--ppr_engine` | `run_recency.py` | `networkx` (default) or `power` |
| `--no_a1` | `run_recency.py` | skip the A1 weight fit (A0 only) |
| `--pos_threshold` | `prepare_amazon.py` | rating cutoff for a positive (default 4.0) |
| `DEFAULT_LAMBDAS` | `run_recency.py` | the λ sweep grid (per day); edit to widen/refine |
| `personalization_mode` | `ppr.py` | `history_items` (default) or `user_node` |

## 11. Troubleshooting

| symptom | fix |
|---|---|
| `prepare_amazon.py` fails with a proxy **403** | your network blocks the host; download `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/<Category>.jsonl.gz` manually into `data/amazon_<cat>/`, then re-run `prepare_amazon.py` (it reuses the cached `.gz`). |
| RecBole crashes at `Config()` | `numpy>=2` installed — pin `numpy==1.26.4`. |
| `evaluate(load_best_model=True)` errors | `torch>=2.6` — pin `torch==2.5.1`. |
| `only <N> users have >=2 positives; need 500` | category too sparse — denser category or lower `--n_active`. |
| MemoryError building the score matrix | catalog too large — smaller/denser category (top-k prefilter is the documented next step). |
| MLflow UI won't start on 5002 | port in use — edit `mlflow-docker/docker-compose.yml` port mapping, or run with `--no_mlflow`. |
| `import simpful` fails | optional; only the reference cross-check test uses it (`pip install simpful==2.12.0`). |
