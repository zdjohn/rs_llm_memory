# Side-Information Recommender Baselines — Experiment Project Plan

A reproducible setup for benchmarking recommender baselines on MovieLens-100K and
LastFM, with a focus on isolating the lift from **side information on both sides** — user
features (age, gender, occupation) and item metadata (genre, release year) — and tracking
per-group fairness.

---

## 1. Goal & Hypotheses

**Primary goal.** Produce a credible, reproducible baseline table comparing
interaction-only models against side-information models, on a fixed shared protocol.
(Evaluation of a custom model is left as a placeholder — see §8.)

**Hypotheses to test.**
- H1: Side information (user and/or item) gives measurable lift in **cold-start** (few interactions) but little lift once interaction history is rich. The effect should be strongest for cold users (user features) and cold items (item features).
- H2: User-feature × item-feature interactions (e.g., occupation × genre) contribute lift beyond either side alone — the case for FM/cross-style models over plain side-info concatenation.
- H3: User demographic features can improve global NDCG while widening the gap between demographic groups (the fairness concern).

**Non-goals.** State-of-the-art chasing; production serving. This is a controlled
comparison harness.

---

## 2. Scope: Models & Datasets

**Datasets.**
| Dataset | User features | Item metadata | Role |
|---|---|---|---|
| MovieLens-100K | age, gender, occupation, zip | genre (19 multi-hot), release year, title* | Primary benchmark |
| LastFM (HetRec 2011) | partial (country/age in some dumps) | artist tags (user-generated, sparse) | Secondary, music domain |

\* Title is free text — used only if embedded (TF-IDF / LM); out of scope for the baseline
table, optionally explored for your model.

**Side-information inventory (what feeds the models).**
| Side | Feature | Type | Encoding | Notes |
|---|---|---|---|---|
| User | age | categorical | 7 buckets (ML convention) | also the cold-user signal |
| User | gender | categorical | binary | fairness slice |
| User | occupation | categorical | 21 classes | fairness slice |
| User | zip | categorical | usually dropped (high-cardinality, leaky) | exclude by default |
| Item | genre | multi-hot | 19 flags | main item feature on ML-100K |
| Item | release year | numeric | bucket to decade | cold-item signal |
| Item | tags (LastFM) | multi-hot | top-k tags | noisy; LastFM only |

**Baselines.**
| Model | Side info used | Framework |
|---|---|---|
| BPR-MF | None — IDs only (interaction-only floor) | RecBole |
| DeepFM | User + item features (incl. feature interactions) | RecBole |
| **Your model** *(placeholder)* | TBD — to be defined later | Your code under same protocol |

---

## 3. Environment

```bash
conda create -n recsys python=3.10
conda activate recsys
# install torch matched to your CUDA first, then:
# (mlflow here is the CLIENT — pin it equal to the Docker server version in §6)
pip install recbole==1.2.0 mlflow==2.16.2 ray[tune] hyperopt
pip freeze > requirements.txt
```

- Pin all versions; record CUDA + driver.
- Apple Silicon (MPS) for iteration; **final numbers on a fixed CUDA GPU** — backends differ.
- Capture environment per run via git commit hash logged to MLflow.

---

## 4. Repository Layout

```
openba-recsys-baselines/
├── data/
│   ├── ml-100k/      # ml-100k.inter, ml-100k.user, ml-100k.item
│   └── lastfm/
├── configs/
│   ├── base.yaml             # shared protocol (split, seed, metrics, eval)
│   ├── bpr.yaml
│   └── deepfm.yaml
├── hyper/                    # per-model HP search spaces
│   ├── bpr.hyper
│   └── deepfm.hyper
├── src/
│   ├── run_one.py            # single run + MLflow logging
│   ├── run_all.py            # the experiment loop
│   ├── group_metrics.py      # per-demographic-group NDCG/Hit/Precision
│   └── aggregate.py          # search_runs → canonical CSV
├── results/
│   ├── baseline_summary.csv  # canonical paper numbers
│   └── group_metrics.csv
├── mlflow-docker/            # MLflow tracking stack (see §6)
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── .env
│   └── mlruns/               # artifact store
├── requirements.txt
└── README.md
```

---

## 5. Shared Protocol (the most important file)

Everything comparable lives in `base.yaml`; per-model configs only add model HPs.

```yaml
# configs/base.yaml
seed: 2020
reproducibility: true

load_col:
  inter: [user_id, item_id, rating, timestamp]
  user:  [user_id, age, gender, occupation]   # user side info; toggled on/off per run
  item:  [item_id, genre, release_year]        # item side info; toggled on/off per run
# NOTE: zip intentionally excluded (high-cardinality, privacy-leaky).
# BPR ignores user/item columns regardless (ID-only); DeepFM consumes them.

# explicit -> implicit for ranking
val_interval:
  rating: "[4,inf)"
unused_col:
  inter: [rating, timestamp]

eval_args:
  split: {RS: [0.8, 0.1, 0.1]}
  order: RO            # random; switch to TO/LS only if adding sequential models
  group_by: user
  mode: full           # full ranking — NEVER sampled negatives for the paper table

metrics: [NDCG, Hit, Precision]
topk: [1, 3]
valid_metric: NDCG@3
```

**Frozen decisions** (keep constant across every run):
- Full ranking at eval (`mode: full`) — sampled negatives inflate and break comparability.
- Random split (RO) — revisit only if the model lineup gains sequential models.
- Positive threshold rating ≥ 4.
- **Side-info preprocessing (fixed once, used everywhere):** `release_year` derived from the raw ML-100K release-date string and bucketed by decade; `genre` kept as the 19 multi-hot flags; `age` in the 7 standard ML buckets; `occupation` as 21 classes; `zip` dropped. When item/user features are toggled "off" for a run, the column is excluded from `load_col` (not zeroed), so the model is genuinely ID-only on that side.

---

## 6. Tracking (MLflow via Docker Compose)

MLflow runs locally as a Docker Compose stack: a **Postgres** backend store (handles
concurrent writes from many seed runs, unlike SQLite) plus the **MLflow server**, with
artifacts on a local `./mlruns/` volume. The training code runs on the host and talks to
the server over HTTP.

### 6.1 Stack layout

```
mlflow-docker/
├── docker-compose.yml   # postgres + mlflow server
├── Dockerfile           # mlflow image with psycopg2
├── .env                 # postgres credentials (edit before non-local use)
└── mlruns/              # artifact store (created on first run)
```

### 6.2 docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:16
    container_name: mlflow-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-mlflow}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlflow}
      POSTGRES_DB: ${POSTGRES_DB:-mlflow}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-mlflow} -d ${POSTGRES_DB:-mlflow}"]
      interval: 5s
      timeout: 5s
      retries: 5
    # DB port not exposed to host by default — only the MLflow server reaches it
    # over the compose network. Uncomment to inspect the DB directly:
    # ports:
    #   - "5432:5432"

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mlflow-server
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      BACKEND_STORE_URI: postgresql://${POSTGRES_USER:-mlflow}:${POSTGRES_PASSWORD:-mlflow}@postgres:5432/${POSTGRES_DB:-mlflow}
      ARTIFACT_ROOT: /mlruns
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: >
      mlflow server
      --backend-store-uri ${BACKEND_STORE_URI}
      --default-artifact-root ${ARTIFACT_ROOT}
      --host 0.0.0.0
      --port 5000

volumes:
  pgdata:
```

### 6.3 Dockerfile

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*
# Keep this mlflow version EQUAL to the client version in the recsys env
RUN pip install --no-cache-dir mlflow==2.16.2 psycopg2-binary==2.9.9
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1
```

### 6.4 .env

```ini
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow
```
Defaults are fine for localhost; change them before this touches any shared host.

### 6.5 Run / stop

```bash
cd mlflow-docker
docker compose up -d --build      # start; UI at http://localhost:5000
docker compose down               # stop, keep data
docker compose down -v            # stop and DELETE run metadata (pgdata volume)
```
`down -v` wipes Postgres metadata but leaves `./mlruns/` artifacts on disk; clear that
directory too for a full reset.

### 6.6 Point experiment code at the server

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("openba-recsys-baselines")
```

### 6.7 What to log per run

- *Params (comparability axes):* model, dataset, seed, user_feats (on/off), item_feats (on/off), split, eval_mode, resolved HPs, git commit.
- *Metrics:* NDCG@1/3, Hit@1/3, Precision@1/3, train_time_sec, peak_mem. Sanitize `@`→`_at_` (MLflow rejects `@`).
- *Per-group metrics:* `ndcg_at_3__{attr}__{group}` flat keys for gender / age_bucket / occupation.

### 6.8 Backups & source of truth

- Metadata: `docker compose exec postgres pg_dump -U mlflow mlflow > backup.sql`
- Artifacts: back up `./mlruns/`.
- **Canonical paper numbers come from a flat CSV you control** (`results/baseline_summary.csv`),
  regenerated by `aggregate.py`. The MLflow UI is the exploration layer, not the source of truth.

### 6.9 Version alignment

Keep the `mlflow` version in the Dockerfile (server, 2.16.2) equal to the `mlflow` you
`pip install` in the recsys client env, or the REST API can mismatch.

---

## 7. Hyperparameter Tuning (fairness of comparison)

Under-tuned baselines invalidate the whole table (Rendle et al.). Therefore:
- **Equal budget** per model — same number of trials (e.g., 50) via Hyperopt/Ray Tune.
- Same search dimensions where applicable: learning rate, embedding size, regularization, # layers, dropout.
- Tune on `valid_metric` (NDCG@3) on the validation split only; report on test.
- Log the search and the resolved best HPs to MLflow.

```bash
python run_hyper.py --model=DeepFM --config_files=configs/base.yaml \
  --params_file=hyper/deepfm.hyper --tool=Hyperopt
```

---

## 8. Experiment Matrix

```
models   = [BPR, DeepFM]                       # custom model added later (placeholder)
datasets = [ml-100k, lastfm]
seeds    = [2020]                              # START with 1 seed; scale to 5 once the pipeline is validated

# side-info configuration = (user_feats, item_feats), each on/off
sideinfo = depends on model:
   BPR                       -> [(off, off)]                          # ID-only floor, fixed
   DeepFM                    -> [(off,off), (on,on)]                  # no side info vs full
```

**Seeds.** Begin with a single seed (2020) to validate the end-to-end pipeline and get
preliminary numbers fast. Once runs are stable and the harness is trusted, expand to 5
seeds `[2020, 2021, 2022, 2023, 2024]` for the final mean ± std reported in the paper.
Single-seed numbers are for development only — never report them as final.

**Custom model (placeholder).** A custom model will be added to this matrix later. Its
side-info configuration and any ablation design are TBD and will be specified once the
model is defined. It will run under the same shared protocol (§5) as the baselines.

**Cold-start sub-study (H1).** Two slices on ml-100k: (a) **cold users** with ≤5 train
interactions — isolates the value of user features; (b) **cold items** with ≤5 train
interactions — isolates the value of item features. This is where side information should
help most.

---

## 9. Execution Loop

```python
for model in models:
    for dataset in datasets:
        for seed in seeds:
            for (user_feats, item_feats) in sideinfo_options(model):
                run_one(model, dataset, seed, user_feats, item_feats)   # logs to MLflow
```
- Each (model, dataset, seed, user_feats, item_feats) = one MLflow run.
- Side info is toggled by including/excluding the `user:` and `item:` lines in `load_col`.
- All models run under RecBole with the same split files and eval metrics — no separate harness needed.

---

## 10. Analysis & Outputs

1. **Main table:** NDCG@3 / Hit@3 / Precision@3 per model × dataset (from `aggregate.py`). Single value during 1-seed development; mean ± std once expanded to 5 seeds for the paper.
2. **Custom model evaluation (placeholder):** ablation/analysis design TBD once the model is defined; will reuse the same metrics and protocol.
3. **Cold-start:** lift on cold-user slice (attributable to user features) and cold-item slice (item features), vs. warm baseline.
4. **Fairness:** per-group NDCG@3 and the group gap as user features toggle on/off.
5. **Cost:** train time / memory per model (efficiency story).

---

## 11. Reproducibility Checklist

- [ ] Seeds set everywhere (`seed` + `reproducibility: true`); 1 seed for development, expand to 5 for final reported numbers.
- [ ] Full-ranking eval, fixed split, fixed positive threshold.
- [ ] Equal HP-tuning budget across baselines; resolved HPs logged.
- [ ] Git commit hash + `requirements.txt` + CUDA version recorded per run.
- [ ] Final numbers regenerated from MLflow into a versioned CSV.
- [ ] Baselines re-run under your protocol (don't copy numbers from other papers).
- [ ] README documents how to reproduce every table from scratch.

---

## 12. Suggested Timeline

| Phase | Work | Est. |
|---|---|---|
| 1 | Env, data atomic files, `base.yaml`, MLflow server | 2–3 days |
| 2 | `run_one` + `run_all` + MLflow logging working on BPR/DeepFM | 2–3 days |
| 3 | HP tuning harness, equal budgets, both RecBole baselines (BPR, DeepFM) | 3–4 days |
| 4 | Per-group metrics + cold-start sub-study | 2–3 days |
| 5 | Aggregation scripts; baseline tables. *(Custom model + its evaluation: placeholder, scoped later.)* | 3–5 days |

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Under-tuned baselines (Rendle critique) | Equal, logged HP budgets; full ranking eval |
| Side info gives negligible lift | Reframe around cold-start (user & item) + fairness, where the story is real |
| Single-run noise | Develop on 1 seed, but expand to 5 seeds (mean ± std) before reporting; test deltas against std |
| MLflow store loss / API drift | Canonical CSV in repo; artifacts backed up |
