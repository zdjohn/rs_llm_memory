# Side-Information Recommender Baselines

A reproducible harness for benchmarking recommender baselines (BPR-MF, DeepFM) on
MovieLens-100K under one fixed protocol, isolating the lift from **side information on
both sides** (user demographics + item metadata) and tracking per-group fairness. Built on
**RecBole 1.2.0**, tracked in **MLflow** (Docker Compose: Postgres + server).

See [`experiment_plan.md`](experiment_plan.md) for the full design, hypotheses, and matrix.

## Status

| Component | State |
|---|---|
| Env (`recsys` conda, py3.10) + pinned `requirements.txt` | ✅ done |
| ml-100k → RecBole atomic files (`src/prepare_ml100k.py`) | ✅ done |
| Shared protocol + model configs (`configs/`) | ✅ done |
| MLflow stack (`mlflow-docker/`, port **5002**) | ✅ up |
| `run_one` / `run_all` + MLflow logging | ✅ done |
| **End-to-end smoke test** (BPR + DeepFM, metrics persisted & read back) | ✅ passing |
| `aggregate.py` → canonical CSV | ✅ done |
| Per-group fairness metrics (`group_metrics.py`) | 🟡 core done; RecBole per-user wiring is Phase 4 |
| HP tuning (`hyper/`, ray/hyperopt installed) | 🟡 Phase 3 (tools installed, not yet validated) |
| LastFM dataset | ⬜ not yet (ml-100k only) |

## Quickstart

```bash
# 0. environment (already created here as the `recsys` conda env)
conda activate recsys                       # python 3.10; deps pinned in requirements.txt

# 1. build RecBole atomic files from raw ml-100k
python src/prepare_ml100k.py                # -> data/ml-100k/ml-100k.{inter,user,item}

# 2. start the MLflow tracking stack (UI at http://localhost:5002)
cd mlflow-docker && docker compose up -d --build && cd ..

# 3. run the end-to-end smoke test (train -> eval -> MLflow -> read-back verify)
python src/smoke_test.py                    # BPR; add --model DeepFM for the side-info path

# 4. run the full baseline matrix and aggregate to the canonical CSV
python src/run_all.py --quick               # drop --quick for the real training budget
python src/aggregate.py --experiment smoke-test   # -> results/baseline_summary.csv
```

## Repository layout

```
data/ml-100k/        ml-100k.{inter,user,item}  (atomic; raw extract under raw/ is gitignored)
configs/             base.yaml (shared protocol) + bpr.yaml, deepfm.yaml (model HPs only)
hyper/               bpr.hyper, deepfm.hyper     (HP search spaces, Phase 3)
src/
  prepare_ml100k.py  raw ml-100k -> RecBole atomic files
  run_one.py         single run + MLflow logging (low-level RecBole API)
  run_all.py         the experiment-matrix loop (§8/§9)
  smoke_test.py      end-to-end smoke + backend read-back verification
  aggregate.py       MLflow search_runs -> results/*.csv (source of truth)
  group_metrics.py   per-demographic-group NDCG (core done; per-user wiring = Phase 4)
results/             baseline_summary.csv, all_runs.csv
mlflow-docker/       docker-compose.yml, Dockerfile, .env   (Postgres + MLflow server)
```

## Protocol (frozen — `configs/base.yaml`)

Full-ranking eval (`mode: full`, never sampled negatives), random split `RS [0.8,0.1,0.1]`,
positive threshold `rating ≥ 4`, metrics `NDCG/Hit/Precision @ [1,3]`, `valid_metric: NDCG@3`,
seed 2020. Side info is toggled per run by including/excluding the `user:`/`item:` lines of
`load_col` (a side that is "off" is genuinely ID-only, not zeroed). `run_one.py` builds that
`load_col` from `--user_feats/--item_feats`.

Side-info encoding: user `age` → 7 standard ML buckets, `gender`, `occupation` (zip dropped);
item `genre` → multi-hot `token_seq` of active genres, `release_year` → decade bucket.

## MLflow stack

Postgres backend (concurrent writes) + MLflow server, artifacts via the server proxy.
Published on host **port 5002** (5000 = macOS ControlCenter/AirPlay, 5001 = another local
stack). Point clients at `http://localhost:5002`.

```bash
cd mlflow-docker
docker compose up -d --build     # start
docker compose down              # stop, keep data
docker compose down -v           # stop and DELETE run metadata (pgdata volume)
```

Two fixes vs. the plan's §6 sketch (both validated):
- The `command:` interpolates the backend URI directly from `.env`'s `POSTGRES_*` (Compose
  does **not** substitute a service's own `environment:` vars into `command:`).
- Artifacts use the server proxy (`--serve-artifacts`, `--default-artifact-root mlflow-artifacts:/`,
  `--artifacts-destination /mlartifacts`) so the host client never needs container-filesystem
  access. Physical artifacts land in `mlflow-docker/mlartifacts/`.

## Environment notes (load-bearing pins)

Validated on Apple Silicon (arm64), CPU/MPS. Key constraints (in `requirements.txt`):

- **`numpy==1.26.4` (`<2`)** — RecBole 1.2.0 monkeypatches `np.float_`/`np.bool`; numpy ≥2
  crashes it at `Config()` construction.
- **`torch==2.5.1` (`<2.6`)** — torch ≥2.6 flipped `torch.load(weights_only=True)`, which
  breaks RecBole's `evaluate(load_best_model=True)` checkpoint reload.
- **`kmeans-pytorch`** — undeclared but mandatory: RecBole's model registry eagerly imports
  `LDiffRec`, so even loading BPR fails without it.
- **`ray[tune]` / `hyperopt`** — only for HP tuning (§7). Avoid `recbole.quick_start` (it
  hard-imports `ray`); `run_one.py` uses the low-level API instead. RecBole declares
  `ray<=2.6.3`; the installed ray is newer, so the §7 ray-tuner path is unverified (use the
  Hyperopt tuner, or pin ray down, when starting Phase 3).

Cosmetic only: pandas `inplace` FutureWarnings and a torch `GradScaler` deprecation warning
appear during runs; neither affects results.
