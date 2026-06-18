"""FIS↔RecBole evaluator seam: FISRecommender wrapper + run_fuzzy orchestrator + sanity gate.

Adapter A (preferred): `FISRecommender` subclasses RecBole's `GeneralRecommender` and returns
a precomputed (n_users, n_items) FIS score matrix from `full_sort_predict`. Because
MODEL_TYPE = GENERAL and the config uses `mode: full`, RecBole masks train-positive items
automatically during full-ranking eval — we do NOT reimplement NDCG or masking.

`run_fuzzy(experiment, weights_mode)` mirrors `run_one`'s RecBole setup (frozen base.yaml),
builds the four concepts from the TRAIN split only, computes the suitability matrix
(A0: weights=None; A1: fitted via fit_weights), wraps it in FISRecommender, runs the SAME
RecBole evaluator over test_data, and returns a `run_one`-shaped dict so `log_to_mlflow` and
`aggregate.py` consume it unchanged.

`bpr_sanity_check` pushes the trained BPR score matrix through the IDENTICAL FISRecommender
path and asserts it reproduces the recorded BPR floor NDCG@3 within |Δ| < tol. HARD BLOCKER:
it RAISES on red so downstream FIS runs do not proceed.

Single source of truth for the BPR floor (review refinement #3): the BPR `ml100k` MLflow run
(params.model=BPR, dataset=ml100k, user_feats=off, item_feats=off) — read back its
`metrics.ndcg_at_3` via `bpr_floor_ndcg3_from_mlflow`. `BPR_FLOOR_NDCG3_FALLBACK` is a
documented fallback constant only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import ModelType

from fuzzy import concepts as concepts_mod
from fuzzy import fis_score, fit_weights
from fuzzy.membership import compute_breakpoints
from run_one import DEFAULT_TRACKING_URI, peak_mem_mb

# Documented fallback only — the live source of truth is the BPR ml100k MLflow run.
BPR_FLOOR_NDCG3_FALLBACK: float | None = None


# --------------------------------------------------------------------------------------
# Adapter A: GeneralRecommender wrapper around a precomputed score matrix.
# --------------------------------------------------------------------------------------
class FISRecommender(GeneralRecommender):
    """Thin RecBole wrapper that serves a precomputed (n_users, n_items) score matrix.

    Performs NO FIS math. The matrix is injected at construction; rows are indexed by
    RecBole internal user id (row 0 = PAD). MODEL_TYPE = GENERAL so RecBole's full-ranking
    evaluator masks train-positive items automatically.
    """

    input_type = None
    MODEL_TYPE = ModelType.GENERAL

    def __init__(self, config, dataset, score_matrix=None):
        super().__init__(config, dataset)
        if score_matrix is None:
            # Fallback injection path: a matrix stashed on the config handle.
            score_matrix = config["fis_score_matrix"]
        if score_matrix is None:
            raise ValueError("FISRecommender needs a score_matrix (constructor arg or "
                             "config['fis_score_matrix'])")
        mat = np.asarray(score_matrix, dtype=np.float32)
        if mat.shape != (self.n_users, self.n_items):
            raise ValueError(
                f"score_matrix shape {mat.shape} != (n_users={self.n_users}, "
                f"n_items={self.n_items})")
        self.score_tensor = torch.as_tensor(mat, dtype=torch.float32, device=self.device)
        # A no-op parameter so RecBole's nn.Module machinery (optimizer/device) is happy.
        self._dummy = torch.nn.Parameter(torch.zeros(1, device=self.device))

    def forward(self, *args, **kwargs):  # pragma: no cover - never trained
        return self.score_tensor

    def calculate_loss(self, interaction):  # pragma: no cover - never trained
        return torch.zeros(1, device=self.device, requires_grad=True)

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        return self.score_tensor[users, items]

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        return self.score_tensor[users]  # (n_batch_users, n_items)


# --------------------------------------------------------------------------------------
# Concept-input extraction from a RecBole dataset / train split.
# --------------------------------------------------------------------------------------
def _build_concept_inputs(dataset, train_data, config):
    """Derive build_concepts inputs from the RecBole dataset + TRAIN split (train-only)."""
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    n_users = dataset.num(uid_field)
    n_items = dataset.num(iid_field)

    item_feat = dataset.item_feat
    # genre is a token_seq -> (n_items, max_len) of internal genre token ids (0 = PAD).
    genre_ids = item_feat["genre"].numpy()
    n_genre_tokens = len(dataset.field2id_token["genre"])  # includes PAD at 0
    # Vectorized multi-hot scatter (genre_ids is fixed-width (n_items, max_len); 0 = PAD).
    item_genres = np.zeros((n_items, n_genre_tokens), dtype=float)
    rows = np.repeat(np.arange(n_items), genre_ids.shape[1])
    cols = genre_ids.ravel()
    mask = cols > 0  # skip PAD
    item_genres[rows[mask], cols[mask]] = 1.0
    # drop the PAD genre column so it never contributes to cosine
    item_genres = item_genres[:, 1:] if n_genre_tokens > 1 else item_genres

    # release_year token id -> decade float (PAD/unknown -> NaN, handled in build_concepts).
    year_ids = item_feat["release_year"].numpy().astype(int)
    year_tokens = dataset.field2id_token["release_year"]
    item_decades = np.full(n_items, np.nan, dtype=float)
    for i in range(n_items):
        tok = year_tokens[year_ids[i]]
        if tok not in ("[PAD]", "unknown"):
            try:
                item_decades[i] = float(tok)
            except ValueError:
                item_decades[i] = np.nan

    # TRAIN interactions (internal ids) — train-split-only.
    inter = train_data._dataset.inter_feat
    train_users = inter[uid_field].numpy().astype(int)
    train_items = inter[iid_field].numpy().astype(int)
    train_inter = np.stack([train_users, train_items], axis=1)

    # User genre profile = sum of genre multi-hots over the user's TRAIN items.
    user_genre_profile = np.zeros((n_users, item_genres.shape[1]), dtype=float)
    np.add.at(user_genre_profile, train_users, item_genres[train_items])

    # User preferred decade = mean decade over the user's TRAIN items (ignoring unknown).
    user_pref_decade = np.zeros(n_users, dtype=float)
    counts = np.zeros(n_users, dtype=float)
    valid = np.isfinite(item_decades[train_items])
    np.add.at(user_pref_decade, train_users[valid], item_decades[train_items][valid])
    np.add.at(counts, train_users[valid], 1.0)
    user_pref_decade = np.where(counts > 0, user_pref_decade / np.where(counts > 0, counts, 1.0), 0.0)

    return dict(train_inter=train_inter, item_genres=item_genres, item_decades=item_decades,
                user_pref_decade=user_pref_decade, user_genre_profile=user_genre_profile,
                n_users=n_users, n_items=n_items)


def _train_pairs(train_data, config, max_pairs: int = 20000, seed: int = 2020) -> np.ndarray:
    """Sample (user, pos_item, neg_item) pairs from TRAIN interactions for weight fitting."""
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    inter = train_data._dataset.inter_feat
    users = inter[uid_field].numpy().astype(int)
    items = inter[iid_field].numpy().astype(int)
    n_items = config["fis_n_items"]
    rng = np.random.default_rng(seed)
    if len(users) > max_pairs:
        sel = rng.choice(len(users), size=max_pairs, replace=False)
        users, items = users[sel], items[sel]
    negs = rng.integers(1, n_items, size=len(users))  # avoid PAD 0
    return np.stack([users, items, negs], axis=1)


# --------------------------------------------------------------------------------------
# Orchestrator + evaluator.
# --------------------------------------------------------------------------------------
def _setup_recbole(dataset_name: str = "ml100k"):
    """Build Config/dataset/loaders exactly like run_one (frozen base.yaml). Returns the tuple."""
    # Shared RecBole setup (Config/create_dataset/data_preparation): keep in sync with
    # run_one.run_one and run_baseline_with_handles if the setup sequence changes.
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    config_files = [str(REPO_ROOT / "configs" / "base.yaml")]
    config_dict = {
        "data_path": str(REPO_ROOT / "data"),
        "seed": 2020,
        # FIS uses item side-info (genre/decade); inter is always loaded.
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"],
                     "item": ["item_id", "genre", "release_year"]},
    }
    # Reuse BPR's model class purely as a structural carrier for Config; FISRecommender is
    # instantiated directly (we bypass get_model).
    config = Config(model="BPR", dataset=dataset_name,
                    config_file_list=config_files, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    rb_dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, rb_dataset)
    return config, rb_dataset, train_data, valid_data, test_data


def _build_fis_model(score_matrix: np.ndarray, config, train_data):
    """Wrap a score matrix in a FISRecommender on the right device (no training)."""
    from recbole.utils import init_seed
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    return FISRecommender(config, train_data._dataset,
                          score_matrix=score_matrix).to(config["device"])


def _evaluate_matrix(score_matrix: np.ndarray, config, train_data, test_data):
    """Wrap a matrix in FISRecommender, run the SAME evaluator. Returns (test_result, model)."""
    from recbole.trainer import Trainer

    model = _build_fis_model(score_matrix, config, train_data)
    trainer = Trainer(config, model)
    # No training — evaluate the injected matrix directly (load_best_model=False).
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    return dict(test_result), model


def run_fuzzy(experiment: str, weights_mode: str, *, dataset_name: str = "ml100k") -> dict:
    """Run one FIS experiment (A0 or A1) and return a run_one-shaped result dict."""
    if weights_mode not in ("A0", "A1"):
        raise ValueError(f"weights_mode must be 'A0' or 'A1', got {weights_mode!r}")

    config, rb_dataset, train_data, valid_data, test_data = _setup_recbole(dataset_name)
    inputs = _build_concept_inputs(rb_dataset, train_data, config)
    config["fis_n_items"] = inputs["n_items"]

    concepts = concepts_mod.build_concepts(**inputs)
    breakpoints = {name: compute_breakpoints(arr) for name, arr in concepts.items()}

    t0 = time.time()
    if weights_mode == "A0":
        weights = None
    else:
        pairs = _train_pairs(train_data, config)
        weights = fit_weights.fit_weights(concepts, breakpoints, pairs)
    matrix = fis_score.score_matrix(concepts, breakpoints, weights=weights)
    fit_time = time.time() - t0

    test_result, model = _evaluate_matrix(matrix, config, train_data, test_data)

    # Surface the chosen mode in the config so log_to_mlflow records `model`.
    config.final_config_dict["model"] = f"FIS-{weights_mode}"
    return {
        "config": config,
        "test_result": dict(test_result),
        "best_valid_score": float("nan"),
        "best_valid_result": {},
        "train_time_sec": fit_time,
        "peak_mem_mb": peak_mem_mb(),
        "stats": {
            "n_users": rb_dataset.user_num,
            "n_items": rb_dataset.item_num,
            "n_inter": rb_dataset.inter_num,
        },
        "user_feats": False,
        "item_feats": True,  # FIS consumes item genre/decade side-info.
        # Non-logged handles for downstream slice-metric computation (ignored by log_to_mlflow).
        "_handles": {"model": model, "train_data": train_data, "test_data": test_data},
    }


def run_baseline_with_handles(model_name: str, *, dataset_name: str = "ml100k",
                              user_feats: bool = False, item_feats: bool = False,
                              quick: bool = False) -> dict:
    """Train a baseline (BPR/DeepFM) and return a run_one-shaped dict PLUS `_handles`.

    Mirrors run_one's low-level training but keeps the trained model + loaders so the
    scheduler can compute slice metrics without retraining. The returned dict is otherwise
    shape-compatible with log_to_mlflow / aggregate.py.
    """
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed

    from run_one import build_load_col

    # Shared RecBole setup (Config/create_dataset/data_preparation/get_model/get_trainer):
    # keep in sync with run_one.run_one and _setup_recbole if the setup sequence changes.
    config_files = [str(REPO_ROOT / "configs" / "base.yaml")]
    model_yaml = REPO_ROOT / "configs" / f"{model_name.lower()}.yaml"
    if model_yaml.exists():
        config_files.append(str(model_yaml))
    # Item-universe parity with the FIS: the FIS loads the .item file to build concepts, so it
    # ranks over the FULL item catalog (1683). Every Track-A role MUST share that candidate set
    # or the cold-item comparison is invalid (the sanity gate enforces this). Force the item
    # file load even for ID-only BPR (BPR ignores the columns; only the candidate universe matters).
    load_col = build_load_col(user_feats, item_feats)
    load_col["item"] = ["item_id", "genre", "release_year"]
    config_dict = {
        "data_path": str(REPO_ROOT / "data"),
        "seed": 2020,
        "load_col": load_col,
    }
    if quick:
        config_dict.update(epochs=1, stopping_step=1)

    config = Config(model=model_name, dataset=dataset_name,
                    config_file_list=config_files, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    rb_dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, rb_dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    net = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, net)

    t0 = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=False)
    train_time_sec = time.time() - t0
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)

    return {
        "config": config,
        "test_result": dict(test_result),
        "best_valid_score": float(best_valid_score),
        "best_valid_result": dict(best_valid_result),
        "train_time_sec": train_time_sec,
        "peak_mem_mb": peak_mem_mb(),
        "stats": {
            "n_users": rb_dataset.user_num,
            "n_items": rb_dataset.item_num,
            "n_inter": rb_dataset.inter_num,
        },
        "user_feats": user_feats,
        "item_feats": item_feats,
        "_handles": {"model": net, "train_data": train_data, "test_data": test_data},
    }


# --------------------------------------------------------------------------------------
# BPR floor reference + sanity gate.
# --------------------------------------------------------------------------------------
def bpr_floor_ndcg3_from_mlflow(*, tracking_uri: str = DEFAULT_TRACKING_URI,
                                experiment: str | None = None,
                                dataset_name: str = "ml100k") -> float:
    """Read the BPR floor NDCG@3 back from the BPR ml100k MLflow run (single source of truth).

    Queries runs with params.model=BPR, params.dataset=<dataset_name>, user_feats=off,
    item_feats=off and returns the most recent run's metrics.ndcg_at_3. Raises if absent
    (and no fallback constant is configured).
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    filter_str = (
        f"params.model = 'BPR' and params.dataset = '{dataset_name}' "
        f"and params.user_feats = 'off' and params.item_feats = 'off'")
    kwargs = dict(filter_string=filter_str, order_by=["attributes.start_time DESC"])
    runs = (mlflow.search_runs(experiment_names=[experiment], **kwargs)
            if experiment else mlflow.search_runs(search_all_experiments=True, **kwargs))
    if runs is not None and not runs.empty and "metrics.ndcg_at_3" in runs.columns:
        val = runs.iloc[0]["metrics.ndcg_at_3"]
        if val is not None and np.isfinite(val):
            return float(val)
    if BPR_FLOOR_NDCG3_FALLBACK is not None:
        return float(BPR_FLOOR_NDCG3_FALLBACK)
    raise RuntimeError(
        "BPR floor NDCG@3 not found in MLflow (no BPR ml100k off/off run) and no "
        "BPR_FLOOR_NDCG3_FALLBACK configured. Run the BPR floor first (TASK-004).")


def assert_within_tolerance(reproduced_ndcg3: float, bpr_reference_ndcg3: float,
                            tol: float = 1e-3) -> bool:
    """Pure tolerance decision shared by the sanity gate. Returns True on green; raises on red.

    HARD BLOCKER: raises RuntimeError when |reproduced - reference| >= tol or the reproduced
    value is non-finite. Extracted so the decision branch is testable without retraining BPR.
    """
    delta = abs(float(reproduced_ndcg3) - float(bpr_reference_ndcg3))
    if not np.isfinite(reproduced_ndcg3) or delta >= tol:
        raise RuntimeError(
            f"sanity gate RED: BPR-through-adapter NDCG@3={reproduced_ndcg3:.6f} vs floor "
            f"{bpr_reference_ndcg3:.6f} (|Δ|={delta:.2e} >= tol {tol:.0e}). HARD BLOCKER.")
    print(f"[ok] sanity gate GREEN: NDCG@3={reproduced_ndcg3:.6f} reproduces floor "
          f"{bpr_reference_ndcg3:.6f} (|Δ|={delta:.2e} < {tol:.0e})")
    return True


def bpr_sanity_check(bpr_reference_ndcg3: float | None = None, tol: float = 1e-3, *,
                     dataset_name: str = "ml100k") -> bool:
    """HARD BLOCKER: prove the FISRecommender adapter reproduces RecBole's native BPR eval.

    Trains BPR under `dataset_name` on the SAME item universe the FIS uses (`_setup_recbole`
    loads the .item file -> full catalog), reloads the BEST checkpoint (mirroring run_one's
    `load_best_model=True`), takes RecBole's native NDCG@3 as the reference, extracts the
    best model's dense score matrix (user_embedding @ item_embedding.T), pushes it through the
    identical FISRecommender + RecBole evaluator path used by run_fuzzy, and asserts the
    reproduced NDCG@3 matches within |Δ| < tol. Returns True on green; RAISES on red.

    The reference is computed self-consistently (same model, same universe) so the gate is not
    confounded by item-universe or best-vs-last-epoch mismatches. Pass `bpr_reference_ndcg3`
    explicitly only to assert against an externally recorded floor on the SAME universe.
    """
    from recbole.trainer import Trainer
    from recbole.utils import get_model, init_seed

    config, rb_dataset, train_data, valid_data, test_data = _setup_recbole(dataset_name)
    # Train a BPR model on the SAME splits/universe the FIS evaluates over.
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    bpr = get_model("BPR")(config, train_data._dataset).to(config["device"])
    trainer = Trainer(config, bpr)
    trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    # Reload the BEST checkpoint and take RecBole's NATIVE eval as the reference (this is the
    # exact path run_one uses for the floor). evaluate(load_best_model=True) loads the best
    # weights back into `bpr`, so the matrix extracted below is the best model, not last-epoch.
    native = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    native_ndcg3 = float(dict(native).get("ndcg@3", dict(native).get("NDCG@3", float("nan"))))
    reference = native_ndcg3 if bpr_reference_ndcg3 is None else float(bpr_reference_ndcg3)

    bpr.eval()
    with torch.no_grad():
        user_e = bpr.user_embedding.weight  # (n_users, dim)
        item_e = bpr.item_embedding.weight  # (n_items, dim)
        matrix = (user_e @ item_e.t()).cpu().numpy()

    reproduced, _ = _evaluate_matrix(matrix, config, train_data, test_data)
    repro_ndcg3 = float(reproduced.get("ndcg@3", reproduced.get("NDCG@3", float("nan"))))
    return assert_within_tolerance(repro_ndcg3, reference, tol=tol)
