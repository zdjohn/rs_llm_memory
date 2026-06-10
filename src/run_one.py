"""Single RecBole run + MLflow logging.

One call = one (model, dataset, seed, user_feats, item_feats) cell of the matrix = one
MLflow run. Uses RecBole's low-level API directly (NOT quick_start.run_recbole, which
hard-imports `ray`) so we control the train/eval boundary and what gets logged.

Example
-------
    python src/run_one.py --model BPR    --dataset ml-100k --user_feats off --item_feats off
    python src/run_one.py --model DeepFM  --dataset ml-100k --user_feats on  --item_feats on
    python src/run_one.py --model BPR    --dataset ml-100k --quick --experiment smoke-test
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "http://localhost:5002"  # host 5000/5001 taken; stack publishes on 5002
DEFAULT_EXPERIMENT = "openba-recsys-baselines"

# Full (side-info ON) load_col template. run_one toggles user/item per run.
LOAD_COL_TEMPLATE = {
    "inter": ["user_id", "item_id", "rating", "timestamp"],
    "user": ["user_id", "age", "gender", "occupation"],
    "item": ["item_id", "genre", "release_year"],
}


def sanitize_key(key: str) -> str:
    """MLflow 2.16.2 allows only [A-Za-z0-9_./\\- :] in metric/param keys ('@' is rejected).
    Normalize to flat lowercase snake_case: NDCG@3 -> ndcg_at_3, Recall@10 -> recall_at_10."""
    s = key.strip().lower().replace("@", "_at_").replace("%", "_pct").replace("&", "_and_")
    s = re.sub(r"[^\w./-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:250]


def _git(*args: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    try:
        return subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
    except Exception:
        return False


def peak_mem_mb() -> float | None:
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return None


def build_load_col(user_feats: bool, item_feats: bool) -> dict:
    load_col = {"inter": list(LOAD_COL_TEMPLATE["inter"])}
    if user_feats:
        load_col["user"] = list(LOAD_COL_TEMPLATE["user"])
    if item_feats:
        load_col["item"] = list(LOAD_COL_TEMPLATE["item"])
    return load_col


def run_one(
    model: str,
    dataset: str = "ml-100k",
    *,
    data_path: str = str(REPO_ROOT / "data"),
    config_files: list[str] | None = None,
    user_feats: bool = True,
    item_feats: bool = True,
    seed: int = 2020,
    epochs: int | None = None,
    quick: bool = False,
) -> dict:
    """Train + evaluate one configuration. Returns a dict with test metrics, resolved
    params and timing. Does NOT touch MLflow (the CLI wrapper handles that)."""
    # Imported lazily so `import run_one` is cheap and MLflow-only callers don't pay for torch.
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed

    if config_files is None:
        config_files = [str(REPO_ROOT / "configs" / "base.yaml")]
        model_yaml = REPO_ROOT / "configs" / f"{model.lower()}.yaml"
        if model_yaml.exists():
            config_files.append(str(model_yaml))

    config_dict: dict = {
        "data_path": data_path,
        "seed": seed,
        "load_col": build_load_col(user_feats, item_feats),
    }
    if quick:
        config_dict.update(epochs=1, stopping_step=1)
    if epochs is not None:
        config_dict["epochs"] = epochs

    config = Config(model=model, dataset=dataset,
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
    }


def log_to_mlflow(result: dict, *, tracking_uri: str, experiment: str, run_name: str) -> str:
    import mlflow

    cfg = result["config"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "git_commit": git_commit(),
            "git_dirty": str(git_dirty()),
            "model_type": str(cfg["MODEL_TYPE"]),
            "device": str(cfg["device"]),
        })
        eval_args = cfg["eval_args"]
        mlflow.log_params({
            "model": cfg["model"],
            "dataset": cfg["dataset"],
            "seed": cfg["seed"],
            "user_feats": "on" if result["user_feats"] else "off",
            "item_feats": "on" if result["item_feats"] else "off",
            "split": str(eval_args.get("split")),
            "order": eval_args.get("order"),
            "eval_mode": eval_args.get("mode"),
            "valid_metric": cfg["valid_metric"],
            "embedding_size": cfg["embedding_size"],
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "train_neg_sample_args": str(cfg["train_neg_sample_args"]),
            **{f"n_{k.split('_')[1]}": v for k, v in result["stats"].items()},
        })
        metrics = {sanitize_key(k): float(v) for k, v in result["test_result"].items()}
        metrics["train_time_sec"] = result["train_time_sec"]
        metrics["best_valid_score"] = result["best_valid_score"]
        if result["peak_mem_mb"] is not None:
            metrics["peak_mem_mb"] = result["peak_mem_mb"]
        mlflow.log_metrics(metrics)
        return run.info.run_id


def _format_metrics(test_result: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in test_result.items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="ml-100k")
    p.add_argument("--data_path", default=str(REPO_ROOT / "data"))
    p.add_argument("--config_files", nargs="*", default=None,
                   help="Override config file list (default: base.yaml + <model>.yaml)")
    p.add_argument("--user_feats", choices=["on", "off"], default="on")
    p.add_argument("--item_feats", choices=["on", "off"], default="on")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="Smoke mode: epochs=1, stopping_step=1")
    p.add_argument("--tracking_uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--no_mlflow", action="store_true", help="Run + print only; skip MLflow logging")
    args = p.parse_args()

    user_feats = args.user_feats == "on"
    item_feats = args.item_feats == "on"

    result = run_one(
        args.model, args.dataset,
        data_path=args.data_path, config_files=args.config_files,
        user_feats=user_feats, item_feats=item_feats,
        seed=args.seed, epochs=args.epochs, quick=args.quick,
    )

    run_name = f"{args.model}-{args.dataset}-s{args.seed}-u{args.user_feats}-i{args.item_feats}"
    print(f"\n=== {run_name} ===")
    print(f"stats: {result['stats']}")
    print(f"test : {_format_metrics(result['test_result'])}")
    print(f"time : {result['train_time_sec']:.1f}s   best_valid(NDCG@3)={result['best_valid_score']:.4f}")

    if not args.no_mlflow:
        run_id = log_to_mlflow(result, tracking_uri=args.tracking_uri,
                               experiment=args.experiment, run_name=run_name)
        print(f"mlflow: logged run {run_id} to '{args.experiment}' @ {args.tracking_uri}")


if __name__ == "__main__":
    main()
