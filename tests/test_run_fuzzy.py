"""Plain-assert tests for the FIS↔RecBole seam (TASK-009/010/012).

Pure unit (runs anywhere — numpy only):
  - assert_within_tolerance returns True when |Δ| < tol, raises when |Δ| >= tol (TASK-012).

Torch/RecBole unit (needs the `recsys` env — guarded behind import availability):
  - FISRecommender construction + full_sort_predict returns the injected matrix rows;
    MODEL_TYPE is GENERAL (TASK-009).

Integration (guarded behind RUN_INTEGRATION — needs recsys env + ml100k atomic files):
  - run_fuzzy("smoke-test", "A0") full RecBole setup -> finite ndcg@3/hit@3; dict is
    log_to_mlflow-shaped; base.yaml byte-unchanged (TASK-010).
  - bpr_sanity_check reproduces the floor then raises on a perturbed matrix (TASK-012).

    python tests/test_run_fuzzy.py                          # pure unit only
    conda run -n recsys python tests/test_run_fuzzy.py      # + torch/recbole unit
    RUN_INTEGRATION=1 conda run -n recsys python tests/test_run_fuzzy.py   # + integration
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN_ONE_SHAPE_KEYS = {
    "config", "test_result", "best_valid_score", "best_valid_result",
    "train_time_sec", "peak_mem_mb", "stats", "user_feats", "item_feats",
}


def _torch_available() -> bool:
    try:
        import recbole  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ---- TASK-012 pure decision-branch unit (no retraining) ----
def test_assert_within_tolerance():
    from fuzzy.run_fuzzy import assert_within_tolerance
    assert assert_within_tolerance(0.1234, 0.1235, tol=1e-3) is True
    raised = False
    try:
        assert_within_tolerance(0.1234, 0.2000, tol=1e-3)  # |Δ| >> tol
    except RuntimeError:
        raised = True
    assert raised, "expected RuntimeError when |Δ| >= tol"
    # Non-finite reproduced value also raises.
    raised = False
    try:
        assert_within_tolerance(float("nan"), 0.1, tol=1e-3)
    except RuntimeError:
        raised = True
    assert raised, "expected RuntimeError on non-finite reproduced NDCG"


# ---- TASK-009 FISRecommender construction unit (torch/recbole) ----
def unit_fis_recommender_construction():
    import torch
    from recbole.data.interaction import Interaction
    from recbole.utils import ModelType

    from fuzzy.run_fuzzy import FISRecommender

    class StubDataset:
        def __init__(self, nu, ni):
            self._nu, self._ni = nu, ni

        def num(self, field):
            return self._nu if field == "user_id" else self._ni

    nu, ni = 4, 5
    config = {"USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
              "NEG_PREFIX": "neg_", "device": torch.device("cpu")}
    mat = np.arange(nu * ni, dtype=float).reshape(nu, ni)
    model = FISRecommender(config, StubDataset(nu, ni), score_matrix=mat)
    assert model.MODEL_TYPE == ModelType.GENERAL, model.MODEL_TYPE

    inter = Interaction({"user_id": torch.tensor([1, 3])})
    out = model.full_sort_predict(inter)
    assert isinstance(out, torch.Tensor) and out.shape == (2, ni), out.shape
    assert np.allclose(out.cpu().numpy()[0], mat[1]), out
    assert np.allclose(out.cpu().numpy()[1], mat[3]), out
    print("[unit] FISRecommender construction + full_sort_predict OK")


# ---- TASK-010 integration ----
def integration_run_fuzzy_a0():  # [INTEGRATION]
    from run_one import log_to_mlflow  # noqa: F401  (shape reference only)
    from fuzzy.run_fuzzy import run_fuzzy

    base_yaml = REPO_ROOT / "configs" / "base.yaml"
    before = hashlib.sha256(base_yaml.read_bytes()).hexdigest()

    result = run_fuzzy("track-a-smoke", "A0")
    tr = result["test_result"]
    assert RUN_ONE_SHAPE_KEYS.issubset(set(result)), set(result)
    ndcg3 = tr.get("ndcg@3", tr.get("NDCG@3"))
    hit3 = tr.get("hit@3", tr.get("Hit@3"))
    assert ndcg3 is not None and np.isfinite(ndcg3), tr
    assert hit3 is not None and np.isfinite(hit3), tr

    after = hashlib.sha256(base_yaml.read_bytes()).hexdigest()
    assert before == after, "configs/base.yaml must be byte-unchanged"
    print(f"[integration] run_fuzzy A0 ndcg@3={ndcg3:.4f} hit@3={hit3:.4f}; base.yaml unchanged")


def integration_sanity_gate():  # [INTEGRATION]
    from fuzzy.run_fuzzy import bpr_floor_ndcg3_from_mlflow, bpr_sanity_check
    floor = bpr_floor_ndcg3_from_mlflow()
    assert bpr_sanity_check(floor) is True
    # A perturbed reference must flip the gate to a raise.
    raised = False
    try:
        bpr_sanity_check(floor + 0.5)
    except RuntimeError:
        raised = True
    assert raised, "expected the sanity gate to raise on a perturbed reference"
    print("[integration] sanity gate green then red-on-perturb OK")


def main() -> int:
    test_assert_within_tolerance()
    if _torch_available():
        unit_fis_recommender_construction()
    else:
        print("(skipped torch/recbole unit — not in recsys env)")
    if os.environ.get("RUN_INTEGRATION"):
        integration_run_fuzzy_a0()
        integration_sanity_gate()
    print("test_run_fuzzy PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
