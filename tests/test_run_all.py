"""Plain-assert tests for the Track-A scheduler in run_all.py (TASK-015).

Unit asserts (run anywhere — the FIS/RecBole deps are imported lazily and stubbed here):
  - the scheduler enumerates the four roles floor(BPR), ceil-ref(DeepFM), A0, A1.
  - existing baseline matrix (BPR + DeepFM cells) remains reachable via iter_matrix.
  - bpr_sanity_check is invoked BEFORE any FIS/baseline run, and a raised gate causes a
    non-zero SystemExit with NO A0/A1 attempted (gate stubbed to raise).

Integration (guarded — needs recsys env + MLflow): live run_all --track_a. [INTEGRATION]

    python tests/test_run_all.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_all  # noqa: E402


def test_track_a_roles_enumerated():
    roles = [r[0] for r in run_all.TRACK_A_ROLES]
    assert roles == ["floor", "ceil-ref", "A0", "A1"], roles
    targets = {r[0]: r[2] for r in run_all.TRACK_A_ROLES}
    assert targets["floor"] == "BPR" and targets["ceil-ref"] == "DeepFM", targets
    assert targets["A0"] == "A0" and targets["A1"] == "A1", targets


def test_baseline_matrix_still_reachable():
    cells = list(run_all.iter_matrix(["BPR", "DeepFM"], ["ml100k"], [2020]))
    models = {c[0] for c in cells}
    assert models == {"BPR", "DeepFM"}, models
    # DeepFM has both off/off and on/on side-info cells.
    deepfm_cells = [c for c in cells if c[0] == "DeepFM"]
    assert len(deepfm_cells) == 2, deepfm_cells


def _install_stub_run_fuzzy(*, gate_raises: bool, call_log: list):
    """Install a fake fuzzy.run_fuzzy module so run_track_a imports it without torch."""
    fuzzy_pkg = types.ModuleType("fuzzy")
    fuzzy_pkg.__path__ = []  # mark as package
    mod = types.ModuleType("fuzzy.run_fuzzy")

    def bpr_floor_ndcg3_from_mlflow(**kwargs):
        call_log.append("floor_lookup")
        return 0.12

    def bpr_sanity_check(*args, **kwargs):
        call_log.append("sanity_gate")
        if gate_raises:
            raise RuntimeError("forced-red sanity gate")
        return True

    def run_fuzzy(experiment, mode, **kwargs):
        call_log.append(f"run_fuzzy:{mode}")
        return {"config": {}, "test_result": {"ndcg@3": 0.1}, "_handles": None}

    def run_baseline_with_handles(model, **kwargs):
        call_log.append(f"run_baseline:{model}")
        return {"config": {}, "test_result": {"ndcg@3": 0.1}, "_handles": None}

    mod.bpr_floor_ndcg3_from_mlflow = bpr_floor_ndcg3_from_mlflow
    mod.bpr_sanity_check = bpr_sanity_check
    mod.run_fuzzy = run_fuzzy
    mod.run_baseline_with_handles = run_baseline_with_handles
    sys.modules["fuzzy"] = fuzzy_pkg
    sys.modules["fuzzy.run_fuzzy"] = mod


class _Args:
    datasets = ["ml100k"]
    experiment = "unit-test"
    tracking_uri = "http://localhost:5002"
    no_mlflow = True
    dry_run = False
    quick = True


def test_gate_raises_aborts_before_fis():
    call_log: list = []
    _install_stub_run_fuzzy(gate_raises=True, call_log=call_log)
    raised = False
    try:
        run_all.run_track_a(_Args())
    except SystemExit as e:
        raised = e.code != 0
    except RuntimeError:
        raised = True  # gate propagates -> still a hard abort before any FIS run
    assert raised, "expected a non-zero abort when the sanity gate raises"
    # The gate ran; NO A0/A1 run was attempted.
    assert "sanity_gate" in call_log, call_log
    assert not any(c.startswith("run_fuzzy") for c in call_log), call_log


def test_gate_runs_before_any_run_when_green():
    call_log: list = []
    _install_stub_run_fuzzy(gate_raises=False, call_log=call_log)
    run_all.run_track_a(_Args())
    # sanity_gate must precede the first FIS/baseline run.
    gate_idx = call_log.index("sanity_gate")
    first_run_idx = next(i for i, c in enumerate(call_log)
                         if c.startswith("run_fuzzy") or c.startswith("run_baseline"))
    assert gate_idx < first_run_idx, call_log
    assert "run_fuzzy:A0" in call_log and "run_fuzzy:A1" in call_log, call_log


def main() -> int:
    test_track_a_roles_enumerated()
    test_baseline_matrix_still_reachable()
    test_gate_raises_aborts_before_fis()
    test_gate_runs_before_any_run_when_green()
    print("test_run_all PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
