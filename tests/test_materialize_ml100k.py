"""Plain-assert tests for src/materialize_ml100k.py (TASK-001). Pure stdlib — runs anywhere.

  - Unit/migration: materialize() creates the three ml100k targets, each byte-identical to
    its ml-100k source; idempotent (second call does not raise, reports skips).
  - Regression: the original ml-100k atomic files are byte-unchanged after materialize.

    python tests/test_materialize_ml100k.py
"""
from __future__ import annotations

import filecmp
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import materialize_ml100k as mat  # noqa: E402

SUFFIXES = ("inter", "user", "item")


def _src(suffix):
    return REPO_ROOT / "data" / "ml-100k" / f"ml-100k.{suffix}"


def _dst(suffix):
    return REPO_ROOT / "data" / "ml100k" / f"ml100k.{suffix}"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_materialize_creates_byte_identical():
    src_digests = {s: _digest(_src(s)) for s in SUFFIXES}
    written = mat.materialize()
    assert len(written) == 3, written
    for s in SUFFIXES:
        assert _dst(s).exists(), _dst(s)
        assert filecmp.cmp(_src(s), _dst(s), shallow=False), s
    # Regression: sources unchanged.
    for s in SUFFIXES:
        assert _digest(_src(s)) == src_digests[s], f"source {s} changed"


def test_idempotent_second_call():
    mat.materialize()
    written = mat.materialize()  # must not raise; reports skips internally
    assert len(written) == 3, written
    for s in SUFFIXES:
        assert filecmp.cmp(_src(s), _dst(s), shallow=False), s


def main() -> int:
    test_materialize_creates_byte_identical()
    test_idempotent_second_call()
    print("test_materialize_ml100k PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
