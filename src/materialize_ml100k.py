"""Materialize the `ml100k`-named atomic files (locked dataset name).

RecBole 1.2.0 special-cases the dataset name `ml-100k` and overrides `data_path` with
its bundled example, so the project pins the name `ml100k` instead. The atomic files
embed no internal dataset name, so we simply byte-copy the already-materialized
`data/ml-100k/ml-100k.{inter,user,item}` into `data/ml100k/ml100k.{inter,user,item}`.

ADDITIVE: `data/ml-100k/` is never deleted or modified. The raw archive
`data/ml-100k/raw/` does not exist in this worktree, so we copy (we do NOT re-run
`prepare_ml100k.py` from raw).

    python src/materialize_ml100k.py
"""
from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_DATASET = "ml-100k"
DST_DATASET = "ml100k"
SUFFIXES = ("inter", "user", "item")


def materialize() -> list[Path]:
    """Copy the three ml-100k atomic files to ml100k/, idempotently.

    Returns the list of destination paths. A target that is already byte-identical to
    its source is skipped (no rewrite); otherwise it is (re)copied. Raises FileNotFoundError
    if a source atomic file is missing.
    """
    src_dir = REPO_ROOT / "data" / SRC_DATASET
    dst_dir = REPO_ROOT / "data" / DST_DATASET
    dst_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for suffix in SUFFIXES:
        src = src_dir / f"{SRC_DATASET}.{suffix}"
        dst = dst_dir / f"{DST_DATASET}.{suffix}"
        if not src.exists():
            raise FileNotFoundError(f"source atomic file missing: {src}")
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            print(f"[skip] {dst} already byte-identical to {src}")
        else:
            shutil.copyfile(src, dst)
            print(f"[copy] {src} -> {dst}")
        written.append(dst)
    return written


def main() -> None:
    paths = materialize()
    print(f"materialized {len(paths)} atomic files under data/{DST_DATASET}/")


if __name__ == "__main__":
    main()
