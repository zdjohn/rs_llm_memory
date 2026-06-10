"""Convert raw MovieLens-100K into RecBole atomic files.

Raw input  : data/ml-100k/raw/ml-100k/{u.data,u.user,u.item,u.genre}
Atomic out : data/ml-100k/{ml-100k.inter,ml-100k.user,ml-100k.item}

RecBole atomic-file conventions (1.2.0):
  - one file per entity, header line is `field:type` cells separated by TAB
  - types used here: token (categorical/id), float (numeric), token_seq (multi-valued)
  - token_seq cells join multiple tokens with a single space

Side-info encoding (frozen per experiment_plan.md sec 5):
  - user: age bucketed to the 7 standard ML buckets, gender, occupation; zip dropped
  - item: genre as a multi-hot token_seq (active genre names), release_year bucketed by decade
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 7 standard MovieLens age buckets, keyed by inclusive lower bound (ML-1M convention).
AGE_BUCKETS = [(0, "1"), (18, "18"), (25, "25"), (35, "35"), (45, "45"), (50, "50"), (56, "56")]


def age_bucket(age: int) -> str:
    label = AGE_BUCKETS[0][1]
    for lo, lab in AGE_BUCKETS:
        if age >= lo:
            label = lab
    return label


def release_decade(release_date: str) -> str:
    # release_date looks like "01-Jan-1995"; a handful of rows are empty.
    release_date = release_date.strip()
    if not release_date:
        return "unknown"
    try:
        year = int(release_date.split("-")[-1])
    except ValueError:
        return "unknown"
    return str(year - (year % 10))


def load_genre_names(raw_dir: Path) -> list[str]:
    # u.genre lines are "name|id"; order the names by their integer id.
    pairs = []
    for line in (raw_dir / "u.genre").read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit("|", 1)
        pairs.append((int(idx), name))
    return [name for _, name in sorted(pairs)]


def write_inter(raw_dir: Path, out_path: Path) -> int:
    rows = ["user_id:token\titem_id:token\trating:float\ttimestamp:float"]
    n = 0
    for line in (raw_dir / "u.data").read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        user_id, item_id, rating, ts = line.split("\t")
        rows.append(f"{user_id}\t{item_id}\t{rating}\t{ts}")
        n += 1
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return n


def write_user(raw_dir: Path, out_path: Path) -> int:
    rows = ["user_id:token\tage:token\tgender:token\toccupation:token"]
    n = 0
    for line in (raw_dir / "u.user").read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        user_id, age, gender, occupation, _zip = line.split("|")
        rows.append(f"{user_id}\t{age_bucket(int(age))}\t{gender}\t{occupation}")
        n += 1
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return n


def write_item(raw_dir: Path, out_path: Path) -> int:
    genre_names = load_genre_names(raw_dir)
    rows = ["item_id:token\tgenre:token_seq\trelease_year:token"]
    n = 0
    for line in (raw_dir / "u.item").read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        item_id, _title, release_date = parts[0], parts[1], parts[2]
        flags = parts[5:]  # 19 binary genre flags
        active = [genre_names[i] for i, f in enumerate(flags) if f == "1"]
        if not active:
            active = ["unknown"]
        rows.append(f"{item_id}\t{' '.join(active)}\t{release_decade(release_date)}")
        n += 1
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/ml-100k/raw/ml-100k")
    parser.add_argument("--out_dir", default="data/ml-100k")
    parser.add_argument("--name", default="ml-100k")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_inter = write_inter(raw_dir, out_dir / f"{args.name}.inter")
    n_user = write_user(raw_dir, out_dir / f"{args.name}.user")
    n_item = write_item(raw_dir, out_dir / f"{args.name}.item")

    print(f"wrote {n_inter} interactions -> {out_dir / (args.name + '.inter')}")
    print(f"wrote {n_user} users        -> {out_dir / (args.name + '.user')}")
    print(f"wrote {n_item} items        -> {out_dir / (args.name + '.item')}")


if __name__ == "__main__":
    main()
