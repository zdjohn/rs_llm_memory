"""Download a mid-size Amazon Reviews 2023 category -> RecBole atomic `.inter` file.

Source: the McAuley-Lab "Amazon Reviews 2023" per-category raw review dumps
(`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/<Cat>.jsonl.gz`).
Each JSONL line carries `user_id`, `parent_asin` (item), `rating` (1-5), and `timestamp`
(unix milliseconds). We keep positives (rating >= threshold), convert timestamp to seconds, and
write `data/amazon_<cat>/amazon_<cat>.inter` with the standard atomic header
`user_id:token  item_id:token  rating:float  timestamp:float`.

Proxy-aware (uses HTTPS_PROXY / the env's CA bundle if set) with retry/backoff. After writing,
reports user/item/interaction counts and how many users have >= 50 positives, so you can confirm
the category yields ~500 active users before running `run_recency.py`.

    python src/prepare_amazon.py --category Digital_Music
    python src/prepare_amazon.py --category Video_Games --pos_threshold 4.0
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
            "review_categories/{cat}.jsonl.gz")

# A few mid-size categories (raw review counts, approximate) suitable for the dense matrix.
SUGGESTED = {
    "Digital_Music": "~130K reviews — smallest; fast iteration",
    "Video_Games": "~4.6M reviews — classic mid-size; dense active users",
    "Musical_Instruments": "~3M reviews",
    "Gift_Cards": "~150K reviews — very small",
    "Subscription_Boxes": "~16K reviews — tiny smoke test",
}


def _download(url: str, dest: Path, *, retries: int = 4) -> Path:
    """Download `url` -> `dest` with exponential backoff; honors env proxy/CA settings."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            print(f"    GET {url} (attempt {attempt + 1}/{retries})")
            with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return dest
        except Exception as exc:  # noqa: BLE001 - retry on any network error
            last = exc
            wait = 2 ** (attempt + 1)
            print(f"    download failed ({type(exc).__name__}: {exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last}")


def _iter_reviews(gz_path: Path):
    """Yield (user_id, item_id, rating, ts_seconds) from a 2023 raw review .jsonl.gz."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            user = rec.get("user_id")
            item = rec.get("parent_asin") or rec.get("asin")
            rating = rec.get("rating")
            ts = rec.get("timestamp")
            if user is None or item is None or rating is None or ts is None:
                continue
            # 2023 timestamps are unix milliseconds -> seconds.
            yield str(user), str(item), float(rating), float(ts) / 1000.0


def prepare(category: str, *, pos_threshold: float = 4.0, keep_gz: bool = False) -> Path:
    """Download + convert one category. Returns the written atomic `.inter` path."""
    cat_key = category.lower()
    out_dir = REPO_ROOT / "data" / f"amazon_{cat_key}"
    gz_path = out_dir / f"{category}.jsonl.gz"
    inter_path = out_dir / f"amazon_{cat_key}.inter"

    if not gz_path.exists():
        _download(BASE_URL.format(cat=category), gz_path)
    else:
        print(f"    reusing cached {gz_path}")

    n_lines = n_pos = 0
    users, items = set(), set()
    user_counts: dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    with inter_path.open("w", encoding="utf-8") as out:
        out.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for user, item, rating, ts in _iter_reviews(gz_path):
            n_lines += 1
            if rating < pos_threshold:
                continue
            n_pos += 1
            users.add(user)
            items.add(item)
            user_counts[user] = user_counts.get(user, 0) + 1
            out.write(f"{user}\t{item}\t{rating:.1f}\t{ts:.1f}\n")

    active50 = sum(1 for c in user_counts.values() if c >= 50)
    active20 = sum(1 for c in user_counts.values() if c >= 20)
    print(f"[done] {category}: {n_lines} reviews, {n_pos} positives (rating>={pos_threshold})")
    print(f"       users={len(users)} items={len(items)}")
    print(f"       users with >=50 positives: {active50}   >=20 positives: {active20}")
    if active50 < 500:
        print(f"       NOTE: only {active50} users have >=50 positives — for n_active=500 the "
              "least-active selected users will have fewer; consider a denser category or lower "
              "--n_active in run_recency.py.")
    print(f"       wrote {inter_path}")
    if not keep_gz:
        gz_path.unlink(missing_ok=True)
    return inter_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="Digital_Music",
                   help=f"Amazon 2023 category. Suggested: {', '.join(SUGGESTED)}")
    p.add_argument("--pos_threshold", type=float, default=4.0)
    p.add_argument("--keep_gz", action="store_true", help="Keep the downloaded .jsonl.gz.")
    args = p.parse_args()
    prepare(args.category, pos_threshold=args.pos_threshold, keep_gz=args.keep_gz)


if __name__ == "__main__":
    main()
