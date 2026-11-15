"""
Create a stratified 200-conversation subset of Aya Red-teaming held-out attacks
for Task D15:
- 100 English (50 global, 50 local)
- 100 Hindi (50 global, 50 local)
Fixed seed ensures exact reproducibility.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import write_jsonl, write_manifest
from src.paths import CURATED_DIR


def make_aya_heldout_200(input_path: str, output_path: str, seed: int = 42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found: {input_path}")

    records = []
    with open(input_path, mode="r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Buckets by (language, global_or_local)
    buckets = {
        ("en", "global"): [],
        ("en", "local"): [],
        ("hi", "global"): [],
        ("hi", "local"): []
    }

    for r in records:
        key = (r.get("language"), r.get("global_or_local"))
        if key in buckets:
            buckets[key].append(r)

    rng = random.Random(seed)
    sampled = []
    for key, items in sorted(buckets.items()):
        if len(items) < 50:
            raise ValueError(f"Bucket {key} has only {len(items)} items, need 50")
        # Sort by id first for stable determinism across platforms before sampling
        sorted_items = sorted(items, key=lambda x: str(x.get("id")))
        chosen = rng.sample(sorted_items, 50)
        sampled.extend(chosen)

    # Sort final sample by id
    sampled.sort(key=lambda x: str(x.get("id")))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = write_jsonl(output_path, sampled)

    counts = Counter((r["language"], r["global_or_local"]) for r in sampled)
    write_manifest(
        os.path.dirname(output_path),
        {
            "script": "make_aya_heldout_200.py",
            "source": input_path,
            "seed": seed,
            "total_sampled": n,
            "counts": {f"{lang}_{gl}": c for (lang, gl), c in counts.items()}
        },
        name="aya_heldout_200_manifest.json"
    )

    print(f"Created {output_path} with {n} stratified held-out conversations:")
    for (lang, gl), c in sorted(counts.items()):
        print(f"  {lang:<6} {gl:<8}: {c}")

    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(CURATED_DIR / "aya_redteaming_heldout_500.jsonl"),
        help="Input 500 or full heldout JSONL"
    )
    ap.add_argument(
        "--out",
        default=str(CURATED_DIR / "aya_redteaming_heldout_200.jsonl"),
        help="Output JSONL"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        alt = str(CURATED_DIR / "aya_redteaming_heldout.jsonl")
        if os.path.exists(alt):
            print(f"{args.input} not found, falling back to {alt}")
            args.input = alt

    make_aya_heldout_200(args.input, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
