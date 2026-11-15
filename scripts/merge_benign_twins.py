"""
Merge topic twins and style twins into final data/03_curated/benign_twins_train.jsonl.
Checks for >= 70 items per surface, unique IDs, and valid schema.
Prints counts only.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SURFACES = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic_twins", default=os.path.join(ROOT, "data", "03_curated", "topic_twins_safe.jsonl"))
    ap.add_argument("--style_twins", default=os.path.join(ROOT, "data", "03_curated", "style_twins_safe.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "03_curated", "benign_twins_train.jsonl"))
    ap.add_argument("--min_per_surface", type=int, default=70)
    args = ap.parse_args()

    rows = []
    seen_ids = set()

    for path, kind in [(args.topic_twins, "topic"), (args.style_twins, "style")]:
        if not os.path.exists(path):
            print(f"[warning] {kind} twins file not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                tid = r["id"]
                if tid in seen_ids:
                    raise SystemExit(f"Duplicate twin ID found: {tid}")
                seen_ids.add(tid)
                rows.append(r)

    stats = Counter((r["profile"], r["category"]) for r in rows)
    surf_totals = Counter(r["profile"] for r in rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(rows)} benign training twins -> {args.out}")
    print(f"{'surface':<12}{'topic':>8}{'style':>8}{'total':>8}{'meets >=70?':>14}")
    print("-" * 50)
    all_ok = True
    for s in SURFACES:
        topic_count = stats[(s, "benign_twin_topic")]
        style_count = stats[(s, "benign_twin_style")]
        tot = surf_totals[s]
        ok = tot >= args.min_per_surface
        if not ok:
            all_ok = False
        print(f"{s:<12}{topic_count:>8}{style_count:>8}{tot:>8}{'YES' if ok else 'NO':>14}")

    print("-" * 50)
    tot_all = len(rows)
    print(f"{'TOTAL':<12}{sum(stats[(s, 'benign_twin_topic')] for s in SURFACES):>8}{sum(stats[(s, 'benign_twin_style')] for s in SURFACES):>8}{tot_all:>8}{'ALL PASS' if all_ok else 'INCOMPLETE':>14}")


if __name__ == "__main__":
    main()
