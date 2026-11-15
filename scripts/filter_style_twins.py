"""
Filter candidate style twins with Qwen3Guard prompt moderation (keeping only Safe).
Pure MPS torch script — does NOT import MTMM.framework.
Prints counts only.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.judge.qwen_guard import QwenGuardJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "03_curated", "style_twins_raw.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "03_curated", "style_twins_safe.jsonl"))
    ap.add_argument("--judge_model", default="Qwen/Qwen3Guard-Gen-4B")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"Source file not found: {args.src}")

    with open(args.src, encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(candidates)} candidate style prompts. Initializing Qwen3Guard...")
    judge = QwenGuardJudge(model_id=args.judge_model)
    judge.load()

    stats = Counter()
    safe_rows = []

    print(f"Moderating {len(candidates)} style prompts with Qwen3Guard...")
    for i, item in enumerate(candidates, 1):
        prompt = item["prompt"]
        profile = item.get("profile", "unknown")
        v = judge.judge_conversation([{"role": "user", "content": prompt}])

        is_safe = v.category.startswith("Safe") and not v.is_violation
        stats[(profile, is_safe)] += 1

        if is_safe:
            safe_rows.append(item)

        if i % 40 == 0 or i == len(candidates):
            print(f"  progress: {i}/{len(candidates)} (safe so far: {len(safe_rows)})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in safe_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(safe_rows)} Safe style twins -> {args.out}")
    print(f"{'surface':<14}{'safe':>8}{'rejected':>10}")
    print("-" * 34)
    surfaces = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]
    for s in surfaces:
        print(f"{s:<14}{stats[(s, True)]:>8}{stats[(s, False)]:>10}")


if __name__ == "__main__":
    main()
