"""
Extract and filter style-matched benign twins from STRATA warm-up attack log (.json or .pkl).
Category: "benign_twin_style".
Format: {id: "twin:<surface>:<n>", profile, language, category: "benign_twin_style", prompt}
Prints counts only.
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LANGUAGE_MAP = {
    "en": "en",
    "hi": "hi",
    "hi_en_mix": "hi",
    "hinglish": "hi",
    "te": "te",
    "te_en_mix": "te",
    "ja": "ja",
    "ja_en_mix": "ja",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata_log", required=True, help="Path to warm_up_attack_log JSON or PKL file")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "03_curated", "style_twins_raw.jsonl"))
    args = ap.parse_args()

    if not os.path.exists(args.strata_log):
        raise SystemExit(f"STRATA log not found: {args.strata_log}")

    if args.strata_log.endswith(".pkl"):
        with open(args.strata_log, "rb") as f:
            data = pickle.load(f)
            entries = data.get("attack_log", []) if isinstance(data, dict) else data
    else:
        with open(args.strata_log, encoding="utf-8") as f:
            entries = json.load(f)

    print(f"Loaded {len(entries)} turns from {args.strata_log}.")
    stats = Counter()
    extracted = []
    seen_prompts = set()

    # Sort entries by request_id, iteration, turn
    valid_entries = [e for e in entries if "turn" in e and not e.get("null_probe") and not e.get("attacker_refusal")]
    valid_entries.sort(key=lambda e: (e.get("request_id", 0), e.get("iteration", 0), e.get("turn", 0)))

    for e in valid_entries:
        prompt = (e.get("clean_prompt") or e.get("prompt") or "").strip()
        profile = e.get("profile") or "unknown"
        if not prompt or profile not in LANGUAGE_MAP:
            stats[f"skipped:{profile}"] += 1
            continue

        prompt_hash = (profile, prompt)
        if prompt_hash in seen_prompts:
            stats[f"duplicate:{profile}"] += 1
            continue
        seen_prompts.add(prompt_hash)

        req_id = e.get("request_id", 0)
        iter_id = e.get("iteration", 0)
        turn_id = e.get("turn", 0)
        twin_id = f"twin:{profile}:style_r{req_id:02d}_i{iter_id}_t{turn_id}"

        extracted.append({
            "id": twin_id,
            "profile": profile,
            "language": LANGUAGE_MAP[profile],
            "category": "benign_twin_style",
            "prompt": prompt,
        })
        stats[f"extracted:{profile}"] += 1

    extracted.sort(key=lambda r: (r["profile"], r["id"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in extracted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(extracted)} candidate style twins -> {args.out}")
    print(f"{'surface':<14}{'count':>8}")
    print("-" * 22)
    for s in ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]:
        print(f"{s:<14}{stats[f'extracted:{s}']:>8}")


if __name__ == "__main__":
    main()
