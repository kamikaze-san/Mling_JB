"""
Import Cohere's Aya Red-teaming dataset as an extra HELD-OUT, human-written,
single-turn attack set (English + Hindi; the dataset has no Telugu/Japanese).

Never used for training or for STRATA discovery. Output matches the
`--attacks` format of run_safety_eval.py, carrying the dataset's
global/local harm label and harm categories for sliced reporting:

  python scripts/import_aya_redteaming.py
  python scripts/run_safety_eval.py --run_name base_q4_aya --gguf ... \
      --attacks data/03_curated/aya_redteaming_heldout.jsonl --slice_by global_or_local

Dataset: CohereLabs/aya_redteaming (Apache-2.0), splits named by language.
Prints counts only.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import write_jsonl, write_manifest  # noqa: E402
from src.paths import CURATED_DIR  # noqa: E402

SPLITS = {"english": "en", "hindi": "hi"}


def devanagari_share(text):
    letters = [c for c in text if c.isalpha() or 0x0900 <= ord(c) <= 0x097F]
    if not letters:
        return 0.0
    return sum(1 for c in letters if 0x0900 <= ord(c) <= 0x097F) / len(letters)


def profile_for(lang_short, text):
    """Map a prompt to our profile ids. Latin-script Hindi prompts are tagged hinglish."""
    if lang_short == "en":
        return "en"
    return "hi" if devanagari_share(text) >= 0.5 else "hinglish"


def parse_categories(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else [str(val)]
    except (TypeError, ValueError):
        return [str(raw)]


def to_records(rows, split_name):
    lang = SPLITS[split_name]
    out = []
    for i, r in enumerate(rows):
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            continue
        out.append({
            "id": f"aya_rt:{split_name}:{i}",
            "profile": profile_for(lang, prompt),
            "language": lang,
            "source": "aya_redteaming",
            "global_or_local": r.get("global_or_local") or "unknown",
            "harm_categories": parse_categories(r.get("harm_category")),
            "turns": [prompt],
        })
    return out


def stratified_sample(records, per_lang, seed=0):
    """Cap per language while keeping the global/local mix proportional."""
    if not per_lang:
        return records
    rng = random.Random(seed)
    out = []
    for lang in sorted({r["language"] for r in records}):
        rs = [r for r in records if r["language"] == lang]
        if len(rs) <= per_lang:
            out += rs
            continue
        groups = {}
        for r in rs:
            groups.setdefault(r["global_or_local"], []).append(r)
        for g, items in sorted(groups.items()):
            k = round(per_lang * len(items) / len(rs))
            out += rng.sample(items, min(k, len(items)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="english,hindi")
    ap.add_argument("--per_lang", type=int, default=0, help="0 = keep all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(CURATED_DIR / "aya_redteaming_heldout.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset
    records = []
    for split in [s.strip() for s in args.splits.split(",")]:
        ds = load_dataset("CohereLabs/aya_redteaming", split=split)
        records += to_records(list(ds), split)
    records = stratified_sample(records, args.per_lang, args.seed)
    n = write_jsonl(args.out, records)
    counts = Counter((r["profile"], r["global_or_local"]) for r in records)
    write_manifest(os.path.dirname(args.out), {"script": "import_aya_redteaming", "args": vars(args),
                                               "counts": {f"{p}|{g}": c for (p, g), c in counts.items()}},
                   name="aya_redteaming_manifest.json")
    print(f"Aya Red-teaming held-out: {n} prompts -> {args.out}")
    for (p, g), c in sorted(counts.items()):
        print(f"  {p:<10}{g:<8}{c:>6}")


if __name__ == "__main__":
    main()
