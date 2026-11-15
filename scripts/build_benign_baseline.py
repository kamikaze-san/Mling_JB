"""
Build the fixed benign set for over-refusal (ORR) in run_safety_eval.py --benign, from the
Aya Evaluation Suite slice fetched by download_benchmarks.py --with_aya:

  en, te : aya_human_annotated (native, human-written)
  hi     : dolly_human_edited  (human-edited translation)
  ja     : dolly_machine_translated (machine-translated; flagged in `category`)

Mixed surfaces (hi_en_mix, hinglish, te_en_mix, ja_en_mix) have no public benign set; their benign twins
come later from build_pair_pool.py (STRATA). This set measures general over-refusal only; it is
not a borderline ("sounds unsafe but is fine") set. Prints counts only.

  python scripts/download_benchmarks.py --with_aya --aya_per_lang 100
  python scripts/build_benign_baseline.py
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval.runs import read_jsonl, write_jsonl  # noqa: E402
from src.paths import CURATED_DIR  # noqa: E402

BENCH = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "aya_eval_suite.jsonl")


def to_benign(rows):
    out = []
    for r in rows:
        prompt = (r.get("prompt") or "").strip()
        if not prompt or r.get("language") not in ("en", "hi", "te", "ja"):
            continue
        # Aya eval suite ids repeat across languages (e.g. Dolly hi/ja share ids), so the language
        # is part of the id; run_safety_eval --compare and the judges pair rows by id.
        out.append({"id": f"benign:{r['language']}:{r['id']}", "profile": r["language"], "language": r["language"],
                    "category": f"aya_eval_suite/{r.get('sub_split')}", "prompt": prompt})
    ids = [o["id"] for o in out]
    assert len(ids) == len(set(ids)), "duplicate benign ids"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=BENCH)
    ap.add_argument("--out", default=str(CURATED_DIR / "benign_baseline.jsonl"))
    args = ap.parse_args()
    rows = to_benign(read_jsonl(args.src))
    n = write_jsonl(args.out, rows)
    print(f"benign baseline: {n} prompts -> {args.out}")
    for (p, c), k in sorted(Counter((r["profile"], r["category"]) for r in rows).items()):
        print(f"  {p:<4}{c:<40}{k:>5}")


if __name__ == "__main__":
    main()
