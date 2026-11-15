"""
Split the benign twins into a TRAINING part and a held-out EVALUATION part.

The checkpoint gate in EVAL_PROTOCOL.md section 4 ("over-refusal on borderline twins rises by at most 5
points") must be measured on twins the adapter never trained on. Deterministic: within each
(surface, category), the first N by sha256(id) are held out. Counts only on stdout.

  python scripts/split_benign_twins.py            # 6 per (surface, category) = 96 eval twins
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paths import CURATED_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(CURATED_DIR / "benign_twins_train.jsonl"))
    ap.add_argument("--per_cell", type=int, default=6)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["profile"], r.get("category"))].append(r)
    held = set()
    for cell, rs in cells.items():
        rs.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        held |= {r["id"] for r in rs[: args.per_cell]}
    ev = [r for r in rows if r["id"] in held]
    tr = [r for r in rows if r["id"] not in held]
    for name, rs in (("benign_twins_eval.jsonl", ev), ("benign_twins_train_split.jsonl", tr)):
        with open(CURATED_DIR / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    print(f"eval {len(ev)} | train {len(tr)} | eval per surface {dict(Counter(r['profile'] for r in ev))}")
    assert not ({r['id'] for r in ev} & {r['id'] for r in tr})


if __name__ == "__main__":
    main()
