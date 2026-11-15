#!/usr/bin/env python3
"""Prompt set for multi-teacher OPD (train_opd.py): translation items (FLORES dev ids only, as in opd_v2_tr) for the
chosen directions, routed to the translation teacher, plus "anchor" items = prompts of the stage-1 TRAINING safety
pairs (never held-out eval attacks), distilled from a frozen copy of the stage-1 adapter to hold safety in place.

  python scripts/build_opd_mt_prompts.py --name opd_v3_mt --directions en-hi,hi-en,en-te,te-en --anchors_per_profile 30
  python scripts/build_opd_mt_prompts.py ... --upload       # + PRIVATE dataset repo (refuses if it is not private)
Writes data/03_curated/opd/<name>.jsonl. Anchor prompts contain attack text: private repo only, never public.
Prints counts only.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPD = ROOT / "data" / "03_curated" / "opd"
PAIRS = ROOT / "data" / "03_curated" / "dpo" / "joint_all_v3.jsonl"
REPO = "kamikaze-san/mjb-dpo-private"
MAX_PROMPT_CHARS = 6000  # keeps OPD sampling batches short; drops the longest multi-turn histories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="opd_v3_mt")
    ap.add_argument("--translation_source", default="opd_v2_tr")
    ap.add_argument("--directions", default="en-hi,hi-en,en-te,te-en")
    ap.add_argument("--anchors_per_profile", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()

    dirs = set(a.directions.split(","))
    src = [json.loads(l) for l in open(OPD / f"{a.translation_source}.jsonl", encoding="utf-8") if l.strip()]
    tr = [it for it in src if it["type"] == "translation" and f"{it['src']}-{it['tgt']}" in dirs]

    by_profile = defaultdict(list)
    too_long = 0
    for r in map(json.loads, open(PAIRS, encoding="utf-8")):
        if r.get("kind") != "safety":
            continue
        if sum(len(m["content"]) for m in r["prompt"]) > MAX_PROMPT_CHARS:
            too_long += 1
            continue
        by_profile[r["profile"]].append(r)
    rng = random.Random(a.seed)
    anchors = []
    for prof in sorted(by_profile):
        for r in rng.sample(by_profile[prof], min(a.anchors_per_profile, len(by_profile[prof]))):
            anchors.append({"type": "anchor", "id": r["pair_id"], "profile": prof, "prompt": r["prompt"]})

    out = OPD / f"{a.name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for it in tr + anchors:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{out.relative_to(ROOT)}: {len(tr)} translation {dict(Counter(f'{i['src']}-{i['tgt']}' for i in tr))} | "
          f"{len(anchors)} anchor {dict(Counter(x['profile'] for x in anchors))} | "
          f"turns {dict(Counter(len(x['prompt']) for x in anchors))} | {too_long} safety prompts over "
          f"{MAX_PROMPT_CHARS} chars skipped")

    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(REPO, repo_type="dataset").private:
            raise SystemExit(f"REFUSING: {REPO} is not private")
        api.upload_file(path_or_fileobj=str(out), path_in_repo=f"opd/{a.name}.jsonl", repo_id=REPO, repo_type="dataset",
                        commit_message=f"OPD multi-teacher prompt set {a.name} (FLORES ids only + training safety prompts)")
        print(f"uploaded -> {REPO}/opd/{a.name}.jsonl (private)")


if __name__ == "__main__":
    main()
