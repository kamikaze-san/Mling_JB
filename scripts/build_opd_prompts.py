#!/usr/bin/env python3
"""Build the prompt set for stage 2 (on-policy distillation of capability back into the safety-tuned adapter).

Only BENIGN prompts, never attacks:
  - translation: FLORES+ **dev** sentences (never devtest, which is the capability test set), all 6 eval
    directions, same prompt wording as the capability eval. Sentences that occur in a Belebele eval passage
    (Belebele is built from FLORES text) are excluded. Stored as FLORES ids only: the HF job fetches the text
    from the official gated source, so FLORES text is never uploaded anywhere.
  - benign: borderline benign TRAINING twins (never the eval twins; any prompt that also appears in the eval
    twins is dropped), balanced over the 8 surfaces.

  python scripts/build_opd_prompts.py --name opd_v1 --per_direction 120 --twins 240          # writes locally
  python scripts/build_opd_prompts.py --name opd_v1 --per_direction 120 --twins 240 --upload # + private repo
Writes data/03_curated/opd/<name>.jsonl and prints counts only.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.capability import DEFAULT_DIRECTIONS  # noqa: E402
from src.eval.runs import read_jsonl  # noqa: E402
from src.paths import BENCHMARKS_DIR, CURATED_DIR  # noqa: E402


def twin_key(p):
    return p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, sort_keys=True)


def build(flores_dev, twins_train, twins_eval, per_direction, n_twins, seed):
    rng = random.Random(seed)
    usable = [r for r in flores_dev if not r.get("in_belebele")]
    if "in_belebele" not in (flores_dev[0] if flores_dev else {}):
        raise SystemExit("flores_dev.jsonl has no in_belebele flags: re-run scripts/download_benchmarks.py")
    items = []
    for src, tgt in DEFAULT_DIRECTIONS:            # a different random sample per direction
        for r in rng.sample(usable, min(per_direction, len(usable))):
            items.append({"type": "translation", "flores_id": r["id"], "src": src, "tgt": tgt})
    eval_keys = {twin_key(t["prompt"]) for t in twins_eval}
    by_surface = defaultdict(list)
    dropped = 0
    for t in twins_train:
        if twin_key(t["prompt"]) in eval_keys:
            dropped += 1
            continue
        by_surface[t["profile"]].append(t)
    per_surface = n_twins // max(1, len(by_surface))
    for prof in sorted(by_surface):
        for t in rng.sample(by_surface[prof], min(per_surface, len(by_surface[prof]))):
            items.append({"type": "benign", "id": t["id"], "profile": prof, "prompt": t["prompt"]})
    rng.shuffle(items)
    stats = {"flores_dev_usable": len(usable), "flores_dev_excluded_belebele": len(flores_dev) - len(usable),
             "twins_dropped_in_eval": dropped,
             "items": Counter(i["type"] for i in items),
             "translation_by_direction": Counter(f"{i['src']}-{i['tgt']}" for i in items if i["type"] == "translation"),
             "benign_by_surface": Counter(i["profile"] for i in items if i["type"] == "benign")}
    return items, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="opd_v1")
    ap.add_argument("--per_direction", type=int, default=120)
    ap.add_argument("--twins", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upload", action="store_true", help="upload opd/<name>.jsonl to the PRIVATE dataset repo")
    ap.add_argument("--dataset_repo", default="kamikaze-san/mjb-dpo-private")
    args = ap.parse_args()

    items, stats = build(read_jsonl(BENCHMARKS_DIR / "flores_dev.jsonl"),
                         read_jsonl(CURATED_DIR / "benign_twins_train_all.jsonl"),
                         read_jsonl(CURATED_DIR / "benign_twins_eval.jsonl"),
                         args.per_direction, args.twins, args.seed)
    out = CURATED_DIR / "opd" / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    for k, v in stats.items():
        print(f"  {k}: {dict(v) if isinstance(v, Counter) else v}")
    print(f"{len(items)} prompts -> {out}")
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(args.dataset_repo, repo_type="dataset").private:
            raise SystemExit(f"REFUSING: {args.dataset_repo} is not private")
        api.upload_file(path_or_fileobj=str(out), path_in_repo=f"opd/{args.name}.jsonl",
                        repo_id=args.dataset_repo, repo_type="dataset")
        print(f"uploaded -> {args.dataset_repo}/opd/{args.name}.jsonl (FLORES as ids only)")


if __name__ == "__main__":
    main()
