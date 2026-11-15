"""
Sample one ablation CONDITION from the pair pool, with identical size and balance
across conditions (the main control for the transfer study).

Rules:
  - exactly --total_pairs pairs, split safety / benign by --safety_frac (default 1:1)
  - equal pairs per training profile within each kind
  - fails loudly if any profile cannot supply its share (use the same
    --total_pairs for every condition so "joint is better" cannot mean "more data")

  python scripts/make_dpo_condition.py --condition hi_only   --profiles hi              --total_pairs 240
  python scripts/make_dpo_condition.py --condition te_only   --profiles te              --total_pairs 240
  python scripts/make_dpo_condition.py --condition joint     --profiles en,hi,hi_en_mix,hinglish,te,te_en_mix,ja,ja_en_mix --total_pairs 240

Output: data/03_curated/dpo/<condition>.jsonl in TRL conversational preference format.
"""

import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, write_jsonl, write_manifest  # noqa: E402
from src.paths import CURATED_DIR  # noqa: E402


class ConditionError(ValueError):
    pass


def allocate(total, profiles):
    """Split `total` across profiles as evenly as possible (deterministic)."""
    base, extra = divmod(total, len(profiles))
    return {p: base + (1 if i < extra else 0) for i, p in enumerate(profiles)}


def sample_condition(pool, profiles, total_pairs, safety_frac=0.5, seed=0):
    rng = random.Random(seed)
    n_safety = round(total_pairs * safety_frac)
    plan = {"safety": allocate(n_safety, profiles),
            "benign": allocate(total_pairs - n_safety, profiles)}
    chosen, shortfalls = [], []
    for kind, per_profile in plan.items():
        for p, k in per_profile.items():
            cands = sorted((r for r in pool if r["kind"] == kind and r["profile"] == p),
                           key=lambda r: r["pair_id"])
            if len(cands) < k:
                shortfalls.append(f"{kind}/{p}: need {k}, have {len(cands)}")
                continue
            chosen += rng.sample(cands, k)
    if shortfalls:
        raise ConditionError("insufficient pairs: " + "; ".join(shortfalls))
    rng.shuffle(chosen)
    return chosen


def to_trl(r):
    return {"prompt": r["prompt"],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
            "pair_id": r["pair_id"], "kind": r["kind"], "profile": r["profile"]}


def union_conditions(condition_rows, seed=0):
    """
    Concatenate already-built conditions into one (exact same pairs), dropping
    duplicate pair_ids. Used for the "mix data" side of mix-vs-merge.
    """
    seen, out = set(), []
    for rows in condition_rows:
        for r in rows:
            if r["pair_id"] not in seen:
                seen.add(r["pair_id"])
                out.append(r)
    random.Random(seed).shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(CURATED_DIR / "pair_pool.jsonl"))
    ap.add_argument("--condition", required=True)
    ap.add_argument("--profiles", default=None, help="comma list of training profiles")
    ap.add_argument("--total_pairs", type=int, default=None)
    ap.add_argument("--safety_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--union", default=None,
                    help="comma list of existing conditions to concatenate (mix-vs-merge control)")
    args = ap.parse_args()

    out_dir = CURATED_DIR / "dpo"
    path = out_dir / f"{args.condition}.jsonl"
    if args.union:
        parts = [read_jsonl(out_dir / f"{c.strip()}.jsonl") for c in args.union.split(",")]
        trl_rows = union_conditions(parts, args.seed)
        write_jsonl(path, trl_rows)
        counts = Counter(f"{r['kind']}:{r['profile']}" for r in trl_rows)
        write_manifest(out_dir, {"script": "make_dpo_condition", "args": vars(args),
                                 "counts": dict(counts), "pair_ids": [r["pair_id"] for r in trl_rows]},
                       name=f"{args.condition}_manifest.json")
        print(f"{args.condition}: {len(trl_rows)} pairs (union of {args.union}) -> {path}")
        for k, v in sorted(counts.items()):
            print(f"  {k:<24}{v:>5}")
        return
    if not args.profiles or not args.total_pairs:
        ap.error("--profiles and --total_pairs are required unless --union is given")

    pool = read_jsonl(args.pool)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    rows = sample_condition(pool, profiles, args.total_pairs, args.safety_frac, args.seed)
    write_jsonl(path, (to_trl(r) for r in rows))
    counts = Counter(f"{r['kind']}:{r['profile']}" for r in rows)
    write_manifest(out_dir, {"script": "make_dpo_condition", "args": vars(args),
                             "counts": dict(counts), "pair_ids": [r["pair_id"] for r in rows]},
                   name=f"{args.condition}_manifest.json")
    print(f"{args.condition}: {len(rows)} pairs -> {path}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<24}{v:>5}")


if __name__ == "__main__":
    main()
