"""
Build the DPO ablation conditions from the pair pool, validate them, and upload them to the PRIVATE HF
dataset repo. Prints counts only, never pair text.

  python scripts/build_conditions.py                 # build + validate
  python scripts/build_conditions.py --upload        # also upload (refuses unless the repo is private)

Conditions (same seed everywhere):
  joint8      all 8 surfaces, equal share           --total_pairs pairs (half safety, half benign)
  en_only     English only                          same size
  te_only     Telugu only                           same size
  hi_only     Hindi only                            same size
  <s>_only    any surface in --singles (e.g. hinglish, hi_en_mix)
  mixed_only  hi_en_mix, hinglish, te_en_mix, ja_en_mix   same size
  joint_all   every safety pair; benign = min(available, safety) per surface (the final-model candidate)
  joint_all_no_<fam>  joint_all minus one language family (--leave_out te,ja): leave-one-language-out
              transfer test with ~700 pairs instead of 100 (te = te + te_en_mix, ja = ja + ja_en_mix, ...)
Equal total size across the first four is the control for the transfer study.
--only writes/uploads just the named conditions, so earlier files are never overwritten by accident.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.eval.runs import read_jsonl  # noqa: E402
from src.paths import CURATED_DIR  # noqa: E402

SURFACES = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]
MIXED = ["hi_en_mix", "hinglish", "te_en_mix", "ja_en_mix"]
FAMILIES = {"en": ["en"], "hi": ["hi", "hi_en_mix", "hinglish"], "te": ["te", "te_en_mix"], "ja": ["ja", "ja_en_mix"],
            "mixed": MIXED}  # joint_all_no_mixed: monolingual-only training, tested on code-mixed surfaces
LEAK = re.compile(r"(?mi)^[\s\*_#>\-•\[\]]*(TACTIC|METHOD|MESSAGE)[\s\*_#>\-•\[\]]*:|</?\s*(scratchpad|payload)\s*>")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def validate(rows, heldout_hashes, name):
    ids = [r["pair_id"] for r in rows]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate pair_id")
    if any(r.get("goal_hash") in heldout_hashes for r in rows if r.get("goal_hash")):
        problems.append("held-out goal in training pairs")
    for r in rows:
        text = " ".join(m.get("content", "") for m in r["prompt"])  # prompt only: replies may say "Method:"
        if LEAK.search(text):
            problems.append("formatting markers in a pair's prompt")
            break
        if not r["chosen"].strip() or not r["rejected"].strip():
            problems.append("empty chosen or rejected")
            break
    if problems:
        raise SystemExit(f"[build_conditions] {name}: " + "; ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(CURATED_DIR / "pair_pool_curated.jsonl"), help="CuratorKIT-filtered pool")
    ap.add_argument("--total_pairs", type=int, default=56)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--singles", default="en,te,hi",
                    help="surfaces that get a single-surface condition <surface>_only (each needs total/2 pairs "
                         "of each kind on that surface)")
    ap.add_argument("--heldout_goals", default=os.path.join(ROOT, "..", "MTMM", "data", "heldout_goals.json"))
    ap.add_argument("--dataset_repo", default="kamikaze-san/mjb-dpo-private")
    ap.add_argument("--out_dir", default=str(CURATED_DIR / "dpo"), help="tests must pass a temp dir")
    ap.add_argument("--suffix", default="", help="file-name suffix, e.g. _p100, so earlier conditions are not overwritten")
    ap.add_argument("--leave_out", default="", help="comma list of families (en,hi,te,ja): adds joint_all_no_<fam>")
    ap.add_argument("--only", default="", help="comma list of condition names to write/upload (default: all)")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    bp = load("build_pair_pool")
    mk = load("make_dpo_condition")
    pool = read_jsonl(args.pool)
    heldout = {bp.goal_hash(g) for g in json.load(open(args.heldout_goals, encoding="utf-8"))}
    cnt = Counter((r["kind"], r["profile"]) for r in pool)
    safety_n = {p: cnt[("safety", p)] for p in SURFACES}
    benign_n = {p: cnt[("benign", p)] for p in SURFACES}
    print(f"[pool] {len(pool)} pairs | safety per surface {safety_n} | benign per surface {benign_n}")

    conds = {}
    singles = [s.strip() for s in args.singles.split(",") if s.strip()]
    keep = [c.strip() for c in args.only.split(",") if c.strip()]
    for name, profiles in [("joint8", SURFACES)] + [(f"{s}_only", [s]) for s in singles] + [("mixed_only", MIXED)]:
        if keep and name not in keep:
            continue  # not requested: do not sample it (it may not have enough pairs)
        conds[name] = mk.sample_condition(pool, profiles, args.total_pairs, safety_frac=0.5, seed=args.seed)
    allp = []
    for p in SURFACES:
        s = sorted((r for r in pool if r["kind"] == "safety" and r["profile"] == p), key=lambda r: r["pair_id"])
        b = sorted((r for r in pool if r["kind"] == "benign" and r["profile"] == p), key=lambda r: r["pair_id"])
        allp += s + b[: len(s)]
    conds["joint_all"] = allp
    for fam in [f.strip() for f in args.leave_out.split(",") if f.strip()]:
        if fam not in FAMILIES:
            raise SystemExit(f"[build_conditions] unknown family {fam!r}; choose from {sorted(FAMILIES)}")
        conds[f"joint_all_no_{fam}"] = [r for r in allp if r["profile"] not in FAMILIES[fam]]
    if keep:
        missing = [c for c in keep if c not in conds]
        if missing:
            raise SystemExit(f"[build_conditions] --only names not built: {missing}; built: {sorted(conds)}")
        conds = {c: conds[c] for c in keep}

    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'condition':<11}{'pairs':>6}{'safety':>7}{'benign':>7}   safety per surface / benign per surface")
    for name, rows in conds.items():
        validate(rows, heldout, name)
        with open(out_dir / f"{name}{args.suffix}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(mk.to_trl(r), ensure_ascii=False) + "\n")
        c = Counter((r["kind"], r["profile"]) for r in rows)
        sp = {p: c[("safety", p)] for p in SURFACES if c[("safety", p)]}
        bn = {p: c[("benign", p)] for p in SURFACES if c[("benign", p)]}
        print(f"{name:<11}{len(rows):>6}{sum(sp.values()):>7}{sum(bn.values()):>7}   {sp} / {bn}")
    print("\nAll conditions passed: unique pair ids, no held-out goals, no formatting markers, no empty replies.")

    if args.upload:
        from huggingface_hub import HfApi, get_token
        api = HfApi(token=get_token())
        info = api.repo_info(args.dataset_repo, repo_type="dataset")
        if not info.private:
            raise SystemExit(f"[build_conditions] REFUSING to upload: {args.dataset_repo} is not private")
        for name in conds:
            api.upload_file(path_or_fileobj=str(out_dir / f"{name}{args.suffix}.jsonl"), path_in_repo=f"dpo/{name}{args.suffix}.jsonl",
                            repo_id=args.dataset_repo, repo_type="dataset")
        print(f"Uploaded {len(conds)} conditions to the private repo {args.dataset_repo}.")


if __name__ == "__main__":
    main()
