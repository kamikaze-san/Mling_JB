#!/usr/bin/env python3
"""Paired bootstrap 95% CI for the FLORES chrF++ change of each adapter run against the base model.

Uses the saved hypotheses (<run>/flores_hypotheses.json, same 200 devtest sentences per direction for every run)
and the local FLORES references; resamples sentence indices (the same indices for both systems) and recomputes
CORPUS chrF++ (sacrebleu CHRF word_order=2, as run_capability_eval.py) from per-sentence statistics.
Prints and saves numbers only (no benchmark text).

  python scripts/flores_bootstrap.py                         # every run with hypotheses
  python scripts/flores_bootstrap.py --runs joint_all_v3_lr5e-5_e2 --n_boot 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.runs import read_jsonl, run_dir  # noqa: E402
from src.paths import BENCHMARKS_DIR  # noqa: E402

BASELINES = {"base_q4", "base_q4_v2", "base_q4_train", "base_q4_aya", "base_q4_hf"}


def sentence_stats(chrf, hyps, refs):
    return np.array(chrf._extract_corpus_statistics(hyps, [refs]), dtype=np.int64)


def corpus(chrf, stats_sum):
    return chrf._compute_score_from_stats(list(stats_sum)).score


def compare(chrf, refs, hyp_a, hyp_b, idx):
    sa, sb = sentence_stats(chrf, hyp_a, refs), sentence_stats(chrf, hyp_b, refs)
    obs = corpus(chrf, sb.sum(0)) - corpus(chrf, sa.sum(0))
    diffs = np.array([corpus(chrf, sb[i].sum(0)) - corpus(chrf, sa[i].sum(0)) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))  # two-sided bootstrap p
    return obs, lo, hi, p


def main():
    from sacrebleu.metrics import CHRF
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--compare", default="base_q4_v2")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chrf = CHRF(word_order=2)
    base = json.load(open(Path(run_dir(args.compare)) / "flores_hypotheses.json", encoding="utf-8"))
    rows = read_jsonl(BENCHMARKS_DIR / "flores_devtest.jsonl")
    cache = Path(run_dir(args.compare)).parent
    runs = args.runs or sorted(p.name for p in cache.iterdir() if p.is_dir() and p.name not in BASELINES
                               and (p / "flores_hypotheses.json").exists())
    rng = np.random.default_rng(args.seed)
    n = len(next(iter(base.values())))
    idx = [rng.integers(0, n, n) for _ in range(args.n_boot)]   # same resamples for every run/direction
    out = {}
    print(f"paired bootstrap, {args.n_boot} resamples, n={n} sentences per direction; delta = run - {args.compare}")
    for name in runs:
        hyp = json.load(open(Path(run_dir(name)) / "flores_hypotheses.json", encoding="utf-8"))
        out[name] = {}
        cells = []
        for d, hyp_b in hyp.items():
            src, tgt = d.split("-")
            refs = [r[tgt] for r in rows[:n]]
            obs, lo, hi, p = compare(chrf, refs, base[d], hyp_b, idx)
            out[name][d] = {"delta": round(obs, 2), "ci95": [round(lo, 2), round(hi, 2)], "p": round(p, 4)}
            flag = "*" if hi < 0 or lo > 0 else " "
            cells.append(f"{d} {obs:+.2f} [{lo:+.2f},{hi:+.2f}]{flag}")
        print(f"{name}\n    " + "\n    ".join(cells))
    dest = cache / "flores_bootstrap.json"
    json.dump({"compare": args.compare, "n_boot": args.n_boot, "seed": args.seed, "runs": out},
              open(dest, "w"), indent=2)
    print(f"\n* = 95% CI excludes 0 (no multiple-comparison correction). saved {dest}")


if __name__ == "__main__":
    main()
