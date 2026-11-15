"""
How many DPO safety pairs survive a harm-centric filter? Counts only; never prints text.

STRATA's scorer (and the compliance rubric) count any substantive engagement as success, so a
"successful" attack can be a harmless rebuttal or an innocent misreading (see
docs/judge_disagreement_qualitative_eval_report.md). Training on those as "rejected" replies would
teach the model to refuse harmless engagement. This script measures, per surface, how many pair
candidates Qwen3Guard also flags, using exactly the candidate selection of build_pair_pool.py.

  .venv/bin/python scripts/pair_yield_check.py \
      --strata_logs ../MTMM/logs/warm_up_attack_log_reduced_seedoff.json \
                    ../MTMM/logs/lifelong_attack_log_reduced_seedoff.json \
                    ../MTMM/logs/lifelong_attack_log_topup_hienmix.json \
      --human_pool ../MTMM/data/human_dpo_pool.jsonl --exclude_goals ../MTMM/data/heldout_goals.json

Crash trap (macOS): this loads torch on MPS, so it must not import MTMM's `framework` package or
llama.cpp. It reads JSON only. Verdicts are cached in data/04_eval_cache/pair_yield_verdicts.json
(keyed by candidate, category only), so a re-run only judges new candidates.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
ORDER = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]


def load_pair_builder():
    spec = importlib.util.spec_from_file_location("build_pair_pool", os.path.join(ROOT, "scripts", "build_pair_pool.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata_logs", nargs="+", required=True)
    ap.add_argument("--human_pool", default=None)
    ap.add_argument("--exclude_goals", default=None)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "04_eval_cache", "pair_yield_verdicts.json"))
    args = ap.parse_args()

    bp = load_pair_builder()
    excluded = set()
    if args.exclude_goals:
        excluded = {bp.goal_hash(g) for g in json.load(open(args.exclude_goals, encoding="utf-8"))}

    cands = []
    all_entries = []  # group ALL logs together: the lifelong log repeats the warm-up entries
    for p in args.strata_logs:
        all_entries += json.load(open(p, encoding="utf-8"))
    cands += list(bp.conversations_from_strata_log(all_entries, args.threshold))
    if args.human_pool:
        for h in bp.read_jsonl(args.human_pool):
            cands.append({"prompt": h["messages"], "rejected": h["rejected"], "profile": h.get("profile", "unknown"),
                          "goal": h.get("goal", ""), "source": "human", "conv_key": h["id"], "turn": None})
    kept = [c for c in cands if bp.goal_hash(c["goal"]) not in excluded]
    print(f"[yield] candidates {len(cands)} | after dropping held-out goals {len(kept)}")

    def key(c):
        return hashlib.sha256(json.dumps([c["prompt"], c["rejected"]], ensure_ascii=False).encode()).hexdigest()[:20]

    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    todo = [c for c in kept if key(c) not in cache]
    print(f"[yield] cached {len(kept) - len(todo)} | to judge {len(todo)}")
    if todo:
        from src.judge.qwen_guard import QwenGuardJudge
        judge = QwenGuardJudge()
        judge.load()
        for i, c in enumerate(todo, 1):
            v = judge.evaluate(c["prompt"][-1]["content"], c["rejected"], history=c["prompt"][:-1])
            cache[key(c)] = (v.category or "Safe|None").split("|")[0]
            if i % 50 == 0 or i == len(todo):
                json.dump(cache, open(args.cache, "w"))
                print(f"  judged {i}/{len(todo)}", flush=True)

    tab = defaultdict(Counter)
    for c in kept:
        lab = cache[key(c)]
        for k in (f"{c['source']}:{c['profile']}", f"ALL:{c['profile']}", "ALL:ALL"):
            tab[k]["n"] += 1
            tab[k][lab] += 1
    print(f"\n{'surface':<11}{'source':<8}{'cands':>6}{'Unsafe':>8}{'Controv':>8}{'Safe':>6}{'strict%':>9}{'loose%':>8}")
    for src in ("strata", "human"):
        for p in ORDER + ["unknown"]:
            r = tab.get(f"{src}:{p}")
            if r:
                print(f"{p:<11}{src:<8}{r['n']:>6}{r['Unsafe']:>8}{r['Controversial']:>8}{r['Safe']:>6}"
                      f"{100 * r['Unsafe'] / r['n']:>8.0f}%{100 * (r['Unsafe'] + r['Controversial']) / r['n']:>7.0f}%")
    print("\nPer surface (strata + human), pairs available if the rejected side must be:")
    print(f"{'surface':<11}{'any success':>12}{'Unsafe only':>13}{'Unsafe/Controv':>16}   (quota: 60 per single-language condition)")
    for p in ORDER:
        r = tab.get(f"ALL:{p}", Counter())
        print(f"{p:<11}{r['n']:>12}{r['Unsafe']:>13}{r['Unsafe'] + r['Controversial']:>16}")
    a = tab["ALL:ALL"]
    print(f"{'ALL':<11}{a['n']:>12}{a['Unsafe']:>13}{a['Unsafe'] + a['Controversial']:>16}")


if __name__ == "__main__":
    main()
