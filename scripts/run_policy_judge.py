"""
Second judge over an existing safety run: judge the cached replies with a bring-your-own-policy
model (default: gpt-oss-safeguard-20b in LM Studio), using our frozen rubric. Then compare it with
Qwen3Guard, per language. Prints aggregates only, never prompts or replies.

  python scripts/run_policy_judge.py --run_name base_q4_aya \
      --attacks data/03_curated/aya_redteaming_heldout_500.jsonl \
      --benign data/03_curated/benign_baseline.jsonl --per_lang 50

Results are cached in data/04_eval_cache/<run>/policy_judge_<model>.jsonl, so the run can resume.
Re-running continues where it stopped; raise --per_lang (0 = all) to extend the sample.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, run_dir  # noqa: E402
from src.eval.stats import wilson_ci  # noqa: E402
from src.judge.policy_judge import PolicyJudge  # noqa: E402

DEFAULT_MODEL = "openai/gpt-oss-safeguard-20b"


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def build_items(results, attacks, benign, per_lang, seed):
    """One item per attack TURN and per benign reply; stratified sample of conversations per language."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in results:
        by[(r["kind"], r.get("language"))].append(r)
    items = []
    for (kind, lang), rows in sorted(by.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        rows = sorted(rows, key=lambda r: r["id"])
        rng.shuffle(rows)
        for r in rows[:per_lang] if per_lang else rows:
            if kind == "attack":
                turns = attacks[r["id"]]["turns"]
                for t, (resp, unsafe, jl) in enumerate(zip(r["responses"], r["turn_unsafe"], r["judge_labels"]), 1):
                    msgs = []
                    for u, a in zip(turns[:t], r["responses"][:t]):
                        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                    items.append({"key": f"attack|{r['id']}|{t}", "kind": kind, "language": lang,
                                  "qg_unsafe": bool(unsafe), "qg_controversial": str(jl).startswith("Controversial"),
                                  "messages": msgs})
            else:
                items.append({"key": f"benign|{r['id']}|1", "kind": kind, "language": lang,
                              "qg_refused": bool(r["over_refused"]),
                              "messages": [{"role": "user", "content": benign[r["id"]]["prompt"]},
                                           {"role": "assistant", "content": r["response"]}]})
    return items


def _f(x, pct=False):
    if x is None:
        return "  n/a"
    return f"{100 * x:5.1f}%" if pct else f"{x:5.2f}"


def report(items, cache):
    print("\n=== policy judge vs Qwen3Guard (per turn for attacks; per reply for benign) ===")
    print("ASR = RC1 complied rate. QG strict = Unsafe; QG loose = Unsafe or Controversial.")
    print(f"{'kind':<7}{'lang':<6}{'n':>5}{'policy':>9}{'95% CI':>15}{'QG strict':>11}{'QG loose':>10}"
          f"{'agree(strict)':>14}{'kappa':>7}{'incoh':>7}{'no label':>9}")
    for kind in ("attack", "benign"):
        groups = defaultdict(list)
        for it in items:
            if it["kind"] == kind:
                groups[str(it["language"])].append(it)
                groups["ALL"].append(it)
        for lang in sorted(groups, key=lambda x: (x == "ALL", x)):
            g = groups[lang]
            lab = [cache.get(it["key"], {}).get("label") for it in g]
            ok = [(it, l) for it, l in zip(g, lab) if l]
            n = len(ok)
            if kind == "attack":
                pol = [l == "RC1" for _, l in ok]
                qs = [it["qg_unsafe"] for it, _ in ok]
                ql = [it["qg_unsafe"] or it["qg_controversial"] for it, _ in ok]
            else:  # over-refusal: policy RC0 vs Qwen3Guard Refusal: Yes
                pol = [l == "RC0" for _, l in ok]
                qs = ql = [it["qg_refused"] for it, _ in ok]
            k = sum(pol)
            ci = wilson_ci(k, n) if n else None
            agree = sum(x == y for x, y in zip(pol, qs)) / n if n else None
            print(f"{kind:<7}{lang:<6}{n:>5}{_f(k / n if n else None, True):>9}"
                  f"{(f'[{ci[0]:.2f},{ci[1]:.2f}]' if ci else '-'):>15}"
                  f"{_f(sum(qs) / n if n else None, True):>11}{_f(sum(ql) / n if n else None, True):>10}"
                  f"{_f(agree):>14}{_f(kappa(pol, qs)):>7}"
                  f"{_f(sum(l == 'RC2' for _, l in ok) / n if n else None, True):>7}{len(g) - n:>9}")
    keys = {it["key"] for it in items}
    cur = [c for k, c in cache.items() if k in keys]  # only items in this sample
    lab_all = Counter(c.get("label") for c in cur)
    rules = Counter(c.get("rule_id", "")[:5] for c in cur if c.get("label"))
    fin = Counter(c.get("finish") for c in cur)
    print(f"\nlabels: {dict(lab_all)} | rule ids: {dict(rules.most_common(8))} | finish: {dict(fin)}")
    print("For benign rows, 'policy' and 'QG' are refusal rates (ORR); agreement is on refused vs answered.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--attacks", required=True)
    ap.add_argument("--benign", required=True)
    ap.add_argument("--api_base", default="http://localhost:1234/v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--policy", default="configs/judge/safeguard_policy.md")
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--per_lang", type=int, default=50, help="conversations per (kind, language); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    d = run_dir(args.run_name)
    results = read_jsonl(d / "safety_results.jsonl")
    attacks = {r["id"]: r for r in read_jsonl(args.attacks)}
    benign = {r["id"]: r for r in read_jsonl(args.benign)}
    items = build_items(results, attacks, benign, args.per_lang, args.seed)

    cache_path = d / f"policy_judge_{re.sub(r'[^A-Za-z0-9._-]+', '_', args.model)}_{args.reasoning}.jsonl"
    cache = {}
    if cache_path.exists():
        for r in read_jsonl(cache_path):
            if r.get("label"):
                cache[r["key"]] = r
    todo = [it for it in items if it["key"] not in cache]
    print(f"[policy_judge] model={args.model} reasoning={args.reasoning} | items={len(items)} "
          f"| cached={len(items) - len(todo)} | to judge={0 if args.report_only else len(todo)}")

    if not args.report_only and todo:
        judge = PolicyJudge(args.api_base, args.model, args.policy, reasoning=args.reasoning,
                            max_tokens=args.max_tokens)
        with open(cache_path, "a", encoding="utf-8") as f:
            for i, it in enumerate(todo, 1):
                res = judge.classify(it["messages"])
                rec = {"key": it["key"], **res}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if res.get("label"):
                    cache[it["key"]] = rec
                if i % 25 == 0 or i == len(todo):
                    print(f"  judged {i}/{len(todo)} (no label so far: {i - sum(1 for x in todo[:i] if x['key'] in cache)})",
                          flush=True)
    report(items, cache)


if __name__ == "__main__":
    main()
