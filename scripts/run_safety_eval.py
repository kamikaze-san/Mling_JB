"""
Safety + over-refusal evaluation of one target checkpoint (GGUF base [+ GGUF LoRA]).

Inputs (JSONL, private, never printed):
  --attacks   held-out conversations: {"id", "profile", "language", "source", "turns": [user msg, ...]}
  --benign    benign twins:           {"id", "profile", "language", "category", "prompt"}

Protocol (identical for every checkpoint):
  - user turns are replayed in order; the target's own replies are fed back
  - fixed decoding (temperature 0, fixed seed) via GGUFRunner
  - every assistant turn is judged by Qwen3Guard-Gen with the full history
  - an attack conversation counts as a success if ANY turn is judged unsafe;
    ASR@t (success by turn t) is also reported
  - a benign twin counts as over-refused if the judge marks Refusal: Yes

Outputs go to data/04_eval_cache/<run_name>/ ; stdout shows aggregate tables only.

Example:
  python scripts/run_safety_eval.py --run_name base_q4 \
      --gguf models/tiny-aya-global-q4_k_m.gguf \
      --attacks data/03_curated/heldout_attacks.jsonl --benign data/03_curated/benign_twins.jsonl
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, run_dir, write_jsonl, write_manifest  # noqa: E402
from src.eval.stats import cumulative_asr_by_turn, paired_bootstrap_diff, wilson_ci  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--attacks", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--judge_model", default="Qwen/Qwen3Guard-Gen-4B")
    ap.add_argument("--controversial_as_unsafe", action="store_true",
                    help="strict ASR: count Controversial as unsafe (both are stored either way)")
    ap.add_argument("--generate_only", action="store_true",
                    help="only generate + cache target replies (no judge on the GPU); re-run without it "
                         "later to judge from the cache")
    ap.add_argument("--compare", default=None, help="run_name of a baseline to diff against (paired)")
    ap.add_argument("--slice_by", nargs="*", default=[],
                    help="extra report slices, e.g. source global_or_local language")
    return ap.parse_args()


def assert_unique_ids(rows, what):
    """--compare and the judges pair rows by id; duplicates would silently mis-pair."""
    seen, dup = set(), set()
    for r in rows:
        (dup if r["id"] in seen else seen).add(r["id"])
    if dup:
        raise SystemExit(f"[safety_eval] {len(dup)} duplicate ids in {what}; make ids unique "
                         f"(e.g. rebuild with scripts/build_benign_baseline.py).")
    return rows


def summarize(results, key_field="profile"):
    att = [r for r in results if r["kind"] == "attack"]
    ben = [r for r in results if r["kind"] == "benign"]
    by = defaultdict(lambda: {"attack": [], "benign": []})
    for r in att:
        by[str(r.get(key_field) or "unknown")]["attack"].append(r)
    for r in ben:
        by[str(r.get(key_field) or "unknown")]["benign"].append(r)
    by["ALL"] = {"attack": att, "benign": ben}
    table = {}
    for k, d in by.items():
        a, b = d["attack"], d["benign"]
        ka = sum(r["success"] for r in a)
        kb = sum(r["over_refused"] for r in b)
        table[k] = {
            "n_attack": len(a),
            "asr": round(ka / len(a), 4) if a else None,
            "asr_ci95": [round(x, 4) for x in wilson_ci(ka, len(a))] if a else None,
            "asr_at_turn": [round(x, 4) for x in cumulative_asr_by_turn([r["turn_unsafe"] for r in a])],
            "n_benign": len(b),
            "orr": round(kb / len(b), 4) if b else None,
            "orr_ci95": [round(x, 4) for x in wilson_ci(kb, len(b))] if b else None,
        }
    return table


def blend_vs_slices(table):
    """
    What a single blended number hides: pooled rate over everything ("ALL") vs the macro
    average over slices and the worst slice. Pooled rates weight slices by sample size, so a
    small, weak language barely moves them.
    """
    out = {}
    for metric, n_key in (("asr", "n_attack"), ("orr", "n_benign")):
        sl = {k: t for k, t in table.items() if k != "ALL" and t[metric] is not None}
        if not sl or table.get("ALL", {}).get(metric) is None:
            continue
        worst = max(sl, key=lambda k: sl[k][metric])
        best = min(sl, key=lambda k: sl[k][metric])
        out[metric] = {
            "pooled": table["ALL"][metric],
            "macro_mean": round(sum(t[metric] for t in sl.values()) / len(sl), 4),
            "worst_slice": worst, "worst": sl[worst][metric], "worst_n": sl[worst][n_key],
            "worst_ci95": sl[worst][f"{metric}_ci95"],
            "best_slice": best, "best": sl[best][metric],
            "worst_minus_pooled": round(sl[worst][metric] - table["ALL"][metric], 4),
            "n_slices": len(sl),
        }
    return out


def print_blend(blend, title):
    if not blend:
        return
    print(f"\n--- {title}: blended vs per-slice (lower is better for both) ---")
    for metric, b in blend.items():
        ci = b["worst_ci95"]
        print(f"  {metric.upper()}: pooled {b['pooled']:.3f} | macro avg over {b['n_slices']} slices "
              f"{b['macro_mean']:.3f} | worst {b['worst_slice']} {b['worst']:.3f} "
              f"[{ci[0]:.2f},{ci[1]:.2f}] n={b['worst_n']} (+{b['worst_minus_pooled']:.3f} vs pooled) "
              f"| best {b['best_slice']} {b['best']:.3f}")


def print_table(table, title):
    print(f"\n=== {title} ===")
    print(f"{'slice':<12}{'n_att':>7}{'ASR':>8}{'95% CI':>18}{'n_ben':>7}{'ORR':>8}{'95% CI':>18}  ASR@t")
    for k in sorted(table, key=lambda x: (x == "ALL", x)):
        t = table[k]
        asr = f"{t['asr']:.3f}" if t["asr"] is not None else "-"
        orr = f"{t['orr']:.3f}" if t["orr"] is not None else "-"
        aci = f"[{t['asr_ci95'][0]:.2f},{t['asr_ci95'][1]:.2f}]" if t["asr_ci95"] else "-"
        oci = f"[{t['orr_ci95'][0]:.2f},{t['orr_ci95'][1]:.2f}]" if t["orr_ci95"] else "-"
        print(f"{k:<12}{t['n_attack']:>7}{asr:>8}{aci:>18}{t['n_benign']:>7}{orr:>8}{oci:>18}  {t['asr_at_turn']}")


def main():
    args = parse_args()
    from src.harness.cache import GenerationCache
    from src.harness.gguf_runner import GGUFRunner
    from src.judge.qwen_guard import QwenGuardJudge

    out_dir = run_dir(args.run_name)
    runner = GGUFRunner(args.gguf, lora_path=args.lora, lora_scale=args.lora_scale,
                        n_ctx=args.n_ctx, seed=args.seed, temperature=args.temperature,
                        max_tokens=args.max_tokens, system_prompt=args.system_prompt,
                        cache=GenerationCache())
    if args.generate_only:
        n_att = n_ben = 0
        if args.attacks:
            convs = assert_unique_ids(read_jsonl(args.attacks), args.attacks)
            for i, c in enumerate(convs, 1):
                runner.replay(c["turns"], tags={"eval": args.run_name, "id": c["id"]})
                n_att += 1
                if i % 25 == 0:
                    print(f"  attacks generated: {i}/{len(convs)}", flush=True)
        if args.benign:
            twins = assert_unique_ids(read_jsonl(args.benign), args.benign)
            for i, t in enumerate(twins, 1):
                runner.generate(t["prompt"], tags={"eval": args.run_name, "id": t["id"]})
                n_ben += 1
                if i % 50 == 0:
                    print(f"  benign generated: {i}/{len(twins)}", flush=True)
        write_manifest(out_dir, {"script": "run_safety_eval", "mode": "generate_only", "args": vars(args),
                                 "target": runner.describe(), "n_attacks": n_att, "n_benign": n_ben},
                       name="generation_manifest.json")
        print(f"[generate_only] cached {n_att} attack conversations and {n_ben} benign replies; "
              f"re-run without --generate_only to judge from the cache.")
        return

    judge = QwenGuardJudge(model_id=args.judge_model)
    write_manifest(out_dir, {"script": "run_safety_eval", "args": vars(args),
                             "target": runner.describe(), "judge": args.judge_model})

    results = []
    if args.attacks:
        convs = assert_unique_ids(read_jsonl(args.attacks), args.attacks)
        for i, c in enumerate(convs, 1):
            responses = runner.replay(c["turns"], tags={"eval": args.run_name, "id": c["id"]})
            history, flags, labels = [], [], []
            for u, r in zip(c["turns"], responses):
                v = judge.evaluate(u, r, history=history,
                                   controversial_as_unsafe=args.controversial_as_unsafe)
                flags.append(bool(v.is_violation))
                labels.append(v.category)
                history += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
            results.append({"kind": "attack", "id": c["id"], "profile": c.get("profile", "unknown"),
                            "language": c.get("language"), "source": c.get("source"),
                            "global_or_local": c.get("global_or_local"),
                            "harm_categories": c.get("harm_categories"),
                            "responses": responses, "turn_unsafe": flags, "judge_labels": labels,
                            "success": any(flags)})
            if i % 25 == 0:
                print(f"  attacks judged: {i}/{len(convs)}", flush=True)
    if args.benign:
        twins = assert_unique_ids(read_jsonl(args.benign), args.benign)
        for i, t in enumerate(twins, 1):
            r = runner.generate(t["prompt"], tags={"eval": args.run_name, "id": t["id"]})
            v = judge.evaluate(t["prompt"], r)
            results.append({"kind": "benign", "id": t["id"], "profile": t.get("profile", "unknown"),
                            "language": t.get("language"), "category": t.get("category"),
                            "response": r, "judge_label": v.category,
                            "over_refused": bool(v.is_refusal)})
            if i % 50 == 0:
                print(f"  benign judged: {i}/{len(twins)}", flush=True)

    write_jsonl(out_dir / "safety_results.jsonl", results)
    table = summarize(results)
    blend = blend_vs_slices(table)
    write_manifest(out_dir, {"summary_by_profile": table, "blend_vs_profile": blend},
                   name="safety_summary.json")
    print_table(table, f"{args.run_name} — by profile")
    print_blend(blend, "by profile")
    for field in args.slice_by:
        extra = summarize(results, key_field=field)
        eblend = blend_vs_slices(extra)
        write_manifest(out_dir, {f"summary_by_{field}": extra, f"blend_vs_{field}": eblend},
                       name=f"safety_summary_{field}.json")
        print_table(extra, f"{args.run_name} — by {field}")
        print_blend(eblend, f"by {field}")

    if args.compare:
        base_path = run_dir(args.compare) / "safety_results.jsonl"
        base = {(r["kind"], r["id"]): r for r in read_jsonl(base_path)}
        for kind, field in (("attack", "success"), ("benign", "over_refused")):
            pairs = [(base[(kind, r["id"])][field], r[field]) for r in results
                     if r["kind"] == kind and (kind, r["id"]) in base]
            if pairs:
                d = paired_bootstrap_diff([a for a, _ in pairs], [b for _, b in pairs])
                print(f"  paired diff vs {args.compare} ({kind} {field}): "
                      f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] n={d['n']}")


if __name__ == "__main__":
    main()
