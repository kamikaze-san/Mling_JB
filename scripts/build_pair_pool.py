"""
Build the preference-pair POOL that every DPO condition samples from.

Safety pairs (on-policy):
  prompt   = conversation up to the turn where the Q4 target complied
  rejected = the target's own compliant reply (from STRATA logs or human traces)
  chosen   = brief, non-preachy refusal in the same language/script/register (writer)

Benign pairs (matched surfaces, from a baseline safety-eval run's benign twins):
  if the base model answered:      chosen = its own answer (on-policy),
                                   rejected = writer's false refusal
  if the base model over-refused:  chosen = writer's helpful answer,
                                   rejected = the base model's own refusal (on-policy)

Inputs are private JSONL/JSON; stdout shows counts only.

  python scripts/build_pair_pool.py \
     --strata_logs ../MTMM/logs/lifelong_attack_log_te.json ../MTMM/logs/warm_up_attack_log_te.json \
     --human_pool data/01_seeds_raw/human_dpo_pool.jsonl \
     --benign data/03_curated/benign_twins_train.jsonl --benign_run base_q4_train \
     --exclude_goals data/03_curated/heldout_goals.json \
     --writer_base_url http://localhost:1234 --writer_model qwen3.6-35b-a3b
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, run_dir, write_jsonl, write_manifest  # noqa: E402
from src.paths import CURATED_DIR  # noqa: E402
from src.curation.strata_logs import conversation_id, group_conversations  # noqa: E402

LEAK = re.compile(r"(?mi)^[\s\*_#>\-•\[\]]*(TACTIC|METHOD|MESSAGE)[\s\*_#>\-•\[\]]*:|</?\s*(scratchpad|payload)\s*>")


def goal_hash(goal: str) -> str:
    return hashlib.sha256((goal or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def candidate_key(prompt, rejected):
    return hashlib.sha256(json.dumps([prompt, rejected], ensure_ascii=False).encode()).hexdigest()[:20]


def has_scaffolding_leak(messages):
    for m in messages:
        if LEAK.search(m.get("content", "")):
            return True
    return False


def conversations_from_strata_log(entries, threshold):
    """
    Group STRATA attack-log entries into conversations and return one candidate
    per conversation: the FIRST turn whose score >= threshold.
    Yields dicts with prompt messages, rejected reply, profile, goal.
    """
    convs, _ = group_conversations(entries)  # lifelong/test logs have iteration == 0 for every pass
    for key, turns in convs.items():
        history = []
        for e in turns:
            user = e.get("clean_prompt") or e.get("prompt") or ""
            if e.get("null_probe"):
                continue
            if float(e.get("score", 0)) >= threshold:
                yield {
                    "prompt": history + [{"role": "user", "content": user}],
                    "rejected": e["response"],
                    "profile": e.get("profile") or "unknown",
                    "goal": e.get("request", ""),
                    "source": "strata",
                    "conv_key": conversation_id(key),
                    "turn": e["turn"],
                }
                break
            history += [{"role": "user", "content": user}, {"role": "assistant", "content": e["response"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata_logs", nargs="*", default=[])
    ap.add_argument("--strata_threshold", type=float, default=8.0,
                    help="STRATA score counted as a successful attack. 8.0 = compliance under the frozen "
                         "rubric (8.0-8.9 complied, 9-10 specific); matches STRATA's break_score.")
    ap.add_argument("--human_pool", default=None,
                    help="JSONL {id, profile, language, goal, messages, rejected} (same GGUF quant only)")
    ap.add_argument("--benign", default=None, help="benign twins JSONL used for TRAINING (not the eval set)")
    ap.add_argument("--benign_run", default=None, help="safety-eval run_name holding base responses to --benign")
    ap.add_argument("--exclude_goals", default=None, help="JSON list of held-out goals (strings)")
    ap.add_argument("--candidate_labels", default=None,
                    help="JSONL of {key, label, rule_id} from label_pair_candidates.py")
    ap.add_argument("--keep_labels", default=None,
                    help="Comma-separated labels to keep (e.g. 'RC1'). Default: keep all.")
    ap.add_argument("--require_judge_unsafe", action="store_true",
                    help="keep a safety candidate only if Qwen3Guard also marks the reply unsafe")
    ap.add_argument("--writer_base_url", required=True)
    ap.add_argument("--writer_model", required=True)
    ap.add_argument("--out", default=str(CURATED_DIR / "pair_pool.jsonl"))
    ap.add_argument("--reuse_chosen", default=None,
                    help="earlier pair_pool.jsonl: pairs with the same pair_id and texts keep their "
                         "written refusal (no new writer call), so earlier conditions stay comparable")
    args = ap.parse_args()

    reuse, reuse_benign = {}, {}
    if args.reuse_chosen:  # benign pairs with the same prompt keep both sides (no new writer call)
        for r in read_jsonl(args.reuse_chosen):
            (reuse if r.get("kind") == "safety" else reuse_benign)[r["pair_id"]] = r

    from src.curation.writer import ResponseWriter
    writer = ResponseWriter(args.writer_base_url, args.writer_model)
    judge = None
    if args.require_judge_unsafe:
        from src.judge.qwen_guard import QwenGuardJudge
        judge = QwenGuardJudge()

    excluded = set()
    if args.exclude_goals:
        with open(args.exclude_goals, encoding="utf-8") as f:
            excluded = {goal_hash(g) for g in json.load(f)}

    candidate_labels_map = {}
    keep_labels_set = None
    if args.candidate_labels:
        for r in read_jsonl(args.candidate_labels):
            candidate_labels_map[r["key"]] = r.get("label")
        if args.keep_labels:
            keep_labels_set = {lab.strip().upper() for lab in args.keep_labels.split(",") if lab.strip()}

    pool, stats = [], Counter()
    for p in ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]:
        stats[f"safety:{p}"] = 0

    # --- safety candidates -------------------------------------------------
    candidates = []
    all_entries = []  # group ALL logs together: the lifelong log repeats the warm-up entries
    for path in args.strata_logs:
        with open(path, encoding="utf-8") as f:
            all_entries += json.load(f)
    candidates += list(conversations_from_strata_log(all_entries, args.strata_threshold))
    if args.human_pool:
        for h in read_jsonl(args.human_pool):
            candidates.append({"prompt": h["messages"], "rejected": h["rejected"],
                               "profile": h.get("profile", "unknown"), "goal": h.get("goal", ""),
                               "source": "human", "conv_key": h["id"], "turn": None})
    for c in candidates:
        stats[f"candidate:{c['source']}:{c['profile']}"] += 1
        gh = goal_hash(c["goal"])
        if gh in excluded:
            stats["safety_dropped_heldout_goal"] += 1
            stats[f"safety_dropped_heldout:{c['profile']}"] += 1
            continue
        if has_scaffolding_leak(c["prompt"]):
            stats["safety_dropped_scaffolding_leak"] += 1
            stats[f"safety_dropped_leak:{c['profile']}"] += 1
            continue
        if keep_labels_set is not None:
            ckey = candidate_key(c["prompt"], c["rejected"])
            lab = candidate_labels_map.get(ckey)
            if lab not in keep_labels_set:
                stats[f"safety_dropped_label_{lab or 'missing'}"] += 1
                stats[f"safety_dropped_label:{c['profile']}"] += 1
                continue
        if judge is not None:
            v = judge.evaluate(c["prompt"][-1]["content"], c["rejected"], history=c["prompt"][:-1])
            if not v.is_violation:
                stats["safety_dropped_judge_safe"] += 1
                stats[f"safety_dropped_judge_safe:{c['profile']}"] += 1
                continue
        old = reuse.get(f"safety:{c['source']}:{c['conv_key']}")
        if old is not None and old["rejected"] == c["rejected"] and old["prompt"] == c["prompt"]:
            chosen = old["chosen"]
            stats["safety_chosen_reused"] += 1
        else:
            chosen = writer.refusal(c["prompt"])
        if not chosen:
            stats["safety_dropped_empty_chosen"] += 1
            stats[f"safety_dropped_empty:{c['profile']}"] += 1
            continue
        pool.append({"pair_id": f"safety:{c['source']}:{c['conv_key']}", "kind": "safety",
                     "profile": c["profile"], "source": c["source"], "goal_hash": gh,
                     "prompt": c["prompt"], "chosen": chosen, "rejected": c["rejected"],
                     "chosen_origin": "writer_refusal", "rejected_origin": "target_on_policy"})
        stats[f"safety:{c['profile']}"] += 1

    # --- benign candidates --------------------------------------------------
    if args.benign and args.benign_run:
        twins = {t["id"]: t for t in read_jsonl(args.benign)}
        base = [r for r in read_jsonl(run_dir(args.benign_run) / "safety_results.jsonl")
                if r["kind"] == "benign" and r["id"] in twins]
        for r in base:
            t = twins[r["id"]]
            msgs = [{"role": "user", "content": t["prompt"]}]
            if has_scaffolding_leak(msgs):  # twins made before the extractor fix may carry formatting
                stats["benign_dropped_scaffolding_leak"] += 1
                continue
            old = reuse_benign.get(f"benign:{t['id']}")
            if old is not None and old["prompt"] == msgs:
                chosen, chosen_o, rejected, rejected_o = (old["chosen"], old["chosen_origin"],
                                                          old["rejected"], old["rejected_origin"])
                stats["benign_pair_reused"] += 1
            elif r["over_refused"]:
                chosen, chosen_o = writer.helpful(msgs), "writer_helpful"
                rejected, rejected_o = r["response"], "target_on_policy_refusal"
            else:
                chosen, chosen_o = r["response"], "target_on_policy_answer"
                rejected, rejected_o = writer.false_refusal(msgs), "writer_false_refusal"
            if not chosen or not rejected:
                stats["benign_dropped_empty"] += 1
                continue
            pool.append({"pair_id": f"benign:{t['id']}", "kind": "benign",
                         "profile": t.get("profile", "unknown"), "source": "benign_twin",
                         "category": t.get("category"), "prompt": msgs,
                         "chosen": chosen, "rejected": rejected,
                         "chosen_origin": chosen_o, "rejected_origin": rejected_o})
            stats[f"benign:{t.get('profile', 'unknown')}"] += 1

    n = write_jsonl(args.out, pool)
    write_manifest(os.path.dirname(args.out), {"script": "build_pair_pool", "args": vars(args),
                                               "counts": dict(stats), "n": n},
                   name="pair_pool_manifest.json")
    print(f"pair pool: {n} pairs -> {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<36}{v:>6}")


if __name__ == "__main__":
    main()
