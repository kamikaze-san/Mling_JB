"""
Label DPO pair candidates with PolicyJudge (gpt-oss-safeguard-20b, safeguard_policy_v2.md)
for Task D10.

Extracts safety pair candidates, filters out held-out goals and scaffolding leaks,
saves them to pair_candidates.jsonl, and labels them into pair_candidate_labels.jsonl.

Resumable: skips already-labeled candidates.
Zero Exposure: reports counts and aggregate distributions only.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, write_jsonl, write_manifest
from src.judge.policy_judge import PolicyJudge
from src.paths import CURATED_DIR
from src.curation.strata_logs import conversation_id, group_conversations

LEAK = re.compile(r"(?mi)^[\s\*_#>\-•\[\]]*(TACTIC|METHOD|MESSAGE)[\s\*_#>\-•\[\]]*:|</?\s*(scratchpad|payload)\s*>")


def goal_hash(goal: str) -> str:
    return hashlib.sha256((goal or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def candidate_key(prompt: List[Dict[str, str]], rejected: str) -> str:
    return hashlib.sha256(json.dumps([prompt, rejected], ensure_ascii=False).encode()).hexdigest()[:20]


def has_scaffolding_leak(messages: List[Dict[str, str]]) -> bool:
    for m in messages:
        content = m.get("content", "")
        if LEAK.search(content):
            return True
    return False


def conversations_from_strata_log(entries, threshold=8.0):
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


def extract_candidates(
    strata_logs: List[str],
    human_pool_path: Optional[str],
    exclude_goals_path: Optional[str],
    threshold: float = 8.0,
) -> Tuple[List[Dict], Counter]:
    stats = Counter()
    excluded = set()
    if exclude_goals_path and os.path.exists(exclude_goals_path):
        with open(exclude_goals_path, encoding="utf-8") as f:
            excluded = {goal_hash(g) for g in json.load(f)}

    raw_candidates = []
    all_entries = []  # group ALL logs together: the lifelong log repeats the warm-up entries
    for p in strata_logs:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                all_entries += json.load(f)
    raw_candidates.extend(list(conversations_from_strata_log(all_entries, threshold)))

    if human_pool_path and os.path.exists(human_pool_path):
        for h in read_jsonl(human_pool_path):
            raw_candidates.append({
                "prompt": h["messages"],
                "rejected": h["rejected"],
                "profile": h.get("profile", "unknown"),
                "goal": h.get("goal", ""),
                "source": "human",
                "conv_key": h.get("id"),
                "turn": None
            })

    stats["total_extracted"] = len(raw_candidates)

    kept = []
    for c in raw_candidates:
        prof = c["profile"]
        stats[f"extracted:{prof}"] += 1

        gh = goal_hash(c["goal"])
        if gh in excluded:
            stats["dropped_heldout_goal"] += 1
            stats[f"dropped_heldout:{prof}"] += 1
            continue

        if has_scaffolding_leak(c["prompt"]):
            stats["dropped_scaffolding_leak"] += 1
            stats[f"dropped_leak:{prof}"] += 1
            continue

        ckey = candidate_key(c["prompt"], c["rejected"])
        full_messages = c["prompt"] + [{"role": "assistant", "content": c["rejected"]}]

        rec = {
            "key": ckey,
            "source": c["source"],
            "profile": prof,
            "goal_hash": gh,
            "messages": full_messages,
            "rejected": c["rejected"]
        }
        kept.append(rec)
        stats[f"kept:{prof}"] += 1

    stats["total_kept"] = len(kept)
    return kept, stats


def label_candidates(
    candidates: List[Dict],
    out_labels_path: str,
    api_base: str,
    model_name: str,
    policy_path: str,
    limit: Optional[int] = None,
    judge_instance: Optional[PolicyJudge] = None,
    max_tokens: int = 1536,
    retry_unknown: bool = False,
    reasoning: str = "medium",
    workers: int = 1,
) -> Counter:
    os.makedirs(os.path.dirname(out_labels_path), exist_ok=True)
    done_keys = set()
    if os.path.exists(out_labels_path):
        latest = {}  # last record per key wins (also in build_pair_pool)
        with open(out_labels_path, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        latest[r["key"]] = r.get("label") or "UNKNOWN"
                    except Exception:
                        pass
        # --retry_unknown: items whose latest label is UNKNOWN (truncated or unparsable) are re-judged
        done_keys = {k for k, lab in latest.items() if not (retry_unknown and lab == "UNKNOWN")}

    judge = judge_instance or PolicyJudge(api_base=api_base, model=model_name, policy_path=policy_path,
                                         max_tokens=max_tokens, reasoning=reasoning)
    label_stats = Counter()

    todo = [c for c in candidates if c["key"] not in done_keys]
    if limit is not None and limit > 0:
        todo = todo[:limit]

    print(f"\n[label_pair_candidates] Total: {len(candidates)} | Already labeled: {len(done_keys)} | To label: {len(todo)}")

    def _results():
        if workers <= 1:
            for c in todo:
                yield c, judge.classify(c["messages"])
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:  # server has parallel slots; writes stay here
            futs = {ex.submit(judge.classify, c["messages"]): c for c in todo}
            for fut in as_completed(futs):
                yield futs[fut], fut.result()

    with open(out_labels_path, mode="a", encoding="utf-8") as out_f:
        for idx, (c, res) in enumerate(_results()):
            lab = res.get("label") or "UNKNOWN"
            rule_id = res.get("rule_id") or ""
            rationale = res.get("rationale") or ""

            out_rec = {
                "key": c["key"],
                "profile": c["profile"],
                "source": c["source"],
                "label": lab,
                "rule_id": rule_id,
                "rationale": rationale
            }
            out_f.write(json.dumps(out_rec) + "\n")
            out_f.flush()

            label_stats[lab] += 1
            label_stats[f"{c['profile']}:{lab}"] += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == len(todo):
                print(f"  Labeled: {idx + 1}/{len(todo)} | RC0: {label_stats['RC0']} | RC1: {label_stats['RC1']} | RC2: {label_stats['RC2']}", flush=True)

    return label_stats


def main():
    ap = argparse.ArgumentParser(description="Extract and label DPO pair candidates (Task D10).")
    ap.add_argument("--strata_logs", nargs="+", default=[
        "MTMM/logs/warm_up_attack_log_reduced_seedoff.json",
        "MTMM/logs/lifelong_attack_log_reduced_seedoff.json",
        "MTMM/logs/lifelong_attack_log_topup_hienmix.json"
    ])
    ap.add_argument("--human_pool", default="MTMM/data/human_dpo_pool.jsonl")
    ap.add_argument("--exclude_goals", default="MTMM/data/heldout_goals.json")
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--out_candidates", default=str(CURATED_DIR / "pair_candidates.jsonl"))
    ap.add_argument("--out_labels", default=str(CURATED_DIR / "pair_candidate_labels.jsonl"))
    ap.add_argument("--api_base", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="openai/gpt-oss-safeguard-20b")
    ap.add_argument("--policy", default="Multilingual_JB/configs/judge/safeguard_policy_v2.md")
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--retry_unknown", action="store_true",
                    help="re-judge items whose latest label is UNKNOWN (use with --reasoning low --max_tokens 4096)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (match the server's parallel slots)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all, >0 limit for smoke/dry tests")
    ap.add_argument("--extract_only", action="store_true")
    args = ap.parse_args()

    candidates, stats = extract_candidates(
        args.strata_logs,
        args.human_pool,
        args.exclude_goals,
        threshold=args.threshold
    )

    write_jsonl(args.out_candidates, candidates)
    print(f"\n[label_pair_candidates] Wrote {len(candidates)} clean candidates to {args.out_candidates}")
    print(f"Extraction stats: {dict(stats)}")

    if not args.extract_only:
        label_stats = label_candidates(
            candidates,
            args.out_labels,
            args.api_base,
            args.model,
            args.policy,
            limit=args.limit if args.limit > 0 else None,
            max_tokens=args.max_tokens,
            retry_unknown=args.retry_unknown,
            reasoning=args.reasoning,
            workers=args.workers,
        )
        print(f"\nLabelling complete. Summary: {dict(label_stats)}")


if __name__ == "__main__":
    main()
