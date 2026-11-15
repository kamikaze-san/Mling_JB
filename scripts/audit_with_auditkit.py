#!/usr/bin/env python3
"""audit_with_auditkit.py: Run an AuditKIT audit on base Q4 or an adapter.

Audits safety (abstention on attacks) and over-refusal (false refusal on benign)
per linguistic surface and pooled, using Lexsi AuditKIT 1.1+.
Outputs run results JSON, summary Markdown, and summary CSV.

Strict adherence to Zero-Exposure:
- Never print raw attack prompts, replies, or toxic text to stdout.
- Aggregate metrics, counts, and confidence intervals only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import auditkit
    from auditkit import Abstention, RAGCase, Sample, ToxicityScore, evaluate
    from auditkit.model import AutoModel
except ImportError as e:
    print(f"[ERROR] auditkit not available in this Python environment: {e}", file=sys.stderr)
    print("Please use Multilingual_JB/.venv-curatorkit/bin/python", file=sys.stderr)
    sys.exit(1)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95%
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def load_attack_samples(path: str, max_samples: Optional[int] = None) -> list[Sample]:
    """Load attack samples from JSONL as AuditKIT Samples with expected_action='abstain'."""
    samples = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Attack file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = item.get("id", f"att_{idx}")
            profile = item.get("profile", item.get("language", "unknown"))

            # Extract user prompt
            prompt = ""
            if "turns" in item and item["turns"]:
                prompt = item["turns"][-1]
            elif "prompt" in item:
                prompt = item["prompt"]
            elif "messages" in item and item["messages"]:
                prompt = item["messages"][-1].get("content", "")

            if not prompt:
                continue

            case = RAGCase(
                case_id=sample_id,
                question=prompt,
                expected_action="abstain",
            )
            s = Sample(
                input=prompt,
                id=sample_id,
                task="attack",
                metadata={
                    "rag_case": case,
                    "profile": profile,
                    "kind": "attack",
                    "goal_hash": item.get("goal_hash"),
                },
                tags=["attack", profile],
            )
            samples.append(s)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_benign_samples(path: str, max_samples: Optional[int] = None) -> list[Sample]:
    """Load benign samples from JSONL as AuditKIT Samples with expected_action='answer'."""
    samples = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benign file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = item.get("id", f"ben_{idx}")
            profile = item.get("profile", item.get("language", "unknown"))

            prompt = item.get("prompt", "")
            if not prompt and "turns" in item and item["turns"]:
                prompt = item["turns"][-1]
            elif not prompt and "messages" in item and item["messages"]:
                prompt = item["messages"][-1].get("content", "")

            if not prompt:
                continue

            case = RAGCase(
                case_id=sample_id,
                question=prompt,
                expected_action="answer",
            )
            s = Sample(
                input=prompt,
                id=sample_id,
                task="benign",
                metadata={
                    "rag_case": case,
                    "profile": profile,
                    "kind": "benign",
                },
                tags=["benign", profile],
            )
            samples.append(s)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def build_mock_model() -> Callable[[list[str]], list[str]]:
    """Deterministic mock model for dry runs and offline verification."""
    refusal = "I cannot fulfill this request as it violates safety guidelines."
    benign_answer = "Here is the factual and helpful information you requested."

    def _call(prompts: list[str]) -> list[str]:
        replies = []
        for p in prompts:
            p_lower = p.lower()
            if any(w in p_lower for w in ("bomb", "exploit", "attack", "hack", "bypass", "jailbreak", "kill")):
                replies.append(refusal)
            else:
                replies.append(benign_answer)
        return replies

    return _call


def run_audit(
    run_name: str,
    attacks_path: Optional[str] = None,
    benign_path: Optional[str] = None,
    endpoint: str = "http://localhost:1234/v1",
    model_name: str = "tiny-aya",
    out_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    mock: bool = False,
    use_toxicity_model: bool = False,
) -> dict[str, Any]:
    """Execute AuditKIT evaluation and produce summary tables."""
    if out_dir is None:
        out_dir = f"data/04_eval_cache/auditkit/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    samples: list[Sample] = []
    if attacks_path:
        att = load_attack_samples(attacks_path, max_samples=max_samples)
        samples.extend(att)
    if benign_path:
        ben = load_benign_samples(benign_path, max_samples=max_samples)
        samples.extend(ben)

    if not samples:
        raise ValueError("No samples loaded. Provide valid --attacks or --benign paths.")

    # Model resolution
    if mock:
        model: Any = build_mock_model()
        model_desc = "mock_model"
    else:
        model = AutoModel.resolve(f"openai:{model_name}", api_base=endpoint, api_key="dummy")
        model_desc = f"openai:{model_name}@{endpoint}"

    scorers = [
        Abstention(),
        ToxicityScore(use_model=use_toxicity_model),
    ]

    print(f"=== [AuditKIT] Running audit '{run_name}' ===")
    print(f"  Target: {model_desc}")
    print(f"  Total samples: {len(samples)} (attacks: {sum(1 for s in samples if s.task == 'attack')}, "
          f"benign: {sum(1 for s in samples if s.task == 'benign')})")

    run_res = evaluate(
        dataset=samples,
        model=model,
        scorers=scorers,
        cache=False,
    )

    # Save raw auditkit run result
    res_path = os.path.join(out_dir, "run_result.json")
    try:
        run_res.save(res_path)
    except Exception as e:
        print(f"  [Notice] run_res.save: {e}")

    # Map samples by ID for slicing
    sample_map = {s.id: s for s in samples}

    # Slice evaluation by surface and kind
    surface_stats: dict[str, dict[str, Any]] = {}

    def _get_slice(prof: str) -> dict[str, Any]:
        if prof not in surface_stats:
            surface_stats[prof] = {
                "n_attack": 0, "n_attack_refused": 0,
                "n_benign": 0, "n_benign_refused": 0,
                "tox_scores": [],
            }
        return surface_stats[prof]

    audit_records = []
    for pred in run_res.predictions:
        s = sample_map.get(pred.sample_id)
        if not s:
            continue
        prof = s.metadata.get("profile", "unknown")
        kind = s.metadata.get("kind", s.task)
        sl = _get_slice(prof)
        all_sl = _get_slice("ALL")

        # Extract abstention outcome
        scores_list = (pred.metadata or {}).get("scores", [])
        abstained = False
        tox = 1.0
        for sc in scores_list:
            if sc.get("name") == "abstention":
                # Abstention metadata carries actual abstained boolean
                sc_meta = sc.get("metadata", {})
                abstained = sc_meta.get("abstained", False)
            elif sc.get("name") == "toxicity_score":
                tox = sc.get("value", 1.0)

        sl["tox_scores"].append(tox)
        all_sl["tox_scores"].append(tox)

        if kind == "attack":
            sl["n_attack"] += 1
            all_sl["n_attack"] += 1
            if abstained:
                sl["n_attack_refused"] += 1
                all_sl["n_attack_refused"] += 1
        elif kind == "benign":
            sl["n_benign"] += 1
            all_sl["n_benign"] += 1
            if abstained:
                sl["n_benign_refused"] += 1
                all_sl["n_benign_refused"] += 1

        audit_records.append({
            "sample_id": pred.sample_id,
            "profile": prof,
            "kind": kind,
            "abstained": abstained,
            "correct": pred.correct,
            "score": pred.score,
            "toxicity_score": tox,
        })

    # Save audit predictions JSONL (aggregate records, no raw prompt/reply text)
    preds_path = os.path.join(out_dir, "audit_predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r in audit_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Format summary table
    summary_rows = []
    for prof in sorted(surface_stats.keys(), key=lambda x: (x == "ALL", x)):
        st = surface_stats[prof]
        n_att = st["n_attack"]
        n_att_ref = st["n_attack_refused"]
        # Abstention on attack: higher is better (model refused attack)
        att_abst_rate = (n_att_ref / n_att) if n_att > 0 else 0.0
        att_ci = wilson_ci(n_att_ref, n_att)

        n_ben = st["n_benign"]
        n_ben_ref = st["n_benign_refused"]
        # Over-refusal on benign: lower is better (model refused harmless)
        ben_orr = (n_ben_ref / n_ben) if n_ben > 0 else 0.0
        ben_ci = wilson_ci(n_ben_ref, n_ben)

        avg_tox = (sum(st["tox_scores"]) / len(st["tox_scores"])) if st["tox_scores"] else 1.0

        summary_rows.append({
            "surface": prof,
            "n_attack": n_att,
            "attack_refusal_rate": att_abst_rate,
            "attack_refusal_ci95": att_ci,
            "n_benign": n_ben,
            "benign_orr": ben_orr,
            "benign_orr_ci95": ben_ci,
            "avg_cleanliness": avg_tox,
        })

    # Write Markdown summary
    md_lines = [
        f"# AuditKIT Evaluation Report: {run_name}",
        f"- **Model Target:** `{model_desc}`",
        f"- **Total Attack Samples:** {surface_stats.get('ALL', {}).get('n_attack', 0)}",
        f"- **Total Benign Samples:** {surface_stats.get('ALL', {}).get('n_benign', 0)}",
        "",
        "| Surface | n_att | Attack Refusal Rate | 95% CI | n_ben | Over-refusal Rate (ORR) | 95% CI | Cleanliness (1-Tox) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        att_str = f"{r['attack_refusal_rate']:.1%}" if r["n_attack"] > 0 else "-"
        att_ci_str = f"[{r['attack_refusal_ci95'][0]:.1%}, {r['attack_refusal_ci95'][1]:.1%}]" if r["n_attack"] > 0 else "-"
        ben_str = f"{r['benign_orr']:.1%}" if r["n_benign"] > 0 else "-"
        ben_ci_str = f"[{r['benign_orr_ci95'][0]:.1%}, {r['benign_orr_ci95'][1]:.1%}]" if r["n_benign"] > 0 else "-"
        tox_str = f"{r['avg_cleanliness']:.3f}"
        md_lines.append(
            f"| **{r['surface']}** | {r['n_attack']} | {att_str} | {att_ci_str} | "
            f"{r['n_benign']} | {ben_str} | {ben_ci_str} | {tox_str} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # Write CSV summary
    csv_lines = [
        "surface,n_attack,attack_refusal_rate,attack_refusal_ci95_low,attack_refusal_ci95_high,"
        "n_benign,benign_orr,benign_orr_ci95_low,benign_orr_ci95_high,avg_cleanliness"
    ]
    for r in summary_rows:
        csv_lines.append(
            f"{r['surface']},{r['n_attack']},{r['attack_refusal_rate']:.4f},"
            f"{r['attack_refusal_ci95'][0]:.4f},{r['attack_refusal_ci95'][1]:.4f},"
            f"{r['n_benign']},{r['benign_orr']:.4f},"
            f"{r['benign_orr_ci95'][0]:.4f},{r['benign_orr_ci95'][1]:.4f},"
            f"{r['avg_cleanliness']:.4f}"
        )
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    print(f"\n[AuditKIT] Audit completed -> written to {out_dir}")
    print(f"  Summary table:\n{md_content}")

    return {
        "run_name": run_name,
        "out_dir": out_dir,
        "summary": summary_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit models with Lexsi AuditKIT")
    parser.add_argument("--run_name", type=str, required=True, help="Name of audit run (e.g. base_q4, joint8)")
    parser.add_argument("--attacks", type=str, default="data/04_eval_cache/base_q4_v2/eval_attacks_combined.jsonl",
                        help="Path to attack samples JSONL")
    parser.add_argument("--benign", type=str, default="data/03_curated/benign_twins_eval.jsonl",
                        help="Path to benign samples JSONL")
    parser.add_argument("--endpoint", type=str, default="http://localhost:1234/v1",
                        help="OpenAI-compatible server endpoint")
    parser.add_argument("--model", type=str, default="tiny-aya", help="Model name on endpoint")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap samples per file (for testing)")
    parser.add_argument("--mock", action="store_true", help="Use offline mock model instead of live endpoint")
    parser.add_argument("--use_toxicity_model", action="store_true", help="Use toxic-bert instead of word list")
    args = parser.parse_args()

    run_audit(
        run_name=args.run_name,
        attacks_path=args.attacks if os.path.exists(args.attacks) else None,
        benign_path=args.benign if os.path.exists(args.benign) else None,
        endpoint=args.endpoint,
        model_name=args.model,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        mock=args.mock,
        use_toxicity_model=args.use_toxicity_model,
    )


if __name__ == "__main__":
    main()
