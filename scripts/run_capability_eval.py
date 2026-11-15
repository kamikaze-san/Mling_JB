"""
Capability evaluation of one target checkpoint: FLORES-200 devtest (chrF++) and Belebele (accuracy).

  python scripts/run_capability_eval.py --run_name base_q4 --gguf models/tiny-aya-global-q4_k_m.gguf
  python scripts/run_capability_eval.py --run_name joint_dpo_s0 --gguf ... --lora adapters/joint_s0.gguf \
      --compare base_q4          # prints BCP = repaired / base per direction and language
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, run_dir, write_manifest  # noqa: E402
from src.paths import BENCHMARKS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--flores_n", type=int, default=200, help="devtest sentences per direction")
    ap.add_argument("--belebele_per_lang", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    from src.eval.capability import eval_belebele, eval_flores, preservation
    from src.harness.gguf_runner import GGUFRunner

    runner = GGUFRunner(args.gguf, lora_path=args.lora, lora_scale=args.lora_scale,
                        seed=args.seed, temperature=0.0, max_tokens=args.max_tokens)
    chat = lambda msgs: runner.chat(msgs, tags={"eval": args.run_name})  # noqa: E731

    flores_path = BENCHMARKS_DIR / "flores_devtest.jsonl"
    if flores_path.exists():
        flores = read_jsonl(flores_path)[: args.flores_n]
    else:  # FLORES+ is gated; Belebele still runs
        print("[capability] flores_devtest.jsonl missing (accept FLORES+ terms, re-run download_benchmarks) "
              "-> FLORES skipped")
        flores = []
    bele_all = read_jsonl(BENCHMARKS_DIR / "belebele_eval.jsonl")
    per_lang = {}
    bele = []
    for r in bele_all:
        if per_lang.get(r["language"], 0) < args.belebele_per_lang:
            bele.append(r)
            per_lang[r["language"]] = per_lang.get(r["language"], 0) + 1

    fl = eval_flores(chat, flores) if flores else {}
    bl = eval_belebele(chat, bele)
    summary = {
        "flores_chrf_pp": {k: v["chrf_pp"] for k, v in fl.items()},
        "belebele_accuracy": {k: v["accuracy"] for k, v in bl.items()},
        "belebele_unparsed": {k: v["unparsed"] for k, v in bl.items()},
        "n": {"flores_per_direction": len(flores), "belebele": per_lang},
    }
    out_dir = run_dir(args.run_name)
    write_manifest(out_dir, {"script": "run_capability_eval", "args": vars(args),
                             "target": runner.describe(), "summary": summary},
                   name="capability_summary.json")
    with open(out_dir / "flores_hypotheses.json", "w", encoding="utf-8") as f:
        json.dump({k: v["hypotheses"] for k, v in fl.items()}, f, ensure_ascii=False)

    print(f"\n=== {args.run_name} — capability ===")
    for k, v in summary["flores_chrf_pp"].items():
        print(f"  FLORES {k:<6} chrF++ {v:6.2f}")
    for k, v in sorted(summary["belebele_accuracy"].items()):
        print(f"  Belebele {k:<4} acc {v:.3f}  (unparsed {summary['belebele_unparsed'][k]})")

    if args.compare:
        with open(run_dir(args.compare) / "capability_summary.json", encoding="utf-8") as f:
            base = json.load(f)["summary"]
        print(f"\n  BCP vs {args.compare} (repaired / base):")
        for name in ("flores_chrf_pp", "belebele_accuracy"):
            for k, v in preservation(base[name], summary[name]).items():
                print(f"    {name:<18} {k:<6} {v}")


if __name__ == "__main__":
    main()
