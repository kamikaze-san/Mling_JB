#!/usr/bin/env python3
"""Recompute the checkpoint gates (EVAL_PROTOCOL.md §4) of finished adapter runs from their saved files.

Needed because eval_adapter.py (before the 14 Nov fix) read capability_summary.json at the wrong level, so the
FLORES and Belebele gates were never evaluated, and the gate printout then crashed before the verdict file was
written. This script only READS safety_results.jsonl and capability_summary.json of each run and its baseline
(no generation, no judging, no GPU) and writes <run>/gate_verdict.json. Counts and deltas only.

  python scripts/recompute_gates.py                      # every run with safety results + capability summary
  python scripts/recompute_gates.py --runs joint_all_v3_lr5e-5_e2 --compare base_q4_v2
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("eval_adapter", ROOT / "scripts" / "eval_adapter.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)

from src.eval.runs import read_jsonl, run_dir  # noqa: E402  (same helpers eval_adapter uses)

BASELINES = {"base_q4", "base_q4_v2", "base_q4_train", "base_q4_aya", "base_q4_hf"}


def recompute(name: str, compare: str) -> dict:
    out_dir, base_dir = run_dir(name), run_dir(compare)
    a_res = read_jsonl(os.path.join(out_dir, "safety_results.jsonl"))
    b_res = read_jsonl(os.path.join(base_dir, "safety_results.jsonl"))
    a_cap = ea.load_capability(os.path.join(out_dir, "capability_summary.json"))
    b_cap = ea.load_capability(os.path.join(base_dir, "capability_summary.json"))
    verdict = ea.evaluate_checkpoint_gates(a_res, b_res, a_cap, b_cap)
    with open(os.path.join(out_dir, "gate_verdict.json"), "w", encoding="utf-8") as f:
        json.dump({"run": name, "compare": compare, "script": "recompute_gates.py", **verdict}, f, indent=2)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--compare", default="base_q4_v2")
    args = ap.parse_args()
    cache = Path(run_dir(args.compare)).parent
    runs = args.runs or sorted(
        p.name for p in cache.iterdir()
        if p.is_dir() and p.name not in BASELINES and (p / "safety_results.jsonl").exists()
        and (p / "capability_summary.json").exists())
    for name in runs:
        v = recompute(name, args.compare)
        parts = []
        for gid, g in v["gates"].items():
            parts.append(f"{gid}={ea.gate_status(g)}")
        print(f"{name:42} overall={'PASS' if v['overall_pass'] else 'FAIL'}  " + " ".join(parts))
        for gid, g in v["gates"].items():
            if isinstance(g.get("deltas"), dict) and ea.gate_status(g) != "SKIPPED":
                print(f"{'':44}{gid}: {ea.describe_deltas(gid, g)}")


if __name__ == "__main__":
    main()
