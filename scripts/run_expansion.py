import argparse
import json
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paths import SEEDS_RAW_DIR, EXPANDED_DIR, ensure_dirs

from collections import Counter
from src.attacker.strategy_library import StrategyLibrary
from src.attacker.expansion_loop import MultilingualExpansionLoop
from src.judge.loop_scorer import LoopScorer

def parse_args():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Zero-exposure automated prompt expansion.")
    parser.add_argument("--seeds", type=str, default=str(SEEDS_RAW_DIR / "validated_seeds.jsonl"), help="Input validated seeds")
    parser.add_argument("--output", type=str, default=str(EXPANDED_DIR / "autodan_expanded_prompts.jsonl"), help="Output path")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.seeds):
        print(f"[ERROR] Seeds file not found: {args.seeds}", file=sys.stderr)
        sys.exit(1)

    seeds = []
    with open(args.seeds, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))

    print("==================================================")
    print("      AUTOMATED EXPANSION (ZERO-EXPOSURE)")
    print("==================================================")
    print(f"Loaded {len(seeds)} validated seed prompts.")
    print("Expanding across registers (brainrot slang, code-mixed, script obfuscation)...")

    lib = StrategyLibrary()
    scorer = LoopScorer()
    loop = MultilingualExpansionLoop(strategy_library=lib, loop_scorer=scorer)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    expanded_results = loop.run_expansion_batch(seeds, output_dir=os.path.dirname(args.output))

    # Aggregate statistics ONLY
    lang_counter = Counter(r["language"] for r in expanded_results)
    reg_counter = Counter(r["register"] for r in expanded_results)
    fam_counter = Counter(r["attack_family"] for r in expanded_results)

    print("\n--- Expansion Results ---")
    print(f"Total Expanded Prompts Generated: {len(expanded_results)}")
    print("\nBy Language:")
    for k, v in lang_counter.items():
        print(f"  {k:<15}: {v}")
    print("\nBy Register:")
    for k, v in reg_counter.items():
        print(f"  {k:<15}: {v}")
    print("\nBy Attack Family:")
    for k, v in fam_counter.items():
        print(f"  {k:<20}: {v}")
    print("==================================================")
    print(f"[SUCCESS] Expanded dataset saved to: {args.output}\n")

if __name__ == "__main__":
    main()
