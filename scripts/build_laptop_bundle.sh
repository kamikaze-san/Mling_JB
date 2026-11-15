#!/usr/bin/env bash
# Assemble the laptop eval bundle (code + eval inputs + benchmarks + base GGUF + adapter GGUFs) under
# data/laptop_bundle/ (data/ is gitignored: the bundle holds attack prompts and FLORES, so it must never be
# committed or uploaded; copy it to the laptop by USB or local network only).
set -euo pipefail
cd "$(dirname "$0")/.."
B=data/laptop_bundle
REF=data/04_eval_cache/en_only_p100_lr5e-5_e3   # trimmed eval set (identical across the 100-pair runs)
mkdir -p $B/scripts $B/src $B/inputs $B/models $B/adapters $B/data/benchmarks $B/data/04_eval_cache/base_q4_v2
cp laptop/run_laptop_eval.py laptop/jobs.txt laptop/BRIEF.md laptop/requirements_laptop.txt $B/
cp scripts/run_safety_eval.py scripts/run_capability_eval.py $B/scripts/
rsync -a --delete --exclude __pycache__ src/__init__.py src/paths.py $B/src/
for sub in eval harness judge; do rsync -a --delete --exclude __pycache__ src/$sub $B/src/; done
cp $REF/eval_attacks_combined.jsonl $REF/eval_benign_combined.jsonl $B/inputs/
cp data/benchmarks/flores_devtest.jsonl data/benchmarks/belebele_eval.jsonl $B/data/benchmarks/
cp data/04_eval_cache/base_q4_v2/capability_summary.json $B/data/04_eval_cache/base_q4_v2/
[ -f $B/models/tiny-aya-global-q4_k_m.gguf ] || cp ~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf $B/models/
grep -v '^\s*#' laptop/jobs.txt | awk 'NF{print $1}' | while read -r ad; do
  mkdir -p $B/adapters/$ad && cp checkpoints/hf/$ad/adapter.gguf $B/adapters/$ad/
done
echo "bundle: $(du -sh $B | cut -f1) at $B ($(ls $B/adapters | wc -l | tr -d ' ') adapters)"
