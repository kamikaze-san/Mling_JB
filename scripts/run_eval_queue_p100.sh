#!/usr/bin/env bash
# Overnight evaluation queue for the 12 Nov 100-pair adapters (Mac, one heavy GPU job at a time).
# Waits for the STRATA top-up (topup3t) to finish, frees LM Studio, then per adapter:
#   eval_adapter.py (llama.cpp Metal + Qwen3Guard)  ->  score_stance_decider.py (MLX, separate process).
# Ablations use last night's joint8 sets (STRATA held-out + human, twins + 100 general benign, no Aya) so they
# stay comparable; joint_all (final-model candidate) also gets Aya 200 and the full 400 general benign.
# A failed adapter is logged and skipped; eval_adapter exits 1 when a gate fails, which is not an error here.
cd "$(dirname "$0")/.."
LOG=logs_eval_queue_p100.log
PPLX=~/Downloads/MARCH-Orz/.venv-pplx/bin/python
C=data/03_curated

while pgrep -f "run_topup3.sh" >/dev/null || pgrep -f "run_id topup3t" >/dev/null; do sleep 60; done
~/.lmstudio/bin/lms unload --all >/dev/null 2>&1 || true
echo "queue start $(date +%H:%M)" >> $LOG

run_one() {  # run_one <adapter folder> <run name> [extra eval args...]
  local ad=$1 name=$2; shift 2
  local lora=checkpoints/hf/$ad/adapter.gguf
  if [ ! -f "$lora" ]; then echo "$name SKIP missing $lora" >> $LOG; return; fi
  if [ -f data/04_eval_cache/$name/stance_decider.jsonl ] && [ -f data/04_eval_cache/$name/capability_summary.json ]; then
    echo "$name SKIP already done" >> $LOG; return; fi
  echo "$name eval start $(date +%H:%M)" >> $LOG
  .venv/bin/python scripts/eval_adapter.py --name "$name" --lora "$lora" --compare base_q4_v2 "$@" \
    > data/04_eval_cache/eval_$name.log 2>&1
  local rc=$?
  if [ ! -f data/04_eval_cache/$name/safety_results.jsonl ]; then
    echo "$name EVAL FAILED rc=$rc (no safety_results; see data/04_eval_cache/eval_$name.log)" >> $LOG; return; fi
  echo "$name eval done rc=$rc $(date +%H:%M)" >> $LOG
  $PPLX scripts/score_stance_decider.py --run "$name" > data/04_eval_cache/stance_$name.log 2>&1
  echo "$name stance done rc=$? $(date +%H:%M)" >> $LOG
}

TRIM=(--aya_attacks $C/empty.jsonl --general_benign $C/benign_baseline_100.jsonl)
run_one joint_all_p100_lr5e-5_e2_s0      joint_all_p100_lr5e-5_e2                         # full sets
run_one en_only_p100_lr5e-5_e3_s0        en_only_p100_lr5e-5_e3        "${TRIM[@]}"
run_one te_only_p100_lr5e-5_e3_s0        te_only_p100_lr5e-5_e3        "${TRIM[@]}"
run_one hi_only_p100_lr5e-5_e3_s0        hi_only_p100_lr5e-5_e3        "${TRIM[@]}"
run_one mixed_only_p100_lr5e-5_e3_s0     mixed_only_p100_lr5e-5_e3     "${TRIM[@]}"
run_one hinglish_only_p100_lr5e-5_e3_s0  hinglish_only_p100_lr5e-5_e3  "${TRIM[@]}"
run_one hi_en_mix_only_p100_lr5e-5_e3_s0 hi_en_mix_only_p100_lr5e-5_e3 "${TRIM[@]}"
run_one joint8_p100_lr5e-5_e3_s0         joint8_p100_lr5e-5_e3         "${TRIM[@]}"
echo "queue end $(date +%H:%M)" >> $LOG
