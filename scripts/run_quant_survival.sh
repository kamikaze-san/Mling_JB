#!/usr/bin/env bash
# Quantization survival for the final adapter (EVAL_PROTOCOL.md §5), 15 Nov.
# Merge the final adapter (stage 1 + multi-teacher OPD, step 50) into the bf16 base, convert, quantize to Q4_K_M;
# re-quantize the base with the identical commands; evaluate both GGUFs (trim sets, as every Mac run) and score
# compliance; the merged model is gated against the re-quantized base. Log: logs_eval_queue_v3.log (counts only).
cd "$(dirname "$0")/.."
LOG=logs_eval_queue_v3.log
PPLX=~/Downloads/MARCH-Orz/.venv-pplx/bin/python
SNAP=~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324
C=data/03_curated
TRIM=(--aya_attacks $C/empty.jsonl --general_benign $C/benign_baseline_100.jsonl)
say() { echo "$* $(date '+%a %H:%M')" >> $LOG; }

while pgrep -f "scripts/eval_adapter.py|scripts/score_stance_decider.py" >/dev/null; do sleep 30; done
say "quant survival start"
[ -f exports/final_mtopd_ck50/model-q4_k_m.gguf ] || {
  .venv/bin/python scripts/export_adapter.py merged --adapter checkpoints/hf/joint_all_v3_mtopd_s0/checkpoint-50 \
    --base_dir $SNAP --llama_cpp ~/llama.cpp --out_dir exports/final_mtopd_ck50 > logs_export_final.log 2>&1
  say "export merged rc=$?"; }
[ -f exports/base_requant/model-q4_k_m.gguf ] || {
  .venv/bin/python scripts/export_adapter.py base --base_dir $SNAP --llama_cpp ~/llama.cpp \
    --out_dir exports/base_requant > logs_export_base.log 2>&1
  say "export base rc=$?"; }

for pair in "base_requant_q4|exports/base_requant/model-q4_k_m.gguf|base_q4_v2" \
            "final_mtopd_ck50_merged_q4|exports/final_mtopd_ck50/model-q4_k_m.gguf|base_requant_q4"; do
  IFS='|' read -r name gguf cmp <<< "$pair"
  [ -f "$gguf" ] || { say "$name SKIPPED (no $gguf)"; continue; }
  if [ ! -f data/04_eval_cache/$name/safety_results.jsonl ]; then
    say "$name eval start"
    .venv/bin/python scripts/eval_adapter.py --name "$name" --gguf "$gguf" --compare "$cmp" "${TRIM[@]}" \
      > data/04_eval_cache/eval_$name.log 2>&1
    say "$name eval done rc=$?"
  fi
  $PPLX scripts/score_stance_decider.py --run "$name" > data/04_eval_cache/stance_$name.log 2>&1
  say "$name stance done rc=$?"
done
say "quant survival end"
