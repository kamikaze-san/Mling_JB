#!/usr/bin/env bash
# Priority eval queue (Mac, one heavy GPU job at a time), 13 Nov.
# Repeatedly picks the FIRST adapter in PRIORITY that is not yet evaluated and is available:
#   - if its GGUF is missing, tries to download + convert it from the private HF repo (cheap no-op while the
#     HF job is still training: the root adapter only appears after the job's final upload);
#   - then eval_adapter.py (llama.cpp Metal + Qwen3Guard) -> score_stance_decider.py (MLX, separate process).
# So new adapters jump ahead of the 100-pair ablations as soon as they land, and the Mac never idles while
# something is runnable. A failed adapter is recorded and skipped (no retry loop). Exits when everything is done
# or failed, or after MAX_IDLE_MIN minutes with nothing runnable.
# Start it only when nothing else needs the GPU (e.g. after pair labelling, which uses LM Studio).
cd "$(dirname "$0")/.."
LOG=logs_eval_queue_v3.log
FAILED=data/04_eval_cache/queue_v3_failed.txt
PPLX=~/Downloads/MARCH-Orz/.venv-pplx/bin/python
C=data/03_curated
MAX_IDLE_MIN=${MAX_IDLE_MIN:-480}
touch "$FAILED"

# adapter folder in the HF repo | run name | test sets
#   full = + Aya 200 and 400 general benign; trim = joint8 sets;
#   lap  = trim sets, generated + capability-tested on the laptop (laptop/run_laptop_eval.py): runnable only after
#          scripts/import_laptop_results.py imported it; the Mac then only judges (any reply missing from the
#          cache is generated here as a fallback)
PRIORITY=(
  "joint_all_v3_lr5e-5_e2_s0|joint_all_v3_lr5e-5_e2|full"
  "joint_all_no_te_v3_lr5e-5_e2_s0|joint_all_no_te_v3_lr5e-5_e2|trim"
  "joint_all_no_ja_v3_lr5e-5_e2_s0|joint_all_no_ja_v3_lr5e-5_e2|trim"
  "joint_all_no_mixed_v3_lr5e-5_e2_s0|joint_all_no_mixed_v3_lr5e-5_e2|trim"
  "joint_all_p100_lr5e-5_e2_ck57_s0|joint_all_p100_lr5e-5_e2_ck57|lap"
  "joint_all_p100_lr5e-5_e2_ck114_s0|joint_all_p100_lr5e-5_e2_ck114|lap"
  "mixed_only_p100_lr5e-5_e3_s0|mixed_only_p100_lr5e-5_e3|lap"
  "joint8_p100_lr5e-5_e3_s0|joint8_p100_lr5e-5_e3|lap"
  "joint_all_p100_lr5e-5_e2_ck171_s0|joint_all_p100_lr5e-5_e2_ck171|lap"
  "te_en_mix_only_p100_lr5e-5_e3_s0|te_en_mix_only_p100_lr5e-5_e3|trim"
  "ja_en_mix_only_p100_lr5e-5_e3_s0|ja_en_mix_only_p100_lr5e-5_e3|trim"
  "hinglish_only_p100_lr5e-5_e3_s0|hinglish_only_p100_lr5e-5_e3|lap"
  "hi_en_mix_only_p100_lr5e-5_e3_s0|hi_en_mix_only_p100_lr5e-5_e3|lap"
)
TRIM=(--aya_attacks $C/empty.jsonl --general_benign $C/benign_baseline_100.jsonl)

is_done()   { [ -f data/04_eval_cache/$1/stance_decider.jsonl ] && [ -f data/04_eval_cache/$1/capability_summary.json ]; }
is_failed() { grep -qx "$1" "$FAILED"; }

run_one() {  # run_one <adapter folder> <run name> <full|trim>
  local ad=$1 name=$2 sets=$3 extra=()
  [ "$sets" = trim ] && extra=("${TRIM[@]}")
  [ "$sets" = lap ] && extra=("${TRIM[@]}" --skip_capability)   # capability came from the laptop
  echo "$name eval start $(date '+%a %H:%M')" >> $LOG
  .venv/bin/python scripts/eval_adapter.py --name "$name" --lora "checkpoints/hf/$ad/adapter.gguf" \
    --compare base_q4_v2 "${extra[@]}" > data/04_eval_cache/eval_$name.log 2>&1
  local rc=$?   # 1 = a gate failed, which is a result, not an error
  if [ ! -f data/04_eval_cache/$name/safety_results.jsonl ]; then
    echo "$name EVAL FAILED rc=$rc (see data/04_eval_cache/eval_$name.log)" >> $LOG; echo "$name" >> "$FAILED"; return; fi
  echo "$name eval done rc=$rc $(date '+%a %H:%M')" >> $LOG
  $PPLX scripts/score_stance_decider.py --run "$name" > data/04_eval_cache/stance_$name.log 2>&1
  local src=$?
  echo "$name stance done rc=$src $(date '+%a %H:%M')" >> $LOG
  [ $src -ne 0 ] && echo "$name" >> "$FAILED"
}

~/.lmstudio/bin/lms unload --all >/dev/null 2>&1 || true
echo "queue v3 start $(date '+%a %H:%M')" >> $LOG
idle=0
while true; do
  ran=false; pending=0
  for entry in "${PRIORITY[@]}"; do
    IFS='|' read -r ad name sets <<< "$entry"
    if is_done "$name" || is_failed "$name"; then continue; fi
    pending=$((pending + 1))
    if [ "$sets" = lap ] && [ ! -f "data/04_eval_cache/$name/generation_manifest_laptop.json" ]; then
      continue   # not imported from the laptop yet
    fi
    if [ ! -f "checkpoints/hf/$ad/adapter.gguf" ]; then
      .venv/bin/python scripts/download_and_convert_adapters.py "$ad" >> logs_eval_queue_v3_fetch.log 2>&1 || continue
      echo "$name fetched + converted $(date '+%a %H:%M')" >> $LOG
    fi
    run_one "$ad" "$name" "$sets"; ran=true; break   # rescan from the top: a new adapter may have landed
  done
  [ $pending -eq 0 ] && { echo "queue v3 end: all done or failed $(date '+%a %H:%M')" >> $LOG; break; }
  if ! $ran; then
    idle=$((idle + 5))
    [ $idle -ge $MAX_IDLE_MIN ] && { echo "queue v3 end: idle ${MAX_IDLE_MIN} min, $pending not available $(date '+%a %H:%M')" >> $LOG; break; }
    sleep 300
  else
    idle=0
  fi
done
