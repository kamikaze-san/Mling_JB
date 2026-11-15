#!/usr/bin/env bash
# Priority eval queue v5 (Mac), 14 Nov 01:15. Same as v4 plus the two stage-2 adapters (retention DPO, OPD) right
# after the leave-one-out runs, evaluated fully on the Mac overnight (trim sets). v4 notes:
#   - a LoRA-scale column (the LoRA-scale comparison, e.g. joint_all_v3 at 0.75);
#   - it first waits until no eval_adapter.py / score_stance_decider.py process is running, so it can replace a
#     stopped v3 loop mid-run without two GPU jobs at once;
#   - a run whose safety judging already finished (safety_results.jsonl + gate verdict in its eval log) but has no
#     stance file gets only the stance step (no second Qwen3Guard pass).
cd "$(dirname "$0")/.."
LOG=logs_eval_queue_v3.log            # same log as v3, so the history stays in one place
FAILED=data/04_eval_cache/queue_v3_failed.txt
PPLX=~/Downloads/MARCH-Orz/.venv-pplx/bin/python
C=data/03_curated
MAX_IDLE_MIN=${MAX_IDLE_MIN:-480}
touch "$FAILED"

# adapter folder in the HF repo | run name | test sets | LoRA scale
#   full = + Aya 200 and 400 general benign; trim = joint8 sets;
#   lap  = trim sets, generated + capability-tested on the laptop: runnable only after import_laptop_results.py
PRIORITY=(
  "joint_all_v3_lr5e-5_e2_s0|joint_all_v3_lr5e-5_e2|full|1.0"
  "joint_all_no_te_v3_lr5e-5_e2_s0|joint_all_no_te_v3_lr5e-5_e2|trim|1.0"
  "joint_all_no_ja_v3_lr5e-5_e2_s0|joint_all_no_ja_v3_lr5e-5_e2|trim|1.0"
  "joint_all_no_mixed_v3_lr5e-5_e2_s0|joint_all_no_mixed_v3_lr5e-5_e2|trim|1.0"
  "joint_all_v3_opd_s0|joint_all_v3_opd|trim|1.0"
  "joint_all_v3_ret_s0|joint_all_v3_ret|trim|1.0"
  "joint_all_p100_lr5e-5_e2_ck57_s0|joint_all_p100_lr5e-5_e2_ck57|lap|1.0"
  "joint_all_p100_lr5e-5_e2_ck114_s0|joint_all_p100_lr5e-5_e2_ck114|lap|1.0"
  "mixed_only_p100_lr5e-5_e3_s0|mixed_only_p100_lr5e-5_e3|lap|1.0"
  "joint8_p100_lr5e-5_e3_s0|joint8_p100_lr5e-5_e3|lap|1.0"
  "joint_all_p100_lr5e-5_e2_ck171_s0|joint_all_p100_lr5e-5_e2_ck171|lap|1.0"
  "te_en_mix_only_p100_lr5e-5_e3_s0|te_en_mix_only_p100_lr5e-5_e3|trim|1.0"
  "ja_en_mix_only_p100_lr5e-5_e3_s0|ja_en_mix_only_p100_lr5e-5_e3|trim|1.0"
  "joint_all_v3_lr5e-5_e2_s0|joint_all_v3_lr5e-5_e2_sc075|lap|0.75"
  "hinglish_only_p100_lr5e-5_e3_s0|hinglish_only_p100_lr5e-5_e3|lap|1.0"
  "hi_en_mix_only_p100_lr5e-5_e3_s0|hi_en_mix_only_p100_lr5e-5_e3|lap|1.0"
)
TRIM=(--aya_attacks $C/empty.jsonl --general_benign $C/benign_baseline_100.jsonl)

is_done()   { [ -f data/04_eval_cache/$1/stance_decider.jsonl ] && [ -f data/04_eval_cache/$1/capability_summary.json ]; }
is_failed() { grep -qx "$1" "$FAILED"; }
judged()    { [ -f data/04_eval_cache/$1/safety_results.jsonl ] && grep -q "CHECKPOINT GATES VERDICT" data/04_eval_cache/eval_$1.log 2>/dev/null; }

stance() {  # stance <run name>
  $PPLX scripts/score_stance_decider.py --run "$1" > data/04_eval_cache/stance_$1.log 2>&1
  local src=$?
  echo "$1 stance done rc=$src $(date '+%a %H:%M')" >> $LOG
  [ $src -ne 0 ] && echo "$1" >> "$FAILED"
}

run_one() {  # run_one <adapter folder> <run name> <full|trim|lap> <scale>
  local ad=$1 name=$2 sets=$3 scale=$4 extra=()
  if judged "$name"; then
    echo "$name already judged, stance only $(date '+%a %H:%M')" >> $LOG; stance "$name"; return; fi
  [ "$sets" = trim ] && extra=("${TRIM[@]}")
  [ "$sets" = lap ] && extra=("${TRIM[@]}" --skip_capability)   # capability came from the laptop
  echo "$name eval start (scale $scale) $(date '+%a %H:%M')" >> $LOG
  .venv/bin/python scripts/eval_adapter.py --name "$name" --lora "checkpoints/hf/$ad/adapter.gguf" \
    --lora_scale "$scale" --compare base_q4_v2 "${extra[@]}" > data/04_eval_cache/eval_$name.log 2>&1
  local rc=$?   # 1 = a gate failed, which is a result, not an error
  if [ ! -f data/04_eval_cache/$name/safety_results.jsonl ]; then
    echo "$name EVAL FAILED rc=$rc (see data/04_eval_cache/eval_$name.log)" >> $LOG; echo "$name" >> "$FAILED"; return; fi
  echo "$name eval done rc=$rc $(date '+%a %H:%M')" >> $LOG
  stance "$name"
}

while pgrep -f "scripts/eval_adapter.py|scripts/score_stance_decider.py" >/dev/null; do sleep 30; done
~/.lmstudio/bin/lms unload --all >/dev/null 2>&1 || true
echo "queue v5 start $(date '+%a %H:%M')" >> $LOG
idle=0
while true; do
  ran=false; pending=0
  for entry in "${PRIORITY[@]}"; do
    IFS='|' read -r ad name sets scale <<< "$entry"
    if is_done "$name" || is_failed "$name"; then continue; fi
    pending=$((pending + 1))
    if [ "$sets" = lap ] && [ ! -f "data/04_eval_cache/$name/generation_manifest_laptop.json" ]; then
      continue   # not imported from the laptop yet
    fi
    if [ ! -f "checkpoints/hf/$ad/adapter.gguf" ]; then
      .venv/bin/python scripts/download_and_convert_adapters.py "$ad" >> logs_eval_queue_v3_fetch.log 2>&1 || continue
      echo "$name fetched + converted $(date '+%a %H:%M')" >> $LOG
    fi
    run_one "$ad" "$name" "$sets" "$scale"; ran=true; break   # rescan from the top
  done
  [ $pending -eq 0 ] && { echo "queue v5 end: all done or failed $(date '+%a %H:%M')" >> $LOG; break; }
  if ! $ran; then
    idle=$((idle + 5))
    [ $idle -ge $MAX_IDLE_MIN ] && { echo "queue v5 end: idle ${MAX_IDLE_MIN} min, $pending not available $(date '+%a %H:%M')" >> $LOG; break; }
    sleep 300
  else
    idle=0
  fi
done
