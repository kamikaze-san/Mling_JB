#!/usr/bin/env bash
# AuditKIT audit (Lexsi kit) of the official base Q4_K_M vs the final model (base + final GGUF LoRA, as evaluated
# everywhere else), on the same held-out attacks (n = 382) and benign prompts (trimmed set). Waits for the
# quantization-survival run so only one GPU job runs at a time. Each model is served by llama-server on a local port
# and audited through AuditKIT's OpenAI-compatible client. Log: logs_eval_queue_v3.log; outputs in
# data/04_eval_cache/auditkit/<run>/ (summary tables are counts only).
cd "$(dirname "$0")/.."
LOG=logs_eval_queue_v3.log
BASE=~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf
FINAL_LORA=checkpoints/hf/joint_all_v3_mtopd_s0/checkpoint-50/adapter.gguf
ATT=data/04_eval_cache/base_q4_v2/eval_attacks_combined.jsonl
BEN=data/04_eval_cache/joint_all_v3_mtopd_ck50/eval_benign_combined.jsonl
PORT=8091
say() { echo "$* $(date '+%a %H:%M')" >> $LOG; }

while pgrep -f "run_quant_survival.sh" >/dev/null; do sleep 60; done
while pgrep -f "scripts/eval_adapter.py|scripts/score_stance_decider.py" >/dev/null; do sleep 30; done

audit() {  # audit <run name> <llama-server extra args...>
  local name=$1; shift
  llama-server -m "$BASE" "$@" --port $PORT -c 8192 -ngl 99 --seed 0 > logs_llama_server_$name.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
  say "auditkit $name start"
  .venv-curatorkit/bin/python scripts/audit_with_auditkit.py --run_name "auditkit_$name" --attacks $ATT --benign $BEN \
    --endpoint http://127.0.0.1:$PORT/v1 --model tiny-aya > data/04_eval_cache/auditkit_$name.log 2>&1
  say "auditkit $name done rc=$?"
  kill $pid; wait $pid 2>/dev/null
}
audit base_q4
audit final_mtopd_ck50 --lora "$FINAL_LORA"
say "auditkit end"
