#!/usr/bin/env bash
# Judge validation on the 128-row human sheet (15 Nov). Runs after eval queue v9 has exited, one GPU job at a time:
#   1. harm judges on all 128 rows (they had only the first 64): Qwen3Guard, decider (eval wording), gpt-oss-safeguard
#   2. pplx-decider flash v1.3 (converted to MLX 4-bit here), 27B v1.1 (mlx-community 4-bit), 27B v1.3 (converted here)
#   3. scripts/judges_vs_human128.py
# Each step is resumable and skipped if its inputs are missing. Log: logs_judge_validation.log (counts only).
cd "$(dirname "$0")/.."
LOG=logs_judge_validation.log
V=data/04_eval_cache/strata_judge_validation
SHEET=$V/judge_labelling_strata.csv
PPLX=~/Downloads/MARCH-Orz/.venv-pplx/bin/python
D27=~/Downloads/decision-models/.venv-decider27/bin/python
DM=~/Downloads/decision-models
LMS=~/.lmstudio/bin/lms
DL=${DECIDER_DL_LOG:-$DM/decider_download.log}   # download log: a model is used only after its DONE line
say() { echo "$* $(date '+%a %H:%M')" >> $LOG; }

while pgrep -f "run_eval_queue_v9.sh" >/dev/null; do sleep 60; done
while pgrep -f "scripts/eval_adapter.py|scripts/score_stance_decider.py" >/dev/null; do sleep 30; done
$LMS unload --all >/dev/null 2>&1 || true
say "chain start"

.venv/bin/python scripts/compare_harm_judges.py --judge qwen3guard --sheet $SHEET >> $LOG 2>&1; say "qwen3guard rc=$?"
$PPLX scripts/compare_harm_judges.py --judge decider --sheet $SHEET >> $LOG 2>&1; say "decider-eval rc=$?"
if $LMS load gpt-oss-safeguard-20b-mlx --yes >/dev/null 2>&1; then
  .venv/bin/python scripts/compare_harm_judges.py --judge safeguard_harm --sheet $SHEET --model gpt-oss-safeguard-20b-mlx >> $LOG 2>&1
  say "safeguard rc=$?"
  $LMS unload --all >/dev/null 2>&1 || true
else
  say "safeguard SKIPPED (lms load failed)"
fi

if grep -q "DONE perplexity-ai/pplx-decider-flash-v1.3" "$DL" 2>/dev/null && [ ! -f $DM/pplx-flash-v1.3-mlx/4bit-g64/model.safetensors ]; then
  $PPLX scripts/convert_decider_mlx.py $DM/pplx-flash-v1.3 $DM/pplx-flash-v1.3-mlx/4bit-g64 >> $LOG 2>&1; say "convert flash v1.3 rc=$?"
fi
[ -f $DM/pplx-flash-v1.3-mlx/4bit-g64/model.safetensors ] && {
  $PPLX scripts/score_sheet_deciders.py --tag flash_v13 --backend marker --model $DM/pplx-flash-v1.3-mlx/4bit-g64 >> $LOG 2>&1
  say "flash v1.3 rc=$?"; }

$D27 scripts/score_sheet_deciders.py --tag v11_27b --backend lasttoken \
  --model ~/.lmstudio/models/mlx-community/pplx-decider-v1.1-27b-4bit >> $LOG 2>&1; say "27B v1.1 rc=$?"

# optional, last on purpose (user, 15 Nov: drop it if it slows things down); needs the flash-style joint/readout format
if grep -q "DONE perplexity-ai/pplx-decider-v1.3 " "$DL" 2>/dev/null && [ -f $DM/pplx-27b-v1.3/readout.safetensors ] && [ -f $DM/pplx-27b-v1.3/joint_config.json ] && [ ! -f $DM/pplx-27b-v1.3-mlx/4bit-g64/model.safetensors ]; then
  $PPLX scripts/convert_decider_mlx.py $DM/pplx-27b-v1.3 $DM/pplx-27b-v1.3-mlx/4bit-g64 >> $LOG 2>&1; say "convert 27B v1.3 rc=$?"
fi
[ -f $DM/pplx-27b-v1.3-mlx/4bit-g64/model.safetensors ] && {
  $PPLX scripts/score_sheet_deciders.py --tag v13_27b --backend marker --model $DM/pplx-27b-v1.3-mlx/4bit-g64 >> $LOG 2>&1
  say "27B v1.3 rc=$?"; }

python3 scripts/judges_vs_human128.py >> $LOG 2>&1
say "chain end"
