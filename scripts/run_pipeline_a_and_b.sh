#!/usr/bin/env bash
# run_pipeline_a_and_b.sh
# Runs Option A (Decision-Model Health Check on Sheet B + Probes)
# then Option B (Candidate Labelling with Safeguard v2).
# Sequential execution prevents GPU contention and latency spikes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PPLX_PYTHON="$HOME/Downloads/MARCH-Orz/.venv-pplx/bin/python"
D1_PYTHON="$HOME/Downloads/MARCH-Orz/.venv-d1/bin/python"
MAIN_PYTHON="$ROOT_DIR/Multilingual_JB/.venv/bin/python"

echo "================================================================="
echo "=== STEP 1: Option A - Decision Model Health Check (Sheet B) ==="
echo "================================================================="

echo -e "\n--- 1.1 Running PPLX-Decider-Flash-v1.2 (MLX 4-bit) ---"
"$PPLX_PYTHON" Multilingual_JB/scripts/eval_d14_pplx.py

echo -e "\n--- 1.2 Running Liquid AI d1-2B (PyTorch MPS) ---"
"$D1_PYTHON" Multilingual_JB/scripts/eval_d14_d1.py

echo -e "\n--- 1.3 Generating Health Check Diagnostic Report ---"
"$MAIN_PYTHON" Multilingual_JB/scripts/analyze_d14_health_check.py

echo -e "\n>>> Option A Complete! Report written to Multilingual_JB/docs/decider_sheetB_health_check.md"

echo -e "\n================================================================="
echo "=== STEP 2: Option B - Safeguard v2 Candidate Labelling       ==="
echo "================================================================="

echo -e "\n--- 2.1 Swapping LM Studio to gpt-oss-safeguard-20b-mlx ---"
lms unload --all || true
lms load gpt-oss-safeguard-20b-mlx --identifier openai/gpt-oss-safeguard-20b -y

echo -e "\n--- 2.2 Waiting 5 seconds for LM Studio endpoint to settle ---"
sleep 5

echo -e "\n--- 2.3 Running Candidate Labelling (542 candidates) ---"
"$MAIN_PYTHON" Multilingual_JB/scripts/label_pair_candidates.py \
    --model openai/gpt-oss-safeguard-20b \
    --api_base http://localhost:1234/v1

echo -e "\n================================================================="
echo "=== Pipeline finished! Pausing as requested.                  ==="
echo "================================================================="
