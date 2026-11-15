#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/Multilingual_JB/.venv/bin/python"

echo "=== Resuming Candidate Labelling with max_tokens=1536 ==="
"$PYTHON" Multilingual_JB/scripts/label_pair_candidates.py \
    --model openai/gpt-oss-safeguard-20b \
    --api_base http://localhost:1234/v1 \
    --max_tokens 1536
