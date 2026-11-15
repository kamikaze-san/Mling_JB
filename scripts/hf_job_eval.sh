#!/bin/bash
# ==============================================================================
# hf_job_eval.sh — Run evaluation on Hugging Face Jobs (CUDA, a10g-small)
#
# Runs in one HF job:
#   1. llama.cpp with CUDA (llama-cpp-python) on Q4_K_M GGUF (+ optional LoRA GGUF)
#   2. run_safety_eval.py --generate_only (582 attacks, 491 benign prompts)
#   3. Qwen3Guard judging phase on CUDA
#   4. run_capability_eval.py (FLORES-200 devtest + Belebele)
#   5. Upload data/04_eval_cache/<run>/ to private dataset repo under runs/<run>/
#
# Usage:
#   ./scripts/hf_job_eval.sh --name base_q4_hf
#   ./scripts/hf_job_eval.sh --name joint8_lr5e-5_hf --lora_repo_path joint8_lr5e-5_e3_s0
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_NAME=""
LORA_REPO_PATH=""
LORA_SCALE="1.0"
FLAVOR="a10g-small"
DATASET_REPO="kamikaze-san/mjb-dpo-private"
ADAPTER_REPO="kamikaze-san/mjb-adapters-private"
DETACH=false
TIMEOUT="2h"
IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      RUN_NAME="$2"
      shift 2
      ;;
    --lora_repo_path)
      LORA_REPO_PATH="$2"
      shift 2
      ;;
    --lora_scale)
      LORA_SCALE="$2"
      shift 2
      ;;
    --flavor)
      FLAVOR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --dataset_repo)
      DATASET_REPO="$2"
      shift 2
      ;;
    --adapter_repo)
      ADAPTER_REPO="$2"
      shift 2
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --name <run_name> [--lora_repo_path <path_in_repo>] [--lora_scale 1.0] [--flavor a10g-small] [--detach]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "Error: --name <run_name> is required"
  exit 1
fi

HF_BIN="$ROOT_DIR/.venv/bin/hf"
if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="hf"
fi

if ! command -v "$HF_BIN" &>/dev/null; then
  echo "Error: hf CLI not found at $HF_BIN or in PATH"
  exit 1
fi

STORED_TOKEN="$(env -u HF_TOKEN "$ROOT_DIR/.venv/bin/python" -c 'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null | tail -1)"
if [[ -n "$STORED_TOKEN" ]]; then
  export HF_TOKEN="$STORED_TOKEN"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN is not set and could not be retrieved from stored credentials."
  exit 1
fi

echo "=== Staging payload for HF Job ($RUN_NAME) ==="
STAGING_DIR="$(mktemp -d /tmp/mjb_eval_staging_XXXXXX)"
trap 'rm -rf "$STAGING_DIR"' EXIT

mkdir -p "$STAGING_DIR/scripts" "$STAGING_DIR/configs" "$STAGING_DIR/src"
cp "$ROOT_DIR/scripts/run_safety_eval.py" "$STAGING_DIR/scripts/"
cp "$ROOT_DIR/scripts/run_capability_eval.py" "$STAGING_DIR/scripts/"
cp "$ROOT_DIR/scripts/download_benchmarks.py" "$STAGING_DIR/scripts/"
cp -r "$ROOT_DIR/configs"/* "$STAGING_DIR/configs/"
cp -r "$ROOT_DIR/src"/* "$STAGING_DIR/src/"

if [[ -f "${LLAMA_CPP_DIR:-$HOME/llama.cpp}/convert_lora_to_gguf.py" ]]; then
  cp "${LLAMA_CPP_DIR:-$HOME/llama.cpp}/convert_lora_to_gguf.py" "$STAGING_DIR/scripts/"
fi

LORA_SETUP_CMD=""
LORA_FLAG=""
if [[ -n "$LORA_REPO_PATH" ]]; then
  LORA_SETUP_CMD=$(cat <<INNER_EOF
echo "=== [HF Job] Downloading adapter: ${LORA_REPO_PATH} ==="
python3 -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN')
snapshot_download(
    repo_id='${ADAPTER_REPO}',
    allow_patterns='${LORA_REPO_PATH}/*',
    local_dir='adapters',
    token=token
)
"
ADAPTER_DIR="/app/adapters/${LORA_REPO_PATH}"
LORA_GGUF="\${ADAPTER_DIR}/adapter.gguf"
if [[ ! -f "\$LORA_GGUF" ]]; then
  echo "=== [HF Job] Converting PEFT adapter to GGUF ==="
  if [[ ! -f "/app/scripts/convert_lora_to_gguf.py" ]]; then
    git clone https://github.com/ggerganov/llama.cpp.git /app/llama_cpp_repo
    git -C /app/llama_cpp_repo checkout a0ad695cb212db3850f163121ddc687513448d42
    cp /app/llama_cpp_repo/convert_lora_to_gguf.py /app/scripts/
  fi
  python3 /app/scripts/convert_lora_to_gguf.py --base-model-id CohereLabs/tiny-aya-global --outfile "\$LORA_GGUF" "\$ADAPTER_DIR"
fi
echo "Adapter GGUF ready at \$LORA_GGUF"
INNER_EOF
)
  LORA_FLAG="--lora /app/adapters/${LORA_REPO_PATH}/adapter.gguf --lora_scale ${LORA_SCALE}"
fi

CONTAINER_CMD=$(cat <<EOF
set -euo pipefail
echo "=== [HF Job] Setting up evaluation workspace ==="
mkdir -p /app/scripts /app/configs /app/src /app/data/04_eval_cache /app/data/benchmarks /app/models /app/adapters
cp -r /workspace/* /app/
cd /app

echo "=== [HF Job] Installing dependencies ==="
pip install --no-cache-dir huggingface_hub datasets sacrebleu transformers accelerate peft
pip install --no-cache-dir nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
pip install --no-cache-dir "llama-cpp-python==0.3.36" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

CUDA_LIBS=\$(python3 -c "import os, site; paths = [os.path.join(p, 'nvidia', sub, 'lib') for p in site.getsitepackages() for sub in ('cuda_runtime', 'cublas')]; print(':'.join([p for p in paths if os.path.exists(p)]))")
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:\${CUDA_LIBS}:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:\${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"
python3 -c "import ctypes; ctypes.CDLL('libcudart.so.12'); print('libcudart.so.12 loaded successfully!')"
python3 -c "from llama_cpp import Llama; print('llama_cpp imported successfully!')"

echo "=== [HF Job] Downloading Tiny Aya Global Q4_K_M GGUF ==="
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='CohereLabs/tiny-aya-global-GGUF',
    filename='tiny-aya-global-q4_k_m.gguf',
    local_dir='models'
)
print('Base GGUF downloaded ->', path)
"
GGUF_PATH="/app/models/tiny-aya-global-q4_k_m.gguf"

echo "=== [HF Job] Downloading evaluation datasets ==="
python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN')
f1 = hf_hub_download(repo_id='${DATASET_REPO}', filename='eval/eval_attacks_combined.jsonl', repo_type='dataset', token=token)
f2 = hf_hub_download(repo_id='${DATASET_REPO}', filename='eval/eval_benign_combined.jsonl', repo_type='dataset', token=token)
f3 = hf_hub_download(repo_id='${DATASET_REPO}', filename='benchmarks/belebele_eval.jsonl', repo_type='dataset', token=token)

os.makedirs('data/04_eval_cache/${RUN_NAME}', exist_ok=True)
os.makedirs('data/benchmarks', exist_ok=True)
shutil.copy(f1, 'data/04_eval_cache/${RUN_NAME}/eval_attacks_combined.jsonl')
shutil.copy(f2, 'data/04_eval_cache/${RUN_NAME}/eval_benign_combined.jsonl')
shutil.copy(f3, 'data/benchmarks/belebele_eval.jsonl')
print('Datasets placed in data/04_eval_cache/${RUN_NAME} and data/benchmarks.')
"
ATTACKS_PATH="/app/data/04_eval_cache/${RUN_NAME}/eval_attacks_combined.jsonl"
BENIGN_PATH="/app/data/04_eval_cache/${RUN_NAME}/eval_benign_combined.jsonl"

echo "=== [HF Job] Fetching FLORES+ devtest on the fly ==="
python3 -c "
from scripts.download_benchmarks import download_flores
from pathlib import Path
download_flores(Path('data/benchmarks'), devtest_n=200, dev_n=0)
print('FLORES+ devtest downloaded successfully.')
"

${LORA_SETUP_CMD}

echo "=== [HF Job] 1. Safety Generation Phase (CUDA) ==="
python3 scripts/run_safety_eval.py \
    --run_name "${RUN_NAME}" \
    --gguf "\$GGUF_PATH" \
    --generate_only \
    --attacks "\$ATTACKS_PATH" \
    --benign "\$BENIGN_PATH" \
    ${LORA_FLAG}

echo "=== [HF Job] Syncing generation cache to ${DATASET_REPO}/runs/${RUN_NAME} ==="
python3 -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
api = HfApi(token=token)
out_dir = 'data/04_eval_cache/${RUN_NAME}'
if os.path.exists(out_dir):
    api.upload_folder(repo_id='${DATASET_REPO}', folder_path=out_dir, path_in_repo='runs/${RUN_NAME}', repo_type='dataset')
    print('Generation cache synced.')
"

echo "=== [HF Job] 2. Safety Judging Phase (Qwen3Guard on CUDA) ==="
python3 scripts/run_safety_eval.py \
    --run_name "${RUN_NAME}" \
    --gguf "\$GGUF_PATH" \
    --judge_model Qwen/Qwen3Guard-Gen-4B \
    --attacks "\$ATTACKS_PATH" \
    --benign "\$BENIGN_PATH" \
    --slice_by profile language global_or_local \
    ${LORA_FLAG}

echo "=== [HF Job] Syncing safety results to ${DATASET_REPO}/runs/${RUN_NAME} ==="
python3 -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
api = HfApi(token=token)
out_dir = 'data/04_eval_cache/${RUN_NAME}'
if os.path.exists(out_dir):
    api.upload_folder(repo_id='${DATASET_REPO}', folder_path=out_dir, path_in_repo='runs/${RUN_NAME}', repo_type='dataset')
    print('Safety evaluation results synced.')
"

echo "=== [HF Job] 3. Capability Evaluation Phase ==="
python3 scripts/run_capability_eval.py \
    --run_name "${RUN_NAME}" \
    --gguf "\$GGUF_PATH" \
    ${LORA_FLAG}

echo "=== [HF Job] 4. Uploading evaluation run results to ${DATASET_REPO}/runs/${RUN_NAME} ==="
python3 -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
api = HfApi(token=token)
out_dir = 'data/04_eval_cache/${RUN_NAME}'
api.upload_folder(
    repo_id='${DATASET_REPO}',
    folder_path=out_dir,
    path_in_repo='runs/${RUN_NAME}',
    repo_type='dataset'
)
print('Uploaded evaluation results to runs/${RUN_NAME} successfully.')
"
echo "=== [HF Job] All evaluation steps completed for ${RUN_NAME} ==="
EOF
)

RUN_ARGS=(
  jobs run
  --flavor "$FLAVOR"
  --timeout "$TIMEOUT"
  --secrets HF_TOKEN
  --name "eval-${RUN_NAME}"
  -v "$STAGING_DIR:/workspace:ro"
)

if [[ "$DETACH" == true ]]; then
  RUN_ARGS+=(--detach)
fi

RUN_ARGS+=("$IMAGE" bash -c "$CONTAINER_CMD")

echo "=== Submitting HF Evaluation Job ($RUN_NAME on $FLAVOR) ==="
START_TIME=$(date +%s)
"$HF_BIN" "${RUN_ARGS[@]}"
END_TIME=$(date +%s)
WALL_TIME=$((END_TIME - START_TIME))

echo "=== Evaluation Job finished (wall time: ${WALL_TIME}s) ==="
