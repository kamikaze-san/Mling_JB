#!/usr/bin/env bash
# scripts/hf_job_train_dpo.sh
# Run DPO training on Hugging Face Jobs (private dataset & private model repo).
#
# Usage:
#   ./scripts/hf_job_train_dpo.sh --condition dryrun --seed 0
#   ./scripts/hf_job_train_dpo.sh --condition dryrun --seed 0 --flavor a10g-small --max_steps 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDITION="dryrun"
SEED=0
FLAVOR="a10g-small"
IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
DATASET_REPO="kamikaze-san/mjb-dpo-private"
ADAPTER_REPO="kamikaze-san/mjb-adapters-private"
MAX_STEPS=""
LR=""
EPOCHS=""
TAG=""
TIMEOUT="1h"
SAVE_FRAC="0.25"
RESUME=false
DETACH=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --condition)
      CONDITION="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --flavor)
      FLAVOR="$2"
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
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"   # suffix for the adapter folder in the repo, e.g. _lr2e-5_e3
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"   # e.g. 2h for joint_all; the job is killed (and billed) up to this
      shift 2
      ;;
    --save_frac)
      SAVE_FRAC="$2"   # resumable checkpoint every this fraction of training (uploaded as saved)
      shift 2
      ;;
    --resume)
      RESUME=true      # continue from the latest uploaded checkpoint of the SAME condition/tag/seed
      shift
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--condition NAME] [--seed INT] [--flavor FLAVOR] [--max_steps INT] [--detach]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

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
  echo "Error: HF_TOKEN environment variable is not set."
  exit 1
fi

echo "=== Ensuring private repos exist ==="
"$HF_BIN" repos create "$DATASET_REPO" --repo-type dataset --private --exist-ok
"$HF_BIN" repos create "$ADAPTER_REPO" --repo-type model --private --exist-ok

echo "=== Checking dataset file exists before paying for a GPU ==="
"$ROOT_DIR/.venv/bin/python" - "$DATASET_REPO" "dpo/${CONDITION}.jsonl" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi
repo, path = sys.argv[1], sys.argv[2]
api = HfApi()
if not api.repo_info(repo, repo_type="dataset").private:
    sys.exit(f"REFUSING: {repo} is not private")
if not api.file_exists(repo, path, repo_type="dataset"):
    sys.exit(f"Missing in {repo}: {path}")
print("ok:", path)
PY
PREFIX="${CONDITION}${TAG}_s${SEED}"
if [[ "$RESUME" == true ]]; then
  "$ROOT_DIR/.venv/bin/python" - "$ADAPTER_REPO" "$PREFIX" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi
repo, prefix = sys.argv[1], sys.argv[2]
ck = sorted({f.split("/")[1] for f in HfApi().list_repo_files(repo) if f.startswith(prefix + "/checkpoint-")},
            key=lambda c: int(c.rsplit("-", 1)[1]))
if not ck:
    sys.exit(f"--resume: no checkpoints under {repo}/{prefix}")
print("resume from:", prefix + "/" + ck[-1])
PY
fi

echo "=== Preparing staging payload directory ==="
STAGING_DIR="$(mktemp -d /tmp/mjb_payload_XXXXXX)"
trap 'rm -rf "$STAGING_DIR"' EXIT

mkdir -p "$STAGING_DIR/scripts" "$STAGING_DIR/configs" "$STAGING_DIR/src"
cp "$ROOT_DIR/scripts/train_dpo.py" "$STAGING_DIR/scripts/"
cp "$ROOT_DIR/configs/dpo.yaml" "$STAGING_DIR/configs/"
cp -r "$ROOT_DIR/src"/* "$STAGING_DIR/src/"

EXTRA_FLAGS="--save_every_frac $SAVE_FRAC --hub_checkpoint_repo $ADAPTER_REPO --hub_checkpoint_prefix $PREFIX"
if [[ "$RESUME" == true ]]; then EXTRA_FLAGS="$EXTRA_FLAGS --resume"; fi
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --max_steps $MAX_STEPS --logging_steps 1 --gradient_accumulation_steps 2"
fi
if [[ -n "$LR" ]]; then EXTRA_FLAGS="$EXTRA_FLAGS --learning_rate $LR"; fi
if [[ -n "$EPOCHS" ]]; then EXTRA_FLAGS="$EXTRA_FLAGS --num_train_epochs $EPOCHS"; fi

CONTAINER_CMD=$(cat <<EOF
set -euo pipefail
echo "=== [HF Job] Setting up workspace ==="
mkdir -p /app
cp -r /workspace/* /app/
cd /app
mkdir -p data/03_curated/dpo checkpoints/dpo

echo "=== [HF Job] Installing pinned dependencies ==="
# Triton JIT-compiles kernels at runtime and needs a C compiler; the -runtime image has none.
(apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev > /dev/null) || echo "WARN: gcc install failed"
export CC=gcc
pip install --no-cache-dir \
    transformers==5.22.2 \
    trl==1.17.1 \
    peft==0.21.2 \
    accelerate==1.15.0 \
    datasets \
    huggingface_hub \
    pyyaml \
    "jinja2>=3.1.6"
# The image's torchvision/torchaudio are built for its original torch; installing transformers
# upgrades torch, leaving them broken ("operator torchvision::nms does not exist"). Text-only
# training needs neither, so remove them.
pip uninstall -y torchvision torchaudio || true
python3 -c "import torch, transformers, trl; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| trl', trl.__version__)"
# Fail fast instead of training on CPU until the timeout.
python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" || { echo "FATAL: no CUDA device"; exit 3; }
# RAM / GPU memory every 2 min, so a silent out-of-memory kill can be diagnosed from the log.
( while sleep 120; do echo "[mem] RAM \$(free -m | awk '/Mem/{print \$3 "/" \$2}') MB | GPU \$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"; done ) &

upload_dir() {  # upload_dir <local dir> <path in repo>, 3 attempts; checkpoints were uploaded as saved
  for attempt in 1 2 3; do
    python3 -c "
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ.get('HF_TOKEN')).upload_folder(repo_id='${ADAPTER_REPO}', folder_path=sys.argv[1],
    path_in_repo=sys.argv[2], repo_type='model', ignore_patterns=['checkpoint-*', 'checkpoint-*/*'])
print('Uploaded ->', sys.argv[2])
" "\$1" "\$2" && return 0
    echo "upload attempt \$attempt failed; retrying in 30 s"; sleep 30
  done
  return 1
}

echo "=== [HF Job] Downloading private dataset: ${DATASET_REPO} ==="
python3 -c "
import os
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN')
path = hf_hub_download(
    repo_id='${DATASET_REPO}',
    filename='dpo/${CONDITION}.jsonl',
    repo_type='dataset',
    local_dir='data/03_curated',  # train_dpo.py reads data/03_curated/dpo/<condition>.jsonl
    token=token
)
print('Dataset downloaded ->', path)
"

OUT_DIR="checkpoints/dpo/${CONDITION}_s${SEED}"
if [ "${RESUME}" = true ]; then
  echo "=== [HF Job] Downloading checkpoints of ${PREFIX} to resume ==="
  python3 -c "
import os, glob, shutil
from huggingface_hub import snapshot_download
d = snapshot_download(repo_id='${ADAPTER_REPO}', repo_type='model', allow_patterns='${PREFIX}/checkpoint-*/*',
                      local_dir='/tmp/resume', token=os.environ.get('HF_TOKEN'))
os.makedirs('\$OUT_DIR', exist_ok=True)
for c in glob.glob('/tmp/resume/${PREFIX}/checkpoint-*'):
    shutil.move(c, os.path.join('\$OUT_DIR', os.path.basename(c)))
print('checkpoints:', sorted(os.listdir('\$OUT_DIR')))
"
fi

echo "=== [HF Job] Running DPO training (${CONDITION}, seed ${SEED}, CUDA bf16) ==="
TRAIN_RC=0
python3 scripts/train_dpo.py \
    --condition "${CONDITION}" \
    --seed "${SEED}" \
    --config configs/dpo.yaml \
    --mixed_precision bf16 \
    ${EXTRA_FLAGS} || TRAIN_RC=\$?

if [ "\$TRAIN_RC" -ne 0 ]; then
  echo "=== [HF Job] TRAINING FAILED (exit \$TRAIN_RC) ==="
  echo "Checkpoints saved so far are already in ${ADAPTER_REPO}/${PREFIX}/; relaunch the same command with --resume."
  exit "\$TRAIN_RC"
fi

echo "=== [HF Job] Uploading trained adapter to ${ADAPTER_REPO} ==="
upload_dir "\$OUT_DIR" "${CONDITION}${TAG}_s${SEED}" || { echo "FATAL: adapter upload failed 3 times"; exit 4; }
echo "=== [HF Job] Done ==="
EOF
)

if [[ "${PRINT_CMD:-0}" == 1 ]]; then  # PRINT_CMD=1: print the container script and stop (no job)
  echo "$CONTAINER_CMD"
  exit 0
fi

RUN_ARGS=(
  jobs run
  --flavor "$FLAVOR"
  --timeout "$TIMEOUT"
  --secrets HF_TOKEN
  --name "dpo-${CONDITION}-s${SEED}"
  -v "$STAGING_DIR:/workspace:ro"
)

if [[ "$DETACH" == true ]]; then
  RUN_ARGS+=(--detach)
fi

RUN_ARGS+=("$IMAGE" bash -c "$CONTAINER_CMD")

echo "=== Submitting HF Job (flavor: $FLAVOR, condition: $CONDITION, seed: $SEED) ==="
START_TIME=$(date +%s)
"$HF_BIN" "${RUN_ARGS[@]}"
END_TIME=$(date +%s)
WALL_TIME=$((END_TIME - START_TIME))

echo "=== Job execution finished (wall time: ${WALL_TIME}s) ==="
