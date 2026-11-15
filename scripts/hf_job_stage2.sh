#!/usr/bin/env bash
# scripts/hf_job_stage2.sh — stage 2 (capability repair) on top of a safety-tuned adapter, on HF Jobs.
#
#   --method opd        on-policy distillation, reverse KL to the original model (scripts/train_opd.py)
#   --method retention  retention pairs + DPO (scripts/train_retention_dpo.py)
# Both start from the stage-1 adapter <init> in the PRIVATE adapter repo, use the benign prompt set
# opd/<prompts>.jsonl from the PRIVATE dataset repo (FLORES as ids; the text is fetched from the official gated
# FLORES+ inside the job and never uploaded), and upload the new adapter as <out>_s0.
#
# Usage:
#   ./scripts/hf_job_stage2.sh --method opd --init joint_all_v3_lr5e-5_e2_s0 --out joint_all_v3_opd \
#       --flavor a10g-large --timeout 2h --detach -- --steps 120 --lr 1e-5
#   ./scripts/hf_job_stage2.sh --method retention --init joint_all_v3_lr5e-5_e2_s0 --out joint_all_v3_ret -- --num_train_epochs 1
#   PRINT_CMD=1 ./scripts/hf_job_stage2.sh ...     # print the container script, submit nothing
# --timeout takes ONE unit ("90m", "2h"); "1h30m" is rejected by hf jobs (no GPU time is used when that happens).
# Everything after "--" is passed to the training script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

METHOD=""; INIT=""; OUT=""; PROMPTS="opd_v1"; FLAVOR="a10g-large"; TIMEOUT="2h"; DETACH=false
IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
DATASET_REPO="kamikaze-san/mjb-dpo-private"
ADAPTER_REPO="kamikaze-san/mjb-adapters-private"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --method) METHOD="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --detach) DETACH=true; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
case "$METHOD" in
  opd) TRAIN_SCRIPT=train_opd.py ;;
  retention) TRAIN_SCRIPT=train_retention_dpo.py ;;
  *) echo "--method must be opd or retention"; exit 1 ;;
esac
[[ -n "$INIT" && -n "$OUT" ]] || { echo "--init and --out are required"; exit 1; }
PREFIX="${OUT}_s0"

HF_BIN="$ROOT_DIR/.venv/bin/hf"; [[ -x "$HF_BIN" ]] || HF_BIN="hf"
STORED_TOKEN="$(env -u HF_TOKEN "$ROOT_DIR/.venv/bin/python" -c 'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null | tail -1)"
[[ -n "$STORED_TOKEN" ]] && export HF_TOKEN="$STORED_TOKEN"
[[ -n "${HF_TOKEN:-}" ]] || { echo "Error: no Hugging Face token"; exit 1; }

echo "=== Checking private repos, prompt set and stage-1 adapter before paying for a GPU ==="
"$ROOT_DIR/.venv/bin/python" - "$DATASET_REPO" "opd/${PROMPTS}.jsonl" "$ADAPTER_REPO" "$INIT" "$PREFIX" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi
ds, path, ad, init, prefix = sys.argv[1:]
api = HfApi()
for repo, t in ((ds, "dataset"), (ad, "model")):
    if not api.repo_info(repo, repo_type=t).private:
        sys.exit(f"REFUSING: {repo} is not private")
if not api.file_exists(ds, path, repo_type="dataset"):
    sys.exit(f"Missing in {ds}: {path} (run build_opd_prompts.py --upload)")
files = api.list_repo_files(ad)
if f"{init}/adapter_model.safetensors" not in files:
    sys.exit(f"Missing stage-1 adapter {ad}/{init}/adapter_model.safetensors")
if any(f.startswith(prefix + "/") for f in files):
    sys.exit(f"REFUSING: {ad}/{prefix}/ already exists (pick another --out)")
print("ok:", path, "|", init, "->", prefix)
PY

STAGING_DIR="$(mktemp -d /tmp/mjb_stage2_XXXXXX)"
trap 'rm -rf "$STAGING_DIR"' EXIT
mkdir -p "$STAGING_DIR/scripts" "$STAGING_DIR/configs" "$STAGING_DIR/src"
cp "$ROOT_DIR/scripts/train_opd.py" "$ROOT_DIR/scripts/train_retention_dpo.py" "$ROOT_DIR/scripts/train_dpo.py" "$STAGING_DIR/scripts/"
cp "$ROOT_DIR/configs/dpo.yaml" "$STAGING_DIR/configs/"
cp -r "$ROOT_DIR/src"/* "$STAGING_DIR/src/"
EXTRA_STR="$(printf '%q ' "${EXTRA[@]+"${EXTRA[@]}"}")"
MP_FLAG=""; [[ "$METHOD" == retention ]] && MP_FLAG="--mixed_precision bf16"
CKPT_FLAGS=""; [[ "$METHOD" == opd ]] && CKPT_FLAGS="--hub_repo ${ADAPTER_REPO} --hub_prefix ${PREFIX}"

CONTAINER_CMD=$(cat <<EOF
set -euo pipefail
mkdir -p /app && cp -r /workspace/* /app/ && cd /app
echo "=== [HF Job] Installing pinned dependencies ==="
(apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev > /dev/null) || echo "WARN: gcc install failed"
export CC=gcc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # OPD step 2 hit fragmentation OOM in the dry run
pip install --no-cache-dir transformers==5.22.2 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets huggingface_hub pyyaml "jinja2>=3.1.6"
pip uninstall -y torchvision torchaudio || true
python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" || { echo "FATAL: no CUDA device"; exit 3; }
( while sleep 120; do echo "[mem] RAM \$(free -m | awk '/Mem/{print \$3 "/" \$2}') MB | GPU \$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"; done ) &

echo "=== [HF Job] Downloading prompt set and stage-1 adapter ${INIT} ==="
python3 - <<'PYDL'
import os
from huggingface_hub import hf_hub_download, snapshot_download
tok = os.environ.get("HF_TOKEN")
hf_hub_download("${DATASET_REPO}", "opd/${PROMPTS}.jsonl", repo_type="dataset", local_dir="data/03_curated", token=tok)
snapshot_download("${ADAPTER_REPO}", repo_type="model", allow_patterns=["${INIT}/adapter_*", "${INIT}/*.json"],
                  local_dir="stage1", token=tok)
print("stage-1 files:", sorted(os.listdir("stage1/${INIT}")))
PYDL

echo "=== [HF Job] Stage 2 (${METHOD}) on top of ${INIT} ==="
TRAIN_RC=0
python3 scripts/${TRAIN_SCRIPT} --prompts data/03_curated/opd/${PROMPTS}.jsonl --init_adapter stage1/${INIT} \
    --out_dir out/${PREFIX} ${MP_FLAG} ${CKPT_FLAGS} ${EXTRA_STR} || TRAIN_RC=\$?
if [ "\$TRAIN_RC" -ne 0 ]; then echo "=== [HF Job] STAGE 2 FAILED (exit \$TRAIN_RC) ==="; exit "\$TRAIN_RC"; fi

echo "=== [HF Job] Uploading adapter -> ${ADAPTER_REPO}/${PREFIX} (no pairs, no reference copy) ==="
for attempt in 1 2 3; do
  python3 -c "
import os
from huggingface_hub import HfApi
HfApi(token=os.environ.get('HF_TOKEN')).upload_folder(repo_id='${ADAPTER_REPO}', folder_path='out/${PREFIX}',
    path_in_repo='${PREFIX}', repo_type='model',
    ignore_patterns=['checkpoint-*', 'checkpoint-*/*', 'ref/*', '*.local.jsonl', 'runs/*'])
print('uploaded')
" && break
  [ \$attempt -eq 3 ] && { echo "FATAL: upload failed 3 times"; exit 4; }
  sleep 30
done
echo "=== [HF Job] Done ==="
EOF
)

if [[ "${PRINT_CMD:-0}" == 1 ]]; then echo "$CONTAINER_CMD"; exit 0; fi
RUN_ARGS=(jobs run --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN --name "stage2-${METHOD}-${OUT}"
          -v "$STAGING_DIR:/workspace:ro")
[[ "$DETACH" == true ]] && RUN_ARGS+=(--detach)
RUN_ARGS+=("$IMAGE" bash -c "$CONTAINER_CMD")
echo "=== Submitting HF Job (stage 2 ${METHOD}: ${INIT} -> ${PREFIX}, ${FLAVOR}) ==="
"$HF_BIN" "${RUN_ARGS[@]}"
