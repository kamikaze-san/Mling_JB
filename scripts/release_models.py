#!/usr/bin/env python3
"""Publish the curated adapter release: one public HF model repo with the final model at the root and every other
adapter in stage1/, stage2/ and ablations/ (only adapter_model.safetensors, adapter_config.json, adapter.gguf; no
optimizer states, checkpoints or pickles). Files are uploaded straight from checkpoints/hf/ (no staging copy).

  python scripts/release_models.py                 # dry run: layout and size
  python scripts/release_models.py --yes           # create the public repo (if needed) and upload
  python scripts/release_models.py --yes --merged  # also upload the merged Q4_K_M GGUF of the final model
The token comes from the Hugging Face login and is never printed.
"""
import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "checkpoints" / "hf"
REPO = "tiny-aya-multilingual-safety"
FILES = ("adapter_model.safetensors", "adapter_config.json", "adapter.gguf")

LAYOUT = {
    "": "joint_all_v3_mtopd_s0/checkpoint-50",           # final model: stage 1 + multi-teacher OPD, step 50
    "stage1/joint_all_v3_lr5e-5_e2": "joint_all_v3_lr5e-5_e2_s0",
    "stage2/joint_all_v3_mtopd_step72": "joint_all_v3_mtopd_s0",
    "stage2/joint_all_v3_ret": "joint_all_v3_ret_s0",
    "stage2/joint_all_v3_opd_step50": "joint_all_v3_opd_s0/checkpoint-50",
    "stage2/joint_all_v3_opd_step100": "joint_all_v3_opd_s0",
    "stage2/joint_all_v3_opdtr_step50": "joint_all_v3_opdtr_s0/checkpoint-50",
    "stage2/joint_all_v3_opdtr_step100": "joint_all_v3_opdtr_s0",
    "stage2/joint_all_v3_opdtr_projected": "joint_all_v3_opdtr_proj_s0",
    "stage2/joint_all_v3_projected_plus_0.1_safety": "joint_all_v3_proj_b010_s0",
    "stage2/joint_all_v3_projected_plus_0.2_safety": "joint_all_v3_proj_b020_s0",
    "ablations/leave_one_out/no_te": "joint_all_no_te_v3_lr5e-5_e2_s0",
    "ablations/leave_one_out/no_ja": "joint_all_no_ja_v3_lr5e-5_e2_s0",
    "ablations/leave_one_out/no_mixed": "joint_all_no_mixed_v3_lr5e-5_e2_s0",
    "ablations/earlier_data_p100/joint_all_p100": "joint_all_p100_lr5e-5_e2_s0",
    "ablations/earlier_data_p100/joint_all_p100_step57": "joint_all_p100_lr5e-5_e2_ck57_s0",
    "ablations/earlier_data_p100/joint_all_p100_step114": "joint_all_p100_lr5e-5_e2_ck114_s0",
    "ablations/earlier_data_p100/joint_all_p100_step171": "joint_all_p100_lr5e-5_e2_ck171_s0",
    "ablations/earlier_data_p100/joint8_p100": "joint8_p100_lr5e-5_e3_s0",
    "ablations/joint8/joint8_lr5e-5": "joint8_lr5e-5_e3_s0",
    "ablations/joint8/joint8_lr2e-5": "joint8_lr2e-5_e3_s0",
}
for s in ["en", "hi", "te", "hinglish", "hi_en_mix", "te_en_mix", "ja_en_mix", "mixed"]:
    LAYOUT[f"ablations/single_surface_100_pairs/{s}"] = f"{s}_only_p100_lr5e-5_e3_s0"
for s in ["en", "hi", "te", "mixed"]:
    for lr in ["2e-5", "5e-5"]:
        LAYOUT[f"ablations/pilot_lr_sweep/{s}_lr{lr}"] = f"{s}_only_lr{lr}_e3_s0"
MERGED = ROOT / "exports" / "final_mtopd_ck50" / "model-q4_k_m.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--merged", action="store_true", help="also upload the merged Q4_K_M GGUF (after the survival check)")
    a = ap.parse_args()
    ops, total = [], 0
    for dst, src in LAYOUT.items():
        for f in FILES:
            p = H / src / f
            if not p.exists():
                raise SystemExit(f"missing {p}")
            ops.append(CommitOperationAdd(path_in_repo=f"{dst}/{f}" if dst else f, path_or_fileobj=str(p)))
            total += p.stat().st_size
    card = ROOT / "release" / "model_card.md"
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)))
    if a.merged:
        ops.append(CommitOperationAdd(path_in_repo="merged/tiny-aya-global-safety-merged-q4_k_m.gguf", path_or_fileobj=str(MERGED)))
        total += MERGED.stat().st_size
    api = HfApi(token=get_token())
    repo = f"{api.whoami()['name']}/{REPO}"
    print(f"{repo} (public): {len(LAYOUT)} adapters, {len(ops)} files, {total / 1e9:.1f} GB")
    if not a.yes:
        print("dry run only; add --yes to upload")
        return
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    for i in range(0, len(ops), 20):  # several commits, so an interrupted upload keeps what finished
        api.create_commit(repo, operations=ops[i:i + 20], repo_type="model",
                          commit_message=f"Adapters {i // 20 + 1}/{(len(ops) + 19) // 20}")
        print(f"  committed {min(i + 20, len(ops))}/{len(ops)}", flush=True)
    print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
