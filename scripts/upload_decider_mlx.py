#!/usr/bin/env python3
"""Upload the prepared MLX 4-bit pplx-decider v1.3 conversions as PUBLIC Hugging Face model repos.

Run scripts/prepare_decider_mlx_release.py first. Without --yes this only prints what would be uploaded.
The token comes from the Hugging Face login (`hf auth login`) and is never printed.

  python scripts/upload_decider_mlx.py                    # dry run: repo names, file list, sizes
  python scripts/upload_decider_mlx.py --yes              # create the public repos and upload
  python scripts/upload_decider_mlx.py --only flash --yes # just one
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, get_token

DM = Path.home() / "Downloads" / "decision-models"
ITEMS = {"flash": (DM / "pplx-flash-v1.3-mlx" / "4bit-g64", "pplx-decider-flash-v1.3-mlx"),
         "27b": (DM / "pplx-27b-v1.3-mlx" / "4bit-g64", "pplx-decider-v1.3-mlx")}
REQUIRED = ["README.md", "ORIGINAL_MODEL_CARD.md", "LICENSE", "NOTICE", "model.safetensors", "readout.safetensors",
            "config.json", "decision_config.json", "joint_config.json", "tokenizer.json", "runner/mlx_decider.py",
            "runner/decider_common.py", "runner/RUNNER_NOTICE.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(ITEMS))
    ap.add_argument("--yes", action="store_true", help="actually create the public repos and upload")
    a = ap.parse_args()
    api = HfApi(token=get_token())
    owner = api.whoami()["name"]
    for key, (folder, name) in ITEMS.items():
        if a.only and key != a.only:
            continue
        missing = [f for f in REQUIRED if not (folder / f).exists()]
        if missing:
            raise SystemExit(f"{folder}: missing {missing}; run prepare_decider_mlx_release.py")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        gb = sum(p.stat().st_size for p in files) / 1e9
        repo = f"{owner}/{name}"
        print(f"{repo} (public): {len(files)} files, {gb:.1f} GB from {folder}")
        if not a.yes:
            continue
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(folder), repo_type="model",
                          ignore_patterns=["**/__pycache__/**", "*.pyc", ".DS_Store"],
                          commit_message="MLX 4-bit text-only conversion of the official v1.3 release")
        print(f"uploaded https://huggingface.co/{repo}")
    if not a.yes:
        print("dry run only; add --yes to upload")


if __name__ == "__main__":
    main()
