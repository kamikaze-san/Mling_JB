#!/usr/bin/env python3
"""
scripts/download_and_convert_adapters.py — Task D17b driver

Downloads completed adapters from kamikaze-san/mjb-adapters-private and
converts them to GGUF LoRA format (f16) for evaluation.

Usage:
  python scripts/download_and_convert_adapters.py
"""

import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, get_token

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--CohereLabs--tiny-aya-global" / "snapshots" / "af89d219b53ed9b13b8a4645f9c8028973510324"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
REPO_ID = "kamikaze-san/mjb-adapters-private"
TARGET_DIR = ROOT_DIR / "checkpoints" / "hf"

TARGET_ADAPTERS = [
    "en_only_lr2e-5_e3_s0",
    "en_only_lr5e-5_e3_s0",
    "te_only_lr2e-5_e3_s0",
    "te_only_lr5e-5_e3_s0",
    "hi_only_lr2e-5_e3_s0",
    "hi_only_lr5e-5_e3_s0",
    "mixed_only_lr2e-5_e3_s0",
    "mixed_only_lr5e-5_e3_s0",
]


def process_adapter(name: str):
    adapter_dir = TARGET_DIR / name
    adapter_gguf = adapter_dir / "adapter.gguf"

    # Step 1: Download root adapter files (skip large intermediate checkpoints)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        print(f"[{name}] Downloading adapter from {REPO_ID}...")
        token = os.environ.get("HF_TOKEN") or get_token()
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=[f"{name}/*"],
            ignore_patterns=[f"{name}/checkpoint-*/*", f"{name}/checkpoint-*"],
            local_dir=str(TARGET_DIR),
            token=token
        )

    if not (adapter_dir / "adapter_model.safetensors").exists():
        print(f"[{name}] ERROR: adapter_model.safetensors missing after download")
        return False

    # Step 2: Convert to GGUF LoRA if not already converted
    if not adapter_gguf.exists():
        print(f"[{name}] Converting PEFT adapter to GGUF LoRA...")
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "export_adapter.py"),
            "lora",
            "--adapter", str(adapter_dir),
            "--base_dir", str(BASE_SNAPSHOT),
            "--llama_cpp", str(LLAMA_CPP_DIR)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[{name}] ERROR during export:\n{res.stderr}")
            return False

    if adapter_gguf.exists():
        size_mb = adapter_gguf.stat().st_size / (1024 * 1024)
        print(f"[D17b] {name:<26} -> {adapter_gguf.name} ({size_mb:.1f} MB) OK")
        return True
    else:
        print(f"[{name}] ERROR: {adapter_gguf.name} was not generated")
        return False


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_SNAPSHOT.exists():
        print(f"Error: Base snapshot directory not found: {BASE_SNAPSHOT}")
        sys.exit(1)
    if not LLAMA_CPP_DIR.exists():
        print(f"Error: llama.cpp checkout not found: {LLAMA_CPP_DIR}")
        sys.exit(1)

    names = sys.argv[1:] or TARGET_ADAPTERS  # adapter folder names in the repo; default: the pilot set
    print("=== [D17b] Downloading & Converting Adapters ===")
    success_count = 0
    for name in names:
        ok = process_adapter(name)
        if ok:
            success_count += 1

    print(f"\n[D17b Summary] {success_count}/{len(names)} adapters converted to GGUF.")
    if success_count < len(names):
        sys.exit(1)


if __name__ == "__main__":
    main()
