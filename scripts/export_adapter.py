"""
Turn a trained PEFT adapter into deployable GGUF artifacts with llama.cpp tools.

Main path (used for all ablation evals):
  PEFT adapter -> GGUF LoRA (f16), applied at runtime on the official Q4_K_M base.
    python scripts/export_adapter.py lora --adapter checkpoints/dpo/joint_s0 \
        --base_dir <local HF snapshot of tiny-aya-global> --llama_cpp ~/llama.cpp

Shipping check (optional):
  merge into BF16 -> convert -> quantize Q4_K_M with a pinned llama.cpp.
  Re-quantize the BASE with the identical command for a fair comparison.
    python scripts/export_adapter.py merged --adapter checkpoints/dpo/joint_s0 \
        --base_dir <snapshot> --llama_cpp ~/llama.cpp --out_dir exports/joint_s0
    python scripts/export_adapter.py base --base_dir <snapshot> --llama_cpp ~/llama.cpp \
        --out_dir exports/base_requant
"""

import argparse
import os
import subprocess
import sys


def sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def lora_cmd(llama_cpp, base_dir, adapter, outfile):
    return [sys.executable, os.path.join(llama_cpp, "convert_lora_to_gguf.py"),
            "--base", base_dir, "--outtype", "f16", "--outfile", outfile, adapter]


def convert_cmd(llama_cpp, hf_dir, outfile):
    return [sys.executable, os.path.join(llama_cpp, "convert_hf_to_gguf.py"), hf_dir,
            "--outtype", "bf16", "--outfile", outfile]


def quantize_cmd(llama_cpp, src, dst, qtype="Q4_K_M"):
    exe = os.path.join(llama_cpp, "build", "bin", "llama-quantize")
    if not os.path.exists(exe):  # unbuilt checkout: use the installed llama-quantize (same binary for base and merged)
        import shutil
        exe = shutil.which("llama-quantize") or exe
    return [exe, src, dst, qtype]


def fix_eos(llama_cpp, gguf_path, base_dir):
    """convert_hf_to_gguf takes EOS from tokenizer_config (<|END_OF_TURN_TOKEN|>, id 6), but the model's
    generation_config and the official GGUF use <EOS_TOKEN> (id 3). With the wrong id, generation does not stop where
    the deployed model stops, so set it from generation_config.json to match the official GGUF."""
    import json
    gc = os.path.join(base_dir, "generation_config.json")
    if not os.path.exists(gc):
        return
    eos = json.load(open(gc)).get("eos_token_id")
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        return
    script = os.path.join(llama_cpp, "gguf-py", "gguf", "scripts", "gguf_set_metadata.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(llama_cpp, "gguf-py"))
    subprocess.run([sys.executable, script, gguf_path, "tokenizer.ggml.eos_token_id", str(eos), "--force"],
                   check=True, env=env)


def merge_adapter(base_dir, adapter, out_hf):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out_hf, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(out_hf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lora", "merged", "base"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base_dir", required=True, help="local HF snapshot directory of the base model")
    ap.add_argument("--llama_cpp", required=True, help="llama.cpp checkout (pin the commit!)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--qtype", default="Q4_K_M")
    args = ap.parse_args()
    llama_cpp = os.path.expanduser(args.llama_cpp)

    if args.mode == "lora":
        out = os.path.join(args.adapter, "adapter.gguf")
        sh(lora_cmd(llama_cpp, args.base_dir, args.adapter, out))
        print(f"GGUF LoRA -> {out}")
        return

    out_dir = args.out_dir or "exports/out"
    os.makedirs(out_dir, exist_ok=True)
    if args.mode == "merged":
        hf_dir = os.path.join(out_dir, "merged_hf")
        merge_adapter(args.base_dir, args.adapter, hf_dir)
    else:
        hf_dir = args.base_dir
    bf16 = os.path.join(out_dir, "model-bf16.gguf")
    q = os.path.join(out_dir, f"model-{args.qtype.lower()}.gguf")
    sh(convert_cmd(llama_cpp, hf_dir, bf16))
    fix_eos(llama_cpp, bf16, args.base_dir)
    sh(quantize_cmd(llama_cpp, bf16, q, args.qtype))
    print(f"{args.mode} -> {q}")


if __name__ == "__main__":
    main()
