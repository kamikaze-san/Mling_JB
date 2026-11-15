#!/usr/bin/env python3
"""Push an adapter further along the stage-1 safety update, with no training: delta' = delta_X + beta * S.

S = stage-1 delta (B_s A_s). X is any adapter built on stage 1 (e.g. the projected stage-2 adapter). The sum is written
EXACTLY as one LoRA of rank r_x + r_s:
    B' = [B_x, beta * B_s],  A' = [A_x ; A_s]      (both adapters must share lora_alpha / r, so the scale is unchanged)

  python scripts/add_safety_direction.py --base joint_all_v3_opdtr_proj_s0 --stage1 joint_all_v3_lr5e-5_e2_s0 \
      --beta 0.1 --out joint_all_v3_proj_b010_s0       # folders under checkpoints/hf/; then download_and_convert_adapters.py
Prints numbers only.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

H = Path(__file__).resolve().parent.parent / "checkpoints" / "hf"


def ip(B1, A1, B2, A2):
    return float(((B1.T @ B2) * (A1 @ A2.T)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="adapter to push (folder under checkpoints/hf/)")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cx, cs = (json.load(open(H / d / "adapter_config.json")) for d in (a.base, a.stage1))
    for c in (cx, cs):
        if c.get("use_rslora") or c.get("use_dora") or c.get("rank_pattern") or c.get("alpha_pattern"):
            raise SystemExit("only plain LoRA is supported")
    if cx["lora_alpha"] * cs["r"] != cs["lora_alpha"] * cx["r"]:
        raise SystemExit(f"lora_alpha / r differs: {cx['lora_alpha']}/{cx['r']} vs {cs['lora_alpha']}/{cs['r']}")
    x, s = load_file(H / a.base / "adapter_model.safetensors"), load_file(H / a.stage1 / "adapter_model.safetensors")
    if sorted(x) != sorted(s):
        raise SystemExit("adapter tensors differ between the two adapters")

    out, SS, XS = {}, 0.0, 0.0
    for ka in sorted(k for k in x if ".lora_A." in k):
        kb = ka.replace(".lora_A.", ".lora_B.")
        Ax, Bx, As, Bs = (t.float() for t in (x[ka], x[kb], s[ka], s[kb]))
        SS += ip(Bs, As, Bs, As)
        XS += ip(Bx, Ax, Bs, As)
        out[ka] = torch.cat([Ax, As], dim=0).to(x[ka].dtype)
        out[kb] = torch.cat([Bx, a.beta * Bs], dim=1).to(x[kb].dtype)

    dst = H / a.out
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    r = cx["r"] + cs["r"]
    cfg = dict(cx, r=r, lora_alpha=cx["lora_alpha"] * r // cx["r"])
    json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (H / a.base / f).exists():
            shutil.copy(H / a.base / f, dst / f)
    json.dump({"script": "add_safety_direction", **vars(a), "SS": SS, "XS": XS}, open(dst / "safety_push_manifest.json", "w"),
              indent=2)
    print(f"{a.base} + {a.beta} * {a.stage1} -> {a.out}: projection on S {XS / SS:.3f} -> {XS / SS + a.beta:.3f} "
          f"(stage 1 = 1.000) | rank {r}, alpha {cfg['lora_alpha']}")


if __name__ == "__main__":
    main()
