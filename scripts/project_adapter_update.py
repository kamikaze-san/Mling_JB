#!/usr/bin/env python3
"""Remove, per module, the part of a stage-2 update that runs AGAINST the stage-1 safety update (no retraining).

For each LoRA module m: S = stage-1 delta (B_s A_s), R = stage-2 delta (B_r A_r), U = R - S (the stage-2 update),
c = <U, S> / ||S||^2 (Frobenius). A negative c means stage 2 partly undid stage 1 along its own direction. The new
delta is R' = R - min(c, 0) * S, written EXACTLY as one LoRA of rank r_s + r_r:
    B' = [B_r, -min(c,0) * B_s],  A' = [A_r ; A_s]       (same lora_alpha / r scale, so alpha doubles with r)
Inner products use the low-rank identity <B1 A1, B2 A2> = sum((B1^T B2) * (A1 A2^T)), so no full matrices are built.

  python scripts/project_adapter_update.py --stage1 joint_all_v3_lr5e-5_e2_s0 --stage2 joint_all_v3_opdtr_s0 \
      --out joint_all_v3_opdtr_proj_s0            # folders under checkpoints/hf/; then download_and_convert_adapters.py
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
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="negative", choices=["negative", "all"],
                    help="negative: remove only the component that opposes stage 1; all: remove any component along it")
    a = ap.parse_args()

    cs, cr = (json.load(open(H / d / "adapter_config.json")) for d in (a.stage1, a.stage2))
    for k in ("lora_alpha", "r", "use_rslora", "use_dora"):
        if cs.get(k) != cr.get(k):
            raise SystemExit(f"adapter configs differ on {k}: {cs.get(k)} vs {cr.get(k)}")
    if cs.get("use_rslora") or cs.get("use_dora") or cs.get("rank_pattern") or cs.get("alpha_pattern"):
        raise SystemExit("only plain LoRA is supported")
    s, r = load_file(H / a.stage1 / "adapter_model.safetensors"), load_file(H / a.stage2 / "adapter_model.safetensors")
    if sorted(s) != sorted(r):
        raise SystemExit("adapter tensors differ between stage 1 and stage 2")

    out, tot = {}, {"SS": 0.0, "US": 0.0, "UU": 0.0, "US_after": 0.0, "removed": 0, "modules": 0}
    for ka in sorted(k for k in s if ".lora_A." in k):
        kb = ka.replace(".lora_A.", ".lora_B.")
        As, Bs, Ar, Br = (t.float() for t in (s[ka], s[kb], r[ka], r[kb]))
        ss = ip(Bs, As, Bs, As)
        rs, rr = ip(Br, Ar, Bs, As), ip(Br, Ar, Br, Ar)
        us, uu = rs - ss, rr - 2 * rs + ss
        c = us / ss if ss > 0 else 0.0
        cu = min(c, 0.0) if a.mode == "negative" else c
        out[ka] = torch.cat([Ar, As], dim=0).to(r[ka].dtype)
        out[kb] = torch.cat([Br, -cu * Bs], dim=1).to(r[kb].dtype)
        tot["SS"] += ss
        tot["US"] += us
        tot["UU"] += uu
        tot["US_after"] += us - cu * ss
        tot["removed"] += cu != 0.0
        tot["modules"] += 1

    dst = H / a.out
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    cfg = dict(cr, r=2 * cr["r"], lora_alpha=2 * cr["lora_alpha"])  # same scale alpha/r
    json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (H / a.stage2 / f).exists():
            shutil.copy(H / a.stage2 / f, dst / f)
    json.dump({"script": "project_adapter_update", **vars(a), **tot}, open(dst / "projection_manifest.json", "w"), indent=2)
    n1 = tot["SS"] ** 0.5
    print(f"{a.stage2} -> {a.out}: {tot['removed']}/{tot['modules']} modules corrected | "
          f"update vs stage 1: cos {tot['US'] / (tot['UU'] ** 0.5 * n1):+.3f}, undone {-tot['US'] / tot['SS'] * 100:.2f}% "
          f"-> after: undone {-tot['US_after'] / tot['SS'] * 100:.2f}% | rank {cfg['r']}, alpha {cfg['lora_alpha']}")


if __name__ == "__main__":
    main()
