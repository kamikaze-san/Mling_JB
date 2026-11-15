#!/usr/bin/env python3
"""Convert an official pplx-decider release (bf16 safetensors + readout.safetensors) to the MLX 4-bit text-only format
our flash runner reads (the same recipe as pplx-flash-mlx/4bit-g64: affine 4-bit, group 64, embeddings 4-bit, readout
kept in float). Uses ~/Downloads/decision-models/pplx-flash-mlx/runner (mlx_decider.load_bf16 / quantize / save_mlx).

  ~/Downloads/MARCH-Orz/.venv-pplx/bin/python scripts/convert_decider_mlx.py \
      ~/Downloads/decision-models/pplx-flash-v1.3 ~/Downloads/decision-models/pplx-flash-v1.3-mlx/4bit-g64
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Downloads/decision-models/pplx-flash-mlx/runner"))
import mlx.core as mx  # noqa: E402
import mlx_decider as D  # noqa: E402

import glob  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from mlx_lm.models import qwen3_5  # noqa: E402


def load_bf16(ckpt):
    """mlx_decider.load_bf16, but releases without an lm_head (the 27B: untied embeddings, the decision readout
    replaces the LM head) are built with tied embeddings so no unused lm_head parameter is expected. The readout only
    reads hidden states, so this does not change any decision."""
    ckpt = Path(ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    if not any(k.endswith("lm_head.weight") for k in idx):
        cfg.setdefault("text_config", {})["tie_word_embeddings"] = True
        cfg["tie_word_embeddings"] = True
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(cfg))
    w = {}
    for f in sorted(glob.glob(str(ckpt / "model-*.safetensors"))):
        for k, v in mx.load(f).items():
            if k.startswith("language_model."):
                w["model.language_model." + k[len("language_model."):]] = v
    w = model.sanitize(w)
    model.load_weights(list(w.items()), strict=True)
    del w
    model.eval()
    return model, cfg


src, dst = (os.path.expanduser(p) for p in sys.argv[1:3])
mx.set_cache_limit(2 << 30)
t = time.time()
model, cfg = load_bf16(src)
D.quantize(model, 4, 64, 4)
mx.eval(model.parameters())
D.save_mlx(model, cfg, dst, src, {"group_size": 64, "bits": 4, "mode": "affine"})
print(f"converted {os.path.basename(src)} -> {dst}: {D.param_bytes(model) / 2**30:.2f} GiB, peak "
      f"{mx.get_peak_memory() / 2**30:.1f} GiB, {time.time() - t:.0f}s")
