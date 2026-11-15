"""
"Merge models" condition: combine single-language DPO adapters into one adapter
(Cohere's "Mix Data or Merge Models?" found language-wise merging can beat
data mixing for safety + general performance).

Fair comparison ("mix vs merge" on IDENTICAL data):
  1. train hi_only, te_only, ja_only adapters (same seed)
  2. merge them here                         -> merged_<...>
  3. build the union condition of the same pairs and train it
       python scripts/make_dpo_condition.py --condition union_hi_te_ja --union hi_only,te_only,ja_only
  4. evaluate both with run_safety_eval / run_capability_eval

  python scripts/merge_adapters.py \
     --adapters checkpoints/dpo/hi_only_s0 checkpoints/dpo/te_only_s0 checkpoints/dpo/ja_only_s0 \
     --method ties --density 0.5 --name merged_hi_te_ja_ties_s0

Methods are PEFT `add_weighted_adapter` combination types. SLERP is not offered
by PEFT; use linear / ties / dare_ties (cat keeps every rank and is exact
"sum of updates"). Output is a normal PEFT adapter directory, so
`scripts/export_adapter.py lora` converts it to GGUF as usual.
"""

import argparse
import os
import shutil
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import write_manifest  # noqa: E402
from src.paths import ROOT_DIR  # noqa: E402

METHODS = {"linear", "cat", "svd", "ties", "ties_svd", "dare_ties", "dare_linear",
           "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"}
NEEDS_DENSITY = {"ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd",
                 "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"}


def build_merge_kwargs(method, adapter_names, weights=None, density=None):
    """Validate and build kwargs for PeftModel.add_weighted_adapter."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    if len(adapter_names) < 2:
        raise ValueError("need at least two adapters to merge")
    weights = weights or [1.0] * len(adapter_names)
    if len(weights) != len(adapter_names):
        raise ValueError("one weight per adapter")
    kw = {"adapters": list(adapter_names), "weights": [float(w) for w in weights],
          "adapter_name": "merged", "combination_type": method}
    if method in NEEDS_DENSITY:
        if density is None or not (0 < density <= 1):
            raise ValueError(f"{method} needs --density in (0, 1]")
        kw["density"] = float(density)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--method", default="ties")
    ap.add_argument("--weights", default=None, help="comma list, default equal")
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(ROOT_DIR / "configs" / "dpo.yaml"))
    ap.add_argument("--out_root", default=str(ROOT_DIR / "checkpoints" / "dpo"))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = [f"a{i}" for i in range(len(args.adapters))]
    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    kw = build_merge_kwargs(args.method, names, weights,
                            args.density if args.method in NEEDS_DENSITY else None)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg.get("base_revision"),
                                                torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")))
    model = PeftModel.from_pretrained(base, args.adapters[0], adapter_name=names[0])
    for n, path in zip(names[1:], args.adapters[1:]):
        model.load_adapter(path, adapter_name=n)
    model.add_weighted_adapter(**kw)
    model.set_adapter("merged")

    out_dir = os.path.join(args.out_root, args.name)
    tmp = tempfile.mkdtemp()
    model.save_pretrained(tmp, selected_adapters=["merged"])
    src = os.path.join(tmp, "merged") if os.path.isdir(os.path.join(tmp, "merged")) else tmp
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists; choose another --name")
    shutil.copytree(src, out_dir)
    shutil.rmtree(tmp)
    AutoTokenizer.from_pretrained(args.adapters[0]).save_pretrained(out_dir)
    write_manifest(out_dir, {"script": "merge_adapters", "sources": args.adapters,
                             "merge_kwargs": kw}, name="merge_manifest.json")
    print(f"merged adapter -> {out_dir}  ({args.method}, {len(args.adapters)} sources)")


if __name__ == "__main__":
    main()
