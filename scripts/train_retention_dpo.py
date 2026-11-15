#!/usr/bin/env python3
"""Stage 2, branch B: capability RETENTION pairs + DPO on top of the safety-tuned LoRA adapter.

The off-policy counterpart of train_opd.py, on the same benign prompt set (build_opd_prompts.py):
  1. greedy answers to every prompt from the ORIGINAL model (adapter switched off) and from the SAFE model
     (adapter on), exactly like the evaluation decodes;
  2. a pair wherever the two differ: chosen = original model's answer, rejected = safe model's drifted answer
     (prompts where they are identical carry no signal and are skipped);
  3. a short DPO run that keeps training THE SAME adapter. With a PEFT model and ref_model=None, TRL (1.17) keeps a
     frozen copy of the STARTING adapter as the reference (the "ref" folder), so DPO's implicit KL anchors to the
     safe stage-1 model, not to the original one: the model is pulled towards the original answers only on these
     benign pairs while staying close to the safe model elsewhere (lower safety risk than OPD, whose target is
     the original model itself).
Benign prompts only (FLORES dev translations + benign training twins). The pairs are written next to the adapter
for inspection on the job machine but never uploaded (they contain FLORES text). Prints counts only.

  python scripts/train_retention_dpo.py --prompts data/03_curated/opd/opd_v1.jsonl \
      --init_adapter checkpoints/hf/joint_all_v3_lr5e-5_e2_s0 --out_dir checkpoints/retention/run
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


opd = _load("train_opd")            # prompt resolution, FLORES fetch, position ids
train_dpo = _load("train_dpo")      # filter_kwargs, HubCheckpointUploader


def greedy(model, tok, prompts, max_new_tokens, batch, eos_ids, device):
    import torch
    outs = []
    model.eval()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            seq = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
        outs += tok.batch_decode(seq[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


def make_pairs(prompts, original, safe):
    pairs, stats = [], Counter()
    for p, o, s in zip(prompts, original, safe):
        key = p["tag"] if p["type"] == "translation" else "benign"
        stats[f"{key}:total"] += 1
        if not o or o == s:
            continue
        stats[f"{key}:pair"] += 1
        pairs.append({"prompt": p["messages"], "chosen": [{"role": "assistant", "content": o}],
                      "rejected": [{"role": "assistant", "content": s}], "type": p["type"], "tag": p["tag"]})
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--init_adapter", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--config", default=str(ROOT / "configs" / "dpo.yaml"))
    ap.add_argument("--flores_local", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_train_epochs", type=float, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_prompts", type=int, default=0, help="dry runs: use only the first N prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--mixed_precision", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--hub_repo", default=None)
    ap.add_argument("--hub_prefix", default=None)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.prompts, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.max_prompts:
        items = items[:args.max_prompts]
    needs_flores = any(i["type"] == "translation" for i in items)
    prompts = opd.resolve_prompts(items, opd.fetch_flores_dev(args.flores_local) if needs_flores else {})

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=getattr(torch, args.dtype)).to(device)
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    gc = model.generation_config
    eos_ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = sorted({e for e in eos_ids + [tok.eos_token_id] if e is not None})

    safe = greedy(model, tok, prompts, args.max_new_tokens, args.gen_batch, eos_ids, device)
    with model.disable_adapter():
        original = greedy(model, tok, prompts, args.max_new_tokens, args.gen_batch, eos_ids, device)
    pairs, stats = make_pairs(prompts, original, safe)
    print(f"[retention] {len(pairs)} pairs from {len(prompts)} prompts: {dict(sorted(stats.items()))}", flush=True)
    if not pairs:
        raise SystemExit("[retention] the safe and original models agree on every prompt: nothing to train")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "retention_pairs.local.jsonl", "w", encoding="utf-8") as f:  # job machine only
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    tok.padding_side = "right"   # DPO training batches
    d = dict(cfg["dpo"])
    d.update(learning_rate=args.learning_rate, num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
             logging_steps=1)
    kw = dict(output_dir=str(out_dir), seed=args.seed, report_to=[], save_strategy="no",
              bf16=args.mixed_precision == "bf16", fp16=args.mixed_precision == "fp16",
              remove_unused_columns=False, **d)
    kw, dropped = train_dpo.filter_kwargs(DPOConfig, kw)
    if dropped:
        print(f"[retention] DPOConfig ignored unsupported keys: {dropped}")
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    trainer_kwargs = dict(model=model, ref_model=None, args=DPOConfig(**kw), train_dataset=ds)
    import inspect
    params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tok
    model.train()
    result = DPOTrainer(**trainer_kwargs).train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    with open(out_dir / "retention_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"script": "train_retention_dpo", "args": vars(args), "n_prompts": len(prompts),
                   "n_pairs": len(pairs), "pair_stats": dict(stats), "dropped_config_keys": dropped,
                   "train_metrics": getattr(result, "metrics", {})}, f, indent=2)
    print(f"[retention] adapter saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
