"""
Train one DPO LoRA adapter for one condition and seed.

  python scripts/train_dpo.py --condition joint --seed 0
  python scripts/train_dpo.py --condition hi_only --seed 1 --config configs/dpo.yaml

Reads data/03_curated/dpo/<condition>.jsonl (TRL conversational preference format)
and writes a PEFT adapter to checkpoints/dpo/<condition>_s<seed>/.
The reference model is the base with the adapter disabled (no second copy).
"""

import argparse
import inspect
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, write_manifest  # noqa: E402
from src.paths import CURATED_DIR, ROOT_DIR  # noqa: E402


def filter_kwargs(cls, kwargs):
    """Drop config keys this TRL/transformers version does not accept (API drift guard)."""
    params = inspect.signature(cls.__init__).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    return accepted, dropped


def _make_uploader_base():
    try:
        from transformers import TrainerCallback
        return TrainerCallback
    except Exception:  # transformers missing (tests): a plain base keeps the module importable
        return object


class HubCheckpointUploader(_make_uploader_base()):
    """Upload each saved checkpoint to a private HF model repo right away. A failed upload only warns:
    losing a checkpoint copy must not kill the training run."""

    def __init__(self, repo_id, prefix):
        self.repo_id, self.prefix = repo_id, prefix

    def on_save(self, args, state, control, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            return
        for attempt in range(3):
            try:
                from huggingface_hub import HfApi
                HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
                    repo_id=self.repo_id, folder_path=ckpt, repo_type="model",
                    path_in_repo=f"{self.prefix}/checkpoint-{state.global_step}")
                print(f"[train_dpo] uploaded checkpoint-{state.global_step} -> {self.prefix}/", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                print(f"[train_dpo] WARN checkpoint upload attempt {attempt + 1} failed: {type(e).__name__}",
                      flush=True)
        print(f"[train_dpo] WARN checkpoint-{state.global_step} NOT uploaded (training continues)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=str(ROOT_DIR / "configs" / "dpo.yaml"))
    ap.add_argument("--out_root", default=str(ROOT_DIR / "checkpoints" / "dpo"))
    ap.add_argument("--max_steps", type=int, default=None, help="override max training steps")
    ap.add_argument("--num_train_epochs", type=int, default=None, help="override num_train_epochs")
    ap.add_argument("--learning_rate", type=float, default=None, help="override learning_rate")
    ap.add_argument("--save_half", action="store_true",
                    help="also save the adapter at 50 %% of training (to evaluate an earlier checkpoint)")
    ap.add_argument("--save_every_frac", type=float, default=None,
                    help="save a RESUMABLE checkpoint (adapter + optimizer + scheduler + RNG) every this fraction "
                         "of training, e.g. 0.25; overrides --save_half")
    ap.add_argument("--hub_checkpoint_repo", default=None,
                    help="upload each checkpoint to this (private) model repo as soon as it is saved, so it "
                         "survives a killed container")
    ap.add_argument("--hub_checkpoint_prefix", default=None, help="folder in that repo: <prefix>/checkpoint-<step>")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest checkpoint-* in the output dir (the job script downloads it)")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None, help="override grad accum steps")
    ap.add_argument("--logging_steps", type=int, default=None, help="override logging steps")
    ap.add_argument("--mixed_precision", choices=["none", "bf16", "fp16"], default=None, help="override mixed precision")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import resource
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    rows = read_jsonl(CURATED_DIR / "dpo" / f"{args.condition}.jsonl")
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
                            for r in rows])

    dtype = getattr(torch, cfg.get("torch_dtype", "bfloat16"))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg.get("base_revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg.get("base_revision"),
                                                 torch_dtype=dtype)

    lc = cfg["lora"]
    peft_cfg = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                          target_modules=lc["target_modules"], task_type="CAUSAL_LM")

    out_dir = os.path.join(args.out_root, f"{args.condition}_s{args.seed}")
    d = dict(cfg["dpo"])
    if args.max_steps is not None:
        d["max_steps"] = args.max_steps
    if args.num_train_epochs is not None:
        d["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        d["learning_rate"] = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        d["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.logging_steps is not None:
        d["logging_steps"] = args.logging_steps

    mp = args.mixed_precision or cfg.get("mixed_precision", "none")
    save_kwargs = dict(save_strategy="no")
    if args.save_half or args.save_every_frac:
        import math
        per_step = d.get("per_device_train_batch_size", 1) * d.get("gradient_accumulation_steps", 1)
        total = d["max_steps"] if d.get("max_steps", -1) and d.get("max_steps", -1) > 0 else \
            math.ceil(len(ds) / per_step) * d.get("num_train_epochs", 1)
        if args.save_every_frac:
            save_kwargs = dict(save_strategy="steps", save_steps=max(1, round(total * args.save_every_frac)),
                               save_only_model=False)  # full state, so training can resume
        else:
            save_kwargs = dict(save_strategy="steps", save_steps=max(1, total // 2), save_only_model=True)
        print(f"[train_dpo] estimated {total} optimizer steps; checkpoint every {save_kwargs['save_steps']} "
              f"({'resumable' if args.save_every_frac else 'model only'})")
    dpo_kwargs = dict(output_dir=out_dir, seed=args.seed, report_to=[], **save_kwargs,
                      bf16=(mp == "bf16"), fp16=(mp == "fp16"), remove_unused_columns=False, **d)
    dpo_kwargs, dropped = filter_kwargs(DPOConfig, dpo_kwargs)
    if dropped:
        print(f"[train_dpo] DPOConfig ignored unsupported keys: {dropped}")
    targs = DPOConfig(**dpo_kwargs)

    trainer_kwargs = dict(model=model, ref_model=None, args=targs, train_dataset=ds, peft_config=peft_cfg)
    params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tok
    if args.hub_checkpoint_repo:
        trainer_kwargs["callbacks"] = [HubCheckpointUploader(args.hub_checkpoint_repo,
                                                             args.hub_checkpoint_prefix or os.path.basename(out_dir))]
    trainer = DPOTrainer(**trainer_kwargs)
    resume = None
    if args.resume:
        import glob
        ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        if not ckpts:
            raise SystemExit(f"[train_dpo] --resume: no checkpoint-* in {out_dir}")
        resume = ckpts[-1]
        print(f"[train_dpo] resuming from {os.path.basename(resume)}")
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # Peak memory measurement
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = (rss_raw / (1024 * 1024)) if sys.platform == "darwin" else (rss_raw / 1024)
    mps_mb = None
    if torch.backends.mps.is_available():
        try:
            mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
    cuda_mb = None
    if torch.cuda.is_available():
        try:
            cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass

    mem_report = {"peak_rss_mb": round(peak_rss_mb, 2)}
    if mps_mb is not None:
        mem_report["mps_allocated_mb"] = round(mps_mb, 2)
    if cuda_mb is not None:
        mem_report["cuda_max_allocated_mb"] = round(cuda_mb, 2)
    extra_dev = f", MPS allocated {mps_mb:.1f} MB" if mps_mb else (f", CUDA peak {cuda_mb:.1f} MB" if cuda_mb else "")
    print(f"[train_dpo] Peak memory: RSS {peak_rss_mb:.1f} MB" + extra_dev)

    write_manifest(out_dir, {"script": "train_dpo", "condition": args.condition, "seed": args.seed,
                             "config": cfg, "n_pairs": len(rows), "dropped_config_keys": dropped,
                             "train_metrics": getattr(result, "metrics", {}),
                             "memory": mem_report},
                   name="train_manifest.json")
    print(f"adapter saved -> {out_dir}  ({len(rows)} pairs, seed {args.seed})")


if __name__ == "__main__":
    main()
