#!/usr/bin/env python3
"""Stage 2: on-policy distillation (OPD) of the base model's capability back into the safety-tuned LoRA adapter.

Stage 1 (train_dpo.py) made the adapter safe. Stage 2 keeps training THE SAME adapter so that, on benign prompts,
the safe model behaves like the original model again:

  for each step:
    1. the student (base + safety adapter, adapter ON) samples completions for a batch of benign prompts
       (translation on FLORES dev + borderline benign training twins; see build_opd_prompts.py);
    2. student logits (adapter ON, with gradients) and teacher logits (the same weights with the adapter
       switched OFF = the original model, no gradients) are computed on those sampled sequences;
    3. loss = exact full-vocabulary REVERSE KL, KL(student || teacher), averaged over the generated tokens
       (up to and including the first end-of-turn token).

Per-token reverse KL on the student's own samples is the on-policy distillation objective of GKD (Agarwal et
al., 2024; lambda = 1, beta = 1): the sampled prefixes are treated as fixed (no policy-gradient term through
the sampling), which is what makes it cheap and stable. Mode-seeking: the student only has to put its mass where
the teacher does on the student's own outputs. One model copy in memory (the teacher is the adapter switched off).
Prompts are benign only; safety is checked afterwards with the full evaluation.

  python scripts/train_opd.py --prompts data/03_curated/opd/opd_v1.jsonl \
      --init_adapter checkpoints/hf/joint_all_v3_lr5e-5_e2_s0 --out_dir checkpoints/opd/run --steps 120
Prints numbers only (KL, lengths, end-of-turn rate), never prompt or completion text.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.capability import translation_prompt  # noqa: E402

FLORES_LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu", "ja": "jpn_Jpan"}


# --------------------------------------------------------------------------- prompts
def fetch_flores_dev(local_path=None):
    """{flores_id: {"en":..., "hi":..., "te":..., "ja":...}} from a local flores_dev.jsonl, or from the official
    gated FLORES+ dataset (HF_TOKEN must have accepted its terms). Never written anywhere."""
    if local_path and os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            return {r["id"]: r for r in map(json.loads, f)}
    from datasets import load_dataset
    by_id = {}
    for short, cfg in FLORES_LANGS.items():
        for row in load_dataset("openlanguagedata/flores_plus", cfg, split="dev"):
            by_id.setdefault(row["id"], {})[short] = row["text"]
    return by_id


def resolve_prompts(items, flores):
    """Prompt-set items -> [{"messages": [...], "type": "translation"|"benign"|"anchor", "tag": direction|surface}].
    "anchor" items (safety prompts from the stage-1 TRAINING pairs) are distilled from a frozen copy of the stage-1
    adapter, so they hold the safe behaviour in place instead of pulling it toward the original model."""
    out, missing = [], 0
    for it in items:
        if it["type"] == "translation":
            row = flores.get(it["flores_id"])
            if not row or it["src"] not in row:
                missing += 1
                continue
            msgs = [{"role": "user", "content": translation_prompt(row[it["src"]], it["src"], it["tgt"])}]
            out.append({"messages": msgs, "type": "translation", "tag": f"{it['src']}-{it['tgt']}"})
        else:
            p = it["prompt"]
            msgs = [{"role": "user", "content": p}] if isinstance(p, str) else list(p)
            out.append({"messages": msgs, "type": "anchor" if it["type"] == "anchor" else "benign",
                        "tag": it.get("profile", "")})
    if missing:
        raise SystemExit(f"[opd] {missing} FLORES ids not found in the dev split")
    return out


# --------------------------------------------------------------------------- loss pieces (tested on CPU)
def completion_mask(gen_tokens, eos_ids, pad_id):
    """gen_tokens [B, G] (generated part only). 1 for every token up to and including the first end token;
    0 after it (padding). A row without an end token (hit max_new_tokens) is kept whole."""
    import torch
    is_end = torch.zeros_like(gen_tokens, dtype=torch.bool)
    for e in eos_ids:
        is_end |= gen_tokens == e
    # positions strictly after the first end token: cumulative count of ends BEFORE this position > 0
    ends_before = torch.cumsum(is_end.long(), dim=1) - is_end.long()
    return (ends_before == 0).float()


def token_kl(student_logits, teacher_logits):
    """Exact per-token KL(student || teacher) over the full vocabulary, in fp32: [B, G, V] x2 -> [B, G]."""
    import torch
    ls = torch.log_softmax(student_logits.float(), dim=-1)
    lt = torch.log_softmax(teacher_logits.float(), dim=-1)
    return (ls.exp() * (ls - lt)).sum(-1)


def reverse_kl(student_logits, teacher_logits, mask):
    """logits [B, G, V], mask [B, G] -> (sum of masked KL, per-row KL sums)."""
    kl = token_kl(student_logits, teacher_logits) * mask
    return kl.sum(), kl.sum(-1)


def transform_kl(kl, how):
    """Per-token loss from per-token KL. 'log' = log(1 + KL): near-linear for small KL, compresses outlier tokens
    (the regulation proposed for large teacher-student gaps in 'Demystifying On-Policy Distillation', 2026)."""
    import torch
    return torch.log1p(kl.clamp(min=0)) if how == "log" else kl


def position_ids_from_mask(attn):
    """Left-padded batches: positions must count real tokens only (as generate() does)."""
    return (attn.long().cumsum(-1) - 1).clamp(min=0)


def gen_logits(model, seq, attn, n_gen):
    """Logits that predict the last n_gen tokens of seq: [B, n_gen, V]."""
    kw = dict(input_ids=seq, attention_mask=attn, position_ids=position_ids_from_mask(attn), use_cache=False)
    try:
        logits = model(**kw, logits_to_keep=n_gen + 1).logits
    except TypeError:                              # older transformers without logits_to_keep
        logits = model(**kw).logits[:, -(n_gen + 1):]
    return logits[:, :n_gen]


# --------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--init_adapter", required=True, help="dir with the safety-tuned PEFT adapter (stage 1)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--flores_local", default=None, help="local flores_dev.jsonl (else fetched from the hub)")
    ap.add_argument("--steps", type=int, default=120, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=8, help="prompts per optimizer step (sampled together)")
    ap.add_argument("--micro", type=int, default=2, help="sequences per forward/backward (memory)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0, help="1.0 = sample from the student's own distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_every", type=int, default=0, help="also save (and upload) the adapter every N steps")
    ap.add_argument("--hub_repo", default=None)
    ap.add_argument("--hub_prefix", default=None)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--tags", default=None,
                    help="keep only translation prompts with these directions (e.g. en-hi,hi-en,en-te,te-en)")
    ap.add_argument("--teacher_model", default=None,
                    help="separate teacher (same tokenizer, e.g. CohereLabs/tiny-aya-fire); default: the original model "
                         "(adapter switched off)")
    ap.add_argument("--teacher_tags", default=None,
                    help="route only prompts with these tags to --teacher_model; the rest use the original model")
    ap.add_argument("--teacher_alpha", type=float, default=1.0,
                    help="target = log_softmax((1-a)*partner + a*teacher) for routed prompts; a<1 shrinks the "
                         "starting KL when the teacher is far from the student")
    ap.add_argument("--mix_with", default="anchor", choices=["anchor", "original"],
                    help="partner for --teacher_alpha < 1: the frozen stage-1 copy (default; does not pull toward "
                         "the unsafe original) or the original model")
    ap.add_argument("--kl_transform", default="none", choices=["none", "log"],
                    help="per-token loss: KL, or log(1+KL) to damp outlier tokens")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    with open(args.prompts, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    needs_flores = any(i["type"] == "translation" for i in items)
    prompts = resolve_prompts(items, fetch_flores_dev(args.flores_local) if needs_flores else {})
    if args.tags:
        keep = set(args.tags.split(","))
        prompts = [p for p in prompts if p["type"] != "translation" or p["tag"] in keep]  # filters translation only
        if not prompts:
            raise SystemExit(f"[opd] no prompts left after --tags {args.tags}")
    print(f"[opd] {len(prompts)} prompts: "
          f"{sum(p['type'] == 'translation' for p in prompts)} translation, "
          f"{sum(p['type'] == 'benign' for p in prompts)} benign, "
          f"{sum(p['type'] == 'anchor' for p in prompts)} anchor", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[opd] trainable params {sum(p.numel() for p in trainable):,} (LoRA, initialised from stage 1)", flush=True)
    if not trainable:
        raise SystemExit("[opd] no trainable parameters: the adapter did not load as trainable")
    # frozen copy of the stage-1 adapter, as a second adapter on the same base (about 60 MB, no second model)
    use_anchor = any(p["type"] == "anchor" for p in prompts) or (
        args.teacher_model and args.teacher_alpha < 1.0 and args.mix_with == "anchor")
    anchor_params = []
    if use_anchor:
        model.load_adapter(args.init_adapter, adapter_name="anchor", is_trainable=False)
        anchor_params = [p for n, p in model.named_parameters() if ".anchor." in n]
        for p in anchor_params:
            p.requires_grad_(False)
        model.set_adapter("default")
        for p in trainable:
            p.requires_grad_(True)
        print(f"[opd] anchor = frozen copy of {args.init_adapter} ({sum(p.numel() for p in anchor_params):,} params)",
              flush=True)

    def anchor_logits(sq, at):
        model.set_adapter("anchor")
        try:
            return gen_logits(model, sq, at, G)
        finally:
            model.set_adapter("default")
            for p in trainable:          # PEFT's set_adapter toggles requires_grad; restore exactly
                p.requires_grad_(True)
            for p in anchor_params:
                p.requires_grad_(False)

    teacher, routed = None, set()
    if args.teacher_model:
        ttok = AutoTokenizer.from_pretrained(args.teacher_model)
        if ttok.get_vocab() != tok.get_vocab():
            raise SystemExit(f"[opd] {args.teacher_model} has a different tokenizer: full-vocab KL is undefined")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        routed = set(args.teacher_tags.split(",")) if args.teacher_tags else {
            p["tag"] for p in prompts if p["type"] == "translation"}
        print(f"[opd] teacher {args.teacher_model} (alpha {args.teacher_alpha}) for tags {sorted(routed)}; "
              f"{sum(p['tag'] in routed for p in prompts)} of {len(prompts)} prompts routed", flush=True)

    def route(p):
        if p["type"] == "anchor":
            return "anchor"
        return "teacher" if teacher is not None and p["type"] == "translation" and p["tag"] in routed else "original"

    def original_logits(sq, at):
        with model.disable_adapter():
            return gen_logits(model, sq, at, G)

    def teacher_logits(rows, sq, at):
        """Target logits per row: original model | frozen stage-1 anchor | routed teacher (optionally mixed)."""
        with torch.no_grad():
            routes = [route(p) for p in rows]
            if len(set(routes)) == 1 and routes[0] != "teacher":
                return (anchor_logits if routes[0] == "anchor" else original_logits)(sq, at)
            out = []
            for k, rt in enumerate(routes):  # row by row when a micro-batch mixes routes
                r = slice(k, k + 1)
                if rt != "teacher":
                    out.append((anchor_logits if rt == "anchor" else original_logits)(sq[r], at[r]).float())
                    continue
                lt = torch.log_softmax(gen_logits(teacher, sq[r], at[r], G).float(), dim=-1)
                if args.teacher_alpha < 1.0:
                    partner = anchor_logits if args.mix_with == "anchor" else original_logits
                    lo = torch.log_softmax(partner(sq[r], at[r]).float(), dim=-1)
                    lt = (1 - args.teacher_alpha) * lo + args.teacher_alpha * lt
                out.append(lt)
            return torch.cat(out, dim=0)

    gc = model.generation_config
    eos_ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = sorted({e for e in eos_ids + [tok.eos_token_id] if e is not None})
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    order = list(range(len(prompts)))
    random.shuffle(order)
    cursor, history = 0, []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(path):
        model.save_pretrained(str(path), selected_adapters=["default"])  # never the frozen anchor copy
        tok.save_pretrained(str(path))
        if args.hub_repo and args.hub_prefix and path != out_dir:
            try:
                from huggingface_hub import HfApi
                HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
                    repo_id=args.hub_repo, folder_path=str(path), repo_type="model",
                    path_in_repo=f"{args.hub_prefix}/{path.name}")
                print(f"[opd] uploaded {path.name}", flush=True)
            except Exception as e:  # noqa: BLE001  a lost checkpoint copy must not kill training
                print(f"[opd] WARN upload of {path.name} failed: {type(e).__name__}", flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(args.batch):
            if cursor == len(order):
                random.shuffle(order)
                cursor = 0
            batch.append(prompts[order[cursor]])
            cursor += 1
        texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        # 1. on-policy sampling from the student
        model.eval()
        with torch.no_grad():
            seq = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                                 eos_token_id=eos_ids)
        P = enc["input_ids"].shape[1]
        gen = seq[:, P:]
        G = gen.shape[1]
        if G == 0:
            print(f"[opd] step {step}: nothing generated, skipped", flush=True)
            continue
        cmask = completion_mask(gen, eos_ids, tok.pad_token_id).to(device)
        attn = torch.cat([enc["attention_mask"], cmask.long()], dim=1)
        n_tok = cmask.sum()

        # 2-3. reverse KL on the sampled tokens, micro-batched; gradients accumulate over the batch
        opt.zero_grad(set_to_none=True)
        kl_total, per_row = 0.0, []
        for i in range(0, len(batch), args.micro):
            sl = slice(i, i + args.micro)
            t_logits = teacher_logits(batch[sl], seq[sl], attn[sl])
            model.train()
            s_logits = gen_logits(model, seq[sl], attn[sl], G)
            tkl = token_kl(s_logits, t_logits) * cmask[sl]
            (transform_kl(tkl, args.kl_transform).sum() / n_tok).backward()
            kl_sum, rows = tkl.sum(), tkl.sum(-1)          # raw KL for logging
            kl_total += kl_sum.item()
            per_row += rows.detach().cpu().tolist()
            del s_logits, t_logits
            model.eval()
        if not math.isfinite(kl_total):
            raise SystemExit(f"[opd] non-finite loss at step {step}; stopping")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        lens = cmask.sum(-1).cpu().tolist()
        ended = [1.0 if l < G or bool(any(int(gen[r, int(l) - 1]) == e for e in eos_ids)) else 0.0
                 for r, l in enumerate(lens)]
        by_type = {}
        for p, k, l in zip(batch, per_row, lens):
            agg = by_type.setdefault(p["type"] + (":teacher" if route(p) == "teacher" else ""), [0.0, 0.0])
            agg[0] += k
            agg[1] += l
        rec = {"step": step, "kl_per_token": kl_total / max(1.0, n_tok.item()),
               "kl_by_type": {t: a / max(1.0, n) for t, (a, n) in by_type.items()},
               "mean_len": sum(lens) / len(lens), "ended_rate": sum(ended) / len(ended),
               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
        history.append(rec)
        kt = " ".join(f"{t}={v:.4f}" for t, v in rec["kl_by_type"].items())
        print(f"[opd] step {step}/{args.steps} kl/token {rec['kl_per_token']:.4f} ({kt}) "
              f"len {rec['mean_len']:.0f} ended {rec['ended_rate']:.2f} {rec['elapsed_s']}s", flush=True)
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save(out_dir / f"checkpoint-{step}")

    save(out_dir)
    with open(out_dir / "opd_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"script": "train_opd", "args": vars(args), "n_prompts": len(prompts), "eos_ids": eos_ids,
                   "history": history}, f, indent=2)
    print(f"[opd] adapter saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
