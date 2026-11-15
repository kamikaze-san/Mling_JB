"""
KL probe (laptop, llama.cpp, all models Q4_K_M): how far apart are the safe model and each candidate OPD teacher?

Student = tiny-aya-global + the stage-1 safety adapter (joint_all_v3). For every probe prompt the student writes its
own greedy reply (as in OPD, where the student samples), then the per-token reverse KL(student || teacher) over the
full vocabulary is measured on that reply, teacher-forced, for each teacher:
  global = the original model (this is also "how far did safety DPO move the model", per prompt type)
  fire   = tiny-aya-fire  (regional: South Asian languages)
  water  = tiny-aya-water (regional: European + Asia-Pacific languages)
Prompt types: translation (FLORES dev, per direction), benign twins (per surface), held-out attacks (first turn,
per surface). A large starting KL means large, noisy OPD gradients on prefixes the teacher would not write, so this
is measured BEFORE paying for any distillation run.

Run from the bundle root:   python kl_probe.py            (about 25-30 min on the RTX 4060)
                            python kl_probe.py --limit 2  (smoke test)
Writes data/04_eval_cache/kl_probe/kl_probe_results.json (aggregates only; --pack includes it).
Prints numbers only - never prompt or reply text.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BASE = MODELS / "tiny-aya-global-q4_k_m.gguf"
STUDENT_ADAPTER = ROOT / "adapters" / "joint_all_v3_lr5e-5_e2_s0" / "adapter.gguf"
TEACHERS = {"global": BASE, "fire": MODELS / "tiny-aya-fire-q4_k_m.gguf", "water": MODELS / "tiny-aya-water-q4_k_m.gguf"}
PROMPTS = ROOT / "inputs" / "kl_prompts.jsonl"  # translation + benign twins, built on the Mac
ATTACKS = ROOT / "inputs" / "eval_attacks_combined.jsonl"
OUT = ROOT / "data" / "04_eval_cache" / "kl_probe" / "kl_probe_results.json"
N_CTX, MAX_NEW, MAX_PROMPT = 1024, 256, 640
ATTACKS_PER_SURFACE = 6


def load(path, lora=None):
    from llama_cpp import Llama
    kw = dict(model_path=str(path), n_ctx=N_CTX, n_gpu_layers=-1, seed=0, logits_all=True, verbose=False)
    if lora:
        kw.update(lora_path=str(lora), lora_scale=1.0)
    return Llama(**kw)


def prompt_ids(llm, messages):
    """Token ids of the chat-formatted prompt, the same way create_chat_completion builds them."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    tok = lambda i: llm.detokenize([i], special=True).decode("utf-8", "ignore")  # noqa: E731
    fmt = Jinja2ChatFormatter(template=llm.metadata["tokenizer.chat_template"], eos_token=tok(llm.token_eos()),
                              bos_token=tok(llm.token_bos()), add_generation_prompt=True)
    res = fmt(messages=messages)
    ids = llm.tokenize(res.prompt.encode("utf-8"), add_bos=not getattr(res, "added_special", False), special=True)
    return ids, res.stop


def log_softmax_rows(llm, seq, start, n):
    """log p(seq[start + k] | seq[:start + k]) distributions over the vocab, k = 0..n-1 -> (n, vocab) float32."""
    llm.reset()
    llm.eval(seq)
    x = np.array(llm.scores[start - 1:start - 1 + n], dtype=np.float32)
    x -= x.max(axis=1, keepdims=True)
    x -= np.log(np.exp(x).sum(axis=1, keepdims=True))
    return x


def load_prompts(limit):
    rows = [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    attacks = [json.loads(l) for l in ATTACKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_surface = defaultdict(list)
    for a in attacks:
        if a.get("source") in ("strata_heldout", "human") and a.get("turns"):
            by_surface[a.get("profile", "")].append(a)
    rng = random.Random(0)
    for s in sorted(by_surface):
        for a in rng.sample(by_surface[s], min(ATTACKS_PER_SURFACE, len(by_surface[s]))):
            rows.append({"messages": [{"role": "user", "content": a["turns"][0]}], "type": "attack", "tag": s})
    if limit:
        seen = defaultdict(int)
        keep = []
        for r in rows:
            if seen[r["type"]] < limit:
                seen[r["type"]] += 1
                keep.append(r)
        rows = keep
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="prompts per type (smoke test)")
    ap.add_argument("--teachers", default="global,fire,water")
    args = ap.parse_args()
    teachers = [t for t in args.teachers.split(",") if t]
    for p in [BASE, STUDENT_ADAPTER, PROMPTS, ATTACKS] + [TEACHERS[t] for t in teachers]:
        if not p.exists():
            raise SystemExit(f"MISSING {p.relative_to(ROOT)}")
    rows = load_prompts(args.limit)
    print(f"[kl] {len(rows)} prompts:", dict(sorted({t: sum(r['type'] == t for r in rows)
                                                       for t in {r['type'] for r in rows}}.items())), flush=True)

    t0 = time.time()
    student = load(BASE, STUDENT_ADAPTER)
    seqs = []  # (row, full token sequence, reply start, reply length)
    skipped = 0
    for i, r in enumerate(rows):
        ids, stop = prompt_ids(student, r["messages"])
        if len(ids) > MAX_PROMPT:
            skipped += 1
            continue
        out = student.create_completion(prompt=ids, max_tokens=MAX_NEW, temperature=0.0, top_k=1, top_p=1.0,
                                        seed=0, stop=stop or None)
        reply = student.tokenize(out["choices"][0]["text"].encode("utf-8"), add_bos=False, special=True)
        reply = reply[:N_CTX - len(ids)]
        if reply:
            seqs.append((r, ids + reply, len(ids), len(reply)))
        if (i + 1) % 25 == 0:
            print(f"[kl] student replies {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[kl] {len(seqs)} replies ({skipped} prompts too long, skipped) in {time.time() - t0:.0f}s", flush=True)

    stats = defaultdict(lambda: {"prompts": 0, "tokens": 0, "kl_sum": 0.0, "per_prompt": [], "max_token_kl": 0.0})
    for t in teachers:
        teacher = load(TEACHERS[t])
        t1 = time.time()
        for r, seq, start, n in seqs:
            ls = log_softmax_rows(student, seq, start, n)
            lt = log_softmax_rows(teacher, seq, start, n)
            kl = (np.exp(ls) * (ls - lt)).sum(axis=1)  # per-token reverse KL(student || teacher), nats
            for key in (f"{t}|{r['type']}|ALL", f"{t}|{r['type']}|{r['tag']}"):
                s = stats[key]
                s["prompts"] += 1
                s["tokens"] += n
                s["kl_sum"] += float(kl.sum())
                s["per_prompt"].append(float(kl.mean()))
                s["max_token_kl"] = max(s["max_token_kl"], float(kl.max()))
        del teacher
        print(f"[kl] teacher {t} done in {time.time() - t1:.0f}s", flush=True)

    results = {}
    for key, s in sorted(stats.items()):
        pp = sorted(s["per_prompt"])
        results[key] = {"prompts": s["prompts"], "tokens": s["tokens"],
                        "kl_per_token": s["kl_sum"] / max(1, s["tokens"]),
                        "per_prompt_median": pp[len(pp) // 2], "per_prompt_p90": pp[min(len(pp) - 1, int(0.9 * len(pp)))],
                        "max_token_kl": s["max_token_kl"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "n_ctx": N_CTX, "max_new": MAX_NEW,
                               "limit": args.limit, "student": "global_q4 + joint_all_v3_lr5e-5_e2_s0",
                               "results": results}, indent=1), encoding="utf-8")
    print(f"\n{'teacher|type|tag':40s} {'prompts':>7s} {'tokens':>7s} {'KL/token':>9s} {'median':>8s} {'p90':>8s}")
    for key, v in results.items():
        print(f"{key:40s} {v['prompts']:7d} {v['tokens']:7d} {v['kl_per_token']:9.4f} "
              f"{v['per_prompt_median']:8.4f} {v['per_prompt_p90']:8.4f}")
    print(f"\nsaved {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
