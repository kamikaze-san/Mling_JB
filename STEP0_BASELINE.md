# Step 0 — baseline runbook (base tiny-aya-global Q4_K_M, no adapter)

Everything is set up. Only the GPU runs remain. All commands are run from `Multilingual_JB/`
with `.venv/bin/python`. Outputs go to `data/04_eval_cache/<run_name>/`, and stdout shows
aggregate tables only.

## What's ready

| Item | Where | Status |
|---|---|---|
| Target GGUF (official Cohere Q4_K_M, `cohere2` arch) | `~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf` (reused, not re-downloaded) | ✅ loads in llama-cpp-python 0.3.36, built with Metal; CPU smoke test passed |
| Eval deps | `.venv`: torch 2.15 (MPS), transformers 5.22, sacrebleu 2.6, llama-cpp-python 0.3.36 | ✅ |
| Judge | `Qwen/Qwen3Guard-Gen-4B` (8.8 GB) in the HF cache | ✅ downloaded; loads on MPS (bf16) on first use |
| Belebele | `data/benchmarks/belebele_eval.jsonl`, 150 per language (en/hi/te/ja) | ✅ |
| FLORES+ devtest/dev | `data/benchmarks/flores_devtest.jsonl` (200 aligned, eval only) and `flores_dev.jsonl` (997, 157 flagged in_belebele) | ✅ terms accepted (no rehosting: `data/` is gitignored, never upload it) |
| Held-out attacks: Aya Red-teaming | `data/03_curated/aya_redteaming_heldout.jsonl`: 1,902 single-turn prompts (en 987, hi 915; global/local labelled) | ✅ |
| Benign over-refusal (ORR) set | `data/03_curated/benign_baseline.jsonl`: 400 prompts, 100 each en/hi/te/ja (Aya eval suite) | ✅ |
| Held-out attacks: human | from `MTMM/scripts/ingest_human_traces.py` (after goals) | ⏳ |
| Held-out attacks: STRATA | from the STRATA test phase on held-out goals | ⏳ |

## Commands

```bash
GGUF=~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf

# 1. capability baseline (FLORES chrF++ en<->hi/te/ja + Belebele accuracy)
.venv/bin/python scripts/run_capability_eval.py --run_name base_q4 --gguf $GGUF

# 2. optional: smaller stratified Aya held-out (keeps the global/local mix) for a faster first pass
.venv/bin/python scripts/import_aya_redteaming.py --per_lang 250 --out data/03_curated/aya_redteaming_heldout_500.jsonl

# 3. safety baseline: ASR on Aya Red-teaming + ORR on the benign set, sliced by global/local and language
.venv/bin/python scripts/run_safety_eval.py --run_name base_q4_aya --gguf $GGUF \
    --attacks data/03_curated/aya_redteaming_heldout_500.jsonl \
    --benign data/03_curated/benign_baseline.jsonl \
    --slice_by global_or_local language

# 4. later, same command per held-out source (human, STRATA) with its own --run_name
```

Every later checkpoint runs the identical commands with `--lora <adapter.gguf>`, a new `--run_name`,
and `--compare base_q4…`. That gives paired bootstrap diffs for ASR/ORR and capability
preservation (BCP).

## Notes

- **GPU sharing.** Steps 1 and 3 use the GPU: tiny-aya on Metal, plus Qwen3Guard on MPS (about
  8–9 GB). Run them when no other GPU job is active (for example the Gemma `llama-server`).
  Otherwise both slow down, and memory pressure can make it worse.
- **Generation cache.** Generations are cached by model fingerprint plus seed plus messages. Re-runs
  and `--compare` re-use them, and only judging re-runs.
- **Linguistic surfaces (8 total).** The active evaluation and attack surfaces are `en, hi, hi_en_mix, hinglish, te, te_en_mix, ja, ja_en_mix`.
  Note that `te`, `hi`, and `ja` designate single-script surfaces rather than strictly monolingual text (unavoidable loanwords in native script are retained).
  Excluded surfaces (such as Romanized Telugu / Tenglish, Romaji Japanese, and `te_hi_mix`) are excluded due to target comprehension collapse or
  attacker generation limitations; detailed rationales are recorded at the top of `MTMM/configs/linguistic_profiles.yaml`.
- **Scheduler formula.** STRATA schedules surface rotation as `(goal_index + iteration) mod 8`, so sweeps must run in iterations of multiples of 8
  for full goal-matched coverage across all 8 surfaces.
- **What ORR here measures.** This benign set measures general over-refusal only. Borderline
  prompts ("sounds unsafe, is fine") and the mixed surfaces (`hi_en_mix`, `hinglish`, `te_en_mix`, `ja_en_mix`) come
  from STRATA benign twins via `build_pair_pool.py`.
- **Aya Red-teaming coverage.** It has en and hi only, and all Hindi prompts are in Devanagari (no
  Hinglish). Telugu, Japanese and mixed-surface attacks come from the human and STRATA held-out
  sets.
- **Judge trust.** Before trusting the judge per language, follow `EVAL_PROTOCOL.md` §2: export
  a blind CSV, have two annotators label it, then `score` gives per-language kappa and
  Krippendorff's α, with human-human agreement as the ceiling.
- **Rules for later checkpoints** (per-slice reporting, checkpoint gate, quantization check):
  `EVAL_PROTOCOL.md`.

## DPO dry run

The training → export → evaluation-loading pipeline is verified end-to-end on synthetic benign pairs before real data training.

### Exact commands

```bash
# 1. Train 3-step DPO LoRA adapter on harmless dryrun pairs (MPS)
.venv/bin/python scripts/train_dpo.py --condition dryrun --seed 0 \
    --max_steps 3 --gradient_accumulation_steps 2 --logging_steps 1

# 2. Export PEFT adapter to GGUF LoRA (F16) using convert_lora_to_gguf.py
HF_BASE=$(ls -d ~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/* | head -n 1)
.venv/bin/python scripts/export_adapter.py lora \
    --adapter checkpoints/dpo/dryrun_s0 \
    --base_dir "$HF_BASE" \
    --llama_cpp ~/llama.cpp

# 3. Verify loading and single-turn generation with GGUFRunner on official Q4_K_M base
.venv/bin/python -c '
import os
from src.harness.gguf_runner import GGUFRunner

gguf = os.path.expanduser("~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf")
lora = os.path.abspath("checkpoints/dpo/dryrun_s0/adapter.gguf")
runner = GGUFRunner(model_path=gguf, lora_path=lora, max_tokens=32)
resp = runner.generate("Hello, how are you?")
print("Response generated successfully (length:", len(resp), "chars)")
'
```

### Dry run verification metrics
- **Loss**: finite logged loss: step 1: `0.7036`, step 2: `0.6956`, step 3: `0.7111`, final: `0.7035`.
- **Peak memory**: RSS `2688.1 MB`, MPS allocated `6734.2 MB`.
- **GGUF LoRA artifact**: `checkpoints/dpo/dryrun_s0/adapter.gguf` (58 MB, 504 tensors exported).
- **Inference exit code**: `0` (loaded via llama-cpp-python Metal runner and generated 83 characters).

