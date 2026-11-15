---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
language: [en, hi, te, ja]
pipeline_tag: text-generation
tags: [lora, peft, gguf, safety, jailbreak, multilingual, code-mixing, dpo, on-policy-distillation, tiny-aya]
---

# Tiny Aya Global: multilingual, multi-turn jailbreak safety, without the capability loss

LoRA adapters for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B). They make the
model resist multi-turn jailbreaks in English, Hindi, Telugu, Japanese and mixes of these with English (Hindi in
Devanagari or Latin script, Telugu and Japanese) without making it worse at translation or reading comprehension.
Made for the Lexsi Labs AI-safety hackathon (Track 2a/2b).

**The final model** (repo root) is stage 1 (DPO on 1,255 multilingual multi-turn pairs) plus stage 2 (50 steps of
on-policy distillation: Telugu translation is learned from Tiny Aya Fire while a frozen copy of the safe adapter
anchors everything else). Every number below is measured on the deployed model: the official Q4_K_M GGUF with this
LoRA, in llama.cpp.

| | Base Q4_K_M | **Final model** |
|---|---|---|
| Held-out multi-turn jailbreak compliance (n = 382, 8 surfaces) | 73.6% | **12.6%** (Δ −61.0 points, 95% CI [−66.2, −55.8]) |
| Per surface | 67–80% | 8–22% |
| FLORES+ chrF++ (6 directions, Δ vs base) | — | −0.14 to +0.54, all at base level or better |
| Belebele (4 languages, Δ vs base) | — | +0.7 to +1.3 points |
| Harmless look-alike prompts refused | 0/91 | 5/91 |

Compliance is judged by pplx-decider-flash (a multi-question decision model) on any turn; see the write-up for the
judge checks against a human annotator. All results are from a single seed.

## Usage
**llama.cpp (how it was evaluated):**
```bash
llama-cli -m tiny-aya-global-q4_k_m.gguf --lora adapter.gguf      # official GGUF from CohereLabs/tiny-aya-global-GGUF
```
**PEFT:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "kamikaze-san/tiny-aya-multilingual-safety")   # final model (root)
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
```
For another adapter, add `subfolder="stage1/joint_all_v3_lr5e-5_e2"` (or any folder below).

## Single-file version (merged, less safe)
`merged/tiny-aya-global-safety-merged-q4_k_m.gguf` (2.0 GB) is the final model merged into the bf16 weights and
re-quantized to Q4_K_M, for apps that load one GGUF file. Rounding to 4 bits partly erases the small safety update, so
it keeps less of the safety:

| Held-out multi-turn jailbreak compliance (n = 382) | |
|---|---|
| Base, re-quantized the same way | 71.7% |
| **Base GGUF + `adapter.gguf` (recommended)** | **12.6%** |
| Merged single file | 19.4% (+6.8 points vs the LoRA, most of it on Hindi) |

Translation and reading comprehension stay at base level in both. **For the best safety, use the base GGUF with
`adapter.gguf`.** The runtime LoRA costs about 6% generation speed.

## Contents
Each folder holds `adapter_model.safetensors` (PEFT), `adapter_config.json` and `adapter.gguf` (GGUF LoRA, f16).

| Folder | What it is |
|---|---|
| `/` (root) | **Final model:** stage 1 + multi-teacher OPD with a safety anchor, step 50 |
| `stage1/joint_all_v3_lr5e-5_e2` | Stage 1 only: DPO, LoRA r16, 1,255 pairs (12.8% compliance; costs ~1.1 chrF++ on te→en) |
| `stage2/joint_all_v3_mtopd_step72` | The same multi-teacher run at its last step (adds nothing over step 50) |
| `stage2/joint_all_v3_ret` | Retention DPO (stays safe, repairs little) |
| `stage2/joint_all_v3_opd_step50`, `_step100` | On-policy distillation from the original model: repairs translation but compliance rises to 33–45% |
| `stage2/joint_all_v3_opdtr_step50`, `_step100` | The same on translation prompts only (+4.2 to +4.7 points) |
| `stage2/joint_all_v3_opdtr_projected` | Translation-only OPD with the component that reverses the safety update projected out, no training (rank 32) |
| `stage2/joint_all_v3_projected_plus_0.1_safety`, `_0.2_` | The projected adapter pushed further along the stage-1 safety update (rank 48) |
| `ablations/single_surface_100_pairs/*` | 100 pairs on one surface (or all code-mixed pooled) |
| `ablations/leave_one_out/*` | Stage-1 data without Telugu, Japanese or code-mixed pairs |
| `ablations/earlier_data_p100/*` | Earlier 912-pair data version and its checkpoints; 56/100-pair joint runs |
| `ablations/joint8/*`, `ablations/pilot_lr_sweep/*` | First small runs and learning-rate pilot |

The run names match the tables in the write-up.

## Intended use and limitations
These are research artifacts for multilingual safety. They reduce jailbreak compliance but do not eliminate it:
roughly 1 in 8 held-out attack conversations still succeeds, and the hardened model was not re-attacked adaptively.
The attacks come from one attacker LLM, so the mixed-language attacks were weaker than a fluent attacker's. The
adapters follow the base model's **CC-BY-NC-4.0** licence (non-commercial).
