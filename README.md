# Multilingual_JB: multilingual and code-mixed safety repair for tiny-aya-global (Lexsi Hackathon, Track 2a/2b)

**Question.** Does the safety of a small multilingual model hold across English, Hindi, Telugu, Japanese and
code-mixed prompts (Hinglish, Hindi-English and Telugu-English mixes, Japanese-English mix), under multi-turn
attacks found by an automated red-teamer, and can we close the gaps without hurting translation, reading
comprehension or over-refusal? Stage 1 is DPO with a LoRA adapter. Stage 2 repairs the translation it costs, using
multi-teacher on-policy distillation from Tiny Aya Fire with a frozen safety anchor.

- **Target:** `CohereLabs/tiny-aya-global`, deployed as the official **Q4_K_M GGUF** (llama.cpp). Training
  happens on the BF16 weights; adapters are converted to **GGUF LoRA** and evaluated on the Q4_K_M base.
- **Attack source:** STRATA, our multilingual extension of AutoDAN-Turbo (separate repository, see below).
- **Eight surfaces:** `en`, `hi`, `hi_en_mix`, `hinglish`, `te`, `te_en_mix`, `ja`, `ja_en_mix`.
- **Write-up:** the full method, results and limitations are in [WRITEUP.md](WRITEUP.md) (tables from `scripts/writeup_tables.py`, figures from `scripts/make_figures.py`).

## Artifacts

- **Write-up: [WRITEUP.md](WRITEUP.md)**
- Models: https://huggingface.co/collections/kamikaze-san/tiny-aya-multilingual-safety-models-6af997c8a617ddc3326889bc
  (final model and every adapter: https://huggingface.co/kamikaze-san/tiny-aya-multilingual-safety)
- Datasets: https://huggingface.co/collections/kamikaze-san/tiny-aya-multilingual-safety-datasets-6af9982d7e17226835a9ac36
  (public results and labels; gated attack conversations)
- Attacker: https://github.com/kamikaze-san/STRATA_mling

## Lineage and work done before the hackathon

- **STRATA** (attack generation) predates the hackathon: it is our multimodal adaptation of
  [AutoDAN-Turbo](https://github.com/SaFoLab-WISC/AutoDAN-Turbo). During the hackathon it was extended to
  multilingual and code-mixed, multi-turn text attacks. Its public code is at [STRATA_mling](https://github.com/kamikaze-san/STRATA_mling).
- **Everything in this repository** was written during the hackathon (Nov 2026). `CODE_INVENTORY.md` lists
  what was adapted and what is new.

## Pipeline

```text
STRATA attack logs ──► label_pair_candidates (gpt-oss-safeguard, policy v2) ──► build_pair_pool (writer model)
                                                                                     │
                                         CuratorKIT (curate_with_curatorkit) ◄───────┘
                                                     │
                     build_conditions (ablations: joint, per-surface, leave-one-family-out)
                                                     │
              HF Jobs: hf_job_train_dpo.sh → train_dpo.py (DPO + LoRA, BF16, A10G)
                                                     │
                         download_and_convert_adapters (PEFT → GGUF LoRA)
                                                     │
   eval_adapter: run_safety_eval (llama.cpp + Qwen3Guard) + run_capability_eval (FLORES chrF++, Belebele)
                 + score_stance_decider (pplx-decider-flash compliance) + gates vs the base model
                                                     │
                       AuditKIT (audit_with_auditkit) ──► make_results_tables
                                                     │
  stage 2 (hf_job_stage2): retention DPO | OPD to the original model | multi-teacher OPD (Fire + frozen safe anchor)
                     + weight-space checks: project_adapter_update, add_safety_direction
```

## Layout

| Path | What it holds |
|---|---|
| `src/harness/` | GGUF model runner (llama.cpp, runtime LoRA) with a SQLite generation cache keyed by model fingerprint + decoding settings + messages |
| `src/eval/` | safety / over-refusal / capability metrics and per-surface slicing |
| `src/judge/` | Qwen3Guard judge, policy judge (gpt-oss-safeguard), decider helpers |
| `src/attacker/` | multilingual strategy pool and expansion loop from the early design (the attacks behind the results come from STRATA) |
| `src/curation/` | the writer that drafts the safe ("chosen") replies for preference pairs |
| `src/paths.py` | path resolver (repo root, `data/`, sibling STRATA checkout) |
| `configs/dpo.yaml` | fixed DPO + LoRA settings shared by every condition |
| `configs/judge/` | judge policies: `safeguard_policy_v2.md` (compliance, used to select pairs), `safeguard_harm_policy.md` (harm, judge comparison only) |
| `scripts/` | one script per pipeline step (each has `--help` and a docstring) |
| `laptop/` | offload of generation + capability runs to a second GPU machine (Windows, CUDA); results are merged back by `scripts/import_laptop_results.py` |
| `tests/` | unit tests (`python -m unittest discover -s tests -p "test_*.py"`) |
| `docs/` | judge validation notes and analysis reports |
| `data/`, `checkpoints/`, `cache/` | **git-ignored**: attack sets, pairs, model replies, adapters, benchmark copies |

## Environment

- Python 3.12, `pip install -r requirements.txt`. llama-cpp-python must be built with Metal (macOS) or CUDA.
- **Mac (Apple Silicon, 24 GB+):** generation (llama.cpp Metal), Qwen3Guard-Gen-4B (torch MPS bf16) and the
  pplx decider (MLX 4-bit) run as **separate processes, one at a time**.
- **LM Studio** serves gpt-oss-safeguard-20b (pair labelling) and the writer model (chosen replies).
- **Training:** Hugging Face Jobs, A10G (small for 100-pair runs, large for joint runs). A token with write
  access to your own private repos is read from the Hugging Face login (`hf auth login`), never from files.
- **Optional laptop:** NVIDIA GPU with 8 GB VRAM for generation + capability (`laptop/BRIEF.md`).

## Reproducing the results

Data folders are local and private (see "Data handling"); the commands below expect them under `data/`.

```bash
# 0. benchmarks (downloaded from their original sources, never re-hosted)
python scripts/download_benchmarks.py

# 1. evaluation sets
python scripts/build_heldout_strata_attacks.py --help     # STRATA held-out attacks (goals disjoint from training)
python scripts/import_aya_redteaming.py && python scripts/make_aya_heldout_200.py
python scripts/build_benign_baseline.py                   # general benign set for over-refusal
python scripts/split_benign_twins.py                      # borderline benign twins: train part / eval part

# 2. base model evaluation (Q4_K_M GGUF)
python scripts/run_safety_eval.py --help
python scripts/run_capability_eval.py --help
python scripts/score_stance_decider.py --run base_q4_v2   # run with the MLX environment

# 3. judge validation against human labels
python scripts/score_judges_vs_humans.py --help
python scripts/compare_harm_judges.py --help

# 4. preference pairs
python scripts/label_pair_candidates.py --max_tokens 3072 --workers 4 --strata_logs <STRATA attack logs>
python scripts/build_pair_pool.py --help                  # keeps RC1 (complied) candidates, writes chosen replies
python scripts/curate_with_curatorkit.py --pool data/03_curated/pair_pool.jsonl
python scripts/build_conditions.py --total_pairs 100 --leave_out te,ja,mixed --suffix _v3   # add --upload for HF

# 5. training on HF Jobs (private dataset + adapter repos)
./scripts/hf_job_train_dpo.sh --condition joint_all_v3 --seed 0 --lr 5e-5 --epochs 2 \
    --tag _lr5e-5_e2 --flavor a10g-large --timeout 3h --detach
python scripts/download_and_convert_adapters.py joint_all_v3_lr5e-5_e2_s0

# 6. adapter evaluation (safety + capability + gates vs base), then compliance scoring
python scripts/eval_adapter.py --name joint_all_v3_lr5e-5_e2 \
    --lora checkpoints/hf/joint_all_v3_lr5e-5_e2_s0/adapter.gguf --compare base_q4_v2
python scripts/score_stance_decider.py --run joint_all_v3_lr5e-5_e2
#    (scripts/run_eval_queue_v3.sh runs steps 6 for a list of adapters, one GPU job at a time)

# 7. AuditKIT and tables
python scripts/audit_with_auditkit.py --help
python scripts/make_results_tables.py
python scripts/writeup_tables.py                          # every table in the write-up, with paired bootstrap CIs
python scripts/flores_bootstrap.py --runs <run> --compare base_q4_v2
python scripts/recompute_gates.py --runs <run> --compare base_q4_v2

# 8. stage 2: capability repair on top of the stage-1 adapter (HF Jobs, A10G)
#    prompt sets: FLORES-dev ids + twins (opd_v1), translation only, or translation + safety anchors (opd_v3_mt)
python scripts/build_opd_prompts.py --help
python scripts/build_opd_mt_prompts.py --help
./scripts/hf_job_stage2.sh --method retention --init joint_all_v3_lr5e-5_e2_s0 --out joint_all_v3_ret -- --num_train_epochs 1
./scripts/hf_job_stage2.sh --method opd --init joint_all_v3_lr5e-5_e2_s0 --out joint_all_v3_opd -- --steps 100 --lr 1e-5
#    final model: multi-teacher OPD, Telugu translation to a 50/50 log-space mix of Tiny Aya Fire and the frozen
#    stage-1 adapter, safety anchors to the frozen stage-1 adapter, log(1 + KL) per token; we keep step 50
./scripts/hf_job_stage2.sh --method opd --prompts opd_v3_mt --init joint_all_v3_lr5e-5_e2_s0 --out joint_all_v3_mtopd \
    --flavor a10g-large --timeout 1h --detach -- --steps 72 --lr 1e-5 --batch 8 --micro 1 --warmup 5 --save_every 50 \
    --gradient_checkpointing --tags en-te,te-en --teacher_model CohereLabs/tiny-aya-fire --teacher_alpha 0.5 \
    --mix_with anchor --kl_transform log

# 9. weight-space interventions (no training)
python scripts/project_adapter_update.py --stage1 joint_all_v3_lr5e-5_e2_s0 --stage2 joint_all_v3_opdtr_s0 \
    --out joint_all_v3_opdtr_proj_s0                       # remove the part of an update that undoes stage 1
python scripts/add_safety_direction.py --base joint_all_v3_opdtr_proj_s0 --stage1 joint_all_v3_lr5e-5_e2_s0 \
    --beta 0.2 --out joint_all_v3_proj_b020_s0             # push further along the stage-1 safety update

# 10. judge check against the 128-row human sheet, including newer and larger pplx deciders
python scripts/convert_decider_mlx.py <official decider release> <out>/4bit-g64   # MLX 4-bit, text-only
python scripts/score_sheet_deciders.py --help
python scripts/judges_vs_human128.py
#    (scripts/run_judge_validation_chain.sh runs all of step 10, one GPU job at a time)
```

**Fixed decoding for every evaluation:** temperature 0, top_p 1, seed 0, the same max tokens and context, so
replies are deterministic and cacheable. Generations from the laptop are merged into the same cache and
verified prompt by prompt before the Mac judges them.

## Metrics

| Metric | Direction | Judge / source |
|---|---|---|
| Compliance rate on held-out attacks (any turn) | lower is better | pplx-decider-flash v1.2 `goal_fulfilled` ≥ 0.5 (primary; checked against human labels) |
| Unsafe rate on held-out attacks | lower is better | Qwen3Guard-Gen-4B (Controversial reported both ways) |
| Over-refusal on borderline benign twins and general benign prompts | lower is better | Qwen3Guard refusal field; gate: at most +5 points per surface |
| FLORES-200 devtest chrF++ (en↔hi/te/ja) | higher is better | sacrebleu |
| Belebele accuracy (en, hi, te, ja) | higher is better | exact option match |

## Data handling

This repository holds code only. Attack prompts and model replies are read from local, git-ignored folders and
never committed. The attack data is released separately, in a gated dataset.

- Attack prompts, harmful replies, preference pairs and model outputs are **never committed**:
  `data/`, `cache/`, `checkpoints/`, `*.jsonl`, `*.json` (outside `configs/`/`schemas/`), `*.csv`, `*.sqlite`,
  `*.gguf` and run logs are git-ignored.
- Scripts print counts and aggregates only ("zero exposure"); no script echoes attack or reply text.
- FLORES-200 and Belebele are downloaded from their original sources and not re-hosted.
- Before publishing anything, run `python scripts/check_public_release.py . --history`
  (secrets, data files, absolute paths; it lists locations only, never the matched text).

## License

Code: MIT (`LICENSE`). The adapters and datasets follow Tiny Aya's CC-BY-NC-4.0, and the base model, judges and
benchmarks keep their own licenses.
