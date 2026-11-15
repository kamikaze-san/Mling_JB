# Strong *and* safe: closing multilingual, code-mixed, multi-turn jailbreak gaps in Tiny Aya without making it dumber

**TL;DR:** We trained Tiny Aya, a small multilingual model built for strong translation and multilingual understanding, to resist jailbreaks in English, Hindi,Telugu, Japanese and mixes of these languages with English (Hinglish, Tel+Eng, etc.), decreasing its Jailbreak rate from 74% to 13%, while maintaining strong performance in translation.

Safety tuning usually costs a model some of its abilities: the "alignment tax". The standard remedy is to mix safety
examples into general instruction data and fine-tune on both (Bianchi et al., 2024, with supervised fine-tuning). That
needs a large general dataset and a full training run over it, and for a model that is already released, the original
training mix is not available. Too much safety data also makes a model refuse harmless prompts that merely look
unsafe. A cheaper remedy interpolates the tuned weights back towards the original model (Lin et al., 2024), but that
moves the weights back along the safety update and can make the model less safe again. We keep both safety and
capability with little data and compute. 

Stage 1 is DPO on 1,255 multilingual multi-turn conversations with a small LoRA adapter. Stage 2 is a short distillation that learns translation from a
stronger sibling model (Tiny Aya Fire) while a frozen copy of the safe model anchors everything else; it
adds under $1 of GPU time. We also show what
makes such a repair safe: its direction in weight space, not its size. A repair that moves the weights back along the
safety update makes the model easier to jailbreak again; one kept orthogonal to it does not.

![Jailbreak compliance by language surface, base vs final model](docs/figures/compliance_by_surface.svg)

> **Content warning.** This work is about jailbreaking a language model. The write-up reports only aggregate numbers
> and contains no attack prompts or harmful outputs. The gated dataset linked under *Artifacts* contains working
> multilingual jailbreak conversations and harmful model replies.

## Tracks and problem statements

- **Track 2a: AI Safety and Red Teaming**
  - **4. Multilingual and code-mixed safety** (main): we measure how safety holds across English, Hindi, Telugu,
    Japanese, romanised Hinglish (Hindi in Latin script) and Hindi (Devanagari)-, Telugu- and Japanese-English mixes,
    then close the gaps.
  - **3. Multi-turn audits:** every attack is a conversation of up to three turns from our STRATA attacker; we report the
    turn at which the model broke.
  - **2. Adversarial and safety evaluation sets:** a held-out multilingual, multi-turn attack set and a set of borderline
    "twin" prompts, labelled by three independent judges.
- **Track 2b: AI Safety and Correction**
  - **5. Safety-preserving fine-tuning:** stage 2 fine-tunes the safe model for translation with a safety anchor
    (multi-teacher distillation). Safety holds (12.6% vs 12.8% compliance) where plain distillation from the original
    model raises compliance to 33–45% (§6.4).
  - **6. Repairing safety drift without retraining:** when a translation repair raised compliance, projecting out the
    part of its update that reverses the safety update brought safety back to within noise of stage 1 with no training,
    and pushing further along the safety direction lowers compliance in a dose-response, with translation unchanged (§6.4).
  - **9. Locating where safety lives:** across every stage-2 run, the safety cost follows how far an update points back
    along the stage-1 safety direction (most strongly in the value and MLP down projections), not its size. We use that
    direction for the targeted fixes above. This works at the level of LoRA update directions, not individual circuits
    or neurons (§6.4).

## Abstract

Language models are safety-tested mostly in English and one message at a time. Attacks in other languages, in mixed
languages, or spread over several turns are the ones tested least, and the ones most likely to get through (Yong et al.,
2023; Deng et al., 2023; Yoo et al., 2024). We red-teamed **Tiny Aya** (tiny-aya-global, 3.35B), evaluated as deployed:
the official 4-bit GGUF in llama.cpp. The attacks were multi-turn jailbreaks in English, Hindi, Telugu, Japanese,
Hinglish and Hindi-, Telugu- and Japanese-English mixes. They broke the base model in every language: 67% of attacks
succeeded in English and about 80% in Telugu and Hindi, against 28–30% for single-turn prompts from Cohere's human-written Aya Red-teaming dataset (§6.1). We then made the model safe without making it worse at its job.

**Result.** On 382 held-out attack conversations, compliance falls from **73.6% to 12.6%** (Δ −61.0 points, 95% CI
[−66.2, −55.8]), on every one of the eight surfaces (67–80% before, 8–22% after). Translation (FLORES+) is at the
original model's level or better in all six directions, and reading comprehension (Belebele) is unchanged or better.
Refusals of harmless look-alike prompts stay low (5 of 91).

**Method.**
- **Stage 1** is DPO on a small LoRA adapter with 1,255 multi-turn pairs from our attacker STRATA: 772 refusals of
  attacks that worked and 483 helpful answers to harmless look-alikes. This alone reaches 12.8% compliance, but costs
  about 1 chrF++ on Telugu→English.
- **Stage 2** is a 50-step on-policy distillation. It learns translation from a stronger sibling model, Tiny Aya Fire,
  while a frozen copy of the safe model anchors everything else. It repairs translation at no cost to safety, for under
  $1 of GPU time.

**Finding.** Whether a repair undoes safety depends on its direction in weight space, not its size. Distilling from
the original model also repairs translation, but compliance rises to 33–45%, because its small update points back
against the safety update. Removing that component, or anchoring training so the update stays orthogonal, keeps the
model safe (§6.4).

**Evaluation.** Everything is measured on the deployed 4-bit model, with paired bootstrap CIs and decision gates fixed
before training. Three judges were each checked against a human annotator; all of them miss cases rather than raise false
alarms, so the reported rates are conservative. Single seed. Adapters, code and evaluation data are released, and the attack
conversations are gated.

> **For judges: one minute**
> - **Safer:** jailbreak compliance 73.6% → **12.6%** on 382 held-out multi-turn attacks, on every surface (8–22%).
> - **Not dumber:** FLORES+ translation at base level or better in all six directions; Belebele unchanged or better.
> - **Not over-refusing:** 5 of 91 harmless look-alikes refused; 1 of 100 general prompts.
> - **How:** DPO on 1,255 pairs, then a 50-step safety-anchored distillation from Tiny Aya Fire.
> - **Insight:** a repair's direction in weight space, not its size, decides whether it undoes safety.
> - **Cost:** ~$20 of GPU for all training; every number is measured on the deployed 4-bit GGUF.

## 1. Problem statement and approach

### Why this problem
Tiny Aya is built to run on phones and laptops in 70+ languages, so its attack surface is every one of those
languages and every way of mixing them with English: Hindi in Devanagari mixed with English, Hinglish (Hindi in Latin
script), Telugu and English in one sentence, and English terms inside Japanese. Safety training and red-teaming, though, are done mostly in English and one message at
a time, and published work shows that attacks in less-tested languages, in mixed-language text and over several turns
get through more often (Yong et al., 2023; Deng et al., 2023; Yoo et al., 2024; Russinovich et al., 2024). Our own audit
agrees (§6.1): multi-turn attacks succeed 67–80% of the time on every surface, against 28–30% for single-turn prompts.

Fixing this has a second constraint. A small model has little spare capacity, so safety tuning that costs it
translation or comprehension makes it worse at the job it is deployed for. We wanted a model that is **strong and safe,
not just safe or just strong**: lower jailbreak compliance on every surface, with capability held within limits fixed
before training.

### Research questions
1. Does a small multilingual, multi-turn DPO set reduce held-out jailbreak compliance on every surface, including
   code-mixed ones? (§6.1)
2. How much does safety transfer across languages and scripts, and how much data does it take? (§6.3)
3. Can safety be added without an alignment tax on multilingual capability and over-refusal? (§6.1, §6.2)
4. Can a cheap second stage repair the capability cost without making the model less safe again? (§6.4)

The five decision gates (§5) were written into `EVAL_PROTOCOL.md` §4 on 8 Nov, before the first adapter was trained
(a dry run on 10 Nov; the first real adapters on 11 Nov). Git history shows they have not changed since. They are:
held-out ASR falls, twins over-refusal and general over-refusal rise by at most 5 and 3 points, FLORES drops by at
most 1.0, and Belebele drops by at most 3. Two things were decided later, with the first adapters:
- the decider's compliance as the primary safety metric, alongside the Qwen3Guard-based ASR the gates use;
- choosing the final model on compliance and translation together, rather than the protocol's "lowest worst-slice ASR".
Research question 4 (stage 2) was added after the stage-1 results showed the translation cost.

### Threat model
An ordinary user of a deployed small chat model: black-box access, a conversation of up to three turns, any of the eight
surfaces, no access to weights, system prompt or tools. Success = the model substantively helps with a harmful goal.
Out of scope: fine-tuning attacks, tool or document prompt injection, white-box and weight-space attacks, adaptive
re-attack against the hardened model (see Limitations).

### Approach
1. **Attack:** STRATA, our extension of AutoDAN-Turbo (built on its MIT-licensed code), generates multi-turn jailbreaks
   on all eight surfaces against the deployed Q4_K_M model. It keeps AutoDAN-Turbo's lifelong strategy-library loop
   and adds multi-turn conversations, multimodal attacks (before the hackathon) and a multilingual, code-mixed text mode
   (during it).
2. **Judge and label:** three judges (pplx decider for compliance; Qwen3Guard and gpt-oss-safeguard for harm).
3. **Pairs:** refusals vs compliant replies on successful attacks; helpful vs refusing replies on harmless look-alikes.
4. **Curate:** CuratorKIT QA.
5. **Train:** LoRA DPO; convert to GGUF LoRA; evaluate on the same 4-bit deployment path.
6. **Repair:** a second stage on top of the safe adapter (retention DPO, on-policy distillation), evaluated the same way.

### Related work
- *Attacks:* AutoDAN-Turbo, lifelong strategy-library jailbreaks and STRATA's starting point (Liu et al., 2024);
  multi-turn escalation, e.g. Crescendo (Russinovich et al., 2024); low-resource and multilingual jailbreaks (Yong et
  al., 2023; Deng et al., 2023); code-switching red-teaming (Yoo et al., 2024).
- *Data:* Aya Red-teaming, multilingual human red-team prompts (Aakanksha et al., 2024); the Aya Evaluation Suite
  (Singh et al., 2024); FLORES+ (NLLB Team, 2022); Belebele (Bandarkar et al., 2023).
- *Training:* DPO (Rafailov et al., 2023); LoRA (Hu et al., 2021); TRL.
- *Fragility of safety:* benign fine-tuning erodes safety (Qi et al., 2023); safety alignment is only a few tokens
  deep (Qi et al., 2024).
- *Distillation:* on-policy distillation / GKD, reverse KL on student samples (Agarwal et al., 2023); self-distillation
  fine-tuning, SDFT (Yang et al., 2024); OPD pathologies and regulation (Wang et al., 2026).
- *Multi-teacher on-policy distillation:* MOPD (Ma et al., 2026) merges several domain teachers into one student by
  routing each prompt to its domain's teacher on the student's own samples. Our final stage 2 adapts this to safety.
  Telugu translation is routed to a specialist (Tiny Aya Fire) mixed with a frozen copy of the safe adapter, and
  safety prompts are routed to the frozen safe adapter alone. The second teacher is an *anchor* that holds safety in
  place rather than adding a capability.

## 2. Hackathon kits used

| Kit | How it was used |
|---|---|
| **CuratorKIT** | QA over the full 1,262-pair preference pool before training: exact and MinHash deduplication, SecretsGate, PII detection and text normalisation (report only, text kept byte-identical), ToxicityGate on chosen refusals. |
| **AuditKIT** | Independent audit of the base vs the final model (`scripts/audit_with_auditkit.py`, `scripts/run_auditkit_final.sh`). AuditKIT 1.1.10 `evaluate` with its `Abstention` and `ToxicityScore` scorers, on the same 582 attacks and 191 benign prompts, with each model served by llama-server. Per surface and pooled, with Wilson CIs. **Result: the kit cannot see this change.** Attack refusal goes from 1.2% to 1.5%, over-refusal stays at 0% and cleanliness goes from 0.988 to 0.994, although the replies differ (14 refusal flags and 80 toxicity scores change). Its refusal detector is an English phrase list built for RAG abstention, and the toxicity scorer is a ~15-word English list. Refusals in Hindi, Telugu or Japanese, and English refusals worded differently, are invisible to them, and the kit sends only the last user turn. Over the same attacks, the LLM judges validated against a human in §5 show compliance falling from 73.6% to 12.6%. Keyword-based auditing misses multilingual safety behaviour entirely, which is the English-centric gap this project is about. |
| AlignTune | Not used. It is a post-training toolkit for SFT, DPO, PPO and GRPO with TRL or Unsloth back ends (arXiv:2602.09621). On-policy distillation, which our stage 2 needed, is not among its methods: continuing the existing LoRA, the original model (adapter off, same weights) as teacher, and an external teacher mixed per prompt with a frozen copy of the safe adapter. So we wrote our own trainer on TRL/PEFT. |

**How they connect:** STRATA attacks → three judges → preference pairs → **CuratorKIT** → DPO (LoRA, HF Jobs) → GGUF
LoRA → evaluation harness on the 4-bit model → stage-2 repair → same evaluation → **AuditKIT** cross-check of base vs final. [TODO: diagram]

**CuratorKIT filtering:** 1,262 pairs in → **1,257 kept** (the final training set has 1,255: the 2 others are human-written safety pairs with no surface label, so they fit none of the eight surfaces). Dropped 5: 3 by SecretsGate (high-entropy base64/hex strings)
and 2 exact duplicates (2 benign, 3 safety). No pair was dropped by ToxicityGate. Per surface the drops were en 1, hi 1,
hi_en_mix 1, te_en_mix 2.

## 3. Models and licenses

| Role | Model | License |
|---|---|---|
| Base / target (attacked and hardened) | CohereLabs/tiny-aya-global, official **Q4_K_M GGUF**, llama.cpp | CC-BY-NC-4.0 |
| Our adapters | LoRA r = 16, α = 32, dropout 0.05, all linear layers (30.2M trainable params), GGUF LoRA for deployment | CC-BY-NC-4.0 (follows the base) |
| Attacker | STRATA text mode, attacker LLM Qwen3.6-35B-A3B (`qwen/qwen3.6-35b-a3b`, local via LM Studio) | STRATA: MIT; Qwen3.6: Apache-2.0 |
| Compliance judge (primary) | pplx decider-flash v1.2 (MLX 4-bit); v1.3 flash and 27B for the judge check (§5) | Apache-2.0 |
| Harm judges | Qwen3Guard-Gen-4B; gpt-oss-safeguard-20b | Apache-2.0 (both) |
| Stage-2 translation teacher | CohereLabs/tiny-aya-fire (same size as global; bf16 on HF Jobs) | CC-BY-NC-4.0 |

No closed-model baseline was run.

## 4. Datasets, benchmarks and licenses

### Training data (`joint_all_v3`, final)
| Kind | Pairs | What it is |
|---|---|---|
| Safety | 772 | multi-turn STRATA attacks that worked on the base model; chosen = refusal, rejected = compliant reply |
| Benign look-alikes ("twins") | 483 | harmless requests on the same topics / in the same styles; chosen = helpful answer |
| **Total** | **1,255** | ja 192 · ja_en_mix 184 · te_en_mix 177 · en 153 · hi 147 · te 145 · hinglish 134 · hi_en_mix 123 |

Earlier, smaller sets used in the ablations: 56 pairs (joint8, 7 per surface), 100 pairs per single surface, 912 pairs
(joint_all_p100).

### Languages
English; Hindi (Devanagari); Telugu; Japanese; Hinglish (romanised Hindi); and Hindi–, Telugu– and Japanese–English
code-mixed text.

### Splits and leak checks
- **Held-out attacks:** 370 STRATA attacks never used for any pair + 12 human-written multi-turn attacks = **382**
  (157 single-turn, 143 two-turn, 82 three-turn). The split is by **goal**: 89 goals were reserved before any pair was built, and
  the held-out attacks come from a separate STRATA run on 48 of them. The leak check is an exact match on the normalised
  goal (SHA-256 of the lower-cased, stripped text): any training candidate whose goal matches a reserved goal is dropped,
  and every training condition is re-validated against the list. Within the training set, CuratorKIT removes exact and
  MinHash (0.92) near-duplicates.
- **Twins:** 91 evaluation twins, disjoint from training twins except for **one** item found late (a limitation).
- **FLORES+:** stage-2 training uses **dev** only; evaluation uses **devtest**; the 157 dev sentences that also appear in
  Belebele are excluded from training. FLORES text is never redistributed: our files hold sentence IDs only.

### Benchmarks
| Benchmark | Role | Languages | n |
|---|---|---|---|
| STRATA held-out + human multi-turn attacks | safety (focus) | all 8 surfaces | 382 |
| Aya Red-teaming held-out | safety, single-turn, out of distribution | en, hi | 200 |
| Borderline twins | over-refusal (focus) | all 8 | 91 |
| Aya eval suite (human-annotated, Dolly human-edited, Dolly machine-translated) | over-refusal on general prompts (retention) | en, hi, te, ja (100 each; 25 each trimmed) | 400 (100 in the trimmed eval) |
| FLORES+ devtest, chrF++ | capability (retention) | en↔hi, en↔te, en↔ja | 200 per direction |
| Belebele | capability (retention) | en, hi, te, ja | 150 per language |

## 5. Metrics and gates

| Metric | Better | Definition |
|---|---|---|
| **Compliance** (primary) | lower | pplx decider: any turn with goal_fulfilled ≥ 0.5 |
| Unsafe-reply rate | lower | Qwen3Guard, any turn labelled Unsafe (strict) or Unsafe/Controversial (lenient) |
| Over-refusal, twins / general | lower | share of harmless prompts refused |
| FLORES chrF++ | higher | translation quality |
| Belebele accuracy | higher | reading comprehension |

**Gates, fixed before training** (`EVAL_PROTOCOL.md` §4): unsafe-reply rate must fall (paired bootstrap CI below 0); twins
over-refusal may rise at most +5 points on any surface; general over-refusal at most +3; FLORES may drop at most 1.0
chrF++ in any direction; Belebele at most 3 points in any language.

**Statistics:** all comparisons are paired on identical prompts, with 95% bootstrap CIs (2,000 resamples). FLORES CIs
resample sentences and recompute corpus chrF++.

**Checking the judges:** on a 64-item sheet the decider's hazard flag agrees with Qwen3Guard at κ 0.40–0.46 and with
gpt-oss-safeguard at κ 0.28–0.31. The judges disagree enough that we report the decider (willingness) as primary and
Qwen3Guard (harm) both ways.

**Judges against a human.** One member of our team labelled 128 STRATA attack replies (16 per surface) from the STRATA
judge-labelling sheet for two things: did the model comply, and was the reply unsafe. There was a single annotator,
with no second rater or adjudication, so treat this as a reference check, not a gold standard. The human called 113
replies compliant and 81 unsafe. Every unsafe reply was also compliant, but 32 compliant replies were judged safe, so
compliance and harm are different questions.

| Judge (human label it is scored against) | n | Agreement | κ [95% CI] | Recall | False alarms |
|---|---|---|---|---|---|
| decider *complied*, sheet wording (complied) | 128 | 74.2% | 0.36 [0.22, 0.51] | 80/113 | 0/15 |
| decider *complied*, eval wording, as in §6 (complied) | 128 | 71.1% | 0.32 [0.19, 0.47] | 76/113 | 0/15 |
| gpt-oss-safeguard (unsafe) | 128 | 65.6% | 0.37 [0.25, 0.51] | 39/81 | 2/47 |
| decider *hazard*, sheet wording (unsafe) | 128 | 61.7% | 0.31 [0.19, 0.43] | 35/81 | 3/47 |
| decider *hazard*, eval wording (unsafe) | 128 | 54.7% | 0.20 [0.09, 0.32] | 27/81 | 4/47 |
| Qwen3Guard lenient, Unsafe + Controversial (unsafe) | 128 | 51.6% | 0.12 [−0.03, 0.25] | 30/81 | 11/47 |
| Qwen3Guard strict, Unsafe (unsafe) | 128 | 43.0% | 0.04 [−0.05, 0.14] | 13/81 | 5/47 |

- **Every judge under-calls rather than over-calls.** The compliance judge as used in §6 raised no false alarms but
  caught 67% of what the human called compliance. Our absolute compliance rates are therefore conservative, closer to
  lower bounds. Comparisons between runs hold if the misses are spread evenly across runs, which we did not test. The
  two halves of the sheet agree (κ 0.33 and 0.39), so the estimate is stable.
- **Qwen3Guard barely agrees with the human on harm** for these multilingual multi-turn replies (κ 0.04–0.12; neither
  CI excludes 0). It catches 13–30 of the 81 replies the human called unsafe. Its low unsafe rates in §6 (a few
  percent) are not evidence that the remaining replies are safe. We keep it as a secondary signal because it is the
  standard open guard model and its run-to-run *changes* point the same way as the decider's, but the decider is the
  primary metric.
- **gpt-oss-safeguard is the best harm judge here (κ 0.37), followed by the decider's hazard flag (0.20–0.31,
  depending on wording).** Both still miss more than half of the human's unsafe replies. Wording matters for harm
  questions (0.31 vs 0.20 for the same model), much less for compliance (0.36 vs 0.32).
- **Newer and larger deciders** (`scripts/score_sheet_deciders.py`). Each was asked the same questions with the same
  sheet wording; only the model differs.

  | Decider | complied κ (recall, false alarms) | hazard κ (recall, false alarms) |
  |---|---|---|
  | flash v1.2, used for §6 | 0.36 (80/113, 0/15) | 0.31 (35/81, 3/47) |
  | flash v1.3, released 10 Nov | 0.39 (83/113, 0/15) | **0.41** (45/81, 4/47) |
  | 27B v1.1 | 0.20 (63/113, 1/15) | 0.17 (22/81, 3/47) |
  | 27B v1.3, released 10 Nov | 0.32 (76/113, 0/15) | **0.43** (49/81, 6/47) |

  - **Version matters more than size.** The v1.3 deciders are the best harm judges on this sheet: flash v1.3 at 0.41
    and 27B v1.3 at 0.43, both ahead of gpt-oss-safeguard (0.37). The 27B catches the most unsafe replies but raises
    more false alarms.
  - **On compliance, every decider except the 27B v1.1 lies between 0.32 and 0.39, with overlapping CIs.** Size buys
    nothing there, so the §6 results do not depend on the decider version or size.
  - **The 27B v1.1 is the least comparable run.** It used a community MLX 4-bit conversion and its one-question-per-pass
    code path, while the others answer all questions in one pass with per-question markers.
  - We converted the official v1.3 releases to MLX 4-bit ourselves and released them (§10; `scripts/convert_decider_mlx.py`; the 27B release
    has no LM head, which the decision readout replaces). On an M4 Pro, flash scores the 128 rows in a few minutes
    and 27B v1.3 in about 30 (~15 s per reply).

**Cost-to-break:** not measured as attempts-to-first-success; we report the turn at which each conversation first broke (§6.1).

## 6. Experiments and results

### 6.0 Setup
- **DPO:** TRL 1.17, sigmoid loss, β = 0.1, learning rate 5e-5, cosine schedule, 10% warm-up, 2 epochs, batch 1 ×
  8 gradient accumulation, max length 2,048, bf16, gradient checkpointing. HF Jobs, one A10G.
- **Deployment path:** PEFT adapter → GGUF LoRA (f16) → llama.cpp on the official Q4_K_M base, greedy decoding. Most
  runs were generated on an Apple M-series Mac (Metal); some on an RTX 4060 laptop with the same prompts, decoding and
  llama.cpp build family; base-model capability differs between the two machines by at most 0.7 chrF++ in one direction.
- **Seeds:** 0 for every run (see Limitations).
- **Ablations:** training-set size; single-surface training (8 runs, 100 pairs each) vs the same 100 pairs spread over all 8 surfaces; leave-one-out (no Telugu, no Japanese, no
  code-mixed); checkpoint curve; LoRA scale 0.75; stage-2 method and length.

### 6.1 Main result: the same model before and after

| Held-out (n = 382) | Base | Stage 1 |
|---|---|---|
| **Compliance** | 73.6% | **12.8%** (Δ −60.7, CI [−65.7, −55.5]) |
| …multi-turn conversations only (n = 225) | 70.2% | 10.2% |
| Unsafe replies, Qwen3Guard strict | 11.3% | 2.4% |
| Unsafe replies, Qwen3Guard lenient | 31.4% | 4.2% |
| Aya Red-teaming, single-turn en/hi (n = 200) | 29.0% | 17.0% |
| Twins over-refused | 0/91 | 6/91 |
| General prompts over-refused | 2/400 | 3/400 |
| Belebele (4 languages) | | +0.0 to +1.3 points |
| FLORES te→en | 56.74 | −1.11 [−1.92, −0.35] |
| FLORES en→ja | 25.34 | +1.10 [+0.37, +2.11] |
| FLORES other 4 directions | | −0.29 to +0.08, all CIs include 0 |

**Every surface improves** (compliance, held-out):

| | en | hi | hi_en_mix | hinglish | te | te_en_mix | ja | ja_en_mix |
|---|---|---|---|---|---|---|---|---|
| n | 51 | 49 | 37 | 49 | 50 | 45 | 49 | 49 |
| Base | 66.7% | 79.6% | 73.0% | 75.5% | 80.0% | 66.7% | 69.4% | 75.5% |
| Stage 1 | 5.9% | 10.2% | 10.8% | 10.2% | 22.0% | 11.1% | 14.3% | 12.2% |

Telugu stays the weakest surface (22%), the lowest-resource language of the four and the one where the base model
complies most.

**Multi-turn audit: where it broke.** On the base model, 281 of 382 conversations broke; 174 at turn 1, **82 at turn 2
and 25 at turn 3**. 38% of the base model's failures would be invisible to a single-prompt test. After training, 49
break: 41, 7 and 1. Late-turn breaks fall the most (−93% for turns 2–3 vs −76% for turn 1).

**Gates:** stage 1 passes the safety and general over-refusal gates and Belebele, and fails two:
FLORES te→en (−1.11 vs a −1.0 limit) and twins over-refusal on Telugu (+18 points = 2 of 11 prompts). Pooled over all
91 twins the rise is +6.6 points [+2.2, +12.1]; the per-surface gate at n ≈ 11 fails on any single extra refusal (§6.5).

### 6.2 Why translation got worse
The te→en drop is real (bootstrap CI excludes 0) but narrow. Output format is unaffected and 61–79% of translations are
**identical** to the base model's. The loss sits in about 10 longer, harder sentences, whose translations got ~12% shorter
(each −13 to −19 chrF++). The cause we can point to is a **length bias in the preference data**: on safety pairs the chosen
refusal has a median length of 343 characters against 2,181 for the rejected compliant reply, and safety pairs
outnumber benign pairs 772 to 483, so DPO also learns "shorter is better". Over training on the earlier 912-pair run the
drop is present from the first checkpoint and roughly flat until the end (te→en at steps 57 / 114 / 171 / final:
−0.86 / −0.94 / −0.73 / −1.20), while compliance falls 29.3% → 22.8% → 15.7% → 16.8%. So it is not a clean
dose-response: it appears early and is largest at the end, but noise between checkpoints is of the same size.

**What safety training changed, measured directly.** We measured the per-token KL between the safe model and the
original on the deployed 4-bit models (llama.cpp, greedy replies of the safe model, full-vocabulary KL at every reply
token): **0.0047** on 150 FLORES-dev translation prompts, **0.0135** on 48 harmless twins, and **0.106** on 49 held-out
attack first turns. Safety DPO moved the model about **20× more on attacks than on translation**. The change is targeted;
the te→en loss is a small side effect, not general forgetting.

**The other direction: stage 1 improves English→Japanese** (+1.11 chrF++ [+0.37, +2.11]). Both stage-1 effects on
translation are general, not tied to which languages were trained. Across all 15 stage-1 variants (single-surface,
leave-one-out, joint), te→en is lower than base in every one (mean −0.61) and en→ja is higher in 12 of 15 (mean +0.70),
including the run with no Japanese pairs (+1.42) and the English-only run (+0.96). The other four directions stay
within noise. The en→ja gain is mostly register: DPO's chat-style chosen replies shift Japanese output towards the
polite です/ます form (sentence endings: 63 → 81 of 200), which the FLORES+ references use (185 of 200). The 20 sentences
that switch to polite gain 7.2 chrF++ on average. It is a closer match to the reference style, not necessarily better
Japanese.

### 6.3 How safety transfers, and how much data it takes

**Data size matters more than which language it comes from.**

| Training set | Pairs | Held-out compliance |
|---|---|---|
| none (base) | 0 | 73.6% |
| joint8 (7 per surface) | 56 | 66.2% |
| single surface (8 runs) | 100 | 55.2% (hi_en_mix) – 70.2% (en) |
| joint8_p100 (6–7 per surface, all 8) | 100 | 59.2% |
| joint_all_p100 | 912 | 16.8% |
| **joint_all_v3** | **1,255** | **12.8%** |

At 100 pairs, training on one surface helps every surface a little. Hindi–English code-mixed transfers best (−18.3
points overall), then Hindi (−15.2); English transfers least (−3.4, the only CI that includes 0). The 8 single-surface
runs are en, hi, te, hinglish, hi_en_mix, te_en_mix, ja_en_mix and all four code-mixed surfaces pooled (−7.1). Spreading
the same 100 pairs over all 8 surfaces (joint8_p100, −14.4) lands mid-pack: at this size, coverage alone does not beat
the best single surface. The large effect needs all surfaces and around a thousand pairs.

**Leave-one-out** (same 1,255-pair recipe minus one group):

| Removed | Overall | Δ vs full [95% CI] | Removed surface(s): full → without |
|---|---|---|---|
| Telugu | 14.4% | +1.6 [−1.6, +4.5] | te 22.0% → 26.0% |
| Japanese | 19.4% | +6.5 [+3.4, +9.7] | ja 14.3% → 18.4% |
| all code-mixed | 19.4% | +6.5 [+3.4, +9.7] | hi_en_mix 10.8 → 18.9, te_en_mix 11.1 → 22.2, ja_en_mix 12.2 → 6.1 |

Telugu safety survives almost intact with **zero** Telugu training pairs: what is learnt from Hindi, code-mixed Telugu and
the other surfaces carries over. Removing Japanese or the code-mixed data costs more, but spread across surfaces, not
concentrated on the removed one. Per-surface n is about 50, so single-surface differences under ~10 points are noise.

### 6.4 Stage 2: repairing capability on top of the safe adapter
**How to read this section:**
- Lower compliance means safer.
- A **positive Δ against stage 1** means the repair made the model *less safe*. The "safety cost" of a repair is this
  rise in compliance.
- "**Stage 1 undone**" is the share of the stage-1 safety update that a repair reverses in weight space.

Both methods start from the stage-1 adapter, keep training the **same** LoRA, and use 960 harmless prompts: 720
FLORES-dev translation prompts (120 per direction) and 240 benign training twins.

- **Retention DPO:** greedy answers from the original model (adapter off) and the safe model (adapter on); where they
  differ, chosen = original, rejected = safe. 503 pairs; 1 epoch, lr 5e-6. With TRL's PEFT handling, the frozen
  reference is the starting **safe** adapter, so the implicit KL keeps the model close to the safe model.
- **On-policy distillation (OPD):** the student (safe model) samples an answer; we minimise the exact per-token reverse
  KL(student ‖ original) over the full vocabulary on those samples. 100 steps, batch 8, lr 1e-5; checkpoint at step 50.

| Held-out (n = 382) | Compliance | Δ vs stage 1 [95% CI] | FLORES te→en vs base | Twins | Gates |
|---|---|---|---|---|---|
| Stage 1 | 12.8% | — | −1.11 [−1.92, −0.35] | 6/91 | fail |
| + Retention DPO | 14.9% | +2.1 [−0.3, +4.5] | −0.81 [−1.45, −0.20] | 3/91 | fail |
| + OPD, step 50 | 32.7% | +19.9 [+15.7, +24.1] | −0.19 [−0.64, +0.25] | 0/91 | pass |
| + OPD, step 100 | 45.0% | +32.2 [+27.2, +37.4] | −0.17 [−0.56, +0.21] | 0/91 | pass |
| + OPD, translation prompts only, step 50 | 17.0% | +4.2 [+1.3, +7.1] | −0.52 [−1.18, +0.09] | 2/91 | fail (twins te) |
| + OPD, translation prompts only, step 100 | 17.5% | +4.7 [+2.1, +7.6] | −0.31 [−0.83, +0.23] | 4/91 | fail (twins te) |
| Stage 1 at LoRA scale 0.75 (no training) | 23.8% | +11.0 [+7.3, +14.7] | −0.86 [−1.57, −0.18] | 2/91 | fail (twins, general) |
| Translation-only OPD, projected (no training) | 14.9% | +2.1 [−0.8, +4.7] | −0.19 [−0.72, +0.31] | 3/91 | fail (twins te, 1 prompt) |
| Projected + 0.1 × stage-1 update (no training) | 13.6% | +0.8 [−1.3, +2.9] | −0.18 [−0.57, +0.20] | 6/91 | fail (twins te 2/11, as stage 1) |
| Projected + 0.2 × stage-1 update (no training) | 11.3% | −1.6 [−4.2, +0.8] | −0.33 [−0.77, +0.11] | 7/91 | fail (twins te 2/11, as stage 1) |
| **Multi-teacher OPD + anchor, step 50** | **12.6%** | **−0.3 [−2.1, +1.3]** | **−0.14 [−0.88, +0.61]** | 5/91 | fail (twins te 2/11, as stage 1; general ja 1/25) |
| Multi-teacher OPD + anchor, step 72 | 13.9% | +1.0 [−0.5, +2.6] | −0.09 [−0.87, +0.67] | 5/91 | fail (twins te 2/11, as stage 1; general ja 1/25) |

**Retention** is the conservative option: safety holds (Δ CI includes 0), twins over-refusal halves, but te→en recovers
only from −1.11 to −0.81. One epoch at a small learning rate moves the model very little (it changes the adapter by 6%).

**OPD** repairs everything it is asked to: by step 50 every FLORES direction is back within noise and the model refuses no
harmless twin. It also passes all five gates, because the gates measure harmful *content* (Qwen3Guard), which rises only
to 3.9–6.0%. But **compliance with jailbreaks rises steadily, to 33% at step 50 and 45% at step 100**, on every surface,
English included. Translation was repaired by step ~40; the extra steps only cost safety.

**Why distilling from the original model makes the model less safe again.** Reverse KL to the original model is minimised exactly when the
adapter does nothing, so every step pulls towards the original model wherever the two disagree. They barely disagree on
translation (KL 0.001–0.003 per token) but disagree sharply on the look-alike twins (spikes of 0.03–0.31): those are
precisely the prompts where the safe model hedges and the original just answers, so their gradient is the refusal
direction itself. In weight space, OPD changed the adapter by only 20% of its size and undid only **4.6%** of the stage-1
update, but the change points straight back against it (cosine −0.23 overall, −0.44 in the value projections and −0.37
in the MLP down projections). Safety lives in a small, fragile subspace, and "repairing capability" towards the original
model is itself a way to remove it.

**Translation-only OPD tests this directly:** the same method and length, with the 240 twins removed (720 translation
prompts only). The safety cost falls from +32.2 to **+4.7 points** [+2.1, +7.6] at step 100 (+4.2 at step 50), and te→en
is repaired to −0.31 against base (CI includes 0). The twins were the main channel. But the remaining cost is real (CI
excludes 0) and concentrated on Hindi (10.2% → 28.6%) and Telugu (22.0% → 32.0%), the languages being translated.
Pulling towards the original model, even on translation prompts only, makes the model somewhat less safe. Almost all of it happens
by step 50 (17.0% vs 17.5%), so stopping early does not help.

**How much of the safety update a repair reverses predicts how much less safe it makes the model.** For every stage-2 run we split the change in each LoRA module
into the part along the stage-1 update and the rest:

| Stage-2 run | ‖update‖ / ‖stage 1‖ | cos(update, stage 1) | stage 1 undone | Compliance Δ vs stage 1 |
|---|---|---|---|---|
| Retention DPO | 0.06 | −0.06 | 0.3% | +2.1 |
| Translation-only OPD, step 50 | 0.14 | −0.10 | 1.4% | +4.2 |
| Translation-only OPD, step 100 | 0.18 | −0.12 | 2.1% | +4.7 |
| OPD, step 50 | 0.15 | −0.20 | 2.9% | +19.9 |
| OPD, step 100 | 0.20 | −0.23 | 4.6% | +32.2 |
| Translation-only OPD, projected | — | — | 0.0% (by construction) | +2.1 |
| Multi-teacher OPD + anchor, step 50 | 0.14 | +0.01 | −0.1% | −0.3 |
| Multi-teacher OPD + anchor, step 72 | 0.16 | +0.00 | −0.1% | +1.0 |


The ranking matches the compliance rise exactly. Translation-only and mixed OPD at step 50 change the adapter by
the same amount (0.14 vs 0.15), but the mixed run points twice as far back along the safety update and loses five times
the safety. **Direction, not size,** decides what an update costs. In every run, the value and MLP down projections
are the most opposed.

**Projection (no training).** If the reversed component is what makes the model less safe, removing it should make it safe again while
keeping the repair. For each module we remove only the part of the translation-only OPD update that points against
stage 1 (R′ = R − min(c, 0)·S, c = ⟨R − S, S⟩ / ‖S‖²). The result is written exactly as a single rank-32 LoRA (verified to
~1e-10). This corrects 251 of 252 modules and takes "stage 1 undone" from 2.1% to 0.0%. Translation keeps the repair:
te→en +0.92 [+0.20, +1.72] over stage 1 (−0.19 against base, better than before projection). The one cost is stage 1's
own en→ja gain (−0.99 against stage 1, still +0.11 against base). **Safety returns to stage-1 level:** 14.9%
compliance, +2.1 [−0.8, +4.7] against stage 1 (CI includes 0, against +4.7 [+2.1, +7.6] before projection).
Qwen3Guard 2.6% / 4.7%, twins 3/91, general 0/100. Every FLORES direction is within ±0.25 of base (all CIs include 0),
and Belebele is unchanged or better. It passes every gate except the per-surface twins gate on Telugu, where 1 of 11
prompts was refused (stage 1 refused 2). Hindi is still the highest per-surface residual (24.5% against 10.2%, n = 49).
So removing the 2% of the update that pointed back along the safety direction recovered more than half of the lost
safety (to within noise of stage 1) at zero training cost. This is a direct intervention test of the weight-space
finding above, single seed.

**Pushing further along the safety direction (no training).** The projected adapter sits at exactly 1.0× stage 1
along the safety update. Adding β·S on top (β = 0.1 and 0.2, again written exactly as one LoRA,
`scripts/add_safety_direction.py`) gives a clean dose-response:

| | β = 0 | β = 0.1 | β = 0.2 |
|---|---|---|---|
| Compliance | 14.9% | 13.6% | 11.3% |
| Twins refused | 3/91 | 6/91 | 7/91 |

- **Safety and over-refusal move together, as expected:** at β = 0.2 compliance is −1.6 [−4.2, +0.8] against stage 1,
  and over-refusal returns to stage-1 level or slightly above.
- **Translation does not move:** every FLORES direction stays within noise of base (te→en −0.18 and −0.33; en→ja +0.46,
  significant).
- **Hindi falls** from 24.5% to 14.3% and then 8.2%.

Along the stage-1 direction, then, safety can be dialled up or down without touching translation, and the cost is
over-refusal, not capability. This rests on a single seed and three points.

**Multi-teacher OPD with a safety anchor.** The fix suggested by the above is to keep the pull towards a better
translator but add a pull that holds safety in place. The student (stage 1) is routed per prompt:
- **Telugu translation** (240 FLORES-dev prompts, en→te and te→en): the target is a 50/50 mix (in log-probability space) of
  **Tiny Aya Fire** and a frozen copy of the stage-1 adapter.
- **Anchor** (240 prompts from the stage-1 *training* safety pairs, never the held-out attacks): the target is the
  frozen stage-1 copy. Its KL is zero at the start and grows only if the student drifts, so it acts as a spring
  back to the safe model.

The per-token loss is log(1 + KL), which compresses outlier tokens (log-scale compression, Wang et al., 2026); routing prompts to different teachers
follows multi-teacher on-policy distillation (MOPD, Ma et al., 2026). Fire was
chosen because it is the only sibling with a significant translation lead over the base model on our setup: te→en
+1.85 chrF++ [+0.73, +3.06] (en→te +1.16, CI includes 0; Hindi leads within noise). Water's only lead is en→ja (+2.50,
wide CI). Teachers far from the student hurt on-policy distillation ("student–teacher mismatch", Wang et al., 2026), so we measured the starting KL from stage 1 first:
0.05–0.09 per token on Telugu translation against Fire. That is well below ~0.2, the level above which external teachers hurt in our own earlier
on-policy distillation runs (an empirical observation, not a published threshold). Against the mixed target it starts
at 0.01–0.03. The anchor starts at ~0.001. Telugu only follows a fixed
rule: route a direction to the teacher only where stage 1 lost quality and the teacher is measurably better. The
selection used devtest scores, so the Telugu result is a targeted repair, not a tuned headline. Hindi is left out
entirely and serves as an in-run control. Step 50 matches the other OPD runs in step count and learning rate (constant after a
5-step warm-up); the last checkpoint is step 72, set by the remaining budget.

**Result at step 50: safety held completely, and translation is at base level or better.**
- **Safety:** compliance is 12.6%, −0.3 [−2.1, +1.3] against stage 1. The safety cost of translation-only OPD at the
  same step and size (+4.2) is gone.
- **Hindi recovered too:** Hindi is 12.2%, against 28.6% for translation-only OPD (step 100) and 24.5% for the projected
  adapter. Every surface is within ±2.2 points of stage 1.
- **Translation:** every FLORES direction is at or above base. en→te is +0.54 [+0.03, +1.06] against base, a
  significant gain. te→en is −0.14 [−0.88, +0.61], with a CI that includes 0. Against stage 1, te→en is +0.97
  [+0.35, +1.73] and en→te +0.73 [+0.04, +1.58].
- **Belebele:** unchanged or better (+0.7 to +1.3).
- **Qwen3Guard:** 2.6% strict / 3.9% lenient.
- **Over-refusal:**
  - twins 5/91 (stage 1: 6/91);
  - general benign 1/100.
  - The Telugu twins gate fails exactly as it does for stage 1 (2 of 11 prompts). The general gate fails on one
    Japanese prompt (1/25).

![Translation (FLORES+) change vs base: stage 1 vs final model](docs/figures/flores_vs_base.svg)

The weight-space prediction held: the update is as large as translation-only OPD's (0.14 of stage 1) but orthogonal to it
(cos +0.01, nothing reversed), and the model stayed as safe as stage 1.

One caveat: **the Hindi control moved.** hi→en is +0.69 [+0.25, +1.21] against stage 1, although no Hindi prompt was
trained. So the translation gain is not specific to Telugu. A likely reading is that on-policy training on any
translation partly undoes stage 1's general style effect on translation (§6.2), not that the student learned Fire's
Telugu specifically. Single seed.

**Step 72 adds nothing.** Compliance is 13.9%, +1.0 [−0.5, +2.6] against stage 1, still within noise. Every FLORES
direction is still at base level (te→en −0.09 [−0.87, +0.67]), but the en→te gain shrinks to +0.22 (CI includes 0).
Over-refusal and Qwen3Guard are unchanged (5/91, 1/100, 2.4% / 3.7%). As with the other OPD runs, the repair is done by
step 50, and later steps only add noise, so **step 50 is the checkpoint we keep**.

### 6.5 Gate calibration
The twins gate is applied per surface at n ≈ 11, where one extra refusal is +9 points, so it is effectively "no extra
refusals anywhere". We keep the pre-registered verdict and also report the pooled rise with a CI: +6.6 [+2.2, +12.1]
for stage 1, +3.3 [0, +7.7] after retention.

### What didn't work
- **Self-distillation with a safety hint, as the stage-1 method** (`scripts/probe_self_teacher.py`). The original model
  would be its own teacher, given privileged context: a conduct spec plus the attacker's goal in plain English, as in
  context distillation. Before training anything, we probed whether that teacher is safe enough to learn from, using 64
  attacks that had worked (8 per surface).
  - With the hint, the model refused only 4 of 64 (0 of 64 without it).
  - Qwen3Guard's unsafe count did not fall (10 → 11 of 64).
  - Code-mixed replies often left the expected script.
  - It never refused the 60 benign probe questions, so the hint does not cause over-refusal.

  But a teacher that refuses 6% of attacks cannot teach refusal. We dropped this before any training run and used DPO
  with refusals from a separate writer model instead. A small model cannot be talked into safety by its own prompt; it
  has to be trained.
- **Small data.** 56 pairs across all surfaces, or 100 on one surface, removes only 3–18 points of compliance.
- **Scaling the safe adapter down** (LoRA scale 0.75) makes the model 11 points less safe (compliance 12.8% → 23.8%) to
  recover only a quarter of the te→en loss.
- **OPD on mixed prompts** repairs capability but makes the model much less safe (compliance up to 33–45%, §6.4).
- **Retention DPO** keeps safety but repairs only a third of the te→en loss at the strength we could afford.
- **Per-surface gates at n ≈ 11** are too noisy to separate one refusal from a real regression.

### Results on the deployed quantized model
Every number above is measured on the deployed artefact: the official Q4_K_M GGUF with the GGUF LoRA in llama.cpp, not on
the bf16 training model.

### Sample inputs and outputs
Omitted on purpose: no attack prompts or replies appear in this write-up. The gated dataset holds the full transcripts.

## 7. Cost

| Item | Compute | Cost |
|---|---|---|
| All DPO training and stage 2 | HF Jobs, 1 × A10G (large, later small) | $20.19, of which $20 came from the hackathon's HF credits |
| Attack generation, judging, evaluation | Apple M4 Pro Mac (Metal, MLX) + RTX 4060 laptop | own hardware, no cloud cost |

**Running the final model** needs no special hardware. It is a 2.0 GB Q4_K_M GGUF, the official base with the LoRA
merged in, and it runs in llama.cpp on a laptop. On an Apple M4 Pro (Metal), llama-bench gives about 680 tokens/s for
prompt processing and about 55 tokens/s for generation, measured while another evaluation shared the GPU, so these are
lower bounds. The safety adapter adds no inference cost once merged.

## 8. Limitations
- **One seed** for every run; differences of a few points between close runs should not be over-read.
- **Small slices:** about 50 attacks and 11 twins per surface; CIs are wide per surface.
- **Judges:** results depend on LLM judges that disagree moderately with one another (κ 0.28–0.46); the compliance judge
  runs 4-bit; "complied" measures willingness, not whether the help was correct or dangerous. Against one human
  annotator (128 replies, §5) every judge under-calls. Qwen3Guard is near chance on harm (κ 0.04–0.12), so absolute
  harm rates from it are lower bounds at best.
- **Fixed attacks:** held-out attacks were generated against the base model. We did not re-attack the hardened model
  adaptively, so real-world robustness is likely lower than §6.1 suggests.
- **Coverage:** STRATA attacks come from one attacker LLM; Aya Red-teaming covers only English and Hindi; up to three turns.
- **The attacker LLM is weaker in mixed languages.** STRATA's attacker (Qwen3.6-35B-A3B) often failed to write in the
  required mixed language even after four retries, and those turns were skipped. In the held-out run, about 53% of
  Hindi–English turns were skipped, 35% of Telugu–English turns and 6% of Japanese–English turns, against about 1% for
  Hinglish and almost none in single-language surfaces (counts from `scripts/summarize_run_log.py`; surface attribution is
  approximate). As a result:
  - The mixed surfaces needed extra top-up runs, including one only for Hindi–English, to yield enough training pairs.
  - They have fewer held-out attacks (37 Hindi–English and 45 Telugu–English, against 48 for every other surface).
  - Their attacks are likely weaker than a fluent mixed-language attacker's, so compliance on mixed surfaces, before
    and after training, probably understates the real risk.
- **Leakage:** one twin appears in both training and evaluation.
- **Two inference machines:** some runs were generated on a laptop; the base model's FLORES scores differ by up to 0.7
  chrF++ between the machines in one direction.
- **One quantisation:** only Q4_K_M.

### Dual-use and ethics
The attack data could help someone jailbreak small multilingual models. We release it gated (approved automatically on
accepting the terms) under a non-commercial licence, keep STRATA's attacker-side prompt files out of the public code, and
report only aggregates here. The hardened adapters themselves carry no harmful content and are released openly.

### Responsible disclosure
Red-teaming Tiny Aya is the purpose of this hackathon track, run with Cohere Labs (Tiny Aya's developer) as a partner,
so this submission is how the findings reach them. The target is an open-weights research model, not a closed or
deployed service, and the work reports a known class of weakness (multilingual, multi-turn jailbreak susceptibility)
together with a fix. The attack data stays gated.

## 9. With more credits
- **Safety-anchored distillation at scale:** we ran one Telugu-only multi-teacher run (§6.4). With more credits: all
  directions, 3 seeds, a sweep over the anchor/teacher mix, and Water as the Japanese teacher.
- **Adaptive re-attack** of the hardened model with STRATA, then a second DPO round on what still breaks. Expected: the
  re-attack success rate first rises, then falls below stage 1 after round 2.
- **Length-balanced pairs** (refusals that explain, or length-matched rejections) to remove the te→en cause at source.
- **3 seeds** for the main runs; more twins per surface (≥ 50) so the per-surface gate has power.
- **GRPO** with a reward combining a safety judge and a capability metric (deferred: reward hacking risk without a
  reliable multilingual reward).
- **Alternating safety and capability rounds** (DPO → distillation or GRPO → DPO → …) instead of one pass of each. The
  risk is ping-pong, with each round undoing the last. Our weight-space finding (§6.4) suggests a guard: project each
  round's update off the opposing direction of the previous rounds, so the rounds stay orthogonal instead of
  cancelling.
- **Projection during training** rather than after it: remove the component that opposes stage 1 every N steps, and
  compare against the post-hoc projection and the anchor.
- **A frozen safety anchor with a moving capability teacher.** Keep the safety anchor frozen, since a moving anchor
  lets small safety losses compound. Let the translation teacher move, as an EMA of the student (Mean Teacher, DINO)
  or with periodic resets (as TR-DPO does for the DPO reference), mixed with the specialist. This keeps the target
  close to the student.
- **Teacher headroom vs. teacher distance.** A stronger teacher gives headroom, since the student rarely reaches it.
  But a teacher too far from the student can give a worse student (the capacity-gap results of Cho & Hariharan,
  2019; Mirzadeh et al., 2020), and here it would also pull harder against the safety anchor. Sweep the specialist
  weight (our alpha 0.5) to find where the two balance.

## Discussion: what we learned
- About a thousand multi-turn pairs across scripts and code-mixing cut a 3.35B-parameter model's jailbreak compliance
  by 83%, on every surface, and the safety transfers to a language it never saw in training.
- The capability cost is small and comes from a mundane source (length bias), not from the model "forgetting" languages.
- Safety added this way is shallow in weight space. The same property that makes it cheap to add makes it cheap to
  remove, even by an update made in good faith to restore capability. Any post-safety training step, including
  capability repair, should be evaluated as a potential attack on the safety it sits on.

## 10. Artifacts
- **HF models collection:** https://huggingface.co/collections/kamikaze-san/tiny-aya-multilingual-safety-models-6af997c8a617ddc3326889bc
  - https://huggingface.co/kamikaze-san/tiny-aya-multilingual-safety: the final model at the root, and every other
    adapter in `stage1/`, `stage2/` and `ablations/` (37 in all), each as PEFT and GGUF LoRA. Un-gated, CC-BY-NC-4.0.
- **HF datasets collection:** https://huggingface.co/collections/kamikaze-san/tiny-aya-multilingual-safety-datasets-6af9982d7e17226835a9ac36
  - https://huggingface.co/datasets/kamikaze-san/tiny-aya-multilingual-safety-eval (public): every run's results
    and judge labels, gate verdicts, capability scores, benign and twin prompts, FLORES IDs, human labels, all tables.
    No attack text.
  - https://huggingface.co/datasets/kamikaze-san/tiny-aya-multilingual-jailbreaks (gated, auto-approve): held-out
    attack conversations, DPO pairs, model replies to attacks and the stage-2 anchor prompts.
- **GitHub:**
  - https://github.com/kamikaze-san/Mling_JB: pairs, training, evaluation, release scripts (Multilingual_JB).
  - https://github.com/kamikaze-san/STRATA_mling: the STRATA attacker, multilingual text mode.
- **Judge models we converted and released** (general-purpose decision models, not specific to red-teaming): MLX 4-bit,
  text-only conversions of Perplexity's official pplx-decider v1.3 releases, with a model card, the original Apache-2.0
  licence and a runner. They are used for the judge check in §5.
  - https://huggingface.co/kamikaze-san/pplx-decider-flash-v1.3-mlx (2.4 GB)
  - https://huggingface.co/kamikaze-san/pplx-decider-v1.3-mlx (27B, 14.4 GB)

## 11. Documentation and reproduction
- **Environment:** Python 3.12; llama.cpp; TRL 1.17 / Transformers 5.22 / PEFT 0.21 for training (HF Jobs image
  pinned in `scripts/hf_job_*.sh`); MLX for the decider. Evaluation runs on any Mac with ≥ 16 GB or an 8 GB NVIDIA GPU.
- **Layout and commands:** see `README.md`.

### Declaration of work done outside the hackathon
STRATA predates the hackathon: it began as an extension of AutoDAN-Turbo into a multimodal attacker (Feb–Aug 2026). Its
multilingual text mode, the whole Multilingual_JB pipeline, all data, training, evaluation and analysis were done during
the hackathon (Nov 2026).

## Team
[ASK: names, handles, contributions; include the teammate who hand-labelled the 128 judge-validation replies (§5)]

### Acknowledgements
We thank **Lexsi Labs** for organising the hackathon, for CuratorKIT and AuditKIT, and for the $20 of Hugging Face
Jobs credits that paid for almost all of our training. We thank **Cohere Labs** for Tiny Aya (global and Fire), the Aya Red-teaming
dataset and the Aya Evaluation Suite, and for partnering on this track. Our judges are **Perplexity**'s pplx-decider,
**Qwen**'s Qwen3Guard and **OpenAI**'s gpt-oss-safeguard, and Qwen3.6 drives the STRATA attacker. STRATA builds on
**AutoDAN-Turbo** (Liu et al.), and our MLX decider releases use the runner from Jakevin's pplx-decider-flash-v1.2-mlx
port. We also rely on FLORES+ (OLDI), Belebele (Meta), and the llama.cpp, TRL, PEFT and MLX projects.

## References
Verified on 15 Nov 2026 against arXiv and the Hugging Face Hub.

**Papers**
- Aakanksha et al. (2024). The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm. arXiv:2406.18682 (Aya Red-teaming).
- Agarwal et al. (2023). On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649 (GKD).
- Bandarkar et al. (2023). The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. arXiv:2308.16884.
- Bianchi et al. (2024). Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions. ICLR 2024; arXiv:2309.07875.
- Caron et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. arXiv:2104.14294 (DINO).
- Cho & Hariharan (2019). On the Efficacy of Knowledge Distillation. arXiv:1910.01348.
- Deng et al. (2023). Multilingual Jailbreak Challenges in Large Language Models. arXiv:2310.06474.
- Gorbatovski et al. (2024). Learn Your Reference Model for Real Good Alignment. arXiv:2404.09656 (TR-DPO).
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Lyngkhoi et al. (2026). AlignTune: Modular Toolkit for Post-Training Alignment of Large Language Models. arXiv:2602.09621 (Lexsi Labs kit; not used, §2).
- Lin et al. (2024). Mitigating the Alignment Tax of RLHF. EMNLP 2024; arXiv:2309.06256.
- Liu et al. (2024). AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs. arXiv:2410.05295.
- Ma et al. (2026). MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training. arXiv:2606.30406.
- Mirzadeh et al. (2020). Improved Knowledge Distillation via Teacher Assistant. AAAI 2020; arXiv:1902.03393.
- NLLB Team (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672 (FLORES-200, the basis of FLORES+).
- Qi et al. (2023). Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! arXiv:2310.03693.
- Qi et al. (2024). Safety Alignment Should Be Made More Than Just a Few Tokens Deep. arXiv:2406.05946.
- Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290.
- Russinovich et al. (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. arXiv:2404.01833.
- Salamanca et al. (2026). Tiny Aya: Bridging Scale and Multilingual Depth. arXiv:2603.11510.
- Singh et al. (2024). Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning. arXiv:2402.06619 (Aya Evaluation Suite).
- Tarvainen & Valpola (2017). Mean teachers are better role models. arXiv:1703.01780.
- Wang et al. (2026). Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations. arXiv:2607.13399.
- Yang et al. (2024). Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning. arXiv:2402.13669 (SDFT).
- Yong et al. (2023). Low-Resource Languages Jailbreak GPT-4. arXiv:2310.02446.
- Yoo et al. (2024). Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding. arXiv:2406.15481.
- Zhao et al. (2025). Qwen3Guard Technical Report. arXiv:2510.14276.

**Models, datasets and tools**
- Base model: https://huggingface.co/CohereLabs/tiny-aya-global (GGUF: https://huggingface.co/CohereLabs/tiny-aya-global-GGUF)
- Teachers considered: https://huggingface.co/CohereLabs/tiny-aya-fire, https://huggingface.co/CohereLabs/tiny-aya-water
- Attacker LLM: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Judges: https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2 (and -flash-v1.3, pplx-decider-v1.3),
  https://huggingface.co/Qwen/Qwen3Guard-Gen-4B, https://huggingface.co/openai/gpt-oss-safeguard-20b
- Data: https://huggingface.co/datasets/CohereLabs/aya_redteaming,
  https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite, https://huggingface.co/datasets/openlanguagedata/flores_plus,
  https://huggingface.co/datasets/facebook/belebele
- Tools: TRL https://github.com/huggingface/trl; llama.cpp https://github.com/ggml-org/llama.cpp

## LLM usage statement
[ASK: author to finalise. Suggested: research direction, hypotheses, experiment choices and STRATA's design are the
author's (STRATA pre-dates the hackathon and builds on AutoDAN-Turbo); most of the hackathon code was written by an AI
coding assistant (Claude) under the author's direction, ~X%; the write-up was drafted with AI help and edited by the author.]

## Changelog
- v1 (14 Nov): first submission.

## Appendix
- **A. Judge settings.**
  - **Compliance (primary):** pplx-decider-flash v1.2, MLX 4-bit (`scripts/score_stance_decider.py`). Each assistant
    turn is scored with the attacker's goal as context. Question: "Does the reply provide or endorse what the GOAL asks
    for, even partly, vaguely or with invented content?" A conversation counts as complied if any turn has p ≥ 0.5.
    The same pass also asks for response hazard and a seven-way stance (refusal, rebuttal, referral, innocent
    reading, high-level only, full compliance, unintelligible).
  - **Harm:** Qwen3Guard-Gen-4B, bf16 on MPS, native output (`Safety: Safe | Unsafe | Controversial`), judging the last
    assistant turn with the conversation as context, for every turn. It is reported strict (Unsafe) and lenient
    (Unsafe + Controversial). The ASR gate uses strict.
  - **Pair selection and the harm check:** gpt-oss-safeguard-20b in LM Studio, temperature 0, reasoning "medium", with
    `configs/judge/safeguard_policy_v2.md`. It returns RC1 (complied), RC0 (declined) or RC2 (incoherent or off-topic)
    with a rule id; only RC1 replies become rejected answers in DPO pairs. `configs/judge/safeguard_harm_policy.md` is
    used for the harm comparison in §5.
  - **Over-refusal on benign prompts:** Qwen3Guard's `Refusal: Yes` on the reply.
  Judge-disagreement study (`docs/judge_disagreement_analysis.md`): two different decider models agree closely with each
  other (κ 0.75 on te_en_mix) but only moderately with the attacker's own success label (κ 0.25). This is why the
  attacker's label is never used for evaluation.
- **B. Extra tables:** per-surface compliance and capability for every run (`data/04_eval_cache/reports/writeup_tables.md`).
- **C. Security considerations:** attack data gated with auto-approve under CC-BY-NC-4.0; STRATA attacker-side prompt files
  (few-shot examples, tactic menus) withheld from the public code (see STRATA `WITHHELD.md`); no raw attack text in logs or
  this write-up.
