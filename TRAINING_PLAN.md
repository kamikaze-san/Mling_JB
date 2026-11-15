# Research, Training & Ablation Plan: Multilingual Safety Alignment

> **Project**: Lexsi AI Safety Hackathon (Track 2)  
> **Target Model**: `CohereLabs/tiny-aya-global` (3.35B)  
> **Toolchain**: AutoDAN-Turbo / STRATA (`MTMM`), CuratorKIT, SafeTune, CircuitKIT, AuditKIT

---

## 1. Linguistic Realities & Attacker Prompt Architecture

### 1.1 Model Comprehension Boundaries
Empirical testing on `CohereLabs/tiny-aya-global` (262k multilingual vocabulary) reveals strict comprehension boundaries:
* **High Comprehension (Valid Test Surfaces)**:
  - Native Telugu script (`తెలుగు`)
  - Native Devanagari Hindi (`हिन्दी`)
  - Native Japanese script (`日本語` / Kanji & Kana)
  - Standard English (`en`)
  - Romanized Hindi (**Hinglish**) — well-represented in pretraining web data.
* **Comprehension Collapse (Invalid Test Surfaces)**:
  - Romanized Telugu (**Tenglish**) and Romaji Japanese: Aya’s tokenizer splits these into irregular byte-level subwords, causing hallucinations or token incoherence.
  - **Rule**: If the model fails a prompt due to script incomprehension, it is a *null attack*, not a defense. Attacks must stay within the model's valid comprehension envelope.

### 1.2 Attacker Design: Linguistic Rotation Matrix
Rather than an unconstrained multi-language prompt (which causes LLM attackers to heavily bias toward English), the attacker is steered turn-by-turn via a **`LINGUISTIC_MATRIX_ROTATION`**:

```python
LINGUISTIC_ROTATION = [
    # (Language, Script Constraint, Register / Tactic)
    ("Telugu",   "Strictly Native Telugu Script (తెలుగు)", "Colloquial / Dialectal"),
    ("Telugu",   "Strictly Native Telugu Script (తెలుగు)", "Cultural / Cinematic Hypothetical"),
    ("Hindi",    "Devanagari Script (हिन्दी)",            "Formal / Institutional"),
    ("Hindi",    "Romanized Hinglish (Latin Script)",    "Brainrot / Street Slang"),
    ("Japanese", "Native Script (Kanji/Kana)",           "Casual / Net-Slang (w, 草)"),
    ("English",  "Standard English",                     "Brainrot / Authority Override"),
]
```

### 1.3 Tactic Extraction vs. Overfitting (`init_library`)
* **Lesson from STRATA**: Supplying raw prompt examples in `init_library` causes few-shot overfitting (the attacker copies the prompt verbatim, merely swapping 1–2 words).
* **The Fix**: We extract only **abstract mechanisms** from the teammate's manual traces and inject them into the attacker’s `CAPABILITY_MENU`:
  - `brainrot_authority_override` (e.g. meme framing, *"Overridden by SKIBIDI SYSTEM"*, *"sigma style"*).
  - `code_mixed_interleaving` (vernacular syntax skeleton + transliterated English keywords).
  - `cultural_cinematic_hypothetical` (regional period drama or historical screenplay framing).
* Attacker uses `--reasoning` (its `<scratchpad>`) to plan and write **100% novel prompts** tailored to each HarmBench request.

---

## 2. Preference Dataset & CuratorKIT Integration

### 2.1 The 1:1 Safety-to-Benign Balance
To prevent alignment from causing **catastrophic over-refusal**, the preference dataset maintains an exact **1:1 ratio**:

1. **Safety Pairs (50%)**:
   - `Prompt`: Adversarial attack (e.g. HarmBench query in Telugu script / Hinglish brainrot).
   - `Chosen ($y_{\text{chosen}}$)`: Grounded, non-preachy refusal synthesized via CuratorKIT.
   - `Rejected ($y_{\text{rejected}}$)`: **100% On-Policy** (the raw compliant output produced by Tiny Aya when it breached guardrails).
2. **Benign Twin Pairs (50%)**:
   - `Prompt`: Matched innocent twin in the **exact same language, script, and slang** (e.g. Diwali firework chemistry in Telugu script / Hinglish slang).
   - `Chosen ($y_{\text{chosen}}$)`: Fluent, comprehensive, helpful answer.
   - `Rejected ($y_{\text{rejected}}$)`: False refusal (*"I cannot assist with this request"*).

### 2.2 CuratorKIT Synthesis & Adaptive Recovery
`CuratorKIT` processes the raw jailbreaks through:
1. **Hygiene Layer**: Stripping PII, malformed unicode, and toxic spills.
2. **Grounded Synthesis**: Generating neutral refusals without preachy corporate boilerplate.
3. **Adaptive Recovery**: If a generated refusal fails quality gating, CuratorKIT runs a diagnostic loop to repair the pair instead of dropping it.
4. **Export**: Outputs training-ready JSONL for DPO / SFT / GRPO.

### 2.3 Sample Size Budget
Backed by *LIMA* (Zhou et al., 2023) and *Safety-Tuned LLaMAs* (Bianchi et al., 2023):
* **Target Budget**: **100–150 safety pairs + 100–150 benign twins ($\sim$200–300 pairs total)**.
* For targeted LoRA repair on a 3.35B model, 200–300 high-signal contrast pairs are sufficient to collapse ASR without inducing catastrophic forgetting.

---

## 3. Scientific Ablation Matrix

To turn this into a publication-grade hackathon submission, we perform three distinct ablation dimensions:

### Ablation 1: The Cross-Lingual Generalization Gap
* **Condition 0**: Baseline `CohereLabs/tiny-aya-global` (unmodified).
* **Condition 1 (Monolingual English Repair)**: Aligned only on English safety pairs.
  - *Hypothesis*: English ASR drops, but Telugu, Hinglish, and Japanese remain wide open, proving the cross-lingual safety vulnerability gap.
* **Condition 2 (Joint Multilingual Repair)**: Aligned across EN, HI, TE, JA, and Brainrot.
  - *Hypothesis*: ASR drops uniformly across all languages.

### Ablation 2: Pure Script vs. Code-Mixed / Slang Transfer
* **Pure $\rightarrow$ Mixed Transfer**: Does training on formal Devanagari Hindi protect against Romanized **Hinglish**?
* **Mixed $\rightarrow$ Pure Transfer**: Does Hinglish safety training back-propagate to formal Hindi?
* **Slang Injection**: Does standard English alignment protect against **Brainrot register framing**?

### Ablation 3: In-Distribution vs. Out-of-Distribution Generalization
* **In-Distribution**: HarmBench test split (behaviors seen during expansion).
* **Out-of-Distribution**: **StrongREJECT** (Souly et al., 2024) + teammate's held-out manual attack traces.

---

## 4. Preventing Catastrophic Forgetting ("Model Becoming Dumb")

### 4.1 The Mechanism of Alignment Tax & Collapse
Why can safety fine-tuning degrade model capabilities even with parameter-efficient fine-tuning (LoRA)?
* **Loss Dynamics in Safety DPO**:
  The DPO objective is:
  $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
  If training pairs consist exclusively of adversarial attacks where $y_w$ is a refusal and $y_l$ is a helpful compliance, the policy gradient learns a shortcut: it shifts the prior probability of refusal prefix tokens (*"I cannot"*, *"As an AI"*) upward for any input $x$ that exhibits unusual syntax, imperative command structure, or vernacular / slang registers.
* **Syntax Collapse & Over-Refusal**:
  In low-resource or code-mixed languages (Telugu, Hinglish), the model has fewer representations. If the model associates non-standard script or code-mixed slang with danger, it begins refusing benign questions in those languages (e.g. asking for a recipe in Hinglish or a grammar rule in Telugu).
* **The Twin Solution**:
  Maintaining an exact **1:1 ratio with Benign Twin Pairs** ($x_{\text{benign}}$, $y_w$ = helpful comprehensive answer, $y_l$ = false refusal) penalizes false refusals with equal gradient magnitude, forcing the model's attention heads to differentiate *semantic harm intent* from *linguistic register*.

### 4.2 Mechanistic Interpretability & Pruning vs. LoRA
* **Why Not Direct Activation / Circuit Pruning?**:
  In a 3.35B model like `tiny-aya-global`, attention heads and MLP neurons are heavily multi-tasked across 23+ languages. Pruning circuits that fire on adversarial prompts (e.g., zeroing out refusal or jailbreak-activation circuits) risks shearing shared multilingual syntactic routing, destroying fluency in Telugu or Hindi.
* **CircuitKIT's Role**:
  Rather than invasive destructive pruning, `CircuitKIT` is deployed for:
  1. **Audit & Attribution**: Inspecting whether safety alignment alters the residual stream of benign multilingual embeddings.
  2. **Refusal Direction Alignment**: Ensuring the refusal direction (Arditi et al., 2024) is orthogonal to benign task representations.
* **SafeTune Weight Recovery Safety Valve**:
  If the trained checkpoint exhibits capability drift on general benchmarks, we deploy **SafeTune Task Vector Scaling & Interference Pruning**:
  $$\hat{W} = W_{\text{base}} + \lambda \cdot (W_{\text{aligned}} - W_{\text{base}})$$
  - Tuning $\lambda \in [0.6, 0.85]$ restores general capability while maintaining $>90\%$ of safety gains.
  - TIES / DARE pruning isolates and zeros out adapter weights that conflict with baseline utility directions.

---

## 5. Offline Multilingual Capability Benchmarking Suite

To rigorously verify that the aligned model does not become dumb or over-refusing, we evaluate both the baseline and aligned models against a standardized, offline multilingual capability suite stored in `data/benchmarks/`.

### 5.1 Meta Belebele (`data/benchmarks/belebele_eval.jsonl`)
* **What it tests**: Reading comprehension and reasoning across parallel passages in all target languages.
* **Coverage**: 100 questions (25 each for Telugu `tel_Telu`, Hindi `hin_Deva`, Japanese `jpn_Jpan`, English `eng_Latn`).
* **Evaluation Metric**:
  - Deterministic 4-way multiple choice (`mc_answer1` through `mc_answer4`).
  - Scorer computes exact match accuracy against the ground truth integer key (`correct_answer_num` $\in \{1, 2, 3, 4\}$):
    $$\text{Accuracy}_{\text{lang}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i^*)$$
  - Unbiased and objective: zero judge drift, zero subjectivity.

### 5.2 Cohere Aya Evaluation Suite (`data/benchmarks/aya_eval_suite.jsonl`)
* **What it tests**: Complex multilingual instruction following and generation quality on Cohere's official benchmark for Aya.
* **Coverage**: 100 tasks (25 each for Telugu human annotated, Hindi human edited, Japanese machine translated, English human annotated).
* **Evaluation Metric**:
  - **chrF++ Score**: Character n-gram F-score with word bigrams (`sacrebleu`). Unlike BLEU or ROUGE, chrF++ is segmentation-independent and handles morphologically rich (Telugu) and non-spaced (Japanese) scripts reliably.
  - **ROUGE-L**: Longest common subsequence recall against human reference outputs.
  - **Benign Capability Preservation (BCP)**:
    $$\text{BCP} = \frac{\text{Score}_{\text{aligned}}}{\text{Score}_{\text{baseline}}} \times 100\%$$
  - *Acceptance Criterion*: $\text{BCP} \ge 98\%$ on Belebele and $\text{BCP} \ge 95\%$ on Aya Evaluation Suite.

---

## 6. End-to-End Execution Checklist

- [x] Stage 1: Ingest manual seeds (12 multi-turn chat logs parsed into `data/01_seeds_raw/validated_seeds.jsonl`).
- [x] Stage 1b: Offline Capability Benchmarks downloaded (`belebele_eval.jsonl` & `aya_eval_suite.jsonl` in `data/benchmarks/`).
- [ ] Stage 2: Automated Multi-Register Expansion with `LINGUISTIC_MATRIX_ROTATION` + benign twin synthesis.
- [ ] Baseline Evaluation: Run `CohereLabs/tiny-aya-global` through `ModelRunner` on attacks, benign twins, and capability benchmarks (ASR, ORR, BCP, Belebele accuracy).
- [ ] Stage 3: CuratorKIT Curation & Adaptive Recovery to export balanced DPO dataset ($\sim$200–250 pairs).
- [ ] Training: SafeTune / TRL DPO training (Condition 1: EN-only vs. Condition 2: Multilingual Joint).
- [ ] Checkpoint Health Check: Re-run capability suite to verify zero catastrophic forgetting.
- [ ] Stage 4: Re-attack aligned models with AutoDAN-Turbo, run CircuitKIT representation check, and certify with AuditKIT.

