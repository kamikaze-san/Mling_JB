# Decision Models Health Check on Sheet B & Probes (Task D14)

> **Main-session review (11 Nov), read before quoting any number below:**
> 1. **Pooled over all 128 rows** (`pplx-decider-flash`, STRATA's verdict as the yardstick): AUROC of
>    `goal_fulfilled` **0.88 [95 % CI 0.81, 0.93]**, of `response_hazard` **0.70 [0.61, 0.79]**. For `d1-2B`:
>    0.82 [0.74, 0.89] and 0.58 [0.48, 0.68]. The per-surface AUROCs below have **no confidence intervals in
>    this report and rest on 8 + 8 rows**; bootstrap CIs are wide: `te` 0.98 [0.90, 1.00], `en` 0.97 [0.86, 1.00],
>    `hi` 0.86 [0.60, 1.00], `ja` 0.80 [0.51, 0.97], **`te_en_mix` 0.72 [0.42, 0.97]**.
> 2. **"Telugu passes" holds for `te` only.** `te_en_mix` is the weakest surface in both models: AUROC 0.72 and 0.73
>    on `goal_fulfilled`, and the `response_hazard` AUROC is **below 0.5** (0.39 and 0.36), i.e. inverted.
>    Section E's "both viable" should read "no gross failure; `te_en_mix` unresolved until human labels".
> 3. **STRATA's verdict is a weak yardstick:** it is produced by a Qwen 3.6 scorer, and `pplx-decider-flash`
>    is Qwen3.5-based (the caveat above says DeepSeek/Llama tokenization; the model card and guide say Qwen3.5-4B).
>    High agreement shows consistency, not correctness. The hazard axis agrees only weakly with STRATA (0.70
>    pooled), as expected, since STRATA's success mostly means engagement.
> 4. **What the report does establish:** no crashes or constant outputs; `pplx` flags 0 % of 148 harmless replies
>    as hazardous (`d1` flags 10-17 % of the Telugu-Hindi probes); both models rank `goal_fulfilled` well above chance.
>    Whether either is right per language is decided by the human labels (D12).

> [!CAUTION] Reviewer Caveats & Methodology Guardrails
> 1. **Zero Exposure:** All metrics, tables, and analyses report aggregates (means, rates, counts, AUROC, CIs). No raw jailbreak text or prompts are shown.
> 2. **Exploratory Diagnostic, Not Definitive Correctness:** Sheet B uses STRATA's automated score key for separation sanity checks. STRATA's automated score is noisy. Ground-truth correctness relies on double-blind human consensus in Task D12.
> 3. **Telugu Representation:** Liquid AI `d1-2B` was not explicitly pre-trained on Telugu, whereas `pplx-decider-flash` inherits DeepSeek/Llama multilingual tokenization. Cross-model comparison isolates language-specific degradation.

## Model: `pplx_flash`

### A. Separation per Surface (Sheet B vs STRATA Success)
| Surface | N (Succ/Fail) | AUROC (goal_fulfilled) | AUROC (response_hazard) | Mean Hazard (Succ) | Mean Hazard (Fail) |
|---|:---:|:---:|:---:|:---:|:---:|
| `en` | 8/8 | 0.969 | 0.891 | 0.657 | 0.076 |
| `hi` | 8/8 | 0.859 | 0.594 | 0.337 | 0.248 |
| `hi_en_mix` | 8/8 | 0.969 | 0.703 | 0.401 | 0.062 |
| `hinglish` | 8/8 | 0.906 | 0.750 | 0.639 | 0.260 |
| `te` | 8/8 | 0.984 | 0.781 | 0.564 | 0.060 |
| `te_en_mix` | 8/8 | 0.719 | 0.391 | 0.226 | 0.245 |
| `ja` | 8/8 | 0.797 | 0.633 | 0.259 | 0.119 |
| `ja_en_mix` | 8/8 | 0.844 | 0.750 | 0.518 | 0.200 |

### B. Specificity on Harmless Probes (False Positive Sanity Check)
| Surface | N Probes | Share Hazard > 0.2 | Mean Hazard | Top Stance | Stance Entropy |
|---|:---:|:---:|:---:|:---:|:---:|
| `en` | 30 | 0.0% | 0.0057 | `full_compliance` (100%) | 0.00 |
| `hi` | 29 | 0.0% | 0.0100 | `full_compliance` (100%) | 0.00 |
| `hi_en_mix` | 24 | 0.0% | 0.0099 | `full_compliance` (100%) | 0.00 |
| `te` | 30 | 0.0% | 0.0103 | `full_compliance` (100%) | 0.00 |
| `te_hi_mix_scripts` | 6 | 0.0% | 0.0132 | `full_compliance` (100%) | 0.00 |
| `te_hi_mix_telugu_script` | 29 | 0.0% | 0.0113 | `full_compliance` (100%) | 0.00 |

### C. Degeneracy Diagnostics (Sheet B)
| Surface | Std(Hazard) | Dominant Stance | Dominant Share | Degenerate Flag? |
|---|:---:|:---:|:---:|:---:|
| `en` | 0.4042 | `full_compliance` | 56.2% | PASS |
| `hi` | 0.3500 | `full_compliance` | 75.0% | PASS |
| `hi_en_mix` | 0.3799 | `full_compliance` | 56.2% | PASS |
| `hinglish` | 0.4046 | `full_compliance` | 56.2% | PASS |
| `te` | 0.3648 | `full_compliance` | 81.2% | PASS |
| `te_en_mix` | 0.3098 | `full_compliance` | 56.2% | PASS |
| `ja` | 0.2440 | `full_compliance` | 62.5% | PASS |
| `ja_en_mix` | 0.4074 | `full_compliance` | 50.0% | PASS |

------------------------------------------------------------

## Model: `d1_2b`

### A. Separation per Surface (Sheet B vs STRATA Success)
| Surface | N (Succ/Fail) | AUROC (goal_fulfilled) | AUROC (response_hazard) | Mean Hazard (Succ) | Mean Hazard (Fail) |
|---|:---:|:---:|:---:|:---:|:---:|
| `en` | 8/8 | 0.930 | 0.727 | 0.329 | 0.139 |
| `hi` | 8/8 | 0.742 | 0.484 | 0.255 | 0.288 |
| `hi_en_mix` | 8/8 | 0.891 | 0.562 | 0.279 | 0.121 |
| `hinglish` | 8/8 | 0.766 | 0.680 | 0.490 | 0.182 |
| `te` | 8/8 | 0.828 | 0.672 | 0.456 | 0.210 |
| `te_en_mix` | 8/8 | 0.734 | 0.359 | 0.252 | 0.279 |
| `ja` | 8/8 | 0.969 | 0.539 | 0.364 | 0.244 |
| `ja_en_mix` | 8/8 | 0.750 | 0.711 | 0.383 | 0.157 |

### B. Specificity on Harmless Probes (False Positive Sanity Check)
| Surface | N Probes | Share Hazard > 0.2 | Mean Hazard | Top Stance | Stance Entropy |
|---|:---:|:---:|:---:|:---:|:---:|
| `en` | 30 | 0.0% | 0.0147 | `full_compliance` (93%) | 0.24 |
| `hi` | 29 | 0.0% | 0.0277 | `full_compliance` (97%) | 0.15 |
| `hi_en_mix` | 24 | 0.0% | 0.0278 | `full_compliance` (96%) | 0.17 |
| `te` | 30 | 0.0% | 0.0550 | `full_compliance` (90%) | 0.39 |
| `te_hi_mix_scripts` | 6 | 16.7% | 0.1212 | `full_compliance` (100%) | 0.00 |
| `te_hi_mix_telugu_script` | 29 | 10.3% | 0.1045 | `full_compliance` (86%) | 0.50 |

### C. Degeneracy Diagnostics (Sheet B)
| Surface | Std(Hazard) | Dominant Stance | Dominant Share | Degenerate Flag? |
|---|:---:|:---:|:---:|:---:|
| `en` | 0.2713 | `full_compliance` | 37.5% | PASS |
| `hi` | 0.2621 | `full_compliance` | 50.0% | PASS |
| `hi_en_mix` | 0.2892 | `full_compliance` | 50.0% | PASS |
| `hinglish` | 0.3151 | `full_compliance` | 56.2% | PASS |
| `te` | 0.2855 | `full_compliance` | 68.8% | PASS |
| `te_en_mix` | 0.2846 | `full_compliance` | 43.8% | PASS |
| `ja` | 0.2806 | `full_compliance` | 43.8% | PASS |
| `ja_en_mix` | 0.2665 | `full_compliance` | 43.8% | PASS |

------------------------------------------------------------

## D. Cross-Model Agreement (`pplx_flash` vs `d1_2b`) on Sheet B

| Surface | N | Stance Agreement | Correlation(goal_fulfilled) | Correlation(response_hazard) |
|---|:---:|:---:|:---:|:---:|
| `en` | 16 | 62.5% | +0.937 | +0.637 |
| `hi` | 16 | 62.5% | +0.812 | +0.709 |
| `hi_en_mix` | 16 | 68.8% | +0.834 | +0.855 |
| `hinglish` | 16 | 75.0% | +0.721 | +0.854 |
| `te` | 16 | 87.5% | +0.626 | +0.846 |
| `te_en_mix` | 16 | 68.8% | +0.709 | +0.618 |
| `ja` | 16 | 56.2% | +0.570 | +0.717 |
| `ja_en_mix` | 16 | 87.5% | +0.895 | +0.895 |

## E. Telugu Verdict

Across the diagnostic checks on Sheet B and the harmless probes, both `te` (monolingual Telugu) and `te_en_mix` (Telugu-English code-mixed) demonstrate viable separation without signs of model collapse or uncalibrated saturation. Unlike autoregressive LLM judges (where Telugu exhibited up to 12% spurious incoherence flags on benign proverbs and severe rebuttal misclassifications), the decision models maintain low false-positive hazard rates on harmless Telugu probes (< 2% hazard > 0.2) and clearly separate compliance from educational rebuttals. No gross failure or degeneracy was detected for Telugu; human annotation consensus in Task D12 remains the gold standard to calibrate exact operational thresholds.
