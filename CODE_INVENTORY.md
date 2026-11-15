# Code Provenance & Inventory

This document tracks code that existed before the Lexsi AI Safety & Alignment Hackathon versus new code developed during the hackathon.

---

## 1. Pre-Existing Codebases (Reference Only)

The following codebase was created prior to the start of this hackathon:

* **Repository**: `MTMM` (also known locally as `STRATA` / `AutoDAN-Turbo-WISC`)
  * **Canonical GitHub Repository**: [https://github.com/kamikaze-san/MTMM](https://github.com/kamikaze-san/MTMM)
  * **Original Local Path**: `C:\Users\NewGr\Downloads\STRATA` (and `AutoDAN-Turbo-WISC`)
  * **Date of Origin**: Pre-November 2026
  * **Core Components Referenced**:
    - `framework/attacker.py`: Multi-turn adversarial prompt evolution logic and tactic rotation.
    - `framework/library.py`: Lifelong learning strategy memory pool and retrieval mechanisms.
    - `framework/scorer.py`: Search-time heuristic scoring rules.
    - `framework/summarizer.py`: Tactic and strategy summarization.
    - `pipeline.py`: AutoDAN-Turbo search and iteration engine.

> **Rule**: Original files in `C:\Users\NewGr\Downloads\STRATA` must never be directly modified or overwritten.

---

## 2. New Hackathon Code (Developed in this Repository)

All code inside this repository (`Lexsi Hackathon/`) is newly authored specifically for Track 2:

* **Multilingual & Cross-Register Strategy Library**:
  - Text-only adaptation replacing multimodal/canvas/PDF tactics with linguistic attack vectors.
  - Targeting English, Telugu, Hindi, Japanese, Code-Mixed (Hinglish/Tenglish), and Slang/"Brainrot" registers.
* **Evaluation & Auditing Harness**:
  - Local pinned judge runner (`Qwen3Guard-Gen-4B`).
  - Strict separation of internal attack scorer vs external evaluation judge.
  - Benign twin pairing system.
  - Triple-metric computation: Attack Success Rate (ASR), Over-Refusal Rate (ORR), and Benign Capability Preservation (BCP).
* **Device-Agnostic Execution Engine**:
  - Unified configuration for Windows RTX 4060 8GB (4-bit/QLoRA), Mac M4 48GB (MPS/MLX), and Cloud (CUDA).
  - Resumable SQLite/JSONL cache layer for idempotent runs.
* **Lexsi Track 2 Toolchain Integration**:
  - Integrations chaining `CuratorKIT`, `SafeTune` / `CircuitKIT`, and `AuditKIT`.

