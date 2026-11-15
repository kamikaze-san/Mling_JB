#!/usr/bin/env python3
"""
scripts/curate_with_curatorkit.py — CuratorKIT Pair QA Driver (Lexsi Track 2)

Applies CuratorKIT data quality checks over a preference pair pool:
  - Exact & MinHash deduplication
  - Text normalization (reporting only; original text preserved)
  - Secrets & high-entropy token detection (SecretsGate)
  - PII identification & pseudonymization (reporting only)
  - Toxicity gating on chosen refusals (ToxicityGate via Detoxify)

Ground Rules:
  - Preserves original input rows byte-identical for kept pairs.
  - Multi-turn conversation prompt structures (list of dicts) remain unchanged.
  - Reports drops and QA detections in a manifest without modifying the original pool.
  - Zero exposure: aggregates and counts only, never prints raw prompts or replies.

Usage:
  .venv-curatorkit/bin/python scripts/curate_with_curatorkit.py --pool data/03_curated/dryrun_pair_pool.jsonl
  .venv-curatorkit/bin/python scripts/curate_with_curatorkit.py --pool data/03_curated/pair_pool.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Self-re-exec into .venv-curatorkit if not running inside it
ROOT_DIR = Path(__file__).resolve().parent.parent
CURATOR_VENV_PY = ROOT_DIR / ".venv-curatorkit" / "bin" / "python"
if CURATOR_VENV_PY.exists() and sys.executable != str(CURATOR_VENV_PY):
    try:
        import curatorkit  # noqa: F401
    except ImportError:
        os.execv(str(CURATOR_VENV_PY), [str(CURATOR_VENV_PY)] + sys.argv)


def compute_pair_hash(pair_dict: dict) -> str:
    """Stable content hash of a pair for fallback identity matching."""
    p = pair_dict.get("prompt")
    prompt_str = json.dumps(p, sort_keys=True, ensure_ascii=False) if isinstance(p, (list, dict)) else str(p or "")
    ch = str(pair_dict.get("chosen") or "")
    rj = str(pair_dict.get("rejected") or "")
    return hashlib.sha256((prompt_str + "\x00" + ch + "\x00" + rj).encode("utf-8")).hexdigest()


def run_curatorkit(pool_path: Path, out_path: Path, manifest_path: Path,
                   minhash_threshold: float = 0.92,
                   tox_pass_threshold: float = 0.5,
                   tox_reject_threshold: float = 0.8,
                   toxicity_field: str = "chosen"):
    from curatorkit import Curator, CuratorConfig

    print(f"=== [CuratorKIT] Initializing Pair QA on {pool_path.name} ===")
    raw_lines = [json.loads(line) for line in open(pool_path, "r", encoding="utf-8") if line.strip()]
    total_input = len(raw_lines)
    print(f"  Input pool: {total_input} pairs")

    # Map raw rows by pair_id and content hash to preserve exact original rows
    raw_by_pair_id = {}
    raw_by_hash = {}
    for idx, r in enumerate(raw_lines):
        pid = r.get("pair_id") or f"pair_{idx}"
        raw_by_pair_id[pid] = r
        raw_by_hash[compute_pair_hash(r)] = r

    with tempfile.TemporaryDirectory(prefix="curatorkit_run_") as tmpdir:
        # Note: schema_gate is disabled so length gating does not discard long valid Japanese/multilingual pairs.
        # Training truncates to max_length 2048 anyway.
        cfg = CuratorConfig(
            dataset=str(pool_path),
            output_dir=tmpdir,
            schema_gate=False,
            dedup="minhash",
            minhash_threshold=minhash_threshold,
            clean=True,
            clean_transforms={"strip_html": False},
            secrets_gate=True,
            pii_pseudonymize=True,
            toxicity_gate=True,
            toxicity_text_field=toxicity_field,
            toxicity_classifier_pass_threshold=tox_pass_threshold,
            toxicity_classifier_reject_threshold=tox_reject_threshold,
        )

        curator = Curator(cfg)
        result = curator.run()

    passed_samples = result.passed
    rejected_samples = result.rejected

    print(f"\n=== [CuratorKIT] Pipeline Completed ===")
    print(f"  Passed: {len(passed_samples)} / {total_input}")
    print(f"  Explicitly Rejected: {len(rejected_samples)} / {total_input}")

    # Build output pool: strictly original input rows unchanged
    out_rows = []
    kept_pair_ids = set()
    for s in passed_samples:
        pid = (s.metadata or {}).get("pair_id") or s.id
        orig = raw_by_pair_id.get(pid)
        if not orig:
            # Fallback to content hash
            chash = compute_pair_hash({
                "prompt": getattr(s, "instruction", ""),
                "chosen": getattr(s, "chosen", ""),
                "rejected": getattr(s, "rejected", ""),
            })
            orig = raw_by_hash.get(chash)

        if orig:
            out_rows.append(orig)
            kept_pair_ids.add(orig.get("pair_id"))
        else:
            raise RuntimeError(f"Could not map CuratorKIT sample {s.id} back to original input pair!")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Curated pool written -> {out_path} ({len(out_rows)} pairs)")

    # Aggregate statistics
    per_surface_input = Counter(r.get("profile", "unknown") for r in raw_lines)
    per_surface_kept = Counter(r.get("profile", "unknown") for r in out_rows)
    per_kind_input = Counter(r.get("kind", "unknown") for r in raw_lines)
    per_kind_kept = Counter(r.get("kind", "unknown") for r in out_rows)

    rejection_reasons = Counter()
    rejection_steps = Counter()
    rejection_by_surface = Counter()

    for rej in rejected_samples:
        step = rej.rejecting_step or "unknown_step"
        reason = rej.rejection_reason or "unknown_reason"
        rejection_steps[step] += 1
        rejection_reasons[f"{step}::{reason}"] += 1

        prof = rej.metadata.get("profile") if hasattr(rej, "metadata") and rej.metadata else None
        rej_pid = (rej.metadata or {}).get("pair_id")
        if not prof and rej_pid and rej_pid in raw_by_pair_id:
            prof = raw_by_pair_id[rej_pid].get("profile")
        rejection_by_surface[prof or "unknown"] += 1

    # Exact duplicates dropped at stage 1 might not be in rejected_samples list
    exact_dup_dropped = total_input - len(out_rows) - len(rejected_samples)
    if exact_dup_dropped > 0:
        rejection_steps["ExactDeduplicator"] += exact_dup_dropped
        rejection_reasons["ExactDeduplicator::exact_duplicate"] += exact_dup_dropped

    manifest = {
        "input_pool": str(pool_path),
        "curated_pool": str(out_path),
        "total_input": total_input,
        "total_kept": len(out_rows),
        "total_dropped": total_input - len(out_rows),
        "stage_counts": result.stage_counts,
        "rejection_by_step": dict(rejection_steps),
        "rejection_reasons": dict(rejection_reasons),
        "per_surface": {
            surf: {
                "input": per_surface_input.get(surf, 0),
                "kept": per_surface_kept.get(surf, 0),
                "dropped": per_surface_input.get(surf, 0) - per_surface_kept.get(surf, 0),
            }
            for surf in sorted(set(list(per_surface_input.keys()) + list(per_surface_kept.keys())))
        },
        "per_kind": {
            k: {
                "input": per_kind_input.get(k, 0),
                "kept": per_kind_kept.get(k, 0),
                "dropped": per_kind_input.get(k, 0) - per_kind_kept.get(k, 0)
            }
            for k in sorted(set(list(per_kind_input.keys()) + list(per_kind_kept.keys())))
        }
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest written -> {manifest_path}")

    # Structured Output Table (Zero Exposure: counts only)
    print("\n" + "=" * 65)
    print(f"{'Surface':<14} | {'Input':<8} | {'Kept':<8} | {'Dropped':<8} | {'Retain %':<8}")
    print("-" * 65)
    for surf, d in manifest["per_surface"].items():
        inp = d["input"]
        kpt = d["kept"]
        drp = d["dropped"]
        pct = (kpt / inp * 100) if inp > 0 else 0.0
        print(f"{surf:<14} | {inp:<8} | {kpt:<8} | {drp:<8} | {pct:6.1f}%")
    print("=" * 65)

    if rejection_reasons:
        print("\nDrop Reasons Breakdown:")
        for rsn, cnt in rejection_reasons.most_common():
            print(f"  - {rsn}: {cnt}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="CuratorKIT Pair QA driver")
    parser.add_argument("--pool", type=Path, default=Path("data/03_curated/pair_pool.jsonl"),
                        help="Input preference pair pool JSONL")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output curated pair pool JSONL (defaults to <pool_stem>_curated.jsonl)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Output manifest JSON (defaults to <pool_stem>_curated_manifest.json)")
    parser.add_argument("--minhash_threshold", type=float, default=0.92,
                        help="MinHash near-dedup threshold (default: 0.92)")
    parser.add_argument("--tox_pass", type=float, default=0.5,
                        help="Toxicity pass threshold (default: 0.5)")
    parser.add_argument("--tox_reject", type=float, default=0.8,
                        help="Toxicity reject threshold (default: 0.8)")
    parser.add_argument("--toxicity_field", type=str, default="chosen",
                        help="Field to evaluate for toxicity (default: chosen)")
    args = parser.parse_args()

    if not args.pool.exists():
        print(f"[ERROR] Pool file not found: {args.pool}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or args.pool.parent / f"{args.pool.stem}_curated.jsonl"
    manifest_path = args.manifest or args.pool.parent / f"{args.pool.stem}_curated_manifest.json"

    run_curatorkit(
        pool_path=args.pool,
        out_path=out_path,
        manifest_path=manifest_path,
        minhash_threshold=args.minhash_threshold,
        tox_pass_threshold=args.tox_pass,
        tox_reject_threshold=args.tox_reject,
        toxicity_field=args.toxicity_field
    )


if __name__ == "__main__":
    main()
