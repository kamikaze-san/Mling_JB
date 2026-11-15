import argparse
import csv
import json
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paths import SEEDS_RAW_DIR, ensure_dirs

import uuid
from collections import Counter
from typing import Dict, Any, List

ALLOWED_LANGS = {"en", "hi", "te", "ja", "mixed_hinglish", "mixed_tenglish", "other"}
ALLOWED_REGISTERS = {"standard", "colloquial", "brainrot_slang", "academic", "bureaucratic", "mixed"}

def parse_args():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Zero-exposure seed validator and ingester.")
    parser.add_argument("--input", type=str, required=True, help="Path to input seed file (.csv or .jsonl)")
    parser.add_argument("--output", type=str, default=str(SEEDS_RAW_DIR / "validated_seeds.jsonl"), help="Output path")
    return parser.parse_args()

def load_records(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    elif ext in [".jsonl", ".json"]:
        with open(file_path, "r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    else:
        print(f"[ERROR] Unsupported file extension '{ext}'. Use .csv or .jsonl", file=sys.stderr)
        sys.exit(1)

    return records

def validate_and_normalize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = []
    lang_counter = Counter()
    register_counter = Counter()
    family_counter = Counter()
    missing_fields_count = 0

    for idx, r in enumerate(records):
        # Support flexible column aliases
        prompt_id = r.get("prompt_id") or r.get("id") or f"seed_{idx:04d}"
        attack_prompt = r.get("attack_prompt") or r.get("prompt") or r.get("attack") or ""
        benign_twin = r.get("benign_twin_prompt") or r.get("benign_twin") or r.get("benign") or ""
        language = (r.get("language") or r.get("lang") or "en").lower().strip()
        register = (r.get("register") or "standard").lower().strip()
        attack_family = r.get("attack_family") or r.get("family") or "manual_curated"
        category = r.get("category") or "general_safety"

        if not attack_prompt.strip():
            missing_fields_count += 1
            continue

        # Normalization
        if language not in ALLOWED_LANGS:
            if "hin" in language:
                language = "hi"
            elif "tel" in language:
                language = "te"
            elif "jp" in language or "jap" in language:
                language = "ja"
            elif "hinglish" in language:
                language = "mixed_hinglish"
            elif "tenglish" in language:
                language = "mixed_tenglish"
            else:
                language = "other"

        if register not in ALLOWED_REGISTERS:
            register = "brainrot_slang" if "slang" in register or "brainrot" in register else "standard"

        clean_record = {
            "prompt_id": str(prompt_id),
            "parent_id": None,
            "language": language,
            "register": register,
            "source": "manual",
            "attack_family": attack_family,
            "category": category,
            "attack_prompt": attack_prompt.strip(),
            "benign_twin_prompt": benign_twin.strip(),
            "notes": r.get("notes", "")
        }

        lang_counter[language] += 1
        register_counter[register] += 1
        family_counter[attack_family] += 1
        validated.append(clean_record)

    # Statistical Report ONLY — NEVER PRINTS PROMPT CONTENT
    print("==================================================")
    print("   SEED INGESTION & VALIDATION REPORT (ZERO-EXPOSURE)")
    print("==================================================")
    print(f"Total raw records read : {len(records)}")
    print(f"Total valid seeds saved: {len(validated)}")
    if missing_fields_count > 0:
        print(f"Skipped invalid/empty  : {missing_fields_count}")
    print("\n--- Language Distribution ---")
    for lang, count in lang_counter.most_common():
        print(f"  {lang:<15}: {count}")
    print("\n--- Register Distribution ---")
    for reg, count in register_counter.most_common():
        print(f"  {reg:<15}: {count}")
    print("\n--- Attack Families ---")
    for fam, count in family_counter.most_common():
        print(f"  {fam:<20}: {count}")
    print("==================================================")

    return validated

def main():
    args = parse_args()
    records = load_records(args.input)
    validated = validate_and_normalize(records)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in validated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Validated seeds written to: {args.output}\n")

if __name__ == "__main__":
    main()
