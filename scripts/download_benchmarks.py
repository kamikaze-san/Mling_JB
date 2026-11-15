import os
import sys
import json
from pathlib import Path
from collections import Counter

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paths import BENCHMARKS_DIR, ensure_dirs

# Reconfigure stdout/stderr for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def download_belebele_slice(output_path: Path, num_per_lang: int = 25):
    """
    Downloads lightweight slice of Meta's Belebele benchmark:
    Telugu (tel_Telu), Hindi (hin_Deva), Japanese (jpn_Jpan), English (eng_Latn).
    Multiple choice format with exact ground truth answer keys (1-4).
    """
    from datasets import load_dataset

    langs = {
        "te": "tel_Telu",
        "hi": "hin_Deva",
        "ja": "jpn_Jpan",
        "en": "eng_Latn"
    }

    records = []
    print("\n[1/2] Fetching Meta Belebele slices (Multiple Choice Reading Comprehension)...")

    for short_lang, config_name in langs.items():
        try:
            print(f"  Downloading {config_name} ({short_lang})...")
            ds = load_dataset("facebook/belebele", config_name, split=f"test[:{num_per_lang}]")
            for row in ds:
                clean_row = {
                    "benchmark": "belebele",
                    "language": short_lang,
                    "lang_config": config_name,
                    "question_number": row.get("question_number"),
                    "passage": row.get("flores_passage", ""),
                    "question": row.get("question", ""),
                    "options": [
                        row.get("mc_answer1", ""),
                        row.get("mc_answer2", ""),
                        row.get("mc_answer3", ""),
                        row.get("mc_answer4", "")
                    ],
                    "correct_answer_num": int(row.get("correct_answer_num", 1))
                }
                records.append(clean_row)
        except Exception as e:
            print(f"  [WARNING] Failed to load {config_name}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} Belebele questions to {output_path}")
    return records

def download_aya_eval_slice(output_path: Path, num_per_lang: int = 25):
    """
    Downloads lightweight slice of Cohere's official Aya Evaluation Suite:
    Telugu (tel), Hindi (hin), Japanese (jpn), English (eng).
    Open-ended multilingual instruction following with human reference targets.
    """
    from datasets import load_dataset

    records = []
    print("\n[2/2] Fetching Cohere Aya Evaluation Suite slices...")

    # 1. Telugu & English from aya_human_annotated
    try:
        print("  Downloading aya_human_annotated (Telugu & English)...")
        ds_annot = load_dataset("CohereLabs/aya_evaluation_suite", "aya_human_annotated", split="test")
        tel_items = [x for x in ds_annot if x.get("language") == "tel"][:num_per_lang]
        eng_items = [x for x in ds_annot if x.get("language") == "eng"][:num_per_lang]
        for item in tel_items + eng_items:
            lang = "te" if item.get("language") == "tel" else "en"
            records.append({
                "benchmark": "aya_eval_suite",
                "sub_split": "aya_human_annotated",
                "language": lang,
                "id": str(item.get("id")),
                "prompt": item.get("inputs", ""),
                "target_reference": item.get("targets", "")
            })
    except Exception as e:
        print(f"  [WARNING] Failed to load aya_human_annotated: {e}")

    # 2. Hindi from dolly_human_edited
    try:
        print("  Downloading dolly_human_edited (Hindi)...")
        ds_dolly = load_dataset("CohereLabs/aya_evaluation_suite", "dolly_human_edited", split="test")
        hin_items = [x for x in ds_dolly if x.get("language") == "hin"][:num_per_lang]
        for item in hin_items:
            records.append({
                "benchmark": "aya_eval_suite",
                "sub_split": "dolly_human_edited",
                "language": "hi",
                "id": str(item.get("id")),
                "prompt": item.get("inputs", ""),
                "target_reference": item.get("targets", "")
            })
    except Exception as e:
        print(f"  [WARNING] Failed to load dolly_human_edited: {e}")

    # 3. Japanese from dolly_machine_translated
    try:
        print("  Downloading dolly_machine_translated (Japanese)...")
        ds_trans = load_dataset("CohereLabs/aya_evaluation_suite", "dolly_machine_translated", split="test")
        jpn_items = [x for x in ds_trans if x.get("language") == "jpn"][:num_per_lang]
        for item in jpn_items:
            records.append({
                "benchmark": "aya_eval_suite",
                "sub_split": "dolly_machine_translated",
                "language": "ja",
                "id": str(item.get("id")),
                "prompt": item.get("inputs", ""),
                "target_reference": item.get("targets", "")
            })
    except Exception as e:
        print(f"  [WARNING] Failed to load dolly_machine_translated: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} Aya Evaluation Suite prompts to {output_path}")
    return records

FLORES_LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu", "ja": "jpn_Jpan"}


def download_flores(output_dir: Path, devtest_n: int = 200, dev_n: int = 0):
    """
    FLORES-200 (FLORES+ on the HF Hub, gated: accept the terms and set HF_TOKEN).

    Writes sentence-aligned rows (one row per sentence id, one field per language):
      flores_devtest.jsonl  -> capability EVALUATION only (never train on it)
      flores_dev.jsonl      -> allowed for rehearsal data / KL-check text
    dev_n=0 means "all dev sentences".
    """
    from datasets import load_dataset

    print("\n[FLORES] Fetching FLORES+ dev/devtest for", ", ".join(FLORES_LANGS.values()))
    out = {}
    for split, n in (("devtest", devtest_n), ("dev", dev_n)):
        by_id = {}
        for short, cfg in FLORES_LANGS.items():
            ds = load_dataset("openlanguagedata/flores_plus", cfg, split=split)
            for row in ds:
                by_id.setdefault(row["id"], {"id": row["id"], "split": split})[short] = row["text"]
        rows = [r for _, r in sorted(by_id.items()) if all(k in r for k in FLORES_LANGS)]
        if n:
            rows = rows[:n]
        path = output_dir / f"flores_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[split] = rows
        print(f"  {split}: {len(rows)} aligned sentences -> {path.name}")
    return out


def flag_flores_belebele_overlap(flores_rows, belebele_records):
    """
    Belebele passages are built from FLORES-200 text. Mark FLORES rows whose
    English sentence appears inside any English Belebele passage, so they are
    never used as rehearsal training data. Returns the number of flagged rows.
    """
    passages = [r["passage"] for r in belebele_records if r["language"] == "en"]
    flagged = 0
    for row in flores_rows:
        sent = row["en"].strip()
        hit = bool(sent) and any(sent in p for p in passages)
        row["in_belebele"] = hit
        flagged += int(hit)
    return flagged


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download capability benchmarks (aggregate output only).")
    ap.add_argument("--belebele_per_lang", type=int, default=150)
    ap.add_argument("--flores_devtest_n", type=int, default=200)
    ap.add_argument("--flores_dev_n", type=int, default=0, help="0 = all dev sentences")
    ap.add_argument("--with_aya", action="store_true", help="Also fetch the (optional) Aya eval suite slice")
    ap.add_argument("--aya_per_lang", type=int, default=25)
    args = ap.parse_args()

    ensure_dirs()
    belebele_path = BENCHMARKS_DIR / "belebele_eval.jsonl"

    print("==================================================")
    print("      OFFLINE CAPABILITY BENCHMARK DOWNLOADER")
    print("==================================================")

    bele_records = download_belebele_slice(belebele_path, num_per_lang=args.belebele_per_lang)
    try:
        flores = download_flores(BENCHMARKS_DIR, devtest_n=args.flores_devtest_n, dev_n=args.flores_dev_n)
    except Exception as e:  # FLORES+ is gated: accept its terms on the HF dataset page, then re-run
        print(f"[FLORES] SKIPPED ({type(e).__name__}). Accept the terms at "
              f"https://huggingface.co/datasets/openlanguagedata/flores_plus (auto-approved), then re-run.")
        flores = None

    flagged = dev_devtest_bele = 0
    if flores is not None:
        # Contamination guard: flag dev sentences that occur inside Belebele passages.
        flagged = flag_flores_belebele_overlap(flores["dev"], bele_records)
        with open(BENCHMARKS_DIR / "flores_dev.jsonl", "w", encoding="utf-8") as f:
            for r in flores["dev"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dev_devtest_bele = flag_flores_belebele_overlap(list(flores["devtest"]), bele_records)

    aya_records = []
    if args.with_aya:
        aya_records = download_aya_eval_slice(BENCHMARKS_DIR / "aya_eval_suite.jsonl",
                                              num_per_lang=args.aya_per_lang)

    print("\n==================================================")
    print("              DOWNLOAD REPORT")
    print("==================================================")
    bele_counts = Counter(r["language"] for r in bele_records)
    print(f"Belebele questions: {len(bele_records)}")
    for lang, c in sorted(bele_counts.items()):
        print(f"  {lang:<10}: {c}")
    if flores is None:
        print("FLORES: not downloaded (gated; see message above)")
    else:
        print(f"FLORES devtest (eval only): {len(flores['devtest'])} aligned sentences "
              f"({dev_devtest_bele} also appear in Belebele passages)")
        print(f"FLORES dev (rehearsal / KL text): {len(flores['dev'])} aligned sentences, "
              f"{flagged} flagged in_belebele (excluded from rehearsal)")
    if aya_records:
        print(f"Aya eval suite (optional): {len(aya_records)}")
    print("==================================================")
    print(f"[SUCCESS] Benchmarks cached in: {BENCHMARKS_DIR}")

if __name__ == "__main__":
    main()

