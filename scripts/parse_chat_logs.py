import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Any, Optional

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paths import SEEDS_RAW_DIR, MANUAL_TRACES_DIR, ensure_dirs

# Reconfigure stdout/stderr for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Unicode ranges for robust multilingual script detection
RE_TELUGU = re.compile(r"[\u0C00-\u0C7F]")
RE_HINDI = re.compile(r"[\u0900-\u097F]")
RE_JAPANESE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")

# Common slang / brainrot markers
BRAINROT_KEYWORDS = {
    "skibidi", "rizz", "gyatt", "sigma", "cooked", "no cap", "fr fr",
    "delulu", "lowkey", "bussin", "fanum", "mewing", "yap", "yapping",
    "bruh", "bro", "bhai", "machan", "mowa", "mama"
}

# Regex to detect user/assistant turn delimiters in markdown / chat logs
TURN_SPLIT_REGEX = re.compile(
    r"(?:^|\n)(?:#{1,6}\s*)?(?:\*{0,2})(user|assistant|human|model|ai)(?:\*{0,2})\s*:?[\r\n]+",
    re.IGNORECASE
)

def detect_language(text: str) -> str:
    """Detect language based on script unicode density and transliteration cues."""
    telugu_count = len(RE_TELUGU.findall(text))
    hindi_count = len(RE_HINDI.findall(text))
    japanese_count = len(RE_JAPANESE.findall(text))

    if telugu_count > 10:
        return "te"
    if hindi_count > 10:
        return "hi"
    if japanese_count > 5:
        return "ja"

    # Transliteration / Romanized detection heuristics
    lower = text.lower()
    tenglish_markers = {"cheppu", "enti", "ela", "chesaru", "ivvandi", "mowa", "babu"}
    hinglish_markers = {"kaise", "karo", "bhai", "karna", "batao", "mujhe", "chahiye"}

    words = set(re.findall(r"\b\w+\b", lower))
    if len(words & tenglish_markers) >= 2:
        return "mixed_tenglish"
    if len(words & hinglish_markers) >= 2:
        return "mixed_hinglish"

    return "en"

def detect_register(text: str) -> str:
    """Detect linguistic register (brainrot_slang, colloquial, standard)."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    matches = words & BRAINROT_KEYWORDS
    if len(matches) >= 2:
        return "brainrot_slang"
    if len(matches) == 1:
        return "colloquial"
    return "standard"

def parse_markdown_chat(file_content: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Parses a markdown chat log containing User: / Assistant: blocks
    into a structured multi-turn conversation trajectory.
    """
    matches = list(TURN_SPLIT_REGEX.finditer(file_content))
    if not matches:
        return None

    turns: List[Dict[str, str]] = []
    for i in range(len(matches)):
        role_raw = matches[i].group(1).lower()
        role = "user" if role_raw in ["user", "human"] else "assistant"
        
        start_idx = matches[i].end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(file_content)
        content = file_content[start_idx:end_idx].strip()

        if content:
            turns.append({"role": role, "content": content})

    if not turns:
        return None

    # Separate user turns
    user_turns = [t["content"] for t in turns if t["role"] == "user"]
    if not user_turns:
        return None

    # Final user turn is the payload / culminating attack prompt
    attack_prompt = user_turns[-1]
    
    # Full concatenated text for language/register detection
    full_user_text = "\n".join(user_turns)
    lang = detect_language(full_user_text)
    register = detect_register(full_user_text)

    # Format multi-turn trajectory for AutoDAN-Turbo summarizer
    trajectory = []
    turn_idx = 1
    current_u = None
    for t in turns:
        if t["role"] == "user":
            current_u = t["content"]
        elif t["role"] == "assistant" and current_u:
            trajectory.append({
                "turn": turn_idx,
                "user": current_u,
                "assistant": t["content"]
            })
            turn_idx += 1
            current_u = None

    stem = os.path.splitext(os.path.basename(filename))[0]
    # Create safe ascii id
    safe_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:30].strip("_")
    file_hash = str(abs(hash(stem)))[:6]
    prompt_id = f"seed_{safe_slug}_{file_hash}" if safe_slug else f"seed_{file_hash}"

    # Default benign twin template (to be paired/curated in downstream step)
    benign_twin = f"[Benign Twin] Educational inquiry regarding context of {prompt_id}"

    return {
        "prompt_id": prompt_id,
        "parent_id": None,
        "source": "manual",
        "language": lang,
        "register": register,
        "attack_family": "multi_turn_social_engineering",
        "category": "safety_compliance",
        "attack_prompt": attack_prompt,
        "benign_twin_prompt": benign_twin,
        "turns_count": len(turns),
        "trajectory": trajectory,
        "filename": filename
    }

def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Parse markdown / chat logs into structured seeds.")
    parser.add_argument("--input_dir", type=str, default=str(MANUAL_TRACES_DIR), help="Directory with .md or .log files (default: env MANUAL_TRACES_DIR or data/01_seeds_raw)")
    parser.add_argument("--output", type=str, default=str(SEEDS_RAW_DIR / "validated_seeds.jsonl"), help="Output JSONL path")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"[ERROR] Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    parsed_records = []
    files_checked = 0

    # Group files by stem to avoid processing duplicates (prefer .md over .txt)
    files_to_process = {}
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith((".md", ".txt")):
                stem = os.path.splitext(f)[0]
                full_path = os.path.join(root, f)
                if stem not in files_to_process or f.endswith(".md"):
                    files_to_process[stem] = full_path

    files_checked = len(files_to_process)
    for stem, fpath in files_to_process.items():
        try:
            content = None
            for enc in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    with open(fpath, "r", encoding=enc) as fh:
                        content = fh.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content:
                record = parse_markdown_chat(content, os.path.basename(fpath))
                if record:
                    parsed_records.append(record)
        except Exception as e:
            print(f"[WARNING] Failed to parse {stem}: {e}", file=sys.stderr)

    # Statistical Report ONLY — NEVER PRINTS PROMPT CONTENT
    lang_counter = Counter(r["language"] for r in parsed_records)
    reg_counter = Counter(r["register"] for r in parsed_records)
    turn_counts = [r["turns_count"] for r in parsed_records]

    print("==================================================")
    print("   CHAT LOG / MARKDOWN SEED INGESTION REPORT")
    print("==================================================")
    print(f"Total files scanned      : {files_checked}")
    print(f"Valid chat logs parsed   : {len(parsed_records)}")
    if turn_counts:
        print(f"Turns per log (min/avg/max): {min(turn_counts)} / {sum(turn_counts)/len(turn_counts):.1f} / {max(turn_counts)}")
    
    print("\n--- Language Distribution ---")
    for lang, count in lang_counter.most_common():
        print(f"  {lang:<15}: {count}")

    print("\n--- Register Distribution ---")
    for reg, count in reg_counter.most_common():
        print(f"  {reg:<15}: {count}")
    print("==================================================")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for r in parsed_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Structured seeds written to: {args.output}\n")

if __name__ == "__main__":
    main()

