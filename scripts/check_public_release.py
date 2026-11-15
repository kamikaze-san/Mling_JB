#!/usr/bin/env python3
"""Pre-publication scan of a repo or folder that is about to become public.

Lists only file paths, line numbers and the kind of finding, never the matched text (a finding may be a
secret, and the repos hold attack material that must not be echoed).

  python scripts/check_public_release.py .                  # git repo: tracked + untracked-not-ignored files
  python scripts/check_public_release.py ../STRATA_public   # plain folder: every file
  python scripts/check_public_release.py . --history        # also: data/model paths and token patterns in git history

Exit code 1 if any BLOCK finding (secrets, data/model files, private keys); WARN findings (absolute
home paths, e-mail addresses, large files) are listed for a manual look.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SECRET_PATTERNS = {
    "hf_token": r"hf_[A-Za-z0-9]{30,}",
    "openai_key": r"sk-[A-Za-z0-9_-]{20,}",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "google_api_key": r"AIza[0-9A-Za-z_-]{35}",
    "slack_token": r"xox[abpr]-[A-Za-z0-9-]{10,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "gcp_service_account": r"\"type\":\s*\"service_account\"",
}
WARN_PATTERNS = {
    "abs_home_path": r"(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|[A-Za-z]:\\\\Users\\\\)",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
}
EMAIL_OK = re.compile(r"(noreply@|@example\.|@users\.noreply\.github\.com|@anthropic\.com)")
DATA_EXT = {".jsonl", ".csv", ".tsv", ".parquet", ".sqlite", ".db", ".pkl", ".pt", ".bin",
            ".safetensors", ".gguf", ".npy", ".npz", ".zip", ".tar", ".gz"}
DATA_DIRS = {"data", "logs", "logs_r", "sft", "wandb", "checkpoints", "cache", "results", "outputs"}
JSON_OK_DIRS = {"configs", "schemas", "chat_templates", "generation_configs", "mock_data"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
LARGE = 5 * 1024 * 1024


def list_files(root: Path):
    if (root / ".git").exists():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             capture_output=True, text=True, check=True).stdout
        return sorted(p for p in out.split("\0") if p)
    files = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".venv")]
        files += [str(Path(d, f).relative_to(root)) for f in fs]
    return sorted(files)


def scan_file(root: Path, rel: str):
    findings = []
    p = root / rel
    parts = Path(rel).parts
    ext = p.suffix.lower()
    if any(x in DATA_DIRS for x in parts[:-1]):
        findings.append(("BLOCK", "inside a data/log folder", 0))
    if ext in DATA_EXT:
        findings.append(("BLOCK", f"data/model file ({ext})", 0))
    if ext == ".json" and not any(x in JSON_OK_DIRS for x in parts[:-1]):
        findings.append(("WARN", "json outside configs/schemas (check it is not data)", 0))
    if p.name.startswith(".env") and p.name != ".env.example":
        findings.append(("BLOCK", "env file", 0))
    try:
        size = p.stat().st_size
    except OSError:
        return findings
    if size > LARGE:
        findings.append(("WARN", f"large file ({size / 1e6:.1f} MB)", 0))
        return findings
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings
    for i, line in enumerate(text.splitlines(), 1):
        for kind, pat in SECRET_PATTERNS.items():
            if re.search(pat, line):
                findings.append(("BLOCK", kind, i))
        for kind, pat in WARN_PATTERNS.items():
            for m in re.finditer(pat, line):
                if kind == "email" and EMAIL_OK.search(m.group(0)):
                    continue
                findings.append(("WARN", kind, i))
                break
    return findings


def scan_history(root: Path):
    out = []
    names = subprocess.run(["git", "-C", str(root), "log", "--all", "--name-only", "--format="],
                           capture_output=True, text=True).stdout.split("\n")
    bad = sorted({n for n in names if n and (Path(n).suffix.lower() in DATA_EXT
                                              or any(x in DATA_DIRS for x in Path(n).parts[:-1]))})
    out.append(("BLOCK" if bad else "OK", f"{len(bad)} data/log/model paths ever committed", bad[:20]))
    patch = subprocess.run(["git", "-C", str(root), "log", "--all", "-p"], capture_output=True, text=True,
                           errors="replace").stdout
    hits = {k: len(re.findall(p, patch)) for k, p in SECRET_PATTERNS.items()}
    hits = {k: v for k, v in hits.items() if v}
    out.append(("BLOCK" if hits else "OK", f"secret patterns in history: {hits or 'none'}", []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--history", action="store_true", help="also scan git history (git repos only)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    files = list_files(root)
    n_block = n_warn = 0
    for rel in files:
        for level, kind, line in scan_file(root, rel):
            n_block += level == "BLOCK"
            n_warn += level == "WARN"
            print(f"{level:5} {rel}{':' + str(line) if line else ''}  {kind}")
    if args.history and (root / ".git").exists():
        for level, msg, sample in scan_history(root):
            n_block += level == "BLOCK"
            print(f"{level:5} [history] {msg}")
            for s in sample:
                print(f"        {s}")
    print(f"\n{len(files)} files scanned | BLOCK {n_block} | WARN {n_warn}")
    sys.exit(1 if n_block else 0)


if __name__ == "__main__":
    main()
