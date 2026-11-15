"""Run-directory, JSONL and manifest helpers shared by the eval / training scripts."""

import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.paths import EVAL_CACHE_DIR, ROOT_DIR


def read_jsonl(path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def run_dir(run_name: str) -> Path:
    d = Path(EVAL_CACHE_DIR) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(directory: Path, payload: Dict[str, Any], name: str = "manifest.json") -> Path:
    """Record everything needed to reproduce a run (config, model fingerprints, git rev)."""
    data = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        **payload,
    }
    path = Path(directory) / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path
