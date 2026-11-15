"""
Import a laptop eval zip (laptop/run_laptop_eval.py --pack) on the Mac:
  1. copy each finished run's files into data/04_eval_cache/<run>/ (capability results + generation manifest);
  2. merge the laptop's generation cache into cache/generations.sqlite (INSERT OR IGNORE; same keys on both
     machines because the key is built from the model files' fingerprints + decoding settings + messages);
  3. VERIFY: replay every attack and benign prompt through a GGUFRunner whose model loader raises, so any reply
     missing from the cache is counted instead of silently regenerated.
A run is "ready for judging" only if it has 0 misses; then eval_adapter.py --skip_capability judges it on the Mac.
Prints counts only.

  python scripts/import_laptop_results.py data/laptop_return/return_1113_2300.zip
"""
import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.runs import read_jsonl  # noqa: E402
from src.harness.cache import GenerationCache  # noqa: E402
from src.harness.gguf_runner import GGUFRunner  # noqa: E402

EVAL_CACHE = ROOT / "data" / "04_eval_cache"
MAC_CACHE = ROOT / "cache" / "generations.sqlite"
REF_INPUTS = ROOT / "data" / "04_eval_cache" / "en_only_p100_lr5e-5_e3"  # the trimmed set the laptop bundle ships
BASE_GGUF = Path.home() / ".lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf"


class _NoModel:
    def __init__(self, **_):
        raise RuntimeError("cache miss")


def merge_cache(src_db: Path):
    with sqlite3.connect(src_db) as s:
        cols = [r[1] for r in s.execute("PRAGMA table_info(generations)")]
        rows = s.execute(f"SELECT {', '.join(cols)} FROM generations").fetchall()
    GenerationCache(str(MAC_CACHE), jsonl_backup=None)  # make sure the table exists
    with sqlite3.connect(MAC_CACHE) as m:
        before = m.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
        ki = cols.index("cache_key")
        ri = cols.index("response")
        conflicts = 0
        for r in rows:
            hit = m.execute("SELECT response FROM generations WHERE cache_key = ?", (r[ki],)).fetchone()
            if hit is not None and hit[0] != r[ri]:
                conflicts += 1  # keep the Mac's reply; report the count
        m.executemany(f"INSERT OR IGNORE INTO generations ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
        after = m.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    print(f"[cache] laptop rows {len(rows)} | new {after - before} | already present {len(rows) - (after - before)} "
          f"| present with a different reply (kept Mac's) {conflicts}")


def verify(run: str):
    man = json.loads((EVAL_CACHE / run / "generation_manifest.json").read_text(encoding="utf-8"))
    lap_lora = man["target"].get("lora_path")
    scale = float(man["target"].get("lora_scale", 1.0))
    lora = None
    if lap_lora:  # None = base model without adapter
        adapter = PureWindowsPath(lap_lora).parent.name if "\\" in lap_lora else Path(lap_lora).parent.name
        lora = ROOT / "checkpoints" / "hf" / adapter / "adapter.gguf"
        if not lora.exists():
            print(f"[verify] {run}: adapter {adapter} not found locally -> cannot verify"); return False
    runner = GGUFRunner(str(BASE_GGUF), lora_path=str(lora) if lora else None, lora_scale=scale,
                        cache=GenerationCache(str(MAC_CACHE), jsonl_backup=None), llama_factory=_NoModel)
    if runner.model_id != man["target"]["model_id"]:
        print(f"[verify] {run}: model fingerprint differs from the laptop's (different base or adapter file) -> not usable")
        return False
    miss = n = 0
    for name in ("eval_attacks_combined.jsonl", "eval_benign_combined.jsonl"):
        for r in read_jsonl(REF_INPUTS / name):
            n += 1
            try:
                runner.replay(r["turns"]) if "turns" in r else runner.generate(r["prompt"])
            except RuntimeError:
                miss += 1
    print(f"[verify] {run}: {n - miss}/{n} prompts found in the cache ({'READY for judging' if miss == 0 else 'NOT ready'})")
    return miss == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--force", action="store_true", help="overwrite a run dir that already has Mac results")
    ap.add_argument("--cache_db", default=None, help="tests only: another generations.sqlite")
    ap.add_argument("--eval_cache_dir", default=None, help="tests only: another 04_eval_cache dir")
    args = ap.parse_args()
    global MAC_CACHE, EVAL_CACHE
    if args.cache_db:
        MAC_CACHE = Path(args.cache_db)
    if args.eval_cache_dir:
        EVAL_CACHE = Path(args.eval_cache_dir)
    ready = []
    for zp in args.zips:
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(zp) as z:
            z.extractall(td)
            td = Path(td)
            man = json.loads((td / "return_manifest.json").read_text(encoding="utf-8"))
            print(f"[import] {Path(zp).name}: {len(man['runs'])} runs, laptop llama_cpp {man.get('llama_cpp')}")
            merge_cache(td / "cache" / "generations.sqlite")
            for run in man["runs"]:
                dst = EVAL_CACHE / run
                if (dst / "safety_results.jsonl").exists() and not args.force:
                    print(f"[import] {run}: already judged on the Mac, leaving it alone (use --force)"); continue
                dst.mkdir(parents=True, exist_ok=True)
                for f in (td / "04_eval_cache" / run).iterdir():
                    shutil.copy2(f, dst / f.name)
                if not (dst / "generation_manifest.json").exists():  # cap_only job: FLORES + Belebele, nothing to judge
                    print(f"[import] {run}: capability only (no safety generation) -> copied, nothing to judge")
                    continue
                shutil.copy2(dst / "generation_manifest.json", dst / "generation_manifest_laptop.json")
                if verify(run):
                    ready.append(run)
    print(f"\nready for Mac judging: {ready}")


if __name__ == "__main__":
    main()
