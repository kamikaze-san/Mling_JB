"""
Laptop (RTX 4060 8 GB, Windows) half of an adapter eval: capability (FLORES + Belebele) and safety GENERATION.
The Mac then judges the replies (Qwen3Guard + pplx decider) straight from the merged generation cache, so the
judges stay byte-identical across all runs.

Run from the bundle root:
  python run_laptop_eval.py --check            # environment + files + CUDA offload check (no generation)
  python run_laptop_eval.py --smoke            # 2 attacks, 2 benign, tiny capability set, on the first job
  python run_laptop_eval.py                    # every job in jobs.txt, in order; resumable (cached replies are reused)
  python run_laptop_eval.py --pack             # zip what the Mac needs into return_<timestamp>.zip

Decoding settings are NOT configurable here on purpose: they must equal the Mac's eval_adapter.py defaults
(temperature 0, seed 0, max_tokens 1024, n_ctx 8192, no system prompt), because the cache key includes them and
the Mac must find every reply in the cache. The LoRA scale is set per job in jobs.txt (default 1.0) and the Mac
judges that run with the same --lora_scale.

jobs.txt line: <adapter folder | none> <run name> [scale=<float>] [cap_only] [model=<file in models/>]
  none      = base model without adapter
  cap_only  = FLORES + Belebele only, no safety generation
  model=    = another GGUF in models/ instead of tiny-aya-global (e.g. a regional Tiny Aya, for teacher checks)
Prints counts and aggregates only - never prompt, reply or FLORES text.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # GenerationCache uses ./cache/ relative to the working directory
BASE_GGUF = ROOT / "models" / "tiny-aya-global-q4_k_m.gguf"
ATTACKS = ROOT / "inputs" / "eval_attacks_combined.jsonl"
BENIGN = ROOT / "inputs" / "eval_benign_combined.jsonl"
EVAL_CACHE = ROOT / "data" / "04_eval_cache"
LOGS = ROOT / "logs"
PY = sys.executable


def parse_job(line):
    """'<adapter|none> <run> [scale=X] [cap_only]' -> dict; raises ValueError on anything else."""
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"jobs.txt line needs '<adapter|none> <run name>': {line!r}")
    job = {"adapter": None if parts[0].lower() == "none" else parts[0], "name": parts[1], "scale": 1.0,
           "cap_only": False, "model": BASE_GGUF}
    for opt in parts[2:]:
        if opt.startswith("scale="):
            job["scale"] = float(opt.split("=", 1)[1])
        elif opt.startswith("model="):
            job["model"] = ROOT / "models" / opt.split("=", 1)[1]
        elif opt == "cap_only":
            job["cap_only"] = True
        else:
            raise ValueError(f"unknown jobs.txt option {opt!r} in: {line!r}")
    return job


def jobs():
    out = []
    for line in (ROOT / "jobs.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(parse_job(line))
    return out


def lora_args(job):
    if job["adapter"] is None:
        return []
    return ["--lora", ROOT / "adapters" / job["adapter"] / "adapter.gguf", "--lora_scale", str(job["scale"])]


def done(job):
    d = EVAL_CACHE / job["name"]
    cap = (d / "capability_summary.json").exists()
    return cap if job["cap_only"] else cap and (d / "generation_manifest.json").exists()


def run(cmd, log):
    LOGS.mkdir(exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n### {datetime.datetime.now():%Y-%m-%d %H:%M} {' '.join(map(str, cmd))}\n")
        f.flush()
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode


def check():
    ok = True
    for p in [BASE_GGUF, ATTACKS, BENIGN, ROOT / "data" / "benchmarks" / "flores_devtest.jsonl",
              ROOT / "data" / "benchmarks" / "belebele_eval.jsonl", EVAL_CACHE / "base_q4_v2" / "capability_summary.json"]:
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p.relative_to(ROOT)}")
        ok &= p.exists()
    for job in jobs():
        what = f"scale {job['scale']}{', capability only' if job['cap_only'] else ''}{'  (done)' if done(job) else ''}"
        if job["model"] != BASE_GGUF:
            print(f"  {'ok ' if job['model'].exists() else 'MISSING'} models/{job['model'].name} -> run {job['name']}")
            ok &= job["model"].exists()
        if job["adapter"] is None:
            print(f"  ok  {job['model'].name} (no adapter) -> run {job['name']} | {what}")
            continue
        p = ROOT / "adapters" / job["adapter"] / "adapter.gguf"
        print(f"  {'ok ' if p.exists() else 'MISSING'} adapters/{job['adapter']}/adapter.gguf -> run {job['name']} | {what}")
        ok &= p.exists()
    try:
        import llama_cpp
        print(f"  llama_cpp {llama_cpp.__version__} | GPU offload supported: {llama_cpp.llama_supports_gpu_offload()}")
        ok &= bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:  # noqa: BLE001
        print(f"  llama_cpp import FAILED: {type(e).__name__}: {e}")
        ok = False
    try:
        import sacrebleu
        print(f"  sacrebleu {sacrebleu.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  sacrebleu import FAILED: {type(e).__name__}")
        ok = False
    print("CHECK", "PASSED" if ok else "FAILED")
    return ok


def smoke():
    job = jobs()[0]
    name, la = job["name"], lora_args(job)
    tmp = ROOT / "inputs" / "_smoke"
    tmp.mkdir(exist_ok=True)
    for src, n in [(ATTACKS, 2), (BENIGN, 2)]:
        rows = src.read_text(encoding="utf-8").splitlines()[:n]
        (tmp / src.name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    sname = name + "_smoke"
    log = LOGS / f"{sname}.log"
    t0 = datetime.datetime.now()
    rc1 = run([PY, "scripts/run_capability_eval.py", "--run_name", sname, "--gguf", BASE_GGUF, *la,
               "--flores_n", "2", "--belebele_per_lang", "2"], log)
    rc2 = run([PY, "scripts/run_safety_eval.py", "--run_name", sname, "--gguf", BASE_GGUF, "--generate_only",
               "--attacks", tmp / ATTACKS.name, "--benign", tmp / BENIGN.name, *la], log)
    secs = (datetime.datetime.now() - t0).total_seconds()
    print(f"smoke: capability rc={rc1} generation rc={rc2} in {secs:.0f}s (log: {log.relative_to(ROOT)})")
    print("SMOKE", "PASSED" if rc1 == 0 and rc2 == 0 else "FAILED")


def run_all():
    for job in jobs():
        name, la = job["name"], lora_args(job)
        if done(job):
            print(f"{name}: already done, skipping")
            continue
        log = LOGS / f"{name}.log"
        t0 = datetime.datetime.now()
        print(f"{name}: capability start {t0:%H:%M} (scale {job['scale']})", flush=True)
        rc = run([PY, "scripts/run_capability_eval.py", "--run_name", name, "--gguf", job["model"], *la,
                  "--compare", "base_q4_v2"], log)
        if rc != 0:
            print(f"{name}: CAPABILITY FAILED rc={rc} (see {log.relative_to(ROOT)}); next job", flush=True)
            continue
        if job["cap_only"]:
            mins = (datetime.datetime.now() - t0).total_seconds() / 60
            print(f"{name}: done (capability only) in {mins:.0f} min", flush=True)
            continue
        print(f"{name}: generation start {datetime.datetime.now():%H:%M}", flush=True)
        rc = run([PY, "scripts/run_safety_eval.py", "--run_name", name, "--gguf", job["model"], "--generate_only",
                  "--attacks", ATTACKS, "--benign", BENIGN, *la], log)
        mins = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"{name}: {'done' if rc == 0 else f'GENERATION FAILED rc={rc}'} in {mins:.0f} min", flush=True)


def pack():
    names = [j["name"] for j in jobs() if done(j)]
    out = ROOT / f"return_{datetime.datetime.now():%m%d_%H%M}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            for f in (EVAL_CACHE / n).iterdir():
                z.write(f, f"04_eval_cache/{n}/{f.name}")
        kl = EVAL_CACHE / "kl_probe" / "kl_probe_results.json"
        if kl.exists():
            z.write(kl, "04_eval_cache/kl_probe/kl_probe_results.json")
        # consistent snapshot via SQLite's backup API: copying the live file while a job writes to it corrupted
        # the 14 Nov 17:52 zip
        import sqlite3
        snap = ROOT / "cache" / "_pack_snapshot.sqlite"
        snap.unlink(missing_ok=True)
        src, dst = sqlite3.connect(ROOT / "cache" / "generations.sqlite"), sqlite3.connect(snap)
        src.backup(dst)
        dst.close(), src.close()
        z.write(snap, "cache/generations.sqlite")
        snap.unlink()
        z.writestr("return_manifest.json", json.dumps({"runs": names, "created": datetime.datetime.now().isoformat(),
                                                       "llama_cpp": __import__("llama_cpp").__version__}))
    print(f"packed {len(names)} finished runs -> {out.name}: {names}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    elif a.smoke:
        smoke()
    elif a.pack:
        pack()
    else:
        run_all()
