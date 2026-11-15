"""
Per-language KL divergence of the deployed Q4_K_M GGUF against the BF16 GGUF.

Why: DPO "rejected" responses come from the Q4 target while the LoRA trains on
BF16 weights. If KL(BF16 || Q4) is small, Q4 outputs are near on-policy for
BF16; if it is large for a language (e.g. Telugu), that is a finding and the
rejected responses for that language should be regenerated from BF16.

Uses llama.cpp's `llama-perplexity`:
  1) BF16 pass saves reference logits   (--kl-divergence-base FILE)
  2) Q4 pass compares against them       (--kl-divergence)
Text: FLORES-200 dev sentences per language (benign, not used for evaluation).

  python scripts/kl_check.py --llama_bin_dir ~/llama.cpp/build/bin \
      --bf16 models/tiny-aya-global-bf16.gguf --q4 models/tiny-aya-global-q4_k_m.gguf
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, run_dir, write_manifest  # noqa: E402
from src.paths import BENCHMARKS_DIR  # noqa: E402

_KLD = re.compile(r"Mean\s+KLD:\s*([0-9.]+)")
_TOP = re.compile(r"Same top p:\s*([0-9.]+)")
_PPL_Q = re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)")
_PPL_BASE = re.compile(r"Mean PPL\(base\)\s*:\s*([0-9.]+)")


def parse_kl_output(text):
    """Extract summary statistics from llama-perplexity --kl-divergence output."""
    out = {}
    for key, rx in (("mean_kld", _KLD), ("same_top_pct", _TOP),
                    ("ppl_q", _PPL_Q), ("ppl_base", _PPL_BASE)):
        m = rx.search(text)
        out[key] = float(m.group(1)) if m else None
    return out


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd[:3])} ...\n{p.stdout[-2000:]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_bin_dir", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--q4", required=True)
    ap.add_argument("--langs", default="en,hi,te,ja")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--run_name", default="kl_check")
    args = ap.parse_args()

    exe = os.path.join(os.path.expanduser(args.llama_bin_dir), "llama-perplexity")
    out_dir = run_dir(args.run_name)
    rows = read_jsonl(BENCHMARKS_DIR / "flores_dev.jsonl")
    results = {}
    for lang in args.langs.split(","):
        txt = out_dir / f"flores_dev_{lang}.txt"
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(r[lang] for r in rows))
        base_logits = out_dir / f"bf16_logits_{lang}.dat"
        run([exe, "-m", args.bf16, "-f", str(txt), "-c", str(args.ctx),
             "--kl-divergence-base", str(base_logits)])
        stdout = run([exe, "-m", args.q4, "-f", str(txt), "-c", str(args.ctx),
                      "--kl-divergence-base", str(base_logits), "--kl-divergence"])
        results[lang] = parse_kl_output(stdout)
        print(f"  {lang}: {results[lang]}", flush=True)

    write_manifest(out_dir, {"script": "kl_check", "args": vars(args), "results": results},
                   name="kl_summary.json")
    print("\n=== KL(BF16 || Q4_K_M) on FLORES dev ===")
    print(f"{'lang':<6}{'mean KLD':>10}{'same top-1 %':>14}{'PPL bf16':>10}{'PPL q4':>10}")
    for lang, r in results.items():
        print(f"{lang:<6}{r['mean_kld'] or float('nan'):>10.4f}{r['same_top_pct'] or float('nan'):>14.2f}"
              f"{r['ppl_base'] or float('nan'):>10.2f}{r['ppl_q'] or float('nan'):>10.2f}")


if __name__ == "__main__":
    main()
