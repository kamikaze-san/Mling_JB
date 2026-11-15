"""
Fetch an evaluation run folder from the private HF dataset repo into local data/04_eval_cache/<run>/
Usage:
  python scripts/hf_fetch_run.py base_q4_hf
"""
import argparse
import os
import shutil
import sys
from huggingface_hub import snapshot_download

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_REPO = "kamikaze-san/mjb-dpo-private"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", help="Run name e.g. base_q4_hf")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Dataset repo ID")
    ap.add_argument("--out_dir", default=os.path.join(ROOT_DIR, "data", "04_eval_cache"))
    args = ap.parse_args()

    target_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"[hf_fetch_run] Fetching runs/{args.run_name} from {args.repo} -> {target_dir}...")

    downloaded = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=f"runs/{args.run_name}/*",
        local_dir=args.out_dir,
    )

    src_subdir = os.path.join(args.out_dir, "runs", args.run_name)
    if os.path.exists(src_subdir) and os.path.abspath(src_subdir) != os.path.abspath(target_dir):
        for item in os.listdir(src_subdir):
            s = os.path.join(src_subdir, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        shutil.rmtree(os.path.join(args.out_dir, "runs"), ignore_errors=True)

    print(f"[hf_fetch_run] Successfully fetched to: {target_dir}")
    safety_file = os.path.join(target_dir, "safety_results.jsonl")
    if os.path.exists(safety_file):
        count = sum(1 for _ in open(safety_file, encoding="utf-8"))
        print(f"  safety_results.jsonl: {count} rows")


if __name__ == "__main__":
    main()
