#!/usr/bin/env python3
"""Score the 128-row human-labelled STRATA sheet with other pplx-decider versions, using exactly the questions and
state text that produced decider_pplx_flash.jsonl (flash v1.2, scripts/eval_d14_pplx.py), so only the model differs.

Two back ends, because the releases differ:
  marker    flash-style: several questions in one pass, readout at <|fim_suffix|> markers (pplx-flash-mlx/runner)
  lasttoken 27B-style: one question per pass, last-token pooling, readout as lm_head (decider_mlx.py next to the model)

  ~/Downloads/MARCH-Orz/.venv-pplx/bin/python scripts/score_sheet_deciders.py --tag flash_v13 --backend marker \
      --model ~/Downloads/decision-models/pplx-flash-v1.3-mlx/4bit-g64
  ~/Downloads/decision-models/.venv-decider27/bin/python scripts/score_sheet_deciders.py --tag v11_27b \
      --backend lasttoken --model ~/.lmstudio/models/mlx-community/pplx-decider-v1.1-27b-4bit

Output: data/04_eval_cache/strata_judge_validation/decider_<tag>.jsonl (resumable). Prints counts and timings only.
"""
import argparse
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "data" / "04_eval_cache" / "strata_judge_validation"
SHEET = V / "judge_labelling_strata.csv"
FLASH_RUNNER = os.path.expanduser("~/Downloads/decision-models/pplx-flash-mlx/runner")

sys.path.insert(0, str(ROOT / "scripts"))
_src = (ROOT / "scripts" / "eval_d14_pplx.py").read_text(encoding="utf-8")
QUESTIONS = ast.literal_eval(_src[_src.index("QUESTIONS = {") + len("QUESTIONS = "):_src.index("\n}\n", _src.index("QUESTIONS = {")) + 2])


def marker_backend(model_dir):
    sys.path.insert(0, FLASH_RUNNER)
    import mlx_decider as D
    from decider_common import Encoder, answers
    enc = Encoder(model_dir)
    model, _ = D.load_mlx_dir(model_dir)
    W = D.load_readout(model_dir)

    def run(state):
        req = {"state": state, "questions": QUESTIONS}
        ids, markers, counts = enc.encode(req)
        a = answers(req, D.readout_logits(model, W, ids, markers, counts), enc.temperature)
        st = a["response_stance"]
        return {k: a[k]["noul"] for k in ("goal_harm", "response_hazard", "goal_fulfilled")}, st["argmax"], dict(zip(st["keys"], st["probs"]))
    return run


def lasttoken_backend(model_dir):
    sys.path.insert(0, str(model_dir))
    from decider_mlx import Decider, answer
    dec = Decider(model_dir)
    names = list(QUESTIONS)

    def run(state):
        rows = [{"state": state, "question": QUESTIONS[n]} for n in names]
        a = {n: answer(QUESTIONS[n], p) for n, p in zip(names, dec.predict(rows, batch_size=4))}
        st = a["response_stance"]
        return {k: a[k]["noul"] for k in ("goal_harm", "response_hazard", "goal_fulfilled")}, st["choice"], st["probabilities"]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", choices=["marker", "lasttoken"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    model_dir = Path(os.path.expanduser(a.model))
    out = V / f"decider_{a.tag}.jsonl"
    done = {json.loads(l)["item_id"] for l in open(out)} if out.exists() else set()
    t0 = time.time()
    run = marker_backend(str(model_dir)) if a.backend == "marker" else lasttoken_backend(model_dir)
    print(f"[{a.tag}] loaded in {time.time() - t0:.0f}s; {len(done)} already done", flush=True)
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8")))[: a.limit]
    t1, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            if r["item_id"] in done:
                continue
            state = f"GOAL: {r['goal']}\n{r['prompt']}\nASSISTANT (turn {r.get('turn', '1')}): {r['response']}"
            p, stance, probs = run(state)
            f.write(json.dumps({"item_id": r["item_id"], "language": r["language"], "turn": r.get("turn", "1"),
                                **{f"{k}_p": round(v, 4) for k, v in p.items()}, "response_stance": stance,
                                "response_stance_probs": {k: round(v, 4) for k, v in probs.items()},
                                "model": str(model_dir.name)}) + "\n")
            f.flush()
            n += 1
            if n % 16 == 0:
                print(f"  {n} done, {(time.time() - t1) / n:.1f} s/item", flush=True)
    print(f"[{a.tag}] {n} new rows in {time.time() - t1:.0f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
