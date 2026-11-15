#!/usr/bin/env python3
"""Probe prompts for laptop/kl_probe.py: FLORES dev translations (25 per direction, from the OPD prompt set) and
benign twins (6 per surface). Writes <out> (contains FLORES text: gitignored data/ only, laptop by USB/LAN only).

  python scripts/build_kl_probe_inputs.py data/laptop_addon_v3d/inputs/kl_prompts.jsonl
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.capability import translation_prompt  # noqa: E402

PER_DIRECTION, PER_SURFACE = 25, 6


def main(out):
    flores = {r["id"]: r for r in map(json.loads, open(ROOT / "data/benchmarks/flores_dev.jsonl", encoding="utf-8"))}
    items = [json.loads(l) for l in open(ROOT / "data/03_curated/opd/opd_v1.jsonl", encoding="utf-8") if l.strip()]
    groups = defaultdict(list)
    for it in items:
        groups[(it["type"], f"{it['src']}-{it['tgt']}" if it["type"] == "translation" else it.get("profile", ""))].append(it)
    rng = random.Random(0)
    rows = []
    for (typ, tag), its in sorted(groups.items()):
        for it in rng.sample(its, min(PER_DIRECTION if typ == "translation" else PER_SURFACE, len(its))):
            if typ == "translation":
                msg = translation_prompt(flores[it["flores_id"]][it["src"]], it["src"], it["tgt"])
            else:
                msg = it["prompt"]
            rows.append({"messages": [{"role": "user", "content": msg}], "type": typ, "tag": tag})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} probe prompts -> {out}:",
          {t: sum(r['type'] == t for r in rows) for t in ('translation', 'benign')})


if __name__ == "__main__":
    main(sys.argv[1])
