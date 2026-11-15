"""build_conditions on a SYNTHETIC pool; always writes to a temp dir, never to data/03_curated/dpo."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]


def run(pool_rows, held=("some other goal",)):
    d = tempfile.mkdtemp()
    pool = os.path.join(d, "pool.jsonl")
    Path(pool).write_text("".join(json.dumps(r) + "\n" for r in pool_rows))
    h = os.path.join(d, "held.json")
    json.dump(list(held), open(h, "w"))
    out = os.path.join(d, "out")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_conditions.py"), "--pool", pool, "--heldout_goals", h,
                        "--out_dir", out], capture_output=True, text=True)
    return r, out


def pool(n=40, **over):
    rows = []
    for p in S:
        for i in range(n):
            for kind in ("safety", "benign"):
                r = {"pair_id": f"{kind}:{p}:{i}", "kind": kind, "profile": p, "goal_hash": f"g{p}{i}",
                     "prompt": [{"role": "user", "content": "q"}], "chosen": "c", "rejected": "r"}
                r.update(over)
                rows.append(r)
    return rows


class TestBuildConditions(unittest.TestCase):
    def test_equal_sizes_and_shares(self):
        r, out = run(pool())
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        sizes = {n: sum(1 for _ in open(os.path.join(out, n + ".jsonl"))) for n in ("joint8", "en_only", "te_only", "mixed_only")}
        self.assertEqual(set(sizes.values()), {56})
        self.assertTrue(os.path.exists(os.path.join(out, "joint_all.jsonl")))

    def test_fails_when_a_surface_cannot_supply_its_share(self):
        rows = [x for x in pool() if not (x["profile"] == "te" and x["kind"] == "safety" and int(x["pair_id"].split(":")[2]) >= 10)]
        r, _ = run(rows)
        self.assertNotEqual(r.returncode, 0)

    def test_fails_on_formatting_markers(self):
        rows = pool(); rows[0]["prompt"] = [{"role": "user", "content": "MESSAGE:\nleaked"}]
        r, _ = run([dict(x) for x in rows])
        self.assertNotEqual(r.returncode, 0)

    def test_reply_text_like_a_recipe_method_line_is_allowed(self):
        rows = pool(); rows[0]["chosen"] = "Ingredients...\nMethod: stir and bake"
        r, _ = run([dict(x) for x in rows])
        self.assertEqual(r.returncode, 0, r.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
