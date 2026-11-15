"""export_strata: what annotators see (synthetic log only)."""
import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load():
    spec = importlib.util.spec_from_file_location("ja", ROOT / "scripts" / "judge_agreement.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e(stage, req, prof, goal, turn, score, **kw):
    d = dict(stage=stage, request_id=req, iteration=0, profile=prof, request=goal, turn=turn,
             clean_prompt=f"USERMSG-{prof}-{req}-{turn}", response=f"REPLY-{prof}-{req}-{turn}", score=score,
             prompt="private attacker reasoning")
    d.update(kw)
    return d


class TestExportStrata(unittest.TestCase):
    def test_sheet_shows_conversation_ending_with_the_latest_user_turn(self):
        m = load()
        entries = []
        for prof in ("en", "hi"):
            entries += [e("lifelong", 0, prof, "goal A", 1, 1.0), e("lifelong", 0, prof, "goal A", 2, 9.0)]  # success t2
            entries += [e("lifelong", 1, prof, "goal B", 1, 1.0)]  # failure after one turn
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.json"
            log.write_text(json.dumps(entries * 2))  # logs repeat entries: duplicates must be dropped
            run = Path(d) / "run"
            m.run_dir = lambda name: run
            m.ORDER_SURFACES = ["en", "hi"]
            args = type("A", (), dict(run_name="r", strata_logs=[str(log)], exclude_goals=None, n_success=1,
                                      n_fail=1, threshold=8.0, seed=0))
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                m.export_strata(args)
            rows = list(csv.DictReader(open(run / "judge_labelling_strata.csv", encoding="utf-8")))
            self.assertEqual(len(rows), 4)  # 2 surfaces x (1 success + 1 fail)
            for r in rows:
                lines = r["prompt"].splitlines()
                self.assertTrue(lines[-1].startswith("USER (turn"), "context must end with the latest user turn")
                self.assertIn(f"USERMSG-{r['language']}", lines[-1])
                n_user = sum(1 for l in lines if l.startswith("USER (turn"))
                n_asst = sum(1 for l in lines if l.startswith("ASSISTANT (turn"))
                self.assertEqual(n_user, n_asst + 1)
                self.assertNotIn("private attacker reasoning", r["prompt"] + r["response"])
                self.assertNotIn("strata_success", r)  # blind
            succ = [r for r in rows if r["response"].endswith("-2")]
            self.assertEqual(len(succ), 2)  # the success rows are the turn-2 replies
            self.assertEqual(len({r["item_id"] for r in rows}), 4)


if __name__ == "__main__":
    unittest.main()
