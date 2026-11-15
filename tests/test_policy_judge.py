"""Offline tests for the bring-your-own-policy judge (fake server, synthetic text only)."""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.judge.policy_judge import PolicyJudge, format_conversation, parse_label  # noqa: E402

POLICY = str(ROOT / "configs" / "judge" / "safeguard_policy.md")


def fake_post(label_by_turn):
    calls = []

    def post(url, payload, timeout):
        calls.append(payload)
        user = payload["messages"][1]["content"]
        lab = label_by_turn(user)
        content = "" if lab is None else json.dumps({"label": lab, "rule_id": lab + ".a", "rationale": "r"})
        return {"choices": [{"message": {"content": content, "reasoning_content": "thinking"},
                             "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}
    return post, calls


class TestPolicyJudge(unittest.TestCase):
    def test_parse_label(self):
        self.assertEqual(parse_label('x {"label": "rc1", "rule_id": "RC1.b"} y')["label"], "RC1")
        self.assertEqual(parse_label('{"label": "RC0"} then {"label": "RC2"}')["label"], "RC2")
        self.assertEqual(parse_label("final answer: RC0")["label"], "RC0")
        self.assertIsNone(parse_label('{"label": "maybe"}'))

    def test_format_marks_only_last_assistant(self):
        t = format_conversation([{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "u2"}, {"role": "assistant", "content": ""}])
        self.assertIn("ASSISTANT (turn 1, context only)", t)
        self.assertIn("ASSISTANT REPLY TO CLASSIFY (turn 2)\n[empty reply]", t)

    def test_system_has_policy_and_reasoning_and_retries_unparseable(self):
        n = {"i": 0}

        def lab(user):
            n["i"] += 1
            return None if n["i"] == 1 else "RC1"
        post, calls = fake_post(lab)
        j = PolicyJudge("http://x/v1", "m", POLICY, reasoning="low", post_fn=post)
        j.retries = 1
        import src.judge.policy_judge as pj
        pj.time.sleep = lambda s: None
        out = j.classify([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        self.assertEqual(out["label"], "RC1")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["messages"][0]["content"].endswith("Reasoning: low"))
        self.assertIn("RC1 COMPLIED", calls[0]["messages"][0]["content"])
        self.assertGreaterEqual(calls[0]["max_tokens"], 4096)


class TestRunner(unittest.TestCase):
    def test_build_items_cache_and_report(self):
        spec = importlib.util.spec_from_file_location("rpj", ROOT / "scripts" / "run_policy_judge.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        results = [{"kind": "attack", "id": f"a{i}", "language": "hi", "responses": ["r1", "r2"],
                    "turn_unsafe": [False, i % 2 == 0], "judge_labels": ["Safe|None", "Unsafe|Violent"]}
                   for i in range(4)]
        results += [{"kind": "benign", "id": f"b{i}", "language": "te", "response": "x", "over_refused": i == 0}
                    for i in range(3)]
        attacks = {f"a{i}": {"turns": ["t1", "t2"]} for i in range(4)}
        benign = {f"b{i}": {"prompt": "p"} for i in range(3)}
        items = m.build_items(results, attacks, benign, per_lang=2, seed=0)
        self.assertEqual(sum(it["kind"] == "attack" for it in items), 4)   # 2 convs x 2 turns
        self.assertEqual(sum(it["kind"] == "benign" for it in items), 2)
        second = [it for it in items if it["key"].endswith("|2") and it["kind"] == "attack"][0]
        self.assertEqual([x["role"] for x in second["messages"]], ["user", "assistant", "user", "assistant"])
        cache = {it["key"]: {"label": "RC1" if it["kind"] == "attack" else "RC0"} for it in items}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.report(items, cache)
        out = buf.getvalue()
        self.assertIn("attack hi", out.replace("  ", " ").replace("  ", " "))
        self.assertNotIn("t1", out)  # zero exposure: no prompt/reply text


class TestJudgeAgreementTwoJudges(unittest.TestCase):
    def test_export_disagreements_and_score_both_judges(self):
        spec = importlib.util.spec_from_file_location("ja", ROOT / "scripts" / "judge_agreement.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            run.mkdir()
            m.run_dir = lambda name: run
            rows = [{"kind": "attack", "id": f"a{i}", "language": "hi", "responses": [f"resp{i}"],
                     "turn_unsafe": [i == 0]} for i in range(6)]
            rows += [{"kind": "benign", "id": f"benign:hi:{i}", "language": "hi", "response": "x",
                      "over_refused": False} for i in range(2)]
            with open(run / "safety_results.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            att = Path(d) / "att.jsonl"
            att.write_text("".join(json.dumps({"id": f"a{i}", "turns": [f"question{i}"]}) + "\n" for i in range(6)))
            pol = Path(d) / "pol.jsonl"  # policy says complied for a0..a2 -> a1, a2 disagree with QG
            pol.write_text("".join(json.dumps({"key": f"attack|a{i}|1", "label": "RC1" if i < 3 else "RC0"}) + "\n"
                                   for i in range(6)))
            args = type("A", (), {"run_name": "r", "attacks": str(att), "benign": None, "policy_cache": str(pol),
                                  "focus": "disagreements", "per_lang": 3, "seed": 0})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m.export(args)
            self.assertNotIn("question", buf.getvalue())  # counts only
            import csv
            rows_out = list(csv.DictReader(open(run / "judge_labelling.csv")))
            key = {r["item_id"]: r for r in csv.DictReader(open(run / "judge_labelling_key.csv"))}
            self.assertEqual(len(rows_out), 3)
            self.assertNotIn("qg_unsafe", rows_out[0])  # annotators are blind to judges
            self.assertEqual(sum(key[r["item_id"]]["disagree"] == "yes" for r in rows_out), 2)
            for r in rows_out:  # humans: complied = policy truth, unsafe = QG truth
                if r["kind"] == "attack":
                    self.assertTrue(r["prompt"].startswith("USER (turn 1): question"))
                    k = key[r["item_id"]]
                    r["human1_complied"] = k["policy_complied"]
                    r["human1_unsafe"] = k["qg_unsafe"]
            with open(run / "judge_labelling.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=m.LABEL_FIELDS)
                w.writeheader()
                w.writerows(rows_out)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m.score(type("A", (), {"run_name": "r"}))
            out = buf.getvalue()
            self.assertIn("complied  policy", out)
            self.assertIn("unsafe    qwen3guard", out)


if __name__ == "__main__":
    unittest.main()
