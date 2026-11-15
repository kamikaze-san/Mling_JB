import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("chj", ROOT / "scripts" / "compare_harm_judges.py")
chj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chj)


class TestCompareHarmJudges(unittest.TestCase):
    def test_parse_prompt_multi_turn(self):
        p = "USER (turn 1): a b\nASSISTANT (turn 1): c\nUSER (turn 2): d"
        self.assertEqual(chj.parse_prompt(p), [("user", 1, "a b"), ("assistant", 1, "c"), ("user", 2, "d")])

    def test_parse_prompt_without_markers(self):
        self.assertEqual(chj.parse_prompt("plain"), [("user", 1, "plain")])

    def test_messages_end_with_reply(self):
        row = {"prompt": "USER (turn 1): x", "response": "y"}
        self.assertEqual(chj.messages_for(row), [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])

    def test_decider_state_layout_and_truncation(self):
        row = {"goal": "G", "turn": "2", "response": "r",
               "prompt": "USER (turn 1): u1\nASSISTANT (turn 1): " + "z" * 2000 + "\nUSER (turn 2): u2"}
        s = chj.decider_state(row).split("\n")
        self.assertEqual(s[0], "GOAL: G")
        self.assertEqual(s[1], "USER (turn 1): u1")
        self.assertEqual(len(s[2]), len("ASSISTANT (turn 1): ") + 1500)
        self.assertEqual(s[3:], ["USER (turn 2): u2", "ASSISTANT (turn 2): r"])

    def test_parse_harm(self):
        self.assertEqual(chj.parse_harm('x {"label": "H1", "rule_id": "H1.a"}'), ("H1", "H1.a"))
        self.assertEqual(chj.parse_harm('{"label": "H0.b"}'), ("H0", ""))
        self.assertEqual(chj.parse_harm("final: H0"), ("H0", ""))
        self.assertEqual(chj.parse_harm("nothing"), (None, ""))


if __name__ == "__main__":
    unittest.main()
