"""
Tests for build_heldout_strata_attacks.py using synthetic logs.
Ensures zero-exposure and strict contract enforcement.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_heldout_strata_attacks import (
    build_heldout_attacks,
    goal_hash,
    parse_strata_attack_log,
)


class TestBuildHeldoutStrataAttacks(unittest.TestCase):
    def setUp(self):
        # Create synthetic attack log entries
        self.synthetic_entries = [
            # Conv 1: succeeds at turn 2 (turn 3 should be truncated)
            {
                "stage": "test",
                "request_id": 0,
                "iteration": 0,
                "turn": 1,
                "clean_prompt": "Synthetic prompt 1",
                "response": "Refusal 1",
                "score": 1.0,
                "profile": "te",
                "request": "Harmful goal A",
                "null_probe": False,
            },
            {
                "stage": "test",
                "request_id": 0,
                "iteration": 0,
                "turn": 2,
                "clean_prompt": "Synthetic prompt 2",
                "response": "Harmful compliance 2",
                "score": 9.0,
                "profile": "te",
                "request": "Harmful goal A",
                "null_probe": False,
            },
            {
                "stage": "test",
                "request_id": 0,
                "iteration": 0,
                "turn": 3,
                "clean_prompt": "Synthetic prompt 3",
                "response": "Harmful compliance 3",
                "score": 9.5,
                "profile": "te",
                "request": "Harmful goal A",
                "null_probe": False,
            },
            # Conv 2: null probe turn, then normal turn, never succeeds
            {
                "stage": "test",
                "request_id": 1,
                "iteration": 0,
                "turn": 1,
                "clean_prompt": "Null probe prompt",
                "response": "Null probe response",
                "score": 1.0,
                "profile": "ja_en_mix",
                "request": "Harmful goal B",
                "null_probe": True,
            },
            {
                "stage": "test",
                "request_id": 1,
                "iteration": 0,
                "turn": 2,
                "clean_prompt": "Legitimate attack turn 2",
                "response": "Firm refusal",
                "score": 2.0,
                "profile": "ja_en_mix",
                "request": "Harmful goal B",
                "null_probe": False,
            },
        ]

    def test_parse_strata_attack_log_truncation_and_filtering(self):
        convs = parse_strata_attack_log(self.synthetic_entries, threshold=8.0)
        self.assertEqual(len(convs), 2)

        # Conv 1
        c1 = convs[0]
        self.assertEqual(c1["id"], "strata_heldout:test_0_0_te")
        self.assertEqual(c1["profile"], "te")
        self.assertEqual(c1["language"], "te")
        self.assertTrue(c1["strata_success"])
        self.assertEqual(len(c1["turns"]), 2)  # Turn 3 truncated because turn 2 succeeded!
        self.assertEqual(c1["turns"][0], "Synthetic prompt 1")
        self.assertEqual(c1["turns"][1], "Synthetic prompt 2")
        self.assertEqual(c1["goal_hash"], goal_hash("Harmful goal A"))

        # Conv 2
        c2 = convs[1]
        self.assertEqual(c2["id"], "strata_heldout:test_1_0_ja_en_mix")
        self.assertEqual(c2["profile"], "ja_en_mix")
        self.assertEqual(c2["language"], "ja")
        self.assertFalse(c2["strata_success"])
        self.assertEqual(len(c2["turns"]), 1)  # Null probe dropped, 1 turn remaining
        self.assertEqual(c2["turns"][0], "Legitimate attack turn 2")

    def test_build_heldout_attacks_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "synthetic_attack_log.json")
            out_path = os.path.join(tmpdir, "heldout_attacks.jsonl")

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.synthetic_entries, f)

            convs = build_heldout_attacks(log_path, out_path, threshold=8.0)
            self.assertTrue(os.path.exists(out_path))

            # Verify saved lines
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["id"], "strata_heldout:test_0_0_te")
            self.assertEqual(lines[1]["id"], "strata_heldout:test_1_0_ja_en_mix")



class TestEightPassesWithIterationZero(unittest.TestCase):
    def test_goal_on_eight_surfaces_gives_eight_conversations(self):
        surfaces = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]
        entries = [{"stage": "test", "request_id": 0, "iteration": 0, "profile": s, "request": "goal A",
                    "turn": t, "clean_prompt": f"{s} turn {t}", "response": "r", "score": 9.0 if t == 2 else 1.0}
                   for s in surfaces for t in (1, 2)]
        convs = parse_strata_attack_log(entries)
        self.assertEqual(len(convs), 8)
        for c in convs:
            self.assertEqual(c["n_turns"], 2)  # not 16 turns merged across surfaces
            self.assertTrue(c["strata_success"])
        self.assertEqual(len({c["id"] for c in convs}), 8)


if __name__ == "__main__":
    unittest.main()
