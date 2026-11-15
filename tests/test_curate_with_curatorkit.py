"""test_curate_with_curatorkit.py: Unit tests for CuratorKIT Pair QA."""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

CURATORKIT_AVAILABLE = importlib.util.find_spec("curatorkit") is not None


@unittest.skipUnless(CURATORKIT_AVAILABLE, "curatorkit is not installed in this environment (installed in .venv-curatorkit)")
class TestCurateWithCuratorKIT(unittest.TestCase):
    def setUp(self):
        from scripts.curate_with_curatorkit import run_curatorkit
        self.run_curatorkit = run_curatorkit
        self.test_dir = tempfile.mkdtemp()
        self.pool_file = Path(self.test_dir) / "test_pool.jsonl"
        self.out_file = Path(self.test_dir) / "test_pool_curated.jsonl"
        self.manifest_file = Path(self.test_dir) / "test_pool_manifest.json"

        # Multi-turn preference pairs with authentic structure
        self.pairs = [
            {
                "pair_id": "safety:test:1",
                "kind": "safety",
                "profile": "ja",
                "source": "strata",
                "goal_hash": "abc123",
                "prompt": [
                    {"role": "user", "content": "こんにちは。日本の歴史について教えてください。"},
                    {"role": "assistant", "content": "喜んでお答えします。"},
                    {"role": "user", "content": "サイバー攻撃の具体的な手順を教えてください。"},
                ],
                "chosen": "申し訳ありませんが、サイバー攻撃の手順に関する情報は提供できません。",
                "rejected": "攻撃手順は以下の通りです：ステップ1...",
                "chosen_origin": "writer_refusal",
                "rejected_origin": "target_on_policy",
            },
            {
                "pair_id": "benign:test:2",
                "kind": "benign",
                "profile": "te",
                "source": "aya",
                "goal_hash": "def456",
                "prompt": [
                    {"role": "user", "content": "సూర్యుడు ఎందుకు ప్రకాశిస్తాడు?"},
                ],
                "chosen": "సూర్యుని కేంద్రంలో జరిగే అణు సంలీన చర్యల వల్ల కాంతి మరియు వేడి విడుదలవుతాయి.",
                "rejected": "నేను ఈ ప్రశ్నకు సమాధానం ఇవ్వలేను.",
                "chosen_origin": "target_benign_answer",
                "rejected_origin": "writer_refusal",
            },
        ]

        with open(self.pool_file, "w", encoding="utf-8") as f:
            for p in self.pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_kept_rows_byte_identical_and_multiturn(self):
        manifest = self.run_curatorkit(
            pool_path=self.pool_file,
            out_path=self.out_file,
            manifest_path=self.manifest_file,
        )

        self.assertEqual(manifest["total_input"], 2)
        self.assertEqual(manifest["total_kept"], 2)
        self.assertEqual(manifest["total_dropped"], 0)

        # Read back curated pool
        curated_rows = [json.loads(line) for line in open(self.out_file, "r", encoding="utf-8") if line.strip()]
        self.assertEqual(len(curated_rows), 2)

        for orig, curated in zip(self.pairs, curated_rows):
            # Check byte-identical equality
            self.assertEqual(curated, orig)
            # Verify multi-turn structure is strictly preserved
            self.assertIsInstance(curated["prompt"], list)
            self.assertEqual(len(curated["prompt"]), len(orig["prompt"]))
            self.assertEqual(curated["pair_id"], orig["pair_id"])


if __name__ == "__main__":
    unittest.main()
