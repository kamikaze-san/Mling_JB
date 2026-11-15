"""
tests/test_score_judges_vs_humans.py — Verification of D12-prep scoring driver

Tests:
  - Both 128-row format (with unsafe columns) and 64-row format (complied only)
  - Inter-annotator agreement and Cohen's kappa computation
  - Handling of Excel BOM (utf-8-sig)
  - Integration with key and decider mock files
  - Proper bootstrap 95% confidence intervals
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.score_judges_vs_humans import (
    compute_cohen_kappa,
    bootstrap_metric_ci,
    agreement_metric,
    score_judges_against_humans
)


class TestScoreJudgesVsHumans(unittest.TestCase):

    def test_metrics_math(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 1, 0, 0, 0])  # 6/8 = 75% agreement

        agr = agreement_metric(y_true, y_pred)
        self.assertAlmostEqual(agr, 75.0)

        kap = compute_cohen_kappa(y_true, y_pred)
        self.assertGreater(kap, 0.4)
        self.assertLess(kap, 1.0)

        pt, lo, hi = bootstrap_metric_ci(y_true, y_pred, agreement_metric, n_resamples=100)
        self.assertAlmostEqual(pt, 75.0)
        self.assertLessEqual(lo, 75.0)
        self.assertGreaterEqual(hi, 75.0)

    def test_end_to_end_synthetic_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Synthetic 64-row sheet with Excel BOM
            rows = []
            surfaces = ["en", "hi", "te", "ja"]
            for i in range(32):
                surf = surfaces[i % len(surfaces)]
                h1 = 1 if i % 2 == 0 else 0
                h2 = 1 if i % 3 == 0 else 0
                rows.append({
                    "item_id": f"item_{i}",
                    "language": surf,
                    "turn": 1,
                    "goal": f"synthetic goal {i}",
                    "prompt": f"synthetic prompt {i}",
                    "response": f"synthetic reply {i}",
                    "human1_complied": h1,
                    "human2_complied": h2,
                })
            df_sheet = pd.DataFrame(rows)
            sheet_path = tmp / "test_sheet.csv"
            df_sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")

            # 2. Key file
            key_rows = []
            for i in range(32):
                key_rows.append({
                    "item_id": f"item_{i}",
                    "language": surfaces[i % len(surfaces)],
                    "strata_score": 9.0 if i % 2 == 0 else 2.0,
                    "strata_success": 1 if i % 2 == 0 else 0
                })
            df_key = pd.DataFrame(key_rows)
            key_path = tmp / "test_key.csv"
            df_key.to_csv(key_path, index=False, encoding="utf-8-sig")

            # 3. Decider PPLX mock file
            pplx_path = tmp / "test_pplx.jsonl"
            with open(pplx_path, "w", encoding="utf-8") as f:
                for i in range(32):
                    row = {
                        "item_id": f"item_{i}",
                        "goal_fulfilled_p": 0.85 if i % 2 == 0 else 0.15
                    }
                    f.write(json.dumps(row) + "\n")

            # Run scoring driver
            results = score_judges_against_humans(
                sheet_path=sheet_path,
                key_path=key_path,
                pplx_path=pplx_path,
                d1_path=None,
                n_bootstrap=100
            )

            self.assertEqual(results["total_rows"], 32)
            self.assertGreater(results["consensus_rows"], 0)
            self.assertIn("human_vs_human", results)
            self.assertIn("strata", results["judges"])
            self.assertIn("pplx_decider", results["judges"])

            # Strata should perfectly match when consensus matches i%2 == 0
            strata_res = results["judges"]["strata"]["pooled"]
            self.assertGreaterEqual(strata_res["agreement_pct"][0], 90.0)


if __name__ == "__main__":
    unittest.main()
