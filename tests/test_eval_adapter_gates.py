import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ea", ROOT / "scripts" / "eval_adapter.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


BASE = [{"id": "x", "kind": "benign"}]  # any non-empty baseline (no baseline returns early)


class TestCapabilityGates(unittest.TestCase):
    def _write(self, d, obj):
        p = os.path.join(d, "capability_summary.json")
        with open(p, "w") as f:
            json.dump(obj, f)
        return p

    def test_nested_summary_is_unwrapped_and_gates_evaluate(self):
        # run_capability_eval.py nests the scores under "summary"; the gates must see them (bug fixed 14 Nov)
        with tempfile.TemporaryDirectory() as d:
            a = ea.load_capability(self._write(d, {"args": {}, "summary": {
                "flores_chrf_pp": {"te-en": 54.0}, "belebele_accuracy": {"te": 0.60}}}))
        b = {"flores_chrf_pp": {"te-en": 55.2}, "belebele_accuracy": {"te": 0.60}}
        gates = ea.evaluate_checkpoint_gates([], BASE, a, b)["gates"]
        self.assertEqual(ea.gate_status(gates["flores_chrf_pp"]), "FAILED")      # -1.2 < -1.0
        self.assertEqual(ea.gate_status(gates["belebele_accuracy"]), "PASSED")
        self.assertIn("-1.2000 (te-en)", ea.describe_deltas("flores_chrf_pp", gates["flores_chrf_pp"]))

    def test_flat_file_still_accepted_and_missing_file_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ea.load_capability(self._write(d, {"flores_chrf_pp": {}})), {"flores_chrf_pp": {}})
        self.assertIsNone(ea.load_capability(None))
        self.assertIsNone(ea.load_capability("/nonexistent/capability_summary.json"))

    def test_missing_capability_reports_skipped_not_passed(self):
        gates = ea.evaluate_checkpoint_gates([], BASE, None, None)["gates"]
        self.assertEqual(ea.gate_status(gates["flores_chrf_pp"]), "SKIPPED")
        self.assertTrue(ea.describe_deltas("flores_chrf_pp", gates["flores_chrf_pp"]).startswith("Not evaluated"))

    def test_rise_gates_report_the_largest_rise(self):
        g = {"deltas": {"hi": 0.10, "te": -0.05}, "threshold": "<= +0.05"}
        self.assertIn("+0.1000 (hi)", ea.describe_deltas("orr_borderline_twins", g))


if __name__ == "__main__":
    unittest.main()
