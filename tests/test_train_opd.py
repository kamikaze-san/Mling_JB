import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("opd", ROOT / "scripts" / "train_opd.py")
opd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opd)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestOpdLoss(unittest.TestCase):
    def test_mask_keeps_up_to_first_end_token(self):
        gen = torch.tensor([[5, 6, 9, 0, 0],    # end token 9 at position 2, then padding
                            [5, 9, 6, 9, 0],    # first end at 1: everything after is masked
                            [5, 6, 7, 8, 4]])   # never ends (max_new_tokens): keep all
        m = opd.completion_mask(gen, eos_ids=[9], pad_id=0)
        self.assertEqual(m.tolist(), [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])

    def test_mask_with_several_end_ids(self):
        m = opd.completion_mask(torch.tensor([[1, 8, 9, 2]]), eos_ids=[8, 9], pad_id=0)
        self.assertEqual(m.tolist(), [[1, 1, 0, 0]])

    def test_reverse_kl_zero_for_identical_and_positive_otherwise(self):
        torch.manual_seed(0)
        s = torch.randn(2, 3, 11)
        mask = torch.ones(2, 3)
        total, rows = opd.reverse_kl(s, s.clone(), mask)
        self.assertAlmostEqual(total.item(), 0.0, places=5)
        total2, _ = opd.reverse_kl(s, torch.randn(2, 3, 11), mask)
        self.assertGreater(total2.item(), 0.0)

    def test_reverse_kl_matches_definition_and_respects_mask(self):
        s = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]]])
        t = torch.tensor([[[0.0, 1.0, 0.0], [5.0, 0.0, 0.0]]])
        ps, lps, lpt = s.softmax(-1)[0, 0], s.log_softmax(-1)[0, 0], t.log_softmax(-1)[0, 0]
        expected = (ps * (lps - lpt)).sum().item()
        total, _ = opd.reverse_kl(s, t, torch.tensor([[1.0, 0.0]]))   # second position masked out
        self.assertAlmostEqual(total.item(), expected, places=5)

    def test_position_ids_ignore_left_padding(self):
        attn = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.assertEqual(opd.position_ids_from_mask(attn).tolist(), [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])


class TestOpdPrompts(unittest.TestCase):
    def test_resolve_translation_and_benign(self):
        items = [{"type": "translation", "flores_id": 7, "src": "te", "tgt": "en"},
                 {"type": "benign", "id": "x", "profile": "hi", "prompt": "hello"}]
        out = opd.resolve_prompts(items, {7: {"te": "T", "en": "E"}})
        self.assertEqual(out[0]["tag"], "te-en")
        self.assertIn("Telugu", out[0]["messages"][0]["content"])
        self.assertEqual(out[1]["messages"], [{"role": "user", "content": "hello"}])

    def test_missing_flores_id_fails_cleanly(self):
        with self.assertRaises(SystemExit):
            opd.resolve_prompts([{"type": "translation", "flores_id": 1, "src": "te", "tgt": "en"}], {})


if __name__ == "__main__":
    unittest.main()
