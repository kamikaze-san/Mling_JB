import unittest
import os
import shutil
from src.harness.device import resolve_device, get_quantization_config

try:
    import torch
    from src.harness.device import resolve_device, get_quantization_config
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.harness.cache import GenerationCache

class TestHarness(unittest.TestCase):
    def setUp(self):
        import uuid
        self.test_dir = f"./tests/temp_cache_{uuid.uuid4().hex[:8]}"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test.sqlite")
        self.jsonl_path = os.path.join(self.test_dir, "test.jsonl")
        self.cache = GenerationCache(db_path=self.db_path, jsonl_backup=self.jsonl_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed in the current environment")
    def test_device_resolution(self):
        device, dtype = resolve_device("auto")
        self.assertIn(device, ["cuda", "mps", "cpu"])
        self.assertIsNotNone(dtype)

    def test_cache_set_and_get(self):
        model_id = "test-model"
        prompt = "Hello world in Telugu: నమస్కారం"
        response = "నమస్కారం! నేను మీకు ఎలా సహాయపడగలను?"
        
        # Initial get should be None
        self.assertIsNone(self.cache.get(model_id, prompt))

        # Set value
        self.cache.set(model_id, prompt, response, tags={"language": "te", "register": "standard"})

        # Get value should return exact response
        cached = self.cache.get(model_id, prompt)
        self.assertEqual(cached, response)

        # Check JSONL backup
        self.assertTrue(os.path.exists(self.jsonl_path))
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)

if __name__ == "__main__":
    unittest.main()
