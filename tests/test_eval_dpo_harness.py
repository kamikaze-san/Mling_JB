"""Unit tests for the eval + DPO harness. No model weights, no network."""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.eval.capability import (belebele_prompt, clean_translation, eval_belebele,
                                 parse_choice, preservation)
from src.eval.stats import cumulative_asr_by_turn, paired_bootstrap_diff, wilson_ci
from src.harness.cache import GenerationCache
from src.harness.gguf_runner import GGUFRunner
from src.judge.qwen_guard import QwenGuardJudge
from src.curation.writer import strip_think

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLlama:
    calls = 0
    init_kwargs = None

    def __init__(self, **kwargs):
        FakeLlama.init_kwargs = kwargs

    def create_chat_completion(self, messages, **kw):
        FakeLlama.calls += 1
        n_user = sum(1 for m in messages if m["role"] == "user")
        return {"choices": [{"message": {"content": f"reply{n_user}"}}]}


class TestGGUFRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, "base.gguf")
        self.lora = os.path.join(self.tmp, "a.gguf")
        for p, b in ((self.base, b"base" * 10), (self.lora, b"lora" * 10)):
            with open(p, "wb") as f:
                f.write(b)
        self.cache = GenerationCache(db_path=os.path.join(self.tmp, "c.sqlite"), jsonl_backup=None)
        FakeLlama.calls = 0

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cache_and_replay(self):
        r = GGUFRunner(self.base, cache=self.cache, llama_factory=FakeLlama)
        self.assertEqual(r.replay(["a", "b", "c"]), ["reply1", "reply2", "reply3"])
        self.assertEqual(FakeLlama.calls, 3)
        r.replay(["a", "b", "c"])  # fully cached
        self.assertEqual(FakeLlama.calls, 3)

    def test_adapter_changes_identity_and_is_passed(self):
        r0 = GGUFRunner(self.base, cache=self.cache, llama_factory=FakeLlama)
        r1 = GGUFRunner(self.base, lora_path=self.lora, cache=self.cache, llama_factory=FakeLlama)
        self.assertNotEqual(r0.model_id, r1.model_id)
        r1.generate("x")
        self.assertEqual(FakeLlama.init_kwargs["lora_path"], self.lora)
        r0.generate("x")  # different model id -> not served from r1's cache
        self.assertEqual(FakeLlama.calls, 2)


class TestCapability(unittest.TestCase):
    def test_clean_translation(self):
        self.assertEqual(clean_translation('Translation: "Hello there"\n\nNote: literal'), "Hello there")
        self.assertEqual(clean_translation("翻訳：こんにちは"), "こんにちは")

    def test_parse_choice(self):
        self.assertEqual(parse_choice("The answer is 3."), 3)
        self.assertEqual(parse_choice("答え：２"), 2)
        self.assertEqual(parse_choice("Paris", ["London", "Paris", "Rome", "Oslo"]), 2)
        self.assertIsNone(parse_choice("no idea"))

    def test_eval_belebele(self):
        recs = [{"language": "te", "passage": "p", "question": "q", "options": ["a", "b", "c", "d"],
                 "correct_answer_num": 2},
                {"language": "te", "passage": "p2", "question": "q2", "options": ["a", "b", "c", "d"],
                 "correct_answer_num": 4}]
        self.assertIn("1. a", belebele_prompt(recs[0]))
        res = eval_belebele(lambda msgs: "2", recs)
        self.assertEqual(res["te"]["accuracy"], 0.5)

    def test_preservation(self):
        self.assertEqual(preservation({"en-hi": 40.0, "x": 0.0}, {"en-hi": 38.0}),
                         {"en-hi": 0.95, "x": None})

    def test_chrf_if_available(self):
        try:
            import sacrebleu  # noqa: F401
        except ImportError:
            self.skipTest("sacrebleu not installed")
        from src.eval.capability import chrf_pp
        self.assertGreater(chrf_pp(["the cat sat"], ["the cat sat"]), 99.0)


class TestJudgeParsing(unittest.TestCase):
    def test_native_format(self):
        j = QwenGuardJudge()
        v = j.parse_native("Safety: Unsafe\nCategories: Violent\nRefusal: No")
        self.assertTrue(v.is_violation)
        self.assertFalse(v.is_refusal)
        v = j.parse_native("Safety: Controversial\nCategories: None\nRefusal: No")
        self.assertFalse(v.is_violation)
        self.assertTrue(j.parse_native("Safety: Controversial\nRefusal: No",
                                       controversial_as_unsafe=True).is_violation)
        v = j.parse_native("Safety: Safe\nCategories: None\nRefusal: Yes")
        self.assertTrue(v.is_refusal)
        self.assertEqual(j.parse_native("garbage").category, "Unparsed|None")

    def test_heuristic_only_never_flags_violation(self):
        v = QwenGuardJudge().evaluate("p", "I cannot help with that.", heuristic_only=True)
        self.assertTrue(v.is_refusal)
        self.assertFalse(v.is_violation)


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_ci(5, 10)
        self.assertTrue(0.2 < lo < 0.5 < hi < 0.8)
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_paired_bootstrap(self):
        a = [True] * 20 + [False] * 20
        b = [False] * 40
        d = paired_bootstrap_diff(a, b, iters=500)
        self.assertAlmostEqual(d["diff"], -0.5)
        self.assertLess(d["hi"], 0)
        with self.assertRaises(ValueError):
            paired_bootstrap_diff([True], [])

    def test_cumulative_asr(self):
        self.assertEqual(cumulative_asr_by_turn([[False, True, False], [False, False], [True]]),
                         [1 / 3, 2 / 3, 2 / 3])


class TestScripts(unittest.TestCase):
    def test_safety_summary(self):
        m = load_script("run_safety_eval")
        res = [
            {"kind": "attack", "profile": "te", "success": True, "turn_unsafe": [False, True]},
            {"kind": "attack", "profile": "te", "success": False, "turn_unsafe": [False, False]},
            {"kind": "benign", "profile": "te", "over_refused": True},
            {"kind": "benign", "profile": "hi", "over_refused": False},
        ]
        t = m.summarize(res)
        self.assertEqual(t["te"]["asr"], 0.5)
        self.assertEqual(t["te"]["asr_at_turn"], [0.0, 0.5])
        self.assertEqual(t["ALL"]["orr"], 0.5)
        self.assertIsNone(t["hi"]["asr"])

    def test_strata_log_to_candidates(self):
        m = load_script("build_pair_pool")
        log = [
            {"stage": "lifelong", "request_id": 1, "iteration": 0, "turn": 2, "clean_prompt": "u2",
             "response": "r2", "score": 9.0, "profile": "te", "request": "g"},
            {"stage": "lifelong", "request_id": 1, "iteration": 0, "turn": 1, "clean_prompt": "u1",
             "response": "r1", "score": 2.0, "profile": "te", "request": "g"},
            {"stage": "lifelong", "request_id": 1, "iteration": 0, "turn": 3, "clean_prompt": "u3",
             "response": "r3", "score": 9.5, "profile": "te", "request": "g"},
            {"stage": "lifelong", "request_id": 2, "iteration": 0, "turn": 1, "clean_prompt": "v1",
             "response": "s1", "score": 3.0, "profile": "hi", "request": "h"},
        ]
        c = list(m.conversations_from_strata_log(log, 8.5))
        self.assertEqual(len(c), 1)  # one candidate per conversation, first winning turn
        self.assertEqual(c[0]["turn"], 2)
        self.assertEqual([x["content"] for x in c[0]["prompt"]], ["u1", "r1", "u2"])
        self.assertEqual(c[0]["rejected"], "r2")
        self.assertEqual(m.goal_hash(" G "), m.goal_hash("g"))

    def test_condition_sampling(self):
        m = load_script("make_dpo_condition")
        pool = []
        for p in ("hi", "te"):
            for kind in ("safety", "benign"):
                for i in range(10):
                    pool.append({"pair_id": f"{kind}:{p}:{i}", "kind": kind, "profile": p,
                                 "prompt": [], "chosen": "c", "rejected": "r"})
        rows = m.sample_condition(pool, ["hi", "te"], 20, seed=3)
        self.assertEqual(len(rows), 20)
        from collections import Counter
        self.assertEqual(set(Counter((r["kind"], r["profile"]) for r in rows).values()), {5})
        self.assertEqual([r["pair_id"] for r in rows],
                         [r["pair_id"] for r in m.sample_condition(pool, ["hi", "te"], 20, seed=3)])
        with self.assertRaises(m.ConditionError):
            m.sample_condition(pool, ["hi"], 40)

    def test_kl_parse(self):
        m = load_script("kl_check")
        out = ("Mean PPL(Q)                   :  12.3456 ±   0.1\n"
               "Mean PPL(base)                :  12.0000 ±   0.1\n"
               "Mean    KLD:   0.012345 ±   0.000123\nSame top p:  95.123 ±  0.123 %\n")
        r = m.parse_kl_output(out)
        self.assertAlmostEqual(r["mean_kld"], 0.012345)
        self.assertAlmostEqual(r["same_top_pct"], 95.123)
        self.assertAlmostEqual(r["ppl_base"], 12.0)

    def test_embedding_metrics(self):
        m = load_script("embedding_check")
        import numpy as np
        e = np.eye(3)
        self.assertEqual(m.recall_at_1(e, e), 1.0)
        self.assertEqual(m.recall_at_1(e, e[[1, 0, 2]]), 1 / 3)
        emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
        self.assertEqual(m.loo_1nn_accuracy(emb, ["a", "a", "b", "b"]), 1.0)

    def test_kappa(self):
        m = load_script("judge_agreement")
        self.assertEqual(m.cohens_kappa([True, False, True, False], [True, False, True, False]), 1.0)
        self.assertAlmostEqual(m.cohens_kappa([True, True, False, False], [True, False, True, False]), 0.0)
        # Krippendorff alpha (nominal): perfect = 1; worked example = 1 - 0.5/(32/56) = 0.125;
        # items with < 2 ratings (missing annotator) are ignored.
        self.assertEqual(m.krippendorff_alpha([[True, True], [False, False], [True, True]]), 1.0)
        self.assertAlmostEqual(m.krippendorff_alpha([[True, True], [True, False], [False, True], [False, False],
                                                     [True, None]]), 0.125)
        self.assertIsNone(m.krippendorff_alpha([[True, None]]))

    def test_filter_kwargs(self):
        try:
            import yaml  # noqa: F401  (train_dpo imports yaml at module level)
        except ImportError:
            self.skipTest("pyyaml missing")
        m = load_script("train_dpo")

        class Cfg:
            def __init__(self, a=1, b=2):
                pass
        acc, dropped = m.filter_kwargs(Cfg, {"a": 1, "c": 3})
        self.assertEqual(acc, {"a": 1})
        self.assertEqual(dropped, ["c"])

    def test_strip_think(self):
        self.assertEqual(strip_think("<think>x</think>\n hello"), "hello")
        self.assertEqual(strip_think("plain"), "plain")


class TestMergeAndAya(unittest.TestCase):
    def test_merge_kwargs(self):
        m = load_script("merge_adapters")
        kw = m.build_merge_kwargs("ties", ["a0", "a1", "a2"], density=0.5)
        self.assertEqual(kw["weights"], [1.0, 1.0, 1.0])
        self.assertEqual(kw["density"], 0.5)
        self.assertNotIn("density", m.build_merge_kwargs("linear", ["a0", "a1"]))
        for bad in (lambda: m.build_merge_kwargs("slerp", ["a0", "a1"]),
                    lambda: m.build_merge_kwargs("ties", ["a0", "a1"]),
                    lambda: m.build_merge_kwargs("linear", ["a0"]),
                    lambda: m.build_merge_kwargs("linear", ["a0", "a1"], weights=[1.0])):
            with self.assertRaises(ValueError):
                bad()

    def test_union_conditions_same_pairs(self):
        m = load_script("make_dpo_condition")
        a = [{"pair_id": "x"}, {"pair_id": "y"}]
        b = [{"pair_id": "y"}, {"pair_id": "z"}]
        u = m.union_conditions([a, b], seed=1)
        self.assertEqual(sorted(r["pair_id"] for r in u), ["x", "y", "z"])
        self.assertEqual([r["pair_id"] for r in u],
                         [r["pair_id"] for r in m.union_conditions([a, b], seed=1)])

    def test_aya_records(self):
        m = load_script("import_aya_redteaming")
        rows = [{"prompt": "नमस्ते दुनिया", "global_or_local": "local",
                 "harm_category": '["A", "B"]'},
                {"prompt": "namaste duniya", "global_or_local": "global", "harm_category": None},
                {"prompt": "  ", "global_or_local": "global"}]
        recs = m.to_records(rows, "hindi")
        self.assertEqual(len(recs), 2)
        self.assertEqual([r["profile"] for r in recs], ["hi", "hinglish"])
        self.assertEqual(recs[0]["harm_categories"], ["A", "B"])
        self.assertEqual(recs[0]["turns"], ["नमस्ते दुनिया"])
        self.assertEqual(m.to_records([{"prompt": "hello there"}], "english")[0]["profile"], "en")

    def test_aya_stratified_sample(self):
        m = load_script("import_aya_redteaming")
        recs = ([{"language": "hi", "global_or_local": "global"} for _ in range(30)]
                + [{"language": "hi", "global_or_local": "local"} for _ in range(10)])
        s = m.stratified_sample(recs, 20, seed=0)
        self.assertEqual(len(s), 20)
        self.assertEqual(sum(r["global_or_local"] == "local" for r in s), 5)

    def test_safety_summary_slices(self):
        m = load_script("run_safety_eval")
        res = [{"kind": "attack", "profile": "hi", "global_or_local": "local", "success": True,
                "turn_unsafe": [True]},
               {"kind": "attack", "profile": "hi", "global_or_local": "global", "success": False,
                "turn_unsafe": [False]},
               {"kind": "attack", "profile": "en", "success": False, "turn_unsafe": [False]}]
        t = m.summarize(res, key_field="global_or_local")
        self.assertEqual(t["local"]["asr"], 1.0)
        self.assertEqual(t["unknown"]["n_attack"], 1)


class TestBlendVsSlices(unittest.TestCase):
    def test_pooled_hides_small_weak_slice(self):
        m = load_script("run_safety_eval")
        res = [{"kind": "attack", "profile": "en", "success": False, "turn_unsafe": [False]} for _ in range(90)]
        res += [{"kind": "attack", "profile": "te", "success": i < 5, "turn_unsafe": [i < 5]} for i in range(10)]
        b = m.blend_vs_slices(m.summarize(res))["asr"]
        self.assertAlmostEqual(b["pooled"], 0.05)
        self.assertAlmostEqual(b["macro_mean"], 0.25)
        self.assertEqual((b["worst_slice"], b["worst"], b["worst_n"]), ("te", 0.5, 10))
        self.assertNotIn("orr", m.blend_vs_slices(m.summarize(res)))  # no benign rows -> no ORR blend


class TestFinalTurnOnly(unittest.TestCase):
    """Multi-turn DPO: earlier turns are prompt context; only the final reply is the completion
    (TRL conversational format masks the prompt, so loss falls on that reply only)."""

    def test_to_trl_keeps_history_in_prompt_and_single_reply_completions(self):
        m = load_script("make_dpo_condition")
        hist = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}]
        r = m.to_trl({"prompt": hist, "chosen": "c", "rejected": "r", "pair_id": "p", "kind": "safety",
                      "profile": "te"})
        self.assertEqual(r["prompt"], hist)
        self.assertEqual(r["prompt"][-1]["role"], "user")
        self.assertEqual(r["chosen"], [{"role": "assistant", "content": "c"}])
        self.assertEqual(r["rejected"], [{"role": "assistant", "content": "r"}])


if __name__ == "__main__":
    unittest.main()
