"""Conversation grouping for STRATA logs (synthetic entries only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.curation.strata_logs import conversation_id, group_conversations  # noqa: E402


def entry(stage, req, it, prof, goal, turn, resp="r", prompt="p", **kw):
    return dict(stage=stage, request_id=req, iteration=it, profile=prof, request=goal, turn=turn,
                clean_prompt=f"{prompt}{turn}", response=f"{resp}{turn}", score=1.0, **kw)


class TestGroupConversations(unittest.TestCase):
    def test_eight_passes_with_iteration_zero_stay_separate(self):
        # the lifelong/test logs: the same goal on 8 surfaces, iteration == 0 for every pass
        surfaces = ["en", "hi", "hi_en_mix", "hinglish", "te", "te_en_mix", "ja", "ja_en_mix"]
        entries = [entry("test", 0, 0, s, "goal A", t) for s in surfaces for t in (1, 2)]
        groups, dup = group_conversations(entries)
        self.assertEqual(len(groups), 8)
        self.assertEqual(dup, 0)
        for turns in groups.values():
            self.assertEqual([e["turn"] for e in turns], [1, 2])
            self.assertEqual(len({e["profile"] for e in turns}), 1)

    def test_same_request_id_in_two_runs_with_different_goals_stays_separate(self):
        a = [entry("lifelong", 0, 0, "en", "goal A", 1)]
        b = [entry("lifelong", 0, 0, "en", "goal B", 1)]  # a top-up run restarts request_id at 0
        groups, _ = group_conversations(a + b)
        self.assertEqual(len(groups), 2)

    def test_entries_repeated_across_logs_are_dropped(self):
        warm = [entry("warm_up", 3, 2, "te", "goal C", t) for t in (1, 2, 3)]
        groups, dup = group_conversations(warm + warm)  # the lifelong log repeats the warm-up entries
        self.assertEqual(dup, 3)
        self.assertEqual([e["turn"] for e in next(iter(groups.values()))], [1, 2, 3])

    def test_ids_are_unique_and_stable(self):
        entries = [entry("test", 0, 0, s, "goal A", 1) for s in ("en", "hi")]
        groups, _ = group_conversations(entries)
        ids = [conversation_id(k) for k in groups]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(ids, [conversation_id(k) for k in groups])

    def test_skips_entries_without_turn_or_response(self):
        groups, _ = group_conversations([{"stage": "x"}, entry("x", 1, 0, "en", "g", 1), "junk"])
        self.assertEqual(len(groups), 1)



class TestCandidatesAcrossLogs(unittest.TestCase):
    def test_warmup_entries_repeated_in_lifelong_log_give_one_candidate(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bpp", os.path.join(os.path.dirname(__file__), "..", "scripts", "build_pair_pool.py"))
        bpp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bpp)
        warm = [dict(entry("warm_up", 0, 1, "te", "goal C", 1), score=9.0)]
        lifelong = warm + [dict(entry("lifelong", 0, 0, p, "goal D", 1), score=9.0) for p in ("en", "hi")]
        cands = list(bpp.conversations_from_strata_log(warm + lifelong, 8.0))  # logs concatenated
        self.assertEqual(len(cands), 3)  # warm-up once, plus the two lifelong surfaces
        self.assertEqual(len({c["conv_key"] for c in cands}), 3)


if __name__ == "__main__":
    unittest.main()
