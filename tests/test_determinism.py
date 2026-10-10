"""Tests: determinism — same seed in, byte-identical artifacts out."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import generate_normal_sessions, session_jsonl  # noqa: E402
from miner import dumps_baseline, mine  # noqa: E402
from traces import canonical  # noqa: E402


class TestDeterminism(unittest.TestCase):
    def test_corpus_byte_identical(self):
        a = [session_jsonl(m, s) for m, s in generate_normal_sessions(60, seed=7)]
        b = [session_jsonl(m, s) for m, s in generate_normal_sessions(60, seed=7)]
        self.assertEqual(a, b)

    def test_corpus_seed_changes_output(self):
        a = [session_jsonl(m, s) for m, s in generate_normal_sessions(20, seed=7)]
        b = [session_jsonl(m, s) for m, s in generate_normal_sessions(20, seed=8)]
        self.assertNotEqual(a, b)

    def test_baseline_byte_identical(self):
        s1 = generate_normal_sessions(40, seed=3)
        s2 = generate_normal_sessions(40, seed=3)
        self.assertEqual(dumps_baseline(mine(s1)), dumps_baseline(mine(s2)))

    def test_eval_report_byte_identical(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))
        import run_evals
        r1 = canonical(run_evals.run())
        r2 = canonical(run_evals.run())
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
