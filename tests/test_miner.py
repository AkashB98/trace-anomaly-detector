"""Tests: baseline mining math."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miner import (dumps_baseline, load_baseline, mean_std, mine,  # noqa: E402
                   percentile, save_baseline)
from tests.common import constant_tool_baseline, make_meta, make_session, make_step  # noqa: E402


class TestPercentile(unittest.TestCase):
    def test_p50(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2)

    def test_p99_nearest_rank(self):
        vals = list(range(1, 101))
        self.assertEqual(percentile(vals, 99), 99)

    def test_single(self):
        self.assertEqual(percentile([7.5], 99), 7.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            percentile([], 50)


class TestMeanStd(unittest.TestCase):
    def test_known(self):
        m, s = mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(m, 5.0)
        self.assertAlmostEqual(s, 2.0)

    def test_empty(self):
        self.assertEqual(mean_std([]), (0.0, 0.0))


class TestMine(unittest.TestCase):
    def test_latency_percentiles(self):
        # durations 10..100 step 10 -> p50=50, p99=100
        sessions = []
        for i, d in enumerate(range(10, 101, 10)):
            meta, steps = make_session(["t"], trace_id=f"trace-{i:04d}",
                                       duration_ms=float(d))
            sessions.append((meta, steps))
        b = mine(sessions)
        self.assertEqual(b["tools"]["t"]["latency"]["p50"], 50.0)
        self.assertEqual(b["tools"]["t"]["latency"]["p99"], 100.0)

    def test_per_session_counts(self):
        sessions = []
        for i in range(10):
            meta, steps = make_session(["t", "t", "u"], trace_id=f"trace-{i:04d}")
            sessions.append((meta, steps))
        b = mine(sessions)
        self.assertEqual(b["tools"]["t"]["per_session"]["mean"], 2.0)
        self.assertEqual(b["tools"]["t"]["per_session"]["std"], 0.0)
        self.assertEqual(b["tools"]["u"]["per_session"]["mean"], 1.0)

    def test_error_rate(self):
        sessions = []
        for i in range(4):
            result = {"error": "boom"} if i < 1 else {"status": "ok"}
            meta, steps = make_session(["t"], trace_id=f"trace-{i:04d}",
                                       result=result)
            sessions.append((meta, steps))
        b = mine(sessions)
        self.assertAlmostEqual(b["tools"]["t"]["error_rate"], 0.25)
        self.assertAlmostEqual(b["global_error_rate"], 0.25)

    def test_bigrams(self):
        sessions = [make_session(["a", "b", "a"], trace_id="trace-0001")]
        b = mine(sessions)
        self.assertEqual(b["bigrams"]["a\x1fb"], 1)
        self.assertEqual(b["bigrams"]["b\x1fa"], 1)

    def test_arg_vocab_counts(self):
        b = constant_tool_baseline(n=10, args={"sku": "HL-1001"})
        entry = b["arg_values"]["t"]["sku"]
        self.assertEqual(entry["n"], 10)
        self.assertEqual(entry["counts"], {'"HL-1001"': 10})
        self.assertFalse(entry["high_cardinality"])

    def test_high_cardinality_arg_detected(self):
        sessions = []
        for i in range(60):
            meta, steps = make_session(["t"], trace_id=f"trace-{i:04d}",
                                       args={"order_id": f"ORD-{10000 + i}"})
            sessions.append((meta, steps))
        b = mine(sessions)
        entry = b["arg_values"]["t"]["order_id"]
        self.assertTrue(entry["high_cardinality"])
        self.assertEqual(entry["counts"], {})  # not stored

    def test_session_length_stats(self):
        sessions = [make_session(["a", "b"], trace_id="t1"),
                    make_session(["a", "b", "c", "d"], trace_id="t2")]
        b = mine(sessions)
        self.assertEqual(b["session"]["mean"], 3.0)
        self.assertEqual(b["session"]["min"], 2)
        self.assertEqual(b["session"]["max"], 4)

    def test_deterministic(self):
        from corpus import generate_normal_sessions
        s1 = generate_normal_sessions(50, seed=11)
        s2 = generate_normal_sessions(50, seed=11)
        self.assertEqual(dumps_baseline(mine(s1)), dumps_baseline(mine(s2)))

    def test_save_load_roundtrip(self):
        b = constant_tool_baseline(n=5)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            save_baseline(b, p)
            b2 = load_baseline(p)
        self.assertEqual(dumps_baseline(b), dumps_baseline(b2))

    def test_bad_baseline_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            with open(p, "w") as fh:
                fh.write('{"format": "nope"}')
            with self.assertRaises(ValueError):
                load_baseline(p)


if __name__ == "__main__":
    unittest.main()
