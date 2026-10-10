"""Tests: unusual_tool_sequence, error_rate_surge, session_length_anomaly."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import score_session  # noqa: E402
from miner import mine  # noqa: E402
from tests.common import make_meta, make_session, make_step  # noqa: E402


def types_of(report):
    return sorted({f["type"] for f in report["findings"]})


def seq_baseline(n=30):
    """Baseline where a->b is the only transition, seen n times."""
    sessions = [make_session(["a", "b"], trace_id=f"trace-{i:04d}") for i in range(n)]
    return mine(sessions)


class TestUnusualSequence(unittest.TestCase):
    def test_fires_on_unseen_transition(self):
        b = seq_baseline(60)
        meta, steps = make_session(["a", "c"])
        # "c" must be a known tool, else new_tool fires instead
        b["tools"]["c"] = {"n": 60, "latency": {"p50": 40, "p99": 100, "n": 60},
                           "tokens": {"p50": 200, "p99": 400, "n": 60},
                           "per_session": {"mean": 1.0, "std": 0.0, "max": 1},
                           "error_rate": 0.0}
        r = score_session(b, meta, steps, config={"bigram_min_prev": 50})
        self.assertIn("unusual_tool_sequence", types_of(r))

    def test_no_fire_on_known_transition(self):
        b = seq_baseline(60)
        meta, steps = make_session(["a", "b"])
        r = score_session(b, meta, steps, config={"bigram_min_prev": 50})
        self.assertNotIn("unusual_tool_sequence", types_of(r))

    def test_self_loop_skipped(self):
        b = seq_baseline(60)
        meta, steps = make_session(["a", "a", "b"])
        r = score_session(b, meta, steps, config={"bigram_min_prev": 50})
        self.assertNotIn("unusual_tool_sequence", types_of(r))

    def test_rare_prev_tool_not_judged(self):
        # "a" seen only 3 times (< bigram_min_prev) -> no verdict on (a, x)
        b = seq_baseline(3)
        b["tools"]["c"] = {"n": 60, "latency": {"p50": 40, "p99": 100, "n": 60},
                           "tokens": {"p50": 200, "p99": 400, "n": 60},
                           "per_session": {"mean": 1.0, "std": 0.0, "max": 1},
                           "error_rate": 0.0}
        meta, steps = make_session(["a", "c"])
        r = score_session(b, meta, steps, config={"bigram_min_prev": 50})
        self.assertNotIn("unusual_tool_sequence", types_of(r))

    def test_first_step_has_no_bigram(self):
        b = seq_baseline(60)
        meta, steps = make_session(["a"])
        r = score_session(b, meta, steps, config={"bigram_min_prev": 50})
        self.assertNotIn("unusual_tool_sequence", types_of(r))


class TestErrorRateSurge(unittest.TestCase):
    def setUp(self):
        self.b = seq_baseline(20)

    def _session_with_errors(self, n_fail, n_total=6):
        steps = []
        for i in range(n_total):
            result = {"error": "UpstreamError: 503"} if i < n_fail else {"status": "ok"}
            steps.append(make_step(i + 1, "a", result=result))
        return make_meta(), steps

    def test_critical_on_mass_failure(self):
        meta, steps = self._session_with_errors(5)
        r = score_session(self.b, meta, steps)
        sev = [f["severity"] for f in r["findings"] if f["type"] == "error_rate_surge"]
        self.assertEqual(sev, ["critical"])
        self.assertEqual(r["verdict"], "critical")

    def test_warning_on_partial_failure(self):
        meta, steps = self._session_with_errors(2)  # 2/6 = 33% >= 30%
        r = score_session(self.b, meta, steps)
        self.assertIn("error_rate_surge", types_of(r))
        sev = [f["severity"] for f in r["findings"] if f["type"] == "error_rate_surge"]
        self.assertEqual(sev, ["warning"])

    def test_no_fire_on_single_error(self):
        meta, steps = self._session_with_errors(1)  # 1/6 = 16.7% < 30%
        r = score_session(self.b, meta, steps)
        self.assertNotIn("error_rate_surge", types_of(r))

    def test_no_fire_when_clean(self):
        meta, steps = self._session_with_errors(0)
        r = score_session(self.b, meta, steps)
        self.assertNotIn("error_rate_surge", types_of(r))

    def test_evidence_numbers(self):
        meta, steps = self._session_with_errors(3)
        r = score_session(self.b, meta, steps)
        f = next(f for f in r["findings"] if f["type"] == "error_rate_surge")
        self.assertEqual(f["evidence"]["failed_steps"], 3)
        self.assertEqual(f["evidence"]["total_steps"], 6)
        self.assertAlmostEqual(f["evidence"]["observed_error_rate"], 0.5)


class TestSessionLength(unittest.TestCase):
    def setUp(self):
        # sessions of length 4..8 -> mean 6, std ~1.4
        sessions = []
        for i, ln in enumerate([4, 5, 6, 6, 7, 8] * 10):
            meta, steps = make_session(["a"] * ln, trace_id=f"trace-{i:04d}")
            sessions.append((meta, steps))
        self.b = mine(sessions)

    def test_fires_on_marathon(self):
        meta, steps = make_session(["a"] * 20)
        r = score_session(self.b, meta, steps)
        self.assertIn("session_length_anomaly", types_of(r))

    def test_fires_on_single_step(self):
        meta, steps = make_session(["a"])
        r = score_session(self.b, meta, steps)
        self.assertIn("session_length_anomaly", types_of(r))

    def test_no_fire_on_normal_length(self):
        meta, steps = make_session(["a"] * 6)
        r = score_session(self.b, meta, steps)
        self.assertNotIn("session_length_anomaly", types_of(r))

    def test_no_fire_on_empty(self):
        r = score_session(self.b, make_meta(), [])
        self.assertNotIn("session_length_anomaly", types_of(r))


if __name__ == "__main__":
    unittest.main()
