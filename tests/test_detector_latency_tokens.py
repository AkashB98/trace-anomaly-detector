"""Tests: latency_spike and token_blowup detection."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import score_session  # noqa: E402
from tests.common import constant_tool_baseline, make_meta, make_session  # noqa: E402


def tuned_baseline():
    b = constant_tool_baseline(n=60)
    b["tools"]["t"]["latency"] = {"p50": 100.0, "p99": 200.0, "n": 60}
    b["tools"]["t"]["tokens"] = {"p50": 100.0, "p99": 300.0, "n": 60}
    return b


def types_of(report):
    return sorted({f["type"] for f in report["findings"]})


class TestLatencySpike(unittest.TestCase):
    def test_fires_above_p99_and_ratio(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], duration_ms=260.0)  # > p99, >= 2.5*p50
        r = score_session(b, meta, steps)
        self.assertIn("latency_spike", types_of(r))
        self.assertEqual(r["verdict"], "suspicious")

    def test_critical_at_extreme(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], duration_ms=1200.0)  # >= 5*p99
        r = score_session(b, meta, steps)
        sev = [f["severity"] for f in r["findings"] if f["type"] == "latency_spike"]
        self.assertEqual(sev, ["critical"])
        self.assertEqual(r["verdict"], "critical")

    def test_no_fire_below_p99(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], duration_ms=190.0)
        r = score_session(b, meta, steps)
        self.assertNotIn("latency_spike", types_of(r))
        self.assertEqual(r["verdict"], "ok")

    def test_no_fire_above_p99_but_below_ratio(self):
        b = tuned_baseline()
        # 210 > p99=200 but < 2.5*p50=250 -> not a spike, just tail noise
        meta, steps = make_session(["t"], duration_ms=210.0)
        r = score_session(b, meta, steps)
        self.assertNotIn("latency_spike", types_of(r))

    def test_missing_duration_skipped(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"])
        del steps[0]["duration_ms"]
        r = score_session(b, meta, steps)
        self.assertNotIn("latency_spike", types_of(r))

    def test_evidence_has_numbers(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], duration_ms=260.0)
        r = score_session(b, meta, steps)
        f = next(f for f in r["findings"] if f["type"] == "latency_spike")
        self.assertEqual(f["evidence"]["observed_ms"], 260.0)
        self.assertEqual(f["evidence"]["baseline_p99_ms"], 200.0)
        self.assertIn("ratio_vs_p99", f["evidence"])


class TestTokenBlowup(unittest.TestCase):
    def test_fires(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], tokens=(200, 200))  # 400 > 300, >= 3*100
        r = score_session(b, meta, steps)
        self.assertIn("token_blowup", types_of(r))

    def test_critical_at_extreme(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], tokens=(1000, 1000))  # 2000 >= 6*300
        r = score_session(b, meta, steps)
        sev = [f["severity"] for f in r["findings"] if f["type"] == "token_blowup"]
        self.assertEqual(sev, ["critical"])

    def test_no_fire_below_p99(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"], tokens=(100, 150))
        r = score_session(b, meta, steps)
        self.assertNotIn("token_blowup", types_of(r))

    def test_no_fire_above_p99_but_below_ratio(self):
        b = tuned_baseline()
        # 320 > p99=300 but < 3*p50=300? 320 >= 300 -> fires. use 305? still >=300.
        # p50=100 -> need total < 300 while > p99=300: impossible, so retune:
        b["tools"]["t"]["tokens"] = {"p50": 500.0, "p99": 300.0, "n": 60}
        meta, steps = make_session(["t"], tokens=(160, 160))  # 320 > 300, < 3*500
        r = score_session(b, meta, steps)
        self.assertNotIn("token_blowup", types_of(r))

    def test_missing_tokens_skipped(self):
        b = tuned_baseline()
        meta, steps = make_session(["t"])
        del steps[0]["tokens"]
        r = score_session(b, meta, steps)
        self.assertNotIn("token_blowup", types_of(r))


if __name__ == "__main__":
    unittest.main()
