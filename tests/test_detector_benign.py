"""Tests: benign sessions stay clean (false-positive traps)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import generate_normal_sessions  # noqa: E402
from detector import score_session  # noqa: E402
from miner import mine  # noqa: E402
from tests.common import make_meta, make_session, make_step  # noqa: E402


class TestBenign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = mine(generate_normal_sessions(400, seed=42))

    def assert_clean(self, meta, steps):
        r = score_session(self.baseline, meta, steps)
        self.assertEqual(r["findings"], [],
                         f"benign session flagged: {[f['type'] for f in r['findings']]}")
        self.assertEqual(r["verdict"], "ok")

    def test_plain_normal_session_clean(self):
        meta, steps = generate_normal_sessions(1, seed=999)[0]
        self.assert_clean(meta, steps)

    def test_high_but_normal_latency_clean(self):
        # latency at 90% of p99 is tail noise, not a spike
        from tests.common import constant_tool_baseline
        b = constant_tool_baseline(n=60)
        b["tools"]["t"]["latency"] = {"p50": 100.0, "p99": 200.0, "n": 60}
        b["tools"]["t"]["tokens"] = {"p50": 100.0, "p99": 200.0, "n": 60}
        meta, steps = make_session(["t"], duration_ms=180.0, tokens=(90, 90))
        r = score_session(b, meta, steps)
        self.assertEqual(r["verdict"], "ok")

    def test_double_tool_call_clean(self):
        from tests.common import constant_tool_baseline
        b = constant_tool_baseline(n=60)
        meta, steps = make_session(["t", "t"])
        r = score_session(b, meta, steps)
        self.assertNotIn("tool_usage_drift",
                         {f["type"] for f in r["findings"]})

    def test_in_vocab_args_clean(self):
        meta, steps = generate_normal_sessions(1, seed=1234)[0]
        r = score_session(self.baseline, meta, steps)
        self.assertNotIn("rare_arg_value", {f["type"] for f in r["findings"]})

    def test_single_error_clean(self):
        # 1 failure in 6 steps (16.7%) is below the warn threshold
        steps = [make_step(i + 1, "get_service_status",
                           result={"error": "UpstreamTimeout: x"} if i == 2
                           else {"status": "ok"})
                 for i in range(6)]
        r = score_session(self.baseline, make_meta(), steps)
        self.assertNotIn("error_rate_surge", {f["type"] for f in r["findings"]})


if __name__ == "__main__":
    unittest.main()
