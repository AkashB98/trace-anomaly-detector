"""Tests: tool_usage_drift, rare_arg_value, new_tool_never_seen."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import score_session  # noqa: E402
from tests.common import constant_tool_baseline, make_meta, make_session  # noqa: E402


def types_of(report):
    return sorted({f["type"] for f in report["findings"]})


class TestToolUsageDrift(unittest.TestCase):
    def setUp(self):
        # "t" appears exactly once per session: mean=1, std=0 -> floor 0.5
        self.b = constant_tool_baseline(n=40)

    def test_fires_on_repetition(self):
        meta, steps = make_session(["t", "t", "t"])  # z = (3-1)/0.5 = 4
        r = score_session(self.b, meta, steps)
        self.assertIn("tool_usage_drift", types_of(r))
        f = next(f for f in r["findings"] if f["type"] == "tool_usage_drift")
        self.assertAlmostEqual(f["evidence"]["z_score"], 4.0)

    def test_critical_on_hammering(self):
        meta, steps = make_session(["t"] * 10)  # z = 18
        r = score_session(self.b, meta, steps)
        sev = [f["severity"] for f in r["findings"] if f["type"] == "tool_usage_drift"]
        self.assertEqual(sev, ["critical"])

    def test_no_fire_on_double_call(self):
        meta, steps = make_session(["t", "t"])  # z = 2 < 3
        r = score_session(self.b, meta, steps)
        self.assertNotIn("tool_usage_drift", types_of(r))

    def test_no_fire_on_single_call(self):
        meta, steps = make_session(["t"])
        r = score_session(self.b, meta, steps)
        self.assertNotIn("tool_usage_drift", types_of(r))

    def test_evidence_numbers(self):
        meta, steps = make_session(["t"] * 4)
        r = score_session(self.b, meta, steps)
        f = next(f for f in r["findings"] if f["type"] == "tool_usage_drift")
        self.assertEqual(f["evidence"]["observed_count"], 4)
        self.assertEqual(f["evidence"]["baseline_mean"], 1.0)


class TestRareArgValue(unittest.TestCase):
    def setUp(self):
        self.b = constant_tool_baseline(n=60, args={"sku": "HL-1001"})

    def test_fires_on_never_seen_value(self):
        meta, steps = make_session(["t"], args={"sku": "ZZZ-9999"})
        r = score_session(self.b, meta, steps)
        self.assertIn("rare_arg_value", types_of(r))
        f = next(f for f in r["findings"] if f["type"] == "rare_arg_value")
        self.assertEqual(f["evidence"]["baseline_count"], 0)

    def test_no_fire_on_common_value(self):
        meta, steps = make_session(["t"], args={"sku": "HL-1001"})
        r = score_session(self.b, meta, steps)
        self.assertNotIn("rare_arg_value", types_of(r))

    def test_fires_on_never_seen_arg_name(self):
        meta, steps = make_session(["t"], args={"sku": "HL-1001", "mystery": 1})
        r = score_session(self.b, meta, steps)
        self.assertIn("rare_arg_value", types_of(r))

    def test_high_cardinality_arg_skipped(self):
        sessions = []
        for i in range(60):
            meta, steps = make_session(["t"], trace_id=f"trace-{i:04d}",
                                       args={"order_id": f"ORD-{10000 + i}"})
            sessions.append((meta, steps))
        from miner import mine
        b = mine(sessions)
        self.assertTrue(b["arg_values"]["t"]["order_id"]["high_cardinality"])
        meta, steps = make_session(["t"], args={"order_id": "ORD-99999"})
        r = score_session(b, meta, steps)
        self.assertNotIn("rare_arg_value", types_of(r))

    def test_small_sample_relfreq_does_not_fire(self):
        # 1/101 < 1% is a small-sample fluke, not a real rarity signal.
        b = constant_tool_baseline(n=101, args={"sku": "HL-1001"})
        # hand-inject one rare observation to simulate the fluke
        b["arg_values"]["t"]["sku"]["counts"]['"HL-1006"'] = 1
        b["arg_values"]["t"]["sku"]["n"] = 101
        meta, steps = make_session(["t"], args={"sku": "HL-1006"})
        r = score_session(b, meta, steps)
        self.assertNotIn("rare_arg_value", types_of(r))

    def test_relfreq_fires_with_enough_samples(self):
        b = constant_tool_baseline(n=300, args={"sku": "HL-1001"})
        b["arg_values"]["t"]["sku"]["counts"]['"HL-1006"'] = 2
        b["arg_values"]["t"]["sku"]["n"] = 300
        meta, steps = make_session(["t"], args={"sku": "HL-1006"})
        r = score_session(b, meta, steps, config={"rare_min_observations": 200})
        self.assertIn("rare_arg_value", types_of(r))


class TestNewTool(unittest.TestCase):
    def test_fires_critical(self):
        b = constant_tool_baseline(n=20)
        meta, steps = make_session(["t", "exec_shell"])
        r = score_session(b, meta, steps)
        self.assertIn("new_tool_never_seen", types_of(r))
        sev = [f["severity"] for f in r["findings"]
               if f["type"] == "new_tool_never_seen"]
        self.assertEqual(sev, ["critical"])
        self.assertEqual(r["verdict"], "critical")

    def test_known_tools_clean(self):
        b = constant_tool_baseline(n=20)
        meta, steps = make_session(["t"])
        r = score_session(b, meta, steps)
        self.assertNotIn("new_tool_never_seen", types_of(r))


if __name__ == "__main__":
    unittest.main()
