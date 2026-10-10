"""One-command end-to-end demo with internal assertions.

Builds a small fictional corpus, mines a baseline, then scores one session
per anomaly type (plus a benign control) and asserts the detector reports
exactly what was scripted. Run with: ``python demo.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import generate_labeled, generate_normal_sessions  # noqa: E402
from detector import score_many  # noqa: E402
from miner import mine  # noqa: E402


def main() -> int:
    print("trace-anomaly-detector demo")
    print("=" * 60)

    print("[1/4] generating 300 benign Helios sessions (seed=42) ...")
    normal = generate_normal_sessions(300, seed=42)

    print("[2/4] mining baseline ...")
    baseline = mine(normal)
    print(f"      tools={len(baseline['tools'])} "
          f"bigrams={len(baseline['bigrams'])} "
          f"global_error_rate={baseline['global_error_rate']}")

    print("[3/4] scoring scripted sessions (one per anomaly type + benign) ...")
    labeled = generate_labeled(seed=2026)
    by_label = {}
    for meta, steps, label in labeled:
        by_label.setdefault(label, []).append((meta, steps))
    demo_sessions = [by_label[l][0] for l in
                     ["latency_spike", "token_blowup", "tool_usage_drift",
                      "rare_arg_value", "new_tool_never_seen", "error_rate_surge",
                      "unusual_tool_sequence", "session_length_anomaly", "benign"]]
    expected = {"latency_spike": "latency_spike", "token_blowup": "token_blowup",
                "tool_usage_drift": "tool_usage_drift",
                "rare_arg_value": "rare_arg_value",
                "new_tool_never_seen": "new_tool_never_seen",
                "error_rate_surge": "error_rate_surge",
                "unusual_tool_sequence": "unusual_tool_sequence",
                "session_length_anomaly": "session_length_anomaly",
                "benign": None}
    reports = score_many(baseline, demo_sessions)

    print("[4/4] verifying ...")
    labels = ["latency_spike", "token_blowup", "tool_usage_drift",
              "rare_arg_value", "new_tool_never_seen", "error_rate_surge",
              "unusual_tool_sequence", "session_length_anomaly", "benign"]
    for label, report in zip(labels, reports):
        types = sorted({f["type"] for f in report["findings"]})
        want = expected[label]
        if want is None:
            assert report["verdict"] == "ok", f"benign session flagged: {types}"
            assert not report["findings"], f"benign session findings: {types}"
        else:
            assert want in types, f"{label}: expected {want}, got {types}"
        print(f"      {label:24s} -> {report['verdict']:10s} {types}")

    print("=" * 60)
    print("demo OK: every scripted anomaly was caught, the benign session was clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
