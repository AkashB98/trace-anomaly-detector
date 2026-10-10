"""Golden evals: scripted anomalies with known labels -> detection metrics.

Builds a baseline from 700 benign sessions (seed=42), then scores 35
labeled sessions (seed=2026): 23 scripted anomalies across the 8 anomaly
types plus 12 benign controls. For each anomaly type we measure *recall*
(plain-English: the fraction of scripted anomalies the detector actually
caught); on the benign controls we measure the *false-positive rate*
(plain-English: the fraction of clean sessions wrongly flagged).

Writes evals/eval_report.json. Fully deterministic: fixed seeds, sorted
keys, no timestamps — the file is byte-identical across runs.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import generate_labeled, generate_normal_sessions  # noqa: E402
from detector import score_session  # noqa: E402
from miner import mine  # noqa: E402
from traces import canonical  # noqa: E402

CORPUS_SEED = 42
CORPUS_SESSIONS = 700
EVAL_SEED = 2026
ANOMALY_LABELS = ["latency_spike", "token_blowup", "tool_usage_drift",
                  "rare_arg_value", "new_tool_never_seen", "error_rate_surge",
                  "unusual_tool_sequence", "session_length_anomaly"]


def run() -> dict:
    normal = generate_normal_sessions(CORPUS_SESSIONS, seed=CORPUS_SEED)
    baseline = mine(normal)
    labeled = generate_labeled(seed=EVAL_SEED)

    per_type = {label: {"total": 0, "detected": 0, "trace_ids": []}
                for label in ANOMALY_LABELS}
    benign_total = 0
    benign_flagged = 0
    session_rows = []

    for meta, steps, label in labeled:
        report = score_session(baseline, meta, steps)
        types = sorted({f["type"] for f in report["findings"]})
        if label == "benign":
            benign_total += 1
            flagged = bool(report["findings"])
            benign_flagged += flagged
            session_rows.append({
                "trace_id": meta["trace_id"], "label": label,
                "verdict": report["verdict"], "flagged": flagged,
                "finding_types": types,
            })
        else:
            entry = per_type[label]
            entry["total"] += 1
            hit = label in types
            entry["detected"] += hit
            entry["trace_ids"].append(meta["trace_id"])
            session_rows.append({
                "trace_id": meta["trace_id"], "label": label,
                "verdict": report["verdict"], "detected": hit,
                "finding_types": types,
            })

    per_type_out = {}
    for label in ANOMALY_LABELS:
        e = per_type[label]
        per_type_out[label] = {
            "total": e["total"],
            "detected": e["detected"],
            "recall": round(e["detected"] / e["total"], 4) if e["total"] else 0.0,
        }

    fp_rate = round(benign_flagged / benign_total, 4) if benign_total else 0.0
    all_passed = (all(e["detected"] == e["total"] and e["total"] > 0
                      for e in per_type.values())
                  and benign_flagged == 0)

    return {
        "dataset": {
            "corpus_seed": CORPUS_SEED,
            "corpus_sessions": CORPUS_SESSIONS,
            "eval_seed": EVAL_SEED,
            "labeled_sessions": len(labeled),
            "anomaly_labels": ANOMALY_LABELS,
        },
        "per_type": per_type_out,
        "benign_controls": {
            "total": benign_total,
            "flagged": benign_flagged,
            "false_positive_rate": fp_rate,
        },
        "sessions": sorted(session_rows, key=lambda r: r["trace_id"]),
        "all_passed": all_passed,
    }


def main() -> int:
    report = run()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(canonical(report) + "\n")
    n = sum(v["total"] for v in report["per_type"].values())
    d = sum(v["detected"] for v in report["per_type"].values())
    print(f"evals: {d}/{n} anomalies detected, "
          f"benign FP {report['benign_controls']['false_positive_rate']}, "
          f"all_passed={report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
