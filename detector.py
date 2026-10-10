"""Anomaly detector: score a trace session against a mined baseline.

Each finding names the anomaly *type*, a *severity* (``warning`` or
``critical``), and *evidence* — the observed number next to the baseline
number it violated, so a human (or CI job) can see exactly why it fired.

Anomaly types
-------------
* ``latency_spike`` — a step took far longer than this tool usually does.
* ``token_blowup`` — a step burned far more tokens than usual.
* ``tool_usage_drift`` — a tool was called far more times in one session
  than its per-session norm (**z-score** — plain-English: how many
  standard deviations away from average the count sits).
* ``rare_arg_value`` — an argument value the baseline rarely or never saw.
* ``new_tool_never_seen`` — a tool absent from the baseline entirely.
* ``error_rate_surge`` — a large share of the session's steps failed.
* ``unusual_tool_sequence`` — a tool-to-tool transition the baseline
  never recorded.
* ``session_length_anomaly`` — a session far longer (or a lone single
  step) versus the usual session length.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from miner import is_error

Session = Tuple[Dict[str, Any], List[Dict[str, Any]]]

#: Tunable detection thresholds. Tests and power users may override per key.
DEFAULT_CONFIG: Dict[str, float] = {
    "latency_warn_ratio": 2.5,   # warn if dur > p99 AND dur >= ratio * p50
    "latency_crit_ratio": 5.0,   # critical if dur >= ratio * p99
    "token_warn_ratio": 3.0,     # warn if toks > p99 AND toks >= ratio * p50
    "token_crit_ratio": 6.0,     # critical if toks >= ratio * p99
    "drift_warn_z": 3.0,         # z-score of per-session call count
    "drift_crit_z": 6.0,
    "std_floor": 0.5,            # floor for std so constant tools still drift
    "rare_relfreq": 0.01,        # arg value rarer than this is suspicious
    "rare_min_observations": 200,  # ...but only with this many baseline samples
    "bigram_min_prev": 50,       # only judge bigrams after a well-seen tool
    "error_warn_rate": 0.30,     # session error-rate thresholds
    "error_crit_rate": 0.50,
    "length_warn_z": 4.0,        # session-length z-score for marathons
}

SEVERITY_ORDER = {"warning": 1, "critical": 2}


def _cfg(config: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


def _finding(ftype: str, severity: str, message: str,
             evidence: Dict[str, Any], tool: Optional[str] = None,
             seq: Optional[int] = None) -> Dict[str, Any]:
    return {
        "type": ftype,
        "severity": severity,
        "tool": tool,
        "seq": seq,
        "message": message,
        "evidence": evidence,
    }


def _known_tool(baseline: Dict[str, Any], tool: Any) -> bool:
    return isinstance(tool, str) and tool in baseline.get("tools", {})


def _step_latency_finding(baseline: Dict[str, Any], step: Dict[str, Any],
                          cfg: Dict[str, float]) -> Optional[Dict[str, Any]]:
    tool = step.get("tool")
    dur = step.get("duration_ms")
    if not _known_tool(baseline, tool):
        return None
    if not isinstance(dur, (int, float)) or dur < 0:
        return None
    prof = baseline["tools"][tool]["latency"]
    p50, p99 = prof.get("p50"), prof.get("p99")
    if p50 is None or p99 is None or p50 <= 0:
        return None
    dur = float(dur)
    if dur > p99 and dur >= cfg["latency_warn_ratio"] * p50:
        severity = "critical" if dur >= cfg["latency_crit_ratio"] * p99 else "warning"
        return _finding(
            "latency_spike", severity,
            f"{tool} took {dur:.1f}ms — {dur / p99:.1f}x its p99 of {p99:.1f}ms",
            {"observed_ms": round(dur, 1), "baseline_p50_ms": p50,
             "baseline_p99_ms": p99, "ratio_vs_p99": round(dur / p99, 2)},
            tool=tool, seq=step.get("seq"))
    return None


def _step_token_finding(baseline: Dict[str, Any], step: Dict[str, Any],
                        cfg: Dict[str, float]) -> Optional[Dict[str, Any]]:
    tool = step.get("tool")
    tk = step.get("tokens")
    if not _known_tool(baseline, tool):
        return None
    if not isinstance(tk, dict):
        return None
    tin, tout = tk.get("input", 0), tk.get("output", 0)
    if not isinstance(tin, (int, float)) or not isinstance(tout, (int, float)):
        return None
    total = float(tin) + float(tout)
    prof = baseline["tools"][tool]["tokens"]
    p50, p99 = prof.get("p50"), prof.get("p99")
    if p50 is None or p99 is None or p50 <= 0:
        return None
    if total > p99 and total >= cfg["token_warn_ratio"] * p50:
        severity = "critical" if total >= cfg["token_crit_ratio"] * p99 else "warning"
        return _finding(
            "token_blowup", severity,
            f"{tool} burned {int(total)} tokens — {total / p99:.1f}x its p99 of {p99:.0f}",
            {"observed_tokens": int(total), "baseline_p50_tokens": p50,
             "baseline_p99_tokens": p99, "ratio_vs_p99": round(total / p99, 2)},
            tool=tool, seq=step.get("seq"))
    return None


def _step_arg_findings(baseline: Dict[str, Any], step: Dict[str, Any],
                       cfg: Dict[str, float]) -> List[Dict[str, Any]]:
    tool = step.get("tool")
    if not _known_tool(baseline, tool):
        return []
    args = step.get("args")
    if not isinstance(args, dict):
        return []
    findings = []
    vocab = baseline.get("arg_values", {}).get(tool, {})
    for name, value in args.items():
        entry = vocab.get(name)
        if entry is None:
            findings.append(_finding(
                "rare_arg_value", "warning",
                f"{tool} called with argument {name!r}, never seen in baseline",
                {"tool": tool, "arg": name, "observed": value,
                 "baseline_observations": 0},
                tool=tool, seq=step.get("seq")))
            continue
        if entry.get("high_cardinality"):
            # Id-like arguments: every value is unique, so rarity is
            # meaningless. Skip.
            continue
        from traces import value_key
        key = value_key(value)
        count = entry["counts"].get(key, 0)
        total = entry["n"]
        if count == 0:
            findings.append(_finding(
                "rare_arg_value", "warning",
                f"{tool}.{name}={value!r} never seen in {total} baseline observations",
                {"tool": tool, "arg": name, "observed": value,
                 "baseline_count": 0, "baseline_observations": total},
                tool=tool, seq=step.get("seq")))
        elif (total >= cfg["rare_min_observations"]
                and count / total < cfg["rare_relfreq"]):
            findings.append(_finding(
                "rare_arg_value", "warning",
                f"{tool}.{name}={value!r} seen {count}/{total} times "
                f"({count / total:.2%} < {cfg['rare_relfreq']:.0%} threshold)",
                {"tool": tool, "arg": name, "observed": value,
                 "baseline_count": count, "baseline_observations": total,
                 "relative_frequency": round(count / total, 6)},
                tool=tool, seq=step.get("seq")))
    return findings


def _new_tool_finding(baseline: Dict[str, Any], step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tool = step.get("tool")
    if isinstance(tool, str) and tool not in baseline.get("tools", {}):
        return _finding(
            "new_tool_never_seen", "critical",
            f"tool {tool!r} was never called in the baseline corpus",
            {"tool": tool, "baseline_tools": sorted(baseline.get("tools", {}))},
            tool=tool, seq=step.get("seq"))
    return None


def _sequence_findings(baseline: Dict[str, Any], steps: List[Dict[str, Any]],
                       cfg: Dict[str, float]) -> List[Dict[str, Any]]:
    findings = []
    bigrams = baseline.get("bigrams", {})
    tools = baseline.get("tools", {})
    prev: Any = None
    for step in steps:
        tool = step.get("tool")
        if (isinstance(prev, str) and isinstance(tool, str) and prev != tool
                and prev in tools and tool in tools):
            key = f"{prev}\x1f{tool}"
            if bigrams.get(key, 0) == 0 and tools[prev].get("n", 0) >= cfg["bigram_min_prev"]:
                findings.append(_finding(
                    "unusual_tool_sequence", "warning",
                    f"{prev} -> {tool}: transition never seen in baseline",
                    {"prev_tool": prev, "tool": tool,
                     "baseline_bigram_count": 0,
                     "baseline_prev_observations": tools[prev].get("n", 0)},
                    tool=tool, seq=step.get("seq")))
        prev = tool
    return findings


def _drift_findings(baseline: Dict[str, Any], steps: List[Dict[str, Any]],
                    cfg: Dict[str, float]) -> List[Dict[str, Any]]:
    counts: Counter = Counter()
    first_seq: Dict[str, int] = {}
    for step in steps:
        tool = step.get("tool")
        if _known_tool(baseline, tool):
            counts[tool] += 1
            first_seq.setdefault(tool, step.get("seq"))
    findings = []
    for tool in sorted(counts):
        prof = baseline["tools"][tool]["per_session"]
        mean, std = prof["mean"], max(prof["std"], cfg["std_floor"])
        if std <= 0:
            continue
        z = (counts[tool] - mean) / std
        if z >= cfg["drift_warn_z"]:
            severity = "critical" if z >= cfg["drift_crit_z"] else "warning"
            findings.append(_finding(
                "tool_usage_drift", severity,
                f"{tool} called {counts[tool]}x in one session "
                f"(baseline {mean:.2f} +/- {prof['std']:.2f}, z={z:.1f})",
                {"tool": tool, "observed_count": counts[tool],
                 "baseline_mean": mean, "baseline_std": prof["std"],
                 "z_score": round(z, 2)},
                tool=tool, seq=first_seq.get(tool)))
    return findings


def _error_finding(baseline: Dict[str, Any], steps: List[Dict[str, Any]],
                   cfg: Dict[str, float]) -> Optional[Dict[str, Any]]:
    n = len(steps)
    if n == 0:
        return None
    errs = sum(1 for s in steps if is_error(s.get("result")))
    rate = errs / n
    if rate >= cfg["error_crit_rate"]:
        severity = "critical"
    elif rate >= cfg["error_warn_rate"]:
        severity = "warning"
    else:
        return None
    return _finding(
        "error_rate_surge", severity,
        f"{errs}/{n} steps failed ({rate:.0%}) vs baseline "
        f"{baseline.get('global_error_rate', 0):.2%}",
        {"failed_steps": errs, "total_steps": n,
         "observed_error_rate": round(rate, 4),
         "baseline_error_rate": baseline.get("global_error_rate", 0)},
        tool=None, seq=None)


def _length_finding(baseline: Dict[str, Any], steps: List[Dict[str, Any]],
                    cfg: Dict[str, float]) -> Optional[Dict[str, Any]]:
    prof = baseline.get("session", {})
    mean, std = prof.get("mean", 0.0), prof.get("std", 0.0)
    n = len(steps)
    if n == 0 or mean <= 0:
        return None
    if n <= 1 and mean >= 3:
        return _finding(
            "session_length_anomaly", "warning",
            f"session ran a single step; baseline averages {mean:.1f} steps",
            {"observed_steps": n, "baseline_mean_steps": mean,
             "baseline_std_steps": std},
            tool=None, seq=None)
    if std > 0:
        z = (n - mean) / std
        if z >= cfg["length_warn_z"]:
            return _finding(
                "session_length_anomaly", "warning",
                f"session ran {n} steps (baseline {mean:.1f} +/- {std:.1f}, z={z:.1f})",
                {"observed_steps": n, "baseline_mean_steps": mean,
                 "baseline_std_steps": std, "z_score": round(z, 2)},
                tool=None, seq=None)
    return None


def score_session(baseline: Dict[str, Any], meta: Dict[str, Any],
                  steps: List[Dict[str, Any]],
                  config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Score one session -> report dict with verdict + findings."""
    cfg = _cfg(config)
    findings: List[Dict[str, Any]] = []

    for step in steps:
        new_tool = _new_tool_finding(baseline, step)
        if new_tool is not None:
            findings.append(new_tool)
            continue  # no baseline exists for this tool; skip per-tool checks
        lat = _step_latency_finding(baseline, step, cfg)
        if lat is not None:
            findings.append(lat)
        tok = _step_token_finding(baseline, step, cfg)
        if tok is not None:
            findings.append(tok)
        findings.extend(_step_arg_findings(baseline, step, cfg))

    findings.extend(_sequence_findings(baseline, steps, cfg))
    findings.extend(_drift_findings(baseline, steps, cfg))
    err = _error_finding(baseline, steps, cfg)
    if err is not None:
        findings.append(err)
    length = _length_finding(baseline, steps, cfg)
    if length is not None:
        findings.append(length)

    findings.sort(key=lambda f: (f["seq"] if f["seq"] is not None else 0, f["type"]))

    if any(f["severity"] == "critical" for f in findings):
        verdict = "critical"
    elif findings:
        verdict = "suspicious"
    else:
        verdict = "ok"

    return {
        "trace_id": meta.get("trace_id"),
        "scenario": meta.get("scenario"),
        "n_steps": len(steps),
        "verdict": verdict,
        "findings": findings,
    }


def score_many(baseline: Dict[str, Any], sessions: List[Session],
               config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    reports = [score_session(baseline, meta, steps, config) for meta, steps in sessions]
    reports.sort(key=lambda r: str(r.get("trace_id")))
    return reports


def summarize(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Counter = Counter()
    by_type: Counter = Counter()
    for r in reports:
        counts[r["verdict"]] += 1
        for f in r["findings"]:
            by_type[f["type"]] += 1
    return {
        "sessions": len(reports),
        "ok": counts.get("ok", 0),
        "suspicious": counts.get("suspicious", 0),
        "critical": counts.get("critical", 0),
        "findings_by_type": dict(sorted(by_type.items())),
    }
