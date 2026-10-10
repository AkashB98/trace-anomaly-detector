"""Baseline miner: learn "normal" from a corpus of benign traces.

The baseline is a statistical profile of how an agent *usually* behaves:

* per tool: latency **percentiles** (p50/p99), token percentiles, and how
  many times the tool is called per session (mean/std) — a *distribution*,
  plain-English: the full spread of observed values, summarized;
* per (tool, argument) pair: a frequency table of argument values seen —
  the tool's *vocabulary*, plain-English: the set of values considered normal;
* per ordered tool pair: how often one tool directly follows another
  (**bigram** — plain-English: a two-step sequence) transitions;
* per session: step-count distribution and the global error rate.

The baseline serializes to deterministic JSON (sorted keys), so mining the
same corpus twice yields byte-identical files.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from traces import canonical, value_key

BASELINE_FORMAT = "trace-anomaly-baseline/1"

#: Cap on stored values per (tool, arg) so one high-cardinality argument
#: (e.g. order_id) cannot blow up the baseline file.
MAX_ARG_VALUES = 200

Session = Tuple[Dict[str, Any], List[Dict[str, Any]]]


def percentile(sorted_vals: List[float], p: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    if not sorted_vals:
        raise ValueError("percentile of empty list")
    n = len(sorted_vals)
    k = math.ceil(p / 100.0 * n)
    return sorted_vals[max(0, min(n - 1, k - 1))]


def mean_std(vals: List[float]) -> Tuple[float, float]:
    """Population mean and std. Empty -> (0.0, 0.0)."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, math.sqrt(var)


def is_error(result: Any) -> bool:
    """A step failed if its result carries an error marker."""
    if not isinstance(result, dict):
        return False
    if "error" in result:
        return True
    return str(result.get("status", "")).lower() in ("error", "failed")


def _round6(x: float) -> float:
    return round(float(x), 6)


def mine(sessions: Iterable[Session]) -> Dict[str, Any]:
    """Build a baseline dict from benign (meta, steps) sessions."""
    lat: Dict[str, List[float]] = defaultdict(list)
    toks: Dict[str, List[float]] = defaultdict(list)
    errors: Dict[str, int] = Counter()
    n_calls: Dict[str, int] = Counter()
    per_session_counts: Dict[str, List[int]] = defaultdict(list)
    arg_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    arg_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bigrams: Counter = Counter()
    session_lengths: List[int] = []
    total_errors = 0
    total_steps = 0
    n_sessions = 0

    for _meta, steps in sessions:
        n_sessions += 1
        session_lengths.append(len(steps))
        counts: Counter = Counter()
        prev_tool: Any = None
        for step in steps:
            tool = step.get("tool")
            if not isinstance(tool, str):
                continue
            total_steps += 1
            counts[tool] += 1
            n_calls[tool] += 1
            dur = step.get("duration_ms")
            if isinstance(dur, (int, float)) and dur >= 0:
                lat[tool].append(round(float(dur), 4))
            tk = step.get("tokens")
            if isinstance(tk, dict):
                tin = tk.get("input", 0)
                tout = tk.get("output", 0)
                if isinstance(tin, (int, float)) and isinstance(tout, (int, float)):
                    toks[tool].append(float(tin) + float(tout))
            if is_error(step.get("result")):
                errors[tool] += 1
                total_errors += 1
            args = step.get("args")
            if isinstance(args, dict):
                for name, value in args.items():
                    arg_counts[tool][name][value_key(value)] += 1
                    arg_totals[tool][name] += 1
            if prev_tool is not None:
                bigrams[(prev_tool, tool)] += 1
            prev_tool = tool
        for tool, c in counts.items():
            per_session_counts[tool].append(c)

    tools: Dict[str, Dict[str, Any]] = {}
    for tool in sorted(set(n_calls) | set(lat) | set(toks)):
        lat_sorted = sorted(lat.get(tool, []))
        tok_sorted = sorted(toks.get(tool, []))
        mean_c, std_c = mean_std([float(c) for c in per_session_counts.get(tool, [])])
        n = n_calls.get(tool, 0)
        tools[tool] = {
            "n": n,
            "latency": {
                "p50": _round6(percentile(lat_sorted, 50)) if lat_sorted else None,
                "p99": _round6(percentile(lat_sorted, 99)) if lat_sorted else None,
                "n": len(lat_sorted),
            },
            "tokens": {
                "p50": _round6(percentile(tok_sorted, 50)) if tok_sorted else None,
                "p99": _round6(percentile(tok_sorted, 99)) if tok_sorted else None,
                "n": len(tok_sorted),
            },
            "per_session": {
                "mean": _round6(mean_c),
                "std": _round6(std_c),
                "max": max(per_session_counts.get(tool, [0])),
            },
            "error_rate": _round6(errors.get(tool, 0) / n) if n else 0.0,
        }

    arg_values: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tool in sorted(arg_counts):
        arg_values[tool] = {}
        for name in sorted(arg_counts[tool]):
            counter = arg_counts[tool][name]
            total = arg_totals[tool][name]
            distinct = len(counter)
            # High-cardinality arguments (ids): every value is unique, so
            # "never seen before" is meaningless. The detector skips
            # rare-value checks for these.
            high_card = distinct > 50 and (distinct / total) > 0.3 if total else False
            top = counter.most_common(MAX_ARG_VALUES)
            # most_common ties broken by insertion order; re-sort by
            # (-count, value) for determinism.
            top = sorted(top, key=lambda kv: (-kv[1], kv[0]))
            arg_values[tool][name] = {
                "n": total,
                "high_cardinality": high_card,
                # Don't waste baseline space on id vocabularies.
                "counts": {} if high_card else {k: v for k, v in top},
            }

    bigram_out = {f"{a}\x1f{b}": c for (a, b), c in
                  sorted(bigrams.items(), key=lambda kv: (kv[0][0], kv[0][1]))}

    s_mean, s_std = mean_std([float(x) for x in session_lengths])

    return {
        "format": BASELINE_FORMAT,
        "meta": {
            "mined_from_sessions": n_sessions,
            "mined_from_steps": total_steps,
            "corpus": "helios-fictional-v1",
        },
        "tools": tools,
        "arg_values": arg_values,
        "bigrams": bigram_out,
        "session": {
            "n": n_sessions,
            "mean": _round6(s_mean),
            "std": _round6(s_std),
            "min": min(session_lengths) if session_lengths else 0,
            "max": max(session_lengths) if session_lengths else 0,
        },
        "global_error_rate": _round6(total_errors / total_steps) if total_steps else 0.0,
    }


def dumps_baseline(baseline: Dict[str, Any]) -> str:
    return canonical(baseline) + "\n"


def save_baseline(baseline: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps_baseline(baseline))


def load_baseline(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict) or obj.get("format") != BASELINE_FORMAT:
        raise ValueError(f"{path}: not a {BASELINE_FORMAT} baseline file")
    return obj
