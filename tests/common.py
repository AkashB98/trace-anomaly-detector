"""Shared test helpers: synthetic sessions and baselines."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miner import mine  # noqa: E402

Session = Tuple[Dict[str, Any], List[Dict[str, Any]]]


def make_meta(trace_id: str = "trace-0001", scenario: str = "test") -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "agent": "test-agent",
        "scenario": scenario,
        "fixture_version": 1,
        "created_at": "2026-01-05T09:00:00+00:00",
        "format": "tool-call-trace/1",
    }


def make_step(seq: int, tool: str, duration_ms: float = 50.0,
              tokens: Tuple[int, int] = (100, 200),
              args: Dict[str, Any] = None, result: Dict[str, Any] = None,
              observation: str = "obs") -> Dict[str, Any]:
    return {
        "seq": seq,
        "observation": observation,
        "tool": tool,
        "args": dict(args) if args else {},
        "result": dict(result) if result is not None else {"status": "ok"},
        "duration_ms": duration_ms,
        "tokens": {"input": tokens[0], "output": tokens[1]},
        "timestamp": "2026-01-05T09:00:00+00:00",
        "request_id": f"req-{seq:02d}",
    }


def make_session(tools: List[str], trace_id: str = "trace-0001",
                 **step_kw) -> Session:
    steps = [make_step(i + 1, tool, **step_kw) for i, tool in enumerate(tools)]
    return make_meta(trace_id), steps


def mine_sessions(sessions: List[Session]) -> Dict[str, Any]:
    return mine(sessions)


def constant_tool_baseline(tool: str = "t", n: int = 60, duration_ms: float = 100.0,
                           tokens: Tuple[int, int] = (100, 200),
                           args: Dict[str, Any] = None) -> Dict[str, Any]:
    """Baseline where *tool* appears once per session with fixed stats."""
    sessions = []
    for i in range(n):
        meta, steps = make_session([tool], trace_id=f"trace-{i:04d}",
                                   duration_ms=duration_ms, tokens=tokens,
                                   args=args or {"sku": "HL-1001"})
        sessions.append((meta, steps))
    return mine(sessions)
