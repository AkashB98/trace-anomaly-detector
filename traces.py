"""Minimal JSONL trace I/O, compatible with tool-call-trace-replay.

Trace files are JSONL: the first line is a ``meta`` record, every following
line is a ``step`` record::

    {"kind": "meta", "trace_id": ..., "agent": ..., "scenario": ..., ...}
    {"kind": "step", "seq": 1, "observation": ..., "tool": ...,
     "args": {...}, "result": {...}, "duration_ms": 42.5,
     "tokens": {"input": 120, "output": 300}, ...}

``duration_ms`` and ``tokens`` are optional. ``tokens`` is an *extension*
field this project adds on top of the trace-replay format (the replay
debugger simply ignores fields it does not know about).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

KIND_META = "meta"
KIND_STEP = "step"

META_REQUIRED = ("trace_id",)
STEP_REQUIRED = ("seq", "tool")


def canonical(obj: Any) -> str:
    """Deterministic JSON encoding: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def value_key(value: Any) -> str:
    """Canonical string key for an argument value (for frequency tables)."""
    return canonical(value)


def load_trace(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a trace file -> (meta, steps). Raises ValueError on bad input."""
    with open(path, "r", encoding="utf-8") as fh:
        return loads_trace(fh.read(), source=path)


def loads_trace(text: str, source: str = "<string>") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{source}:{lineno}: record is not a JSON object")
        kind = obj.get("kind")
        if kind == KIND_META:
            meta = {k: v for k, v in obj.items() if k != "kind"}
        elif kind == KIND_STEP:
            steps.append(obj)
        else:
            raise ValueError(f"{source}:{lineno}: unknown record kind {kind!r}")
    if not meta:
        raise ValueError(f"{source}: trace has no meta record")
    for field in META_REQUIRED:
        if field not in meta:
            raise ValueError(f"{source}: meta record missing {field!r}")
    steps.sort(key=lambda s: s.get("seq", 0))
    for i, step in enumerate(steps, start=1):
        for field in STEP_REQUIRED:
            if field not in step:
                raise ValueError(f"{source}: step {i} missing {field!r}")
    return meta, steps


def write_trace(path: str, meta: Dict[str, Any], steps: List[Dict[str, Any]]) -> None:
    """Write (meta, steps) to a JSONL trace file, canonical form."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps_trace(meta, steps))


def dumps_trace(meta: Dict[str, Any], steps: List[Dict[str, Any]]) -> str:
    lines = [canonical({"kind": KIND_META, **meta})]
    for step in steps:
        lines.append(canonical({"kind": KIND_STEP, **step}))
    return "\n".join(lines) + "\n"
