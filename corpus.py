"""Seeded fictional trace corpus generator (Helios theme).

Generates deterministic agent-trace corpora in the tool-call-trace-replay
JSONL format. Two fictional domains, mirroring the sibling builds:

* **Helios Home store** — storefront tools (search/get products, inventory,
  orders, customers, tickets), mirroring mcp-tool-server.
* **Helios Cloud devops** — incident-response tools (service status, logs,
  restarts, scaling, alert acks), mirroring trace-replay's backend.

Everything is derived from an explicit ``random.Random(seed)``: the same
seed always produces byte-identical JSONL. Timestamps and request ids are
synthesized from the session index, never from the wall clock or uuid.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from traces import dumps_trace

# ---------------------------------------------------------------- constants

BASE_TIME = datetime(2026, 1, 5, 9, 0, 0, tzinfo=timezone.utc)
TRACE_FORMAT = "tool-call-trace/1"
AGENT = "helios-agent"
FIXTURE_VERSION = 1

SKUS = [f"HL-{1001 + i}" for i in range(20)]
SERVICES = ["payments-api", "orders-api", "auth-service", "inventory-db", "cdn-edge"]
REGIONS = ["us-east", "us-west", "eu-west"]
CUSTOMERS = [f"C-{101 + i}" for i in range(12)]
QUERIES = ["helios lamp", "smart bulb", "thermostat", "desk lamp", "led strip", "motion sensor"]
SUBJECTS = ["delivery late", "bulb flickers", "app won't pair", "wrong item shipped", "refund request"]

# tool -> (median latency ms, lognormal sigma, mean input tokens, mean output tokens)
TOOL_PROFILES: Dict[str, Tuple[float, float, int, int]] = {
    # --- Helios Home store ---
    "search_products": (40.0, 0.35, 120, 300),
    "get_product": (25.0, 0.30, 80, 220),
    "check_inventory": (30.0, 0.30, 90, 120),
    "place_order": (180.0, 0.40, 200, 260),
    "get_order": (35.0, 0.30, 110, 240),
    "lookup_customer": (45.0, 0.30, 100, 200),
    "open_ticket": (120.0, 0.35, 180, 160),
    # --- Helios Cloud devops ---
    "get_service_status": (60.0, 0.35, 150, 420),
    "list_deployments": (90.0, 0.35, 140, 500),
    "tail_logs": (250.0, 0.45, 160, 1500),
    "restart_service": (900.0, 0.40, 120, 180),
    "scale_service": (700.0, 0.40, 140, 200),
    "ack_alert": (50.0, 0.30, 90, 120),
}

# scenario name -> (tool sequence, relative weight)
SCENARIOS: List[Tuple[str, List[str], int]] = [
    ("store_browse", ["search_products", "get_product", "check_inventory", "place_order"], 3),
    ("order_status", ["lookup_customer", "get_order"], 2),
    ("support_ticket", ["lookup_customer", "open_ticket"], 2),
    ("long_support", ["lookup_customer", "search_products", "get_product",
                      "check_inventory", "place_order", "get_order"], 1),
    ("incident", ["get_service_status", "tail_logs", "restart_service", "ack_alert"], 3),
    ("deploy_check", ["list_deployments", "get_service_status"], 2),
    ("mixed_support", ["search_products", "get_product", "lookup_customer", "open_ticket"], 1),
]

# Chance a normal step records a benign upstream error (keeps the baseline
# error rate small but nonzero so the error math is exercised).
BENIGN_ERROR_P = 0.004
# Chance a normal session duplicates one middle step (benign retry).
BENIGN_RETRY_P = 0.05

Session = Tuple[Dict[str, Any], List[Dict[str, Any]]]


# ------------------------------------------------------------- generation

def tool_args(rng: random.Random, tool: str) -> Dict[str, Any]:
    """Fictional but plausible arguments for one tool call."""
    if tool == "search_products":
        return {"query": rng.choice(QUERIES)}
    if tool in ("get_product", "check_inventory"):
        return {"sku": rng.choice(SKUS)}
    if tool == "place_order":
        return {"sku": rng.choice(SKUS), "quantity": rng.randint(1, 5)}
    if tool == "get_order":
        return {"order_id": f"ORD-{rng.randint(10000, 19999)}"}
    if tool == "lookup_customer":
        return {"customer_id": rng.choice(CUSTOMERS)}
    if tool == "open_ticket":
        return {"customer_id": rng.choice(CUSTOMERS),
                "subject": rng.choice(SUBJECTS),
                "priority": rng.choice(["low", "medium", "high"])}
    if tool in ("get_service_status", "list_deployments", "restart_service"):
        return {"service": rng.choice(SERVICES), "region": rng.choice(REGIONS)}
    if tool == "tail_logs":
        return {"service": rng.choice(SERVICES), "lines": rng.choice([50, 100, 200])}
    if tool == "scale_service":
        return {"service": rng.choice(SERVICES), "replicas": rng.randint(2, 8)}
    if tool == "ack_alert":
        return {"alert_id": f"ALT-{rng.randint(1000, 9999)}"}
    raise ValueError(f"unknown tool {tool!r}")


def tool_result(rng: random.Random, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Fictional tool result. Mostly ok; rarely a benign upstream error."""
    if rng.random() < BENIGN_ERROR_P:
        return {"error": "UpstreamTimeout: inventory-db read timed out"}
    if tool == "search_products":
        return {"status": "ok", "hits": rng.randint(3, 12)}
    if tool in ("get_product", "get_order", "lookup_customer"):
        return {"status": "ok"}
    if tool == "check_inventory":
        return {"status": "ok", "in_stock": rng.randint(0, 40)}
    if tool == "place_order":
        return {"status": "ok", "order_id": f"ORD-{rng.randint(10000, 19999)}"}
    if tool == "open_ticket":
        return {"status": "ok", "ticket_id": f"T-{rng.randint(5000, 9999)}"}
    if tool in ("get_service_status",):
        return {"status": "ok", "healthy": rng.random() > 0.2}
    return {"status": "ok"}


def step_latency_ms(rng: random.Random, tool: str) -> float:
    median, sigma, _, _ = TOOL_PROFILES[tool]
    return round(rng.lognormvariate(0.0, sigma) * median, 1)


def step_tokens(rng: random.Random, tool: str) -> Dict[str, int]:
    _, _, tok_in, tok_out = TOOL_PROFILES[tool]
    return {
        "input": max(1, int(rng.gauss(tok_in, tok_in * 0.25))),
        "output": max(1, int(rng.gauss(tok_out, tok_out * 0.25))),
    }


def make_step(rng: random.Random, sess_idx: int, seq: int, tool: str,
              scenario: str) -> Dict[str, Any]:
    args = tool_args(rng, tool)
    return {
        "seq": seq,
        "observation": f"[{scenario}] step {seq}: calling {tool}",
        "tool": tool,
        "args": args,
        "result": tool_result(rng, tool, args),
        "duration_ms": step_latency_ms(rng, tool),
        "tokens": step_tokens(rng, tool),
        "timestamp": (BASE_TIME + timedelta(seconds=sess_idx * 37 + seq * 3)).isoformat(),
        "request_id": f"req-{sess_idx:04d}-{seq:02d}",
    }


def make_meta(sess_idx: int, scenario: str, label: Optional[str] = None) -> Dict[str, Any]:
    meta = {
        "trace_id": f"trace-{sess_idx:04d}",
        "agent": AGENT,
        "scenario": scenario,
        "fixture_version": FIXTURE_VERSION,
        "created_at": (BASE_TIME + timedelta(seconds=sess_idx * 37)).isoformat(),
        "format": TRACE_FORMAT,
    }
    if label is not None:
        meta["label"] = label  # eval-only marker, never present in shipped traces
    return meta


def build_session(rng: random.Random, sess_idx: int, scenario: str,
                  tools: List[str], label: Optional[str] = None) -> Session:
    steps = [make_step(rng, sess_idx, seq, tool, scenario)
             for seq, tool in enumerate(tools, start=1)]
    # Benign retry: duplicate one middle step right after itself (self-loop).
    if label is None and len(tools) >= 3 and rng.random() < BENIGN_RETRY_P:
        at = rng.randrange(1, len(tools) - 1)
        retry = make_step(rng, sess_idx, at + 1, tools[at], scenario)
        retry["observation"] += " (retry)"
        steps.insert(at + 1, retry)
        for i, s in enumerate(steps, start=1):
            s["seq"] = i
            s["request_id"] = f"req-{sess_idx:04d}-{i:02d}"
    return make_meta(sess_idx, scenario, label), steps


def _pick_scenario(rng: random.Random) -> Tuple[str, List[str]]:
    names = [name for name, _, _ in SCENARIOS]
    weights = [w for _, _, w in SCENARIOS]
    idx = rng.choices(range(len(names)), weights=weights, k=1)[0]
    return SCENARIOS[idx][0], SCENARIOS[idx][1]


def generate_normal_sessions(n: int, seed: int = 42) -> List[Session]:
    """Generate *n* benign sessions. Deterministic for a fixed seed."""
    rng = random.Random(seed)
    sessions = []
    for i in range(n):
        scenario, tools = _pick_scenario(rng)
        sessions.append(build_session(rng, i, scenario, tools))
    return sessions


def session_jsonl(meta: Dict[str, Any], steps: List[Dict[str, Any]]) -> str:
    return dumps_trace(meta, steps)


# ------------------------------------------------------- labeled eval set

def generate_labeled(seed: int = 2026) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]], str]]:
    """Scripted anomaly sessions, each carrying its expected finding type.

    Returns (meta, steps, label). Every label names the anomaly type the
    detector is expected to report for that session. ``benign`` sessions
    must produce no findings at all.
    """
    rng = random.Random(seed)
    out: List[Tuple[Dict[str, Any], List[Dict[str, Any]], str]] = []
    idx = 9000  # keep eval trace ids disjoint from corpus ids

    def base(scenario: str, tools: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        nonlocal idx
        idx += 1
        return build_session(rng, idx, scenario, tools, label=None)

    def emit(meta: Dict[str, Any], steps: List[Dict[str, Any]], label: str) -> None:
        meta = dict(meta)
        meta["label"] = label
        out.append((meta, steps, label))

    # --- latency_spike x3: one step 12x slower than usual ---
    for _ in range(3):
        meta, steps = base("incident", ["get_service_status", "tail_logs",
                                        "restart_service", "ack_alert"])
        victim = rng.choice(steps[1:3])
        victim["duration_ms"] = round(victim["duration_ms"] * 12, 1)
        emit(meta, steps, "latency_spike")

    # --- token_blowup x3: one step 25x the tokens ---
    for _ in range(3):
        meta, steps = base("incident", ["get_service_status", "tail_logs",
                                        "restart_service", "ack_alert"])
        victim = rng.choice(steps)
        victim["tokens"] = {"input": victim["tokens"]["input"] * 25,
                            "output": victim["tokens"]["output"] * 25}
        emit(meta, steps, "token_blowup")

    # --- tool_usage_drift x3: one tool hammered 8 extra times ---
    for _ in range(3):
        meta, steps = base("store_browse", ["search_products", "get_product",
                                            "check_inventory", "place_order"])
        for _ in range(8):
            steps.append(make_step(rng, idx, len(steps) + 1, "check_inventory", "store_browse"))
        emit(meta, steps, "tool_usage_drift")

    # --- rare_arg_value x3: an arg value the baseline never saw ---
    for _ in range(3):
        meta, steps = base("store_browse", ["search_products", "get_product",
                                            "check_inventory", "place_order"])
        for step in steps:
            if "sku" in step["args"]:
                step["args"]["sku"] = "ZZZ-9999"
                break
        emit(meta, steps, "rare_arg_value")

    # --- new_tool_never_seen x2: a tool absent from the baseline ---
    for _ in range(2):
        meta, steps = base("incident", ["get_service_status", "tail_logs",
                                        "restart_service", "ack_alert"])
        evil = make_step(rng, idx, 3, "get_service_status", "incident")
        evil["tool"] = "exec_shell"
        evil["args"] = {"command": "rm -rf /tmp/cache"}
        evil["result"] = {"status": "ok", "exit_code": 0}
        evil["observation"] = "[incident] step 3: calling exec_shell"
        steps.insert(2, evil)
        for i, s in enumerate(steps, start=1):
            s["seq"] = i
        emit(meta, steps, "new_tool_never_seen")

    # --- error_rate_surge x3: 5 of 6 steps fail ---
    for _ in range(3):
        meta, steps = base("long_support", ["lookup_customer", "search_products",
                                            "get_product", "check_inventory",
                                            "place_order", "get_order"])
        for step in steps[:5]:
            step["result"] = {"error": "UpstreamError: 503 Service Unavailable"}
        emit(meta, steps, "error_rate_surge")

    # --- unusual_tool_sequence x3: cross-domain jumps the baseline never saw ---
    for _ in range(3):
        meta, steps = base("incident", ["get_service_status", "tail_logs",
                                        "restart_service", "ack_alert"])
        swap = make_step(rng, idx, 2, "place_order", "incident")
        swap["observation"] = "[incident] step 2: calling place_order"
        steps[1] = swap
        for i, s in enumerate(steps, start=1):
            s["seq"] = i
        emit(meta, steps, "unusual_tool_sequence")

    # --- session_length_anomaly x3: 2 marathon sessions + 1 single-step ---
    for _ in range(2):
        meta, steps = base("incident", ["get_service_status", "tail_logs",
                                        "restart_service", "ack_alert"] * 4)
        emit(meta, steps, "session_length_anomaly")
    meta, steps = base("incident", ["ack_alert"])
    emit(meta, steps, "session_length_anomaly")

    # --- benign x12: normal sessions with mild, in-vocabulary variation ---
    for b in range(12):
        scenario, tools = _pick_scenario(rng)
        meta, steps = build_session(rng, 9500 + b, scenario, tools)
        if b == 10:
            # a single benign error in a 6-step session (16.7% < warn threshold)
            meta, steps = base("long_support", ["lookup_customer", "search_products",
                                                "get_product", "check_inventory",
                                                "place_order", "get_order"])
            steps[2]["result"] = {"error": "UpstreamTimeout: inventory-db read timed out"}
        if b == 11:
            # a benign retry: one tool called twice in a row (self-loop)
            dup = make_step(rng, 9511, 3, steps[1]["tool"], scenario)
            dup["observation"] += " (retry)"
            steps.insert(2, dup)
            for i, s in enumerate(steps, start=1):
                s["seq"] = i
        emit(meta, steps, "benign")

    return out
