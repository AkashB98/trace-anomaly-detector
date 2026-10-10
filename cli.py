"""CLI: mine baselines, scan traces, explain findings, demo, serve.

Exit codes (for CI): 0 = clean, 1 = warnings (suspicious), 2 = critical.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import score_many, summarize  # noqa: E402
from miner import dumps_baseline, load_baseline, mine, save_baseline  # noqa: E402
from traces import load_trace  # noqa: E402

EXIT_OK, EXIT_WARN, EXIT_CRIT = 0, 1, 2


def iter_trace_files(path: str):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    else:
        files = sorted(glob.glob(path)) if any(c in path for c in "*?[") else [path]
    for f in files:
        if os.path.isfile(f):
            yield f


def _read_sessions(path: str):
    sessions = []
    for f in iter_trace_files(path):
        meta, steps = load_trace(f)
        sessions.append((meta, steps))
    return sessions


def _load_inputs(args):
    """Load baseline + sessions, returning (baseline, sessions) or an error string."""
    try:
        baseline = load_baseline(args.baseline)
    except (OSError, ValueError) as exc:
        return None, None, f"error: cannot load baseline: {exc}"
    try:
        sessions = _read_sessions(args.traces)
    except (OSError, ValueError) as exc:
        return None, None, f"error: cannot read traces: {exc}"
    if not sessions:
        return None, None, f"error: no trace files found under {args.traces}"
    return baseline, sessions, None


def cmd_mine(args) -> int:
    try:
        sessions = _read_sessions(args.traces)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read traces: {exc}", file=sys.stderr)
        return 3
    if not sessions:
        print(f"error: no trace files found under {args.traces}", file=sys.stderr)
        return 3
    baseline = mine(sessions)
    save_baseline(baseline, args.out)
    m = baseline["meta"]
    print(f"mined baseline from {m['mined_from_sessions']} sessions "
          f"({m['mined_from_steps']} steps) -> {args.out}")
    print(f"tools: {len(baseline['tools'])}, bigrams: {len(baseline['bigrams'])}")
    return EXIT_OK


def _verdict_exit(report: dict) -> int:
    if report["verdict"] == "critical":
        return EXIT_CRIT
    if report["verdict"] == "suspicious":
        return EXIT_WARN
    return EXIT_OK


def cmd_scan(args) -> int:
    baseline, sessions, err = _load_inputs(args)
    if err:
        print(err, file=sys.stderr)
        return 3
    reports = score_many(baseline, sessions)
    summary = summarize(reports)
    payload = {"summary": summary, "sessions": reports}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    s = summary
    print(f"scanned {s['sessions']} sessions: {s['ok']} ok, "
          f"{s['suspicious']} suspicious, {s['critical']} critical",
          file=sys.stderr)
    worst = max((_verdict_exit(r) for r in reports), default=EXIT_OK)
    return worst


def cmd_explain(args) -> int:
    try:
        baseline = load_baseline(args.baseline)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load baseline: {exc}", file=sys.stderr)
        return 3
    try:
        meta, steps = load_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read trace: {exc}", file=sys.stderr)
        return 3
    from detector import score_session
    report = score_session(baseline, meta, steps)
    print(f"trace {report['trace_id']} ({report['scenario']}): "
          f"{report['n_steps']} steps -> {report['verdict'].upper()}")
    if not report["findings"]:
        print("no findings: session matches the baseline.")
        return EXIT_OK
    for f in report["findings"]:
        where = f"seq={f['seq']} " if f["seq"] is not None else ""
        print(f"\n[{f['severity'].upper()}] {f['type']} {where}({f['tool']})")
        print(f"  {f['message']}")
        for k in sorted(f["evidence"]):
            print(f"    {k}: {f['evidence'][k]}")
    return _verdict_exit(report)


def cmd_demo(_args) -> int:
    import demo
    return demo.main()


def make_handler(baseline):
    """Build the HTTP request handler class bound to a baseline (testable)."""
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep stdout clean
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, sort_keys=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/baseline":
                self._json(200, baseline)
            else:
                self._json(404, {"error": "unknown path; try GET /baseline or POST /scan"})

        def do_POST(self):
            if self.path != "/scan":
                self._json(404, {"error": "unknown path; try GET /baseline or POST /scan"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, OSError):
                self._json(400, {"error": "invalid JSON body"})
                return
            try:
                from detector import score_session
                from traces import loads_trace
                if "trace_jsonl" in body:
                    meta, steps = loads_trace(body["trace_jsonl"])
                elif "steps" in body:
                    meta, steps = body.get("meta", {}), body["steps"]
                else:
                    self._json(400, {"error": "body needs trace_jsonl or steps"})
                    return
                report = score_session(baseline, meta, steps)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, report)

    return Handler


def cmd_serve(args) -> int:
    from http.server import ThreadingHTTPServer

    try:
        baseline = load_baseline(args.baseline)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load baseline: {exc}", file=sys.stderr)
        return 3

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(baseline))
    print(f"serving on http://127.0.0.1:{args.port} "
          f"(GET /baseline, POST /scan) — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="trace-anomaly-detector",
        description="Statistical anomaly detection over agent tool-call traces.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mine", help="build a baseline from benign traces")
    p.add_argument("--traces", required=True, help="trace file, glob, or directory")
    p.add_argument("--out", required=True, help="output baseline JSON path")
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("scan", help="score traces against a baseline")
    p.add_argument("--baseline", required=True)
    p.add_argument("--traces", required=True, help="trace file, glob, or directory")
    p.add_argument("--out", default=None, help="write JSON report here (else stdout)")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("explain", help="show findings for one trace in detail")
    p.add_argument("--baseline", required=True)
    p.add_argument("--trace", required=True)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("demo", help="one-command end-to-end demo")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("serve", help="tiny JSON API: GET /baseline, POST /scan")
    p.add_argument("--baseline", required=True)
    p.add_argument("--port", type=int, default=8137)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
