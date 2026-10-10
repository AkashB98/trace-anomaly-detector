"""Tests: CLI commands and exit codes (0 clean / 1 warnings / 2 critical)."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli  # noqa: E402
from corpus import generate_labeled, generate_normal_sessions, session_jsonl  # noqa: E402
from traces import dumps_trace  # noqa: E402


def write_corpus(d, sessions, prefix="trace"):
    for i, (meta, steps) in enumerate(sessions):
        meta = dict(meta)
        meta["trace_id"] = f"{prefix}-{i:04d}"
        with open(os.path.join(d, f"{prefix}-{i:04d}.jsonl"), "w") as fh:
            fh.write(session_jsonl(meta, steps))


def _has_error(steps):
    return any("error" in (s.get("result") or {}) for s in steps)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name
        all_sessions = generate_normal_sessions(150, seed=42)
        write_corpus(self.d, all_sessions)
        self.baseline = os.path.join(self.d, "baseline.json")
        rc = cli.main(["mine", "--traces", self.d, "--out", self.baseline])
        self.assertEqual(rc, 0)
        # "Clean" scan dir: drop sessions with benign injected errors, since
        # a 2-step session with 1 error is a genuine 50% error rate (critical).
        self.clean_dir = os.path.join(self.d, "clean")
        os.mkdir(self.clean_dir)
        clean = [(m, s) for m, s in all_sessions if not _has_error(s)]
        write_corpus(self.clean_dir, clean, prefix="clean")
        self.n_clean = len(clean)

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_mine_writes_baseline(self):
        self.assertTrue(os.path.isfile(self.baseline))
        with open(self.baseline) as fh:
            obj = json.load(fh)
        self.assertEqual(obj["format"], "trace-anomaly-baseline/1")

    def test_mine_no_traces_errors(self):
        empty = os.path.join(self.d, "empty")
        os.mkdir(empty)
        rc, _, _ = self.run_cli(["mine", "--traces", empty,
                                 "--out", os.path.join(self.d, "b.json")])
        self.assertEqual(rc, 3)

    def test_scan_clean_exit_0(self):
        rc, out, err = self.run_cli(["scan", "--baseline", self.baseline,
                                    "--traces", self.clean_dir])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("summary", payload)

    def test_scan_warnings_exit_1(self):
        labeled = [s for s in generate_labeled(seed=2026) if s[2] == "rare_arg_value"]
        wdir = os.path.join(self.d, "warn")
        os.mkdir(wdir)
        write_corpus(wdir, [(m, s) for m, s, _ in labeled], prefix="w")
        rc, _, _ = self.run_cli(["scan", "--baseline", self.baseline,
                                 "--traces", wdir])
        self.assertEqual(rc, 1)

    def test_scan_critical_exit_2(self):
        labeled = [s for s in generate_labeled(seed=2026) if s[2] == "new_tool_never_seen"]
        cdir = os.path.join(self.d, "crit")
        os.mkdir(cdir)
        write_corpus(cdir, [(m, s) for m, s, _ in labeled], prefix="c")
        rc, _, _ = self.run_cli(["scan", "--baseline", self.baseline,
                                 "--traces", cdir])
        self.assertEqual(rc, 2)

    def test_scan_missing_baseline_exit_3(self):
        rc, _, _ = self.run_cli(["scan", "--baseline", "/nope.json",
                                 "--traces", self.d])
        self.assertEqual(rc, 3)

    def test_scan_writes_out_file(self):
        outp = os.path.join(self.d, "report.json")
        rc, _, _ = self.run_cli(["scan", "--baseline", self.baseline,
                                 "--traces", self.clean_dir, "--out", outp])
        self.assertEqual(rc, 0)
        with open(outp) as fh:
            payload = json.load(fh)
        self.assertEqual(payload["summary"]["sessions"], self.n_clean)

    def test_explain_shows_evidence(self):
        labeled = [s for s in generate_labeled(seed=2026) if s[2] == "latency_spike"]
        meta, steps, _ = labeled[0]
        p = os.path.join(self.d, "spike.jsonl")
        with open(p, "w") as fh:
            fh.write(dumps_trace(meta, steps))
        rc, out, _ = self.run_cli(["explain", "--baseline", self.baseline,
                                   "--trace", p])
        self.assertIn("latency_spike", out)
        self.assertIn("baseline_p99_ms", out)
        self.assertTrue(rc in (1, 2))

    def test_explain_clean_trace(self):
        from tests.common import make_session
        # Hand-built 2-step store session: in-vocab everything, no anomalies.
        meta, steps = make_session(["search_products", "get_product"],
                                   trace_id="clean-0001")
        p = os.path.join(self.d, "clean.jsonl")
        with open(p, "w") as fh:
            fh.write(dumps_trace(meta, steps))
        rc, out, _ = self.run_cli(["explain", "--baseline", self.baseline,
                                   "--trace", p])
        self.assertEqual(rc, 0)
        self.assertIn("no findings", out)

    def test_demo_runs(self):
        rc, out, _ = self.run_cli(["demo"])
        self.assertEqual(rc, 0)
        self.assertIn("demo OK", out)


if __name__ == "__main__":
    unittest.main()
