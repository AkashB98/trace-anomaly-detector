"""Tests: the tiny JSON API (localhost only — hermetic)."""

import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import ThreadingHTTPServer  # noqa: E402

from cli import make_handler  # noqa: E402
from corpus import generate_labeled, generate_normal_sessions  # noqa: E402
from miner import mine  # noqa: E402
from traces import dumps_trace  # noqa: E402


class TestServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = mine(generate_normal_sessions(150, seed=42))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(baseline))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        labeled = [s for s in generate_labeled(seed=2026) if s[2] == "latency_spike"]
        cls.spike_meta, cls.spike_steps, _ = labeled[0]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_get_baseline(self):
        with urllib.request.urlopen(self.url("/baseline")) as resp:
            self.assertEqual(resp.status, 200)
            obj = json.load(resp)
        self.assertEqual(obj["format"], "trace-anomaly-baseline/1")

    def test_post_scan_finds_anomaly(self):
        body = json.dumps({
            "trace_jsonl": dumps_trace(self.spike_meta, self.spike_steps),
        }).encode()
        req = urllib.request.Request(self.url("/scan"), data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            report = json.load(resp)
        types = {f["type"] for f in report["findings"]}
        self.assertIn("latency_spike", types)

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.request.HTTPError) as ctx:
            urllib.request.urlopen(self.url("/nope"))
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
