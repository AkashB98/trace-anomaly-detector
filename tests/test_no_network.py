"""Tests: no network use anywhere in the offline pipeline.

Sockets are stubbed to raise; corpus generation, mining, scanning, and the
eval runner must all work without touching the network. (The `serve`
command is the only network surface and is covered by test_serve.py with a
real localhost socket.)
"""

import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _raise(*a, **k):
    raise AssertionError("network socket use is forbidden in offline code")


class TestNoNetwork(unittest.TestCase):
    def setUp(self):
        self._real_socket = socket.socket
        socket.socket = _raise

    def tearDown(self):
        socket.socket = self._real_socket

    def test_offline_pipeline_needs_no_sockets(self):
        from corpus import generate_labeled, generate_normal_sessions
        from detector import score_many
        from miner import mine
        sessions = generate_normal_sessions(30, seed=5)
        baseline = mine(sessions)
        labeled = [(m, s) for m, s, _ in generate_labeled(seed=2026)[:6]]
        reports = score_many(baseline, labeled)
        self.assertEqual(len(reports), 6)

    def test_eval_runner_needs_no_sockets(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))
        import run_evals
        report = run_evals.run()
        self.assertTrue(report["all_passed"])


if __name__ == "__main__":
    unittest.main()
