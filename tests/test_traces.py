"""Tests: trace JSONL I/O (the trace-replay-compatible format)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traces import dumps_trace, loads_trace, load_trace, write_trace  # noqa: E402
from tests.common import make_meta, make_step  # noqa: E402


class TestTraceIO(unittest.TestCase):
    def test_roundtrip(self):
        meta = make_meta()
        steps = [make_step(1, "search_products"), make_step(2, "get_product")]
        text = dumps_trace(meta, steps)
        m2, s2 = loads_trace(text)
        self.assertEqual(m2["trace_id"], "trace-0001")
        self.assertEqual([s["tool"] for s in s2], ["search_products", "get_product"])
        self.assertEqual(s2[0]["args"], {})

    def test_canonical_deterministic(self):
        meta = make_meta()
        steps = [make_step(1, "t", args={"b": 1, "a": 2})]
        self.assertEqual(dumps_trace(meta, steps), dumps_trace(meta, steps))
        # sorted keys in the serialized step line
        self.assertIn('"a":2', dumps_trace(meta, steps).splitlines()[1])

    def test_write_load_file(self):
        meta = make_meta()
        steps = [make_step(1, "t")]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            write_trace(p, meta, steps)
            m2, s2 = load_trace(p)
            self.assertEqual(m2["trace_id"], meta["trace_id"])
            self.assertEqual(len(s2), 1)

    def test_bad_json_raises(self):
        with self.assertRaises(ValueError):
            loads_trace('{"kind": "meta", "trace_id": "x"}\n{not json}\n')

    def test_missing_meta_raises(self):
        with self.assertRaises(ValueError):
            loads_trace('{"kind": "step", "seq": 1, "tool": "t"}\n')

    def test_optional_fields_may_be_absent(self):
        text = ('{"kind":"meta","trace_id":"t1"}\n'
                '{"kind":"step","seq":1,"tool":"t","args":{},"result":{}}\n')
        meta, steps = loads_trace(text)
        self.assertNotIn("duration_ms", steps[0])
        self.assertNotIn("tokens", steps[0])


if __name__ == "__main__":
    unittest.main()
