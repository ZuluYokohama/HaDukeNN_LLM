"""Contract tests for exact reconstruction, prefix causality, and graph bounds."""

import hashlib
import json
import unittest
from collections import Counter
from copy import deepcopy

from hadukenn_exec.spans import build_prefix, build_span_graph, decode_graph


class SpanGraphTests(unittest.TestCase):
    def test_round_trip_arbitrary_bytes(self):
        for data in (b"", bytes(range(256)), b"\xff\xfe\x00\x80", "café 🌍 中文".encode()):
            with self.subTest(data=data[:20]):
                self.assertEqual(decode_graph(build_span_graph(data)), data)

    def test_prefix_causality_including_split_utf8(self):
        observed = "café 🌍".encode()
        for cut in range(len(observed) + 1):
            for suffix in (b"", b" future words", b"\xff\x00" * 20):
                prefix_graph = build_prefix(observed + suffix, cut)
                standalone = build_span_graph(observed[:cut])
                self.assertEqual(prefix_graph, standalone)
                self.assertEqual(decode_graph(prefix_graph), observed[:cut])

    def test_future_suffix_does_not_spend_visible_budget(self):
        data = b"visible" + b"x" * 20_000
        self.assertEqual(build_prefix(data, 7, max_bytes=7), build_span_graph(b"visible", max_bytes=7))
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            build_span_graph(data, max_bytes=7)

    def test_overlapping_candidates_and_ordered_ports(self):
        data = b"red green blue"
        graph = build_span_graph(data)
        spans = {(span["start"], span["end"]) for span in graph["span_nodes"]}
        self.assertIn((0, 3), spans)
        self.assertIn((0, 9), spans)
        self.assertIn((0, 14), spans)
        self.assertIn((4, 14), spans)
        for edge in graph["hyperedges"]:
            start, end = edge["start"], edge["end"]
            self.assertEqual(edge["ports"][0], {"role": "unit", "node": f"s:{start}:{end}"})
            self.assertEqual(edge["ports"][1:], [
                {"role": f"part_{i}", "node": f"b:{offset}"}
                for i, offset in enumerate(range(start, end))
            ])

    def test_candidate_and_length_bounds_preserve_byte_fallback(self):
        data = b"red green blue " + b"z" * 100
        for per_start in (1, 2, 4):
            graph = build_span_graph(data, max_span_bytes=8, max_spans_per_start=per_start)
            counts = Counter(span["start"] for span in graph["span_nodes"])
            self.assertLessEqual(max(counts.values()), per_start)
            self.assertTrue(all(1 <= span["end"] - span["start"] <= 8 for span in graph["span_nodes"]))
            spans = {(span["start"], span["end"]) for span in graph["span_nodes"]}
            self.assertTrue(all((i, i + 1) in spans for i in range(len(data))))
            self.assertEqual(decode_graph(graph), data)

    def test_source_digest_covers_only_visible_bytes(self):
        graph = build_prefix(b"abcNOT_VISIBLE", 3)
        expected = hashlib.sha256(b"abc").hexdigest()
        self.assertEqual(graph["source"], {"id": expected, "sha256": expected, "byte_length": 3, "hex": "616263"})
        self.assertEqual([node["value"] for node in graph["byte_nodes"]], [97, 98, 99])

    def test_independent_results_and_deepcopy(self):
        first = build_span_graph(b"a b")
        second = build_span_graph(b"a b")
        copied = deepcopy(first)
        first["byte_nodes"][0]["value"] = 99
        self.assertEqual(second, copied)
        self.assertEqual(decode_graph(second), b"a b")
        with self.assertRaises(ValueError):
            decode_graph(first)

    def test_reordered_ports_rejected_even_with_recomputed_graph_hash(self):
        graph = build_span_graph(b"ab")
        edge = next(item for item in graph["hyperedges"] if item["start"] == 0 and item["end"] == 2)
        edge["ports"][1], edge["ports"][2] = edge["ports"][2], edge["ports"][1]
        body = {key: value for key, value in graph.items() if key != "graph_sha256"}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
        graph["graph_sha256"] = hashlib.sha256(encoded).hexdigest()
        with self.assertRaisesRegex(ValueError, "deterministic"):
            decode_graph(graph)

    def test_invalid_inputs_and_limits(self):
        for data in ("abc", bytearray(b"abc"), None):
            with self.subTest(data=repr(data)), self.assertRaises(TypeError):
                build_span_graph(data)
        for key in ("max_bytes", "max_span_bytes", "max_spans_per_start"):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    build_span_graph(b"abc", **{key: value})
        for value in (-1, 4, True, 1.5):
            with self.subTest(visible_bytes=value), self.assertRaises(ValueError):
                build_span_graph(b"abc", visible_bytes=value)

    def test_mutated_source_and_malformed_graph_rejected(self):
        graph = build_span_graph(b"abc")
        graph["source"]["hex"] = "616264"
        with self.assertRaisesRegex(ValueError, "digest"):
            decode_graph(graph)
        for value in (None, {}, {"schema": "wrong"}, {"schema": "hadukenn.span_graph.v1"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_graph(value)


if __name__ == "__main__":
    unittest.main()
