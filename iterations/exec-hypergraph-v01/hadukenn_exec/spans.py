"""Bounded reversible bytes-to-overlapping-spans foundation.

This is a deterministic candidate representation, not a learned tokenizer,
entropy patcher, semantic parser, or demonstrated performance improvement. A
future learned component may select or process these candidates. Byte nodes
remain the exact fallback, including incomplete or invalid UTF-8 sequences.

All model-facing fields are computed after slicing the available prefix. The
graph contains no hash or length of unobserved suffix bytes. Digests detect
corruption relative to a retained graph; they are not authenticated signatures.
The source bytes and deterministic graph can be reconstructed and validated.
"""

import hashlib
import json
import re

SCHEMA = "hadukenn.span_graph.v1"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def build_span_graph(data: bytes, visible_bytes=None, max_span_bytes=64, max_spans_per_start=4, max_bytes=16_384):
    """Return an independent JSON-compatible graph of the available bytes.

    Each byte gets one byte node and one mandatory single-byte span. Additional
    candidates are ASCII-whitespace-delimited byte runs, bounded fragments of
    long runs, and adjacent two-/three-run spans. The cap includes the mandatory
    byte span. These rules give multiple overlapping views without assigning
    semantic labels. Offsets are byte offsets, never decoded character offsets.

    ``visible_bytes`` must be an integer in [0, len(data)]. The byte budget is
    applied to the visible prefix only; an unobserved suffix cannot change the
    model-facing graph or cause the visible prefix to exceed that budget.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be immutable bytes")
    max_span_bytes = _positive_integer(max_span_bytes, "max_span_bytes")
    max_spans_per_start = _positive_integer(max_spans_per_start, "max_spans_per_start")
    max_bytes = _positive_integer(max_bytes, "max_bytes")
    if visible_bytes is not None:
        if isinstance(visible_bytes, bool) or not isinstance(visible_bytes, int):
            raise ValueError("visible_bytes must be an integer or None")
        if visible_bytes < 0 or visible_bytes > len(data):
            raise ValueError("visible_bytes is outside the available input")
        available = data[:visible_bytes]
    else:
        available = data
    # Nothing below this line inspects data or the original visible_bytes value.
    size = len(available)
    if size > max_bytes:
        raise ValueError("visible bytes exceed max_bytes")
    source_hash = hashlib.sha256(available).hexdigest()
    by_start = {i: {(i, i + 1): "byte"} for i in range(size)}

    def candidate(start, end, kind):
        if start < 0 or end > size or start >= end or end - start > max_span_bytes:
            return
        candidates = by_start[start]
        if (start, end) not in candidates and len(candidates) < max_spans_per_start:
            candidates[(start, end)] = kind

    runs = [(match.start(), match.end()) for match in re.finditer(rb"\S+", available)]
    for start, end in runs:
        if end - start <= max_span_bytes:
            candidate(start, end, "byte_run")
        else:
            for chunk_start in range(start, end, max_span_bytes):
                candidate(chunk_start, min(chunk_start + max_span_bytes, end), "bounded_run_fragment")
    for width in (2, 3):
        for i in range(len(runs) - width + 1):
            candidate(runs[i][0], runs[i + width - 1][1], f"adjacent_{width}_runs")

    byte_nodes = [
        {"id": f"b:{i}", "kind": "byte", "offset": i, "value": value, "source_id": source_hash}
        for i, value in enumerate(available)
    ]
    span_nodes = []
    edges = []
    for start in range(size):
        for (_, end), kind in sorted(by_start[start].items()):
            span_id = f"s:{start}:{end}"
            span_nodes.append({
                "id": span_id, "kind": "span", "candidate": kind,
                "start": start, "end": end, "byte_length": end - start, "source_id": source_hash,
            })
            edges.append({
                "id": f"e:{start}:{end}", "type": "span_contains_bytes",
                "source_id": source_hash, "start": start, "end": end,
                "ports": [{"role": "unit", "node": span_id}] + [
                    {"role": f"part_{i}", "node": f"b:{offset}"}
                    for i, offset in enumerate(range(start, end))
                ],
            })
    graph = {
        "schema": SCHEMA,
        "construction": "deterministic byte runs and bounded adjacent runs; no learned semantics",
        "limits": {
            "max_span_bytes": max_span_bytes,
            "max_spans_per_start": max_spans_per_start,
            "max_bytes": max_bytes,
        },
        "source": {"id": source_hash, "sha256": source_hash, "byte_length": size, "hex": available.hex()},
        "byte_nodes": byte_nodes,
        "span_nodes": span_nodes,
        "hyperedges": edges,
    }
    graph["graph_sha256"] = _digest(graph)
    return graph


def build_prefix(data: bytes, visible_bytes: int, **kwargs):
    """Build exactly the graph obtained from the same standalone byte prefix."""
    return build_span_graph(data, visible_bytes=visible_bytes, **kwargs)


def decode_graph(graph) -> bytes:
    """Validate a graph and reconstruct its exact bytes.

    Validation includes source and graph digests, byte-node values, candidate
    spans, and ordered incidence ports. Rebuilding the deterministic candidates
    checks consistency beyond a checksum. A caller changing bytes and rebuilding
    all fields can create a different valid graph; this is not authentication.
    """
    if not isinstance(graph, dict) or graph.get("schema") != SCHEMA:
        raise ValueError("unsupported span graph")
    try:
        limits = graph["limits"]
        if not isinstance(limits, dict) or set(limits) != {"max_span_bytes", "max_spans_per_start", "max_bytes"}:
            raise ValueError("invalid limits")
        for name, value in limits.items():
            _positive_integer(value, name)
        source = graph["source"]
        size = source["byte_length"]
        payload = source["hex"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or size > limits["max_bytes"]:
            raise ValueError("invalid source length")
        if not isinstance(payload, str) or len(payload) != 2 * size:
            raise ValueError("invalid source encoding")
        available = bytes.fromhex(payload)
        if len(available) != size or hashlib.sha256(available).hexdigest() != source["sha256"]:
            raise ValueError("source digest mismatch")
        body = {key: value for key, value in graph.items() if key != "graph_sha256"}
        if _digest(body) != graph["graph_sha256"]:
            raise ValueError("graph digest mismatch")
        if build_span_graph(available, **limits) != graph:
            raise ValueError("span graph is not the declared deterministic representation")
        return available
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("malformed span graph") from error
