"""Command line entry point for the isolated executable-state pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="HaDukeNN executable-state engineering pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="Run semantic, intervention, replay and descriptive timing checks")
    evaluate.add_argument("--out", type=Path, default=Path("results"))
    evaluate.add_argument("--no-timing", action="store_true")
    evaluate.add_argument("--repeats", type=int, default=3)
    replay = sub.add_parser("replay", help="Verify and replay a JSON execution trace")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--expected-head", help="An independently retained final receipt hash")
    spans = sub.add_parser("spans", help="Build a reversible byte-to-span incidence graph")
    spans.add_argument("source", type=Path)
    spans.add_argument("--visible-bytes", type=int)
    spans.add_argument("--out", type=Path, default=Path("span-graph.json"))
    demo = sub.add_parser("run", help="Execute a JSON list of formal commands")
    demo.add_argument("scenario", type=Path)
    demo.add_argument("--out", type=Path, default=Path("demo-results"))
    args = parser.parse_args()
    if args.command == "evaluate":
        if not 1 <= args.repeats <= 20:
            parser.error("--repeats must be between 1 and 20")
        from .runner import cli_exit
        from .runner import evaluate as run_evaluation
        return cli_exit(run_evaluation(args.out, benchmark=not args.no_timing, repeats=args.repeats))
    if args.command == "replay":
        from .runner import replay_file
        print(json.dumps(replay_file(args.trace, expected_head=args.expected_head), indent=2))
        return 0
    if args.command == "spans":
        from .runner import write_json
        from .spans import build_span_graph, decode_graph
        data = args.source.read_bytes()
        graph = build_span_graph(data, visible_bytes=args.visible_bytes)
        recovered = decode_graph(graph)
        write_json(args.out, graph)
        print(json.dumps({"status": "pass", "bytes": len(recovered), "span_nodes": len(graph["span_nodes"]), "hyperedges": len(graph["hyperedges"]), "graph_sha256": graph["graph_sha256"]}, indent=2))
        return 0
    from .core import Kernel
    from .runner import write_json
    data = json.loads(args.scenario.read_text())
    commands = data if isinstance(data, list) else data["commands"]
    kernel = Kernel()
    outcomes = [kernel.execute(command) for command in commands]
    write_json(args.out / "outcomes.json", outcomes)
    write_json(args.out / "trace.json", kernel.export_trace())
    write_json(args.out / "graph.json", kernel.graph())
    write_json(args.out / "anchor.json", {"head": kernel.head, "state_digest": kernel.state_digest(), "receipt_count": len(kernel.export_trace()), "notice": "Retain independently to detect truncation; this file is not a signature."})
    print(json.dumps({"commands": len(commands), "outcomes": outcomes, "state_digest": kernel.state_digest(), "trace_head": kernel.head}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
