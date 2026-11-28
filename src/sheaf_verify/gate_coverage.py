"""Fail unless every control-graph node and dependency edge is tested.

A pass counts when the gate's own criteria ran and the gate passed. A fail
counts when those own criteria ran and at least one of them failed. A
downstream gate that only records ``upstream.<dep>`` because a predecessor
failed has not exercised its own fail path. That observation does count as
the dependency edge being blocked.

A dependency edge is tested when some run satisfies it and some run blocks it.

Each trace event carries a tri-state ``outcome``: ``pass``, ``fail`` (own
criteria ran and one failed), or ``blocked`` (own criteria never ran because a
dependency had no passing record). Older traces without ``outcome`` are
classified the same way from ``status``, ``own_evaluated``, and ``upstream``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sheaf_verify.gate_trace import trace_outcome
from sheaf_verify.gates import all_gates


@dataclass
class Report:
    ok: bool
    problems: list[str]
    gates: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    trace_error: str = ""


def load_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def assess(events: list[dict]) -> Report:
    passes: dict[str, set[str]] = {gate.id: set() for gate in all_gates()}
    fails: dict[str, set[str]] = {gate.id: set() for gate in all_gates()}
    blocked_gates: dict[str, set[str]] = {gate.id: set() for gate in all_gates()}
    satisfied: dict[tuple[str, str], set[str]] = {}
    blocked: dict[tuple[str, str], set[str]] = {}
    for gate in all_gates():
        for dep in gate.depends_on:
            satisfied[(dep, gate.id)] = set()
            blocked[(dep, gate.id)] = set()
    for event in events:
        test = event.get("test")
        gate_id = event.get("gate_id")
        if not isinstance(test, str) or "::" not in test or gate_id not in passes:
            continue
        upstream = event.get("upstream") if isinstance(event.get("upstream"), dict) else {}
        for dep, open_edge in upstream.items():
            key = (dep, gate_id)
            if key not in satisfied:
                continue
            if open_edge is True:
                satisfied[key].add(test)
            elif open_edge is False:
                blocked[key].add(test)
        outcome = event_outcome(event, upstream)
        if outcome == "blocked":
            blocked_gates[gate_id].add(test)
        elif not event.get("own_evaluated"):
            continue
        elif outcome == "pass" and not event.get("own_failed"):
            passes[gate_id].add(test)
        elif outcome == "fail" and event.get("own_failed"):
            fails[gate_id].add(test)
    problems = []
    gate_rows = []
    for gate in all_gates():
        if not passes[gate.id]:
            problems.append(f"{gate.id} pass path has no test")
        if not fails[gate.id]:
            problems.append(f"{gate.id} fail path has no test that exercises its own criteria")
        gate_rows.append({
            "blocked": _files(blocked_gates[gate.id]),
            "fail": _files(fails[gate.id]),
            "gate": gate.id,
            "pass": _files(passes[gate.id]),
            "tests": {
                "blocked": sorted(blocked_gates[gate.id]),
                "fail": sorted(fails[gate.id]),
                "pass": sorted(passes[gate.id]),
            },
        })
    edge_rows = []
    for gate in all_gates():
        for dep in gate.depends_on:
            key = (dep, gate.id)
            label = f"{dep} -> {gate.id}"
            if not satisfied[key]:
                problems.append(f"dependency {label} was never satisfied")
            if not blocked[key]:
                problems.append(f"dependency {label} was never blocked")
            edge_rows.append({
                "blocked": _files(blocked[key]),
                "edge": label,
                "satisfied": _files(satisfied[key]),
                "tests": {"blocked": sorted(blocked[key]), "satisfied": sorted(satisfied[key])},
            })
    return Report(ok=not problems, problems=problems, gates=gate_rows, edges=edge_rows)


def render_matrix(report: Report) -> str:
    lines = [
        "# Gate coverage matrix",
        "",
        "Columns are the gate, the pass path, the fail path, the blocked path, and the test file that exercised it.",
        "A fail counts only when the gate's own criteria ran and one of them failed.",
        "A blocked run is one where an upstream gate had no passing record, so the gate's own criteria never ran.",
        "",
    ]
    if report.trace_error:
        lines.extend([f"Trace error: {report.trace_error}", ""])
    if report.problems:
        lines.append("The control graph is not covered.")
        lines.append("")
        for problem in report.problems:
            lines.append(f"- {problem}")
        lines.append("")
    else:
        lines.append("Every gate has a pass test and a fail test. Every dependency edge was satisfied and blocked.")
        lines.append("")
    lines.extend([
        "| Gate | Pass | Fail | Blocked | Test file |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in report.gates:
        blocked = row.get("blocked", [])
        files = _join(_join_list(row["pass"], row["fail"]), blocked)
        lines.append(
            f"| {row['gate']} | {_cell(row['pass'])} | {_cell(row['fail'])} | {_cell(blocked)} | {files} |"
        )
    lines.extend([
        "",
        "| Edge | Satisfied | Blocked | Test file |",
        "| --- | --- | --- | --- |",
    ])
    for row in report.edges:
        files = _join(row["satisfied"], row["blocked"])
        lines.append(
            f"| {row['edge']} | {_cell(row['satisfied'])} | {_cell(row['blocked'])} | {files} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sheaf-verify.gate_coverage")
    parser.add_argument("trace")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    matrix = Path(args.matrix)
    try:
        events = load_events(Path(args.trace))
        report = assess(events)
    except (OSError, json.JSONDecodeError) as error:
        report = Report(ok=False, problems=[f"trace could not be read: {error}"], trace_error=str(error))
    matrix.write_text(render_matrix(report), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(
            json.dumps({
                "edges": report.edges,
                "gates": report.gates,
                "ok": report.ok,
                "problems": report.problems,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    if not report.ok:
        print("\n".join(report.problems), file=sys.stderr)
        return 1
    return 0


def event_outcome(event: dict, upstream: dict | None = None) -> str:
    """Return the event's tri-state outcome, deriving it for older traces."""
    outcome = event.get("outcome")
    if outcome in ("pass", "fail", "blocked"):
        return outcome
    if upstream is None:
        upstream = event.get("upstream") if isinstance(event.get("upstream"), dict) else {}
    return trace_outcome(
        str(event.get("status")),
        bool(event.get("own_evaluated")),
        bool(event.get("own_failed")),
        upstream,
    )


def _join_list(left: list[str], right: list[str]) -> list[str]:
    return sorted(set(left) | set(right))


def _files(nodeids: set[str]) -> list[str]:
    return sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})


def _cell(files: list[str]) -> str:
    return ", ".join(files) if files else "—"


def _join(left: list[str], right: list[str]) -> str:
    return ", ".join(sorted(set(left) | set(right))) or "—"


if __name__ == "__main__":
    raise SystemExit(main())
