"""Render ``docs/control-graph.mmd`` from ``gates.all_gates()``.

The dependency edges come from each gate's ``depends_on`` so the diagram can
not drift from the code. The dotted comparison edges are not dependencies;
they mark the primary comparisons and live here as data.

``python -m sheaf_verify.control_graph --check`` exits 1 when the committed
diagram, or the copies in ``GATES.md`` and ``docs/graph.html``, differ from
the rendered graph. ``--write`` rewrites all three.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sheaf_verify.gates import Gate, all_gates

COMPARISONS: tuple[tuple[str, str], ...] = (
    ("hntCtl", "hnt vs equal-size control"),
    ("bestArm", "best arm vs struct32k / GPT-2"),
    ("peHeat", "pe_heat vs baseline"),
)
COMPARISON_EDGES: tuple[tuple[str, str], ...] = (
    ("hntCtl", "G1"),
    ("bestArm", "G1"),
    ("bestArm", "H1"),
    ("bestArm", "G4"),
    ("peHeat", "P2"),
)
PRIMARY_STYLE = "fill:#f3e0c4,stroke:#8a4b08,color:#1b1914,stroke-width:2px"

ROOT = Path(__file__).resolve().parents[2]
DIAGRAM = Path("docs") / "control-graph.mmd"
EMBEDDED = (Path("GATES.md"), Path("docs") / "graph.html")


def dependency_edges(gates: tuple[Gate, ...] = all_gates()) -> list[tuple[str, str]]:
    return [(dep, gate.id) for gate in gates for dep in gate.depends_on]


def render(gates: tuple[Gate, ...] = all_gates()) -> str:
    ids = {gate.id for gate in gates}
    for source, target in COMPARISON_EDGES:
        if target not in ids:
            raise ValueError(f"comparison edge {source} -> {target} names an unknown gate")
    primary_gates = sorted({target for _, target in COMPARISON_EDGES}, key=[g.id for g in gates].index)
    lines = ["flowchart TD", '  subgraph comparisons["Primary comparisons"]', "    direction LR"]
    lines += [f'    {node}["{label}"]' for node, label in COMPARISONS]
    lines.append("  end")
    lines += [f"  {source} -.-> {target}" for source, target in COMPARISON_EDGES]
    lines.append("")
    lines += [f"  {dep} --> {gate}" for dep, gate in dependency_edges(gates)]
    lines.append("")
    lines.append(f"  classDef primary {PRIMARY_STYLE}")
    lines.append("  class " + ",".join(primary_gates + [node for node, _ in COMPARISONS]) + " primary")
    return "\n".join(lines) + "\n"


def _replace_block(text: str, new: str) -> str:
    start = text.find("flowchart TD")
    if start < 0:
        raise ValueError("no flowchart block found")
    marker = " primary"
    end = text.find(marker, text.find("  class ", start))
    if end < 0:
        raise ValueError("flowchart block has no class line")
    return text[:start] + new.strip() + text[end + len(marker):]


def problems(root: Path = ROOT) -> list[str]:
    rendered = render()
    found = []
    if (root / DIAGRAM).read_text(encoding="utf-8") != rendered:
        found.append(f"{DIAGRAM.as_posix()} differs from the declared control graph")
    for rel in EMBEDDED:
        if rendered.strip() not in (root / rel).read_text(encoding="utf-8"):
            found.append(f"{rel.as_posix()} does not embed the rendered control graph")
    return found


def write(root: Path = ROOT) -> None:
    rendered = render()
    (root / DIAGRAM).write_text(rendered, encoding="utf-8", newline="\n")
    for rel in EMBEDDED:
        path = root / rel
        path.write_text(_replace_block(path.read_text(encoding="utf-8"), rendered), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sheaf-verify.control_graph")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="exit 1 if committed diagrams drift")
    group.add_argument("--write", action="store_true", help="rewrite the committed diagrams")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.write:
        write(args.root)
        return 0
    if args.check:
        found = problems(args.root)
        if found:
            print("\n".join(found), file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
