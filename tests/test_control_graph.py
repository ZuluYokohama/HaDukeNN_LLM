"""control-graph.mmd is generated from gates.GATES and the trace is tri-state."""

import re
import shutil
from pathlib import Path

import pytest

from sheaf_verify import control_graph, gate_trace
from sheaf_verify.gate_coverage import assess, event_outcome, render_matrix
from sheaf_verify.gate_trace import trace_outcome
from sheaf_verify.gates import Gate, all_gates
from sheaf_verify.runner import _trace_event

ROOT = Path(__file__).resolve().parents[1]


def _solid_edges(text: str) -> set[tuple[str, str]]:
    edges = set()
    for line in text.splitlines():
        if "-->" not in line or "-.->" in line:
            continue
        nodes = [part.strip() for part in line.split("-->")]
        edges.update(zip(nodes, nodes[1:], strict=False))
    return edges


def test_committed_diagram_matches_gates():
    assert control_graph.problems(ROOT) == []
    committed = (ROOT / "docs" / "control-graph.mmd").read_text(encoding="utf-8")
    assert committed == control_graph.render()
    assert _solid_edges(committed) == {(dep, gate.id) for gate in all_gates() for dep in gate.depends_on}


def test_every_gate_appears_and_comparisons_hit_real_gates():
    text = control_graph.render()
    for gate in all_gates():
        assert re.search(rf"\b{gate.id}\b", text), gate.id
    for source, target in control_graph.COMPARISON_EDGES:
        assert f"{source} -.-> {target}" in text


def test_render_rejects_unknown_comparison_target():
    lone = (Gate("G1", "x", (), (), lambda ctx: []),)
    with pytest.raises(ValueError):
        control_graph.render(lone)


def test_check_detects_drift_and_write_repairs_it(tmp_path, capsys):
    for rel in ("docs/control-graph.mmd", "docs/graph.html", "GATES.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    assert control_graph.main(["--check", "--root", str(tmp_path)]) == 0
    mmd = tmp_path / "docs" / "control-graph.mmd"
    mmd.write_text(mmd.read_text(encoding="utf-8").replace("  G0 --> H0\n", ""), encoding="utf-8")
    gates_md = tmp_path / "GATES.md"
    gates_md.write_text(gates_md.read_text(encoding="utf-8").replace("  G0 --> H0\n", ""), encoding="utf-8")
    assert control_graph.main(["--check", "--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "control-graph.mmd" in err and "GATES.md" in err
    assert control_graph.main(["--write", "--root", str(tmp_path)]) == 0
    assert control_graph.problems(tmp_path) == []
    assert control_graph.main([]) == 0
    assert capsys.readouterr().out == control_graph.render()


def test_replace_block_requires_a_flowchart():
    with pytest.raises(ValueError):
        control_graph._replace_block("no diagram here", control_graph.render())
    with pytest.raises(ValueError):
        control_graph._replace_block("flowchart TD\n  A --> B\n", control_graph.render())


def test_trace_outcome_is_tri_state():
    assert trace_outcome("pass", True, False, {"G0": True}) == "pass"
    assert trace_outcome("fail", True, True, {"G0": True}) == "fail"
    assert trace_outcome("fail", False, False, {"G0": False}) == "blocked"
    assert trace_outcome("fail", False, False, {}) == "fail"


def test_runner_trace_labels_upstream_block_as_blocked():
    decision = {
        "criteria": [{"id": "upstream.G0", "passed": False, "detail": "none"}],
        "gate_id": "G1",
        "status": "fail",
    }
    event = _trace_event(decision)
    assert event["outcome"] == "blocked"
    assert event["status"] == "fail"
    own = {
        "criteria": [
            {"id": "upstream.G0", "passed": True, "detail": "sha"},
            {"id": "bpb.margin", "passed": False, "detail": "bad"},
        ],
        "gate_id": "G1",
        "status": "fail",
    }
    assert _trace_event(own)["outcome"] == "fail"


def test_coverage_counts_blocked_separately_and_derives_old_traces():
    old = {"gate_id": "G1", "status": "fail", "own_evaluated": False, "upstream": {"G0": False}}
    assert event_outcome(old) == "blocked"
    assert event_outcome({**old, "outcome": "pass"}) == "pass"
    events = [
        {**old, "test": "tests/a.py::t1"},
        {"gate_id": "G1", "status": "fail", "own_evaluated": True, "own_failed": True,
         "outcome": "fail", "upstream": {"G0": True}, "test": "tests/b.py::t2"},
    ]
    report = assess(events)
    row = next(item for item in report.gates if item["gate"] == "G1")
    assert row["blocked"] == ["tests/a.py"]
    assert row["fail"] == ["tests/b.py"]
    assert "G1 fail path has no test that exercises its own criteria" not in report.problems
    matrix = render_matrix(report)
    assert "| Gate | Pass | Fail | Blocked | Test file |" in matrix
    assert "| G1 | — | tests/b.py | tests/a.py | tests/a.py, tests/b.py |" in matrix


def test_real_run_emits_blocked_downstream(tmp_path):
    from sheaf_verify.synthetic import build

    was = gate_trace.is_enabled()
    gate_trace.enable(clear=not was)
    start = len(gate_trace.events())
    build(tmp_path / "store")
    fresh = gate_trace.events()[start:]
    if not was:
        gate_trace.disable()
    assert fresh and all(item["outcome"] in {"pass", "fail", "blocked"} for item in fresh)
    for item in fresh:
        if item["outcome"] == "blocked":
            assert item["own_evaluated"] is False
            assert False in item["upstream"].values()
