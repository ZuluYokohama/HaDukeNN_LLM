"""The control-graph checker fails unless every node and every edge is exercised."""

import json

import pytest

from sheaf_verify.gate_coverage import assess, load_events, main, render_matrix
from sheaf_verify.gates import all_gates

REQUIRED = {
    "G0", "G1", "G2", "G3", "G4", "G5", "G6",
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
    "P0", "P1", "P2", "P3", "P4", "P5", "P6",
    "HAL0", "HAL1", "HAL2", "HAL3", "HAL4", "HAL5", "HAL6",
}


def _event(gate, *, status, test, own_failed=False, upstream=None, own_evaluated=True):
    return {
        "gate_id": gate,
        "own_evaluated": own_evaluated,
        "own_failed": own_failed,
        "status": status,
        "test": test,
        "upstream": upstream or {},
    }


def _covered():
    events = []
    for gate in all_gates():
        events.append(_event(gate.id, status="pass", test=f"tests/pass_{gate.id}.py::test_ok"))
        events.append(_event(
            gate.id,
            status="fail",
            own_failed=True,
            test=f"tests/fail_{gate.id}.py::test_bad",
        ))
        for dep in gate.depends_on:
            events.append(_event(
                gate.id,
                status="pass",
                test=f"tests/edge_{dep}_{gate.id}.py::test_open",
                upstream={dep: True},
            ))
            events.append(_event(
                gate.id,
                status="fail",
                own_evaluated=False,
                test=f"tests/edge_{dep}_{gate.id}.py::test_shut",
                upstream={dep: False},
            ))
    return events


def test_required_nodes_are_the_control_graph():
    assert {gate.id for gate in all_gates()} == REQUIRED


def test_a_complete_trace_passes_and_names_the_test_file():
    report = assess(_covered())
    assert report.ok
    assert report.problems == []
    text = render_matrix(report)
    assert "| G0 |" in text
    assert "tests/pass_G0.py" in text
    assert "tests/fail_G0.py" in text
    assert "G0 -> H0" in text
    row = next(item for item in report.gates if item["gate"] == "G2")
    assert "tests/pass_G2.py" in row["pass"]
    assert row["fail"] == ["tests/fail_G2.py"]


def test_upstream_block_does_not_count_as_the_gates_own_failure():
    events = _covered()
    events = [item for item in events if not (item["gate_id"] == "G2" and item["own_failed"])]
    events.append(_event(
        "G2",
        status="fail",
        own_evaluated=False,
        test="tests/blocked.py::test_upstream",
        upstream={"G1": False},
    ))
    report = assess(events)
    assert not report.ok
    assert any("G2" in problem and "fail" in problem for problem in report.problems)


def test_a_missing_blocked_edge_fails():
    events = [
        item for item in _covered()
        if not (item["gate_id"] == "H0" and item["upstream"].get("G0") is False)
    ]
    report = assess(events)
    assert not report.ok
    assert any("G0 -> H0" in problem and "blocked" in problem for problem in report.problems)


def test_a_missing_pass_fails():
    events = [item for item in _covered() if not (item["gate_id"] == "P6" and item["status"] == "pass")]
    report = assess(events)
    assert any("P6" in problem and "pass" in problem for problem in report.problems)


def test_cli_writes_the_matrix_and_fails_closed(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("", encoding="utf-8")
    matrix = tmp_path / "matrix.md"
    payload = tmp_path / "matrix.json"
    code = main([str(trace), "--matrix", str(matrix), "--json", str(payload)])
    assert code == 1
    assert "G0" in matrix.read_text(encoding="utf-8")
    body = json.loads(payload.read_text(encoding="utf-8"))
    assert body["ok"] is False
    assert any(item["gate"] == "G1" for item in body["gates"])


def test_cli_accepts_a_complete_trace(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(item) + "\n" for item in _covered()), encoding="utf-8")
    matrix = tmp_path / "matrix.md"
    assert load_events(trace)[0]["gate_id"] == "P0"
    assert main([str(trace), "--matrix", str(matrix)]) == 0
    assert "pass" in matrix.read_text(encoding="utf-8").lower()


def test_a_broken_trace_line_is_a_failure(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{not json}\n", encoding="utf-8")
    assert main([str(trace), "--matrix", str(tmp_path / "matrix.md")]) == 1


def test_unscoped_and_unknown_edges_are_ignored():
    events = _covered()
    events.append(_event("G0", status="pass", test="not-a-test"))
    events.append(_event("G1", status="pass", test="tests/x.py::test_x", upstream={"not-a-gate": True}))
    assert assess(events).ok


def test_running_the_module_as_main_exits(tmp_path, monkeypatch):
    import runpy
    import sys

    trace = tmp_path / "trace.jsonl"
    trace.write_text("", encoding="utf-8")
    matrix = tmp_path / "matrix.md"
    monkeypatch.setattr(
        "sys.argv",
        ["sheaf_verify.gate_coverage", str(trace), "--matrix", str(matrix)],
    )
    sys.modules.pop("sheaf_verify.gate_coverage", None)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("sheaf_verify.gate_coverage", run_name="__main__")
    assert caught.value.code == 1
    assert matrix.is_file()


def test_runner_records_a_real_pass(tmp_path):
    from sheaf_verify import gate_trace
    from sheaf_verify.synthetic import build

    was = gate_trace.is_enabled()
    gate_trace.enable(clear=not was)
    gate_trace.set_test("tests/test_gate_coverage.py::test_runner_records_a_real_pass")
    start = len(gate_trace.events())
    build(tmp_path / "store")
    fresh = gate_trace.events()[start:]
    if not was:
        gate_trace.disable()
    g0 = [item for item in fresh if item["gate_id"] == "G0"]
    assert g0 and g0[0]["status"] == "pass" and g0[0]["own_evaluated"] is True
    assert g0[0]["test"].endswith("test_runner_records_a_real_pass")
