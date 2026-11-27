"""Tests for v0.5.0 IncrementalHypergraphKernel, unified span-sheaf pre-gate, and T-42 campaign gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from hadukenn_core.exec_bridge import (
    AuditedCachedDAG,
    IncrementalHypergraphKernel,
    verify_unified_span_sheaf_bridge,
)
from sheaf_verify.campaign import verify_t42_exec_bridge
from sheaf_verify.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
T42_JSON = ROOT / "iterations" / "t42-unified-neuro-symbolic" / "T42_UNIFIED_EXEC_RESULTS.json"


def test_incremental_kernel_fork_refusal_retract_and_tamper_detection() -> None:
    kernel = IncrementalHypergraphKernel()
    assert kernel.execute({"action": "source", "id": "s1", "text": "A holds 20 L, B holds 5 L, transfer 8 L."})["status"] == "ok"
    assert kernel.execute({"action": "set", "scope": "main", "name": "A0", "value": "20", "unit": "L", "source": "s1"})["status"] == "ok"
    assert kernel.execute({"action": "set", "scope": "main", "name": "T1", "value": "8", "unit": "L", "source": "s1"})["status"] == "ok"
    assert kernel.execute(
        {
            "action": "derive",
            "scope": "main",
            "name": "A1",
            "op": "sub",
            "args": {"left": "A0", "right": "T1"},
            "source": "s1",
        }
    )["status"] == "ok"

    q1 = kernel.execute({"action": "query", "scope": "main", "name": "A1"})
    assert q1["status"] == "ok" and q1["value"] == "12"
    last_receipt = kernel.export_trace()[-1]
    assert last_receipt["state_before"] == last_receipt["state_after"]

    assert kernel.execute({"action": "fork", "scope": "main", "new_scope": "hypo"})["status"] == "ok"
    assert kernel.execute({"action": "set", "scope": "hypo", "name": "A0", "value": "30", "unit": "L", "source": "s1"})["status"] == "ok"
    q_hypo = kernel.execute({"action": "query", "scope": "hypo", "name": "A1"})
    q_main = kernel.execute({"action": "query", "scope": "main", "name": "A1"})
    assert q_hypo["value"] == "22"
    assert q_main["value"] == "12"

    bad_assert = kernel.execute({"action": "assert", "scope": "main", "name": "A1", "value": "99", "unit": "L"})
    assert bad_assert["status"] == "error" and bad_assert["error"] == "conflict"

    assert kernel.execute({"action": "retract", "scope": "main", "name": "A0"})["status"] == "ok"
    q_unres = kernel.execute({"action": "query", "scope": "main", "name": "A1"})
    assert q_unres["status"] == "error" and q_unres["error"] == "missing_reference"

    trace_records = kernel.export_trace()
    replayed = IncrementalHypergraphKernel.replay(trace_records, expected_head=kernel.head)
    assert replayed.head == kernel.head

    tampered = copy.deepcopy(trace_records)
    for rec in tampered:
        if rec["result"].get("value") == "12":
            rec["result"]["value"] = "999"
            break
    prev = IncrementalHypergraphKernel.ZERO_HASH
    for rec in tampered:
        rec["previous_hash"] = prev
        payload = {k: v for k, v in rec.items() if k != "hash"}
        rec["hash"] = kernel._digest(payload)
        prev = rec["hash"]
    failed = False
    try:
        IncrementalHypergraphKernel.replay(tampered, expected_head=prev)
    except ValueError:
        failed = True
    assert failed is True

    dag = AuditedCachedDAG()
    assert dag.execute({"action": "source", "id": "s1", "text": "X=7"})["status"] == "ok"
    assert dag.execute({"action": "set", "scope": "main", "name": "X", "value": "7", "unit": "L", "source": "s1"})["status"] == "ok"
    assert len(dag.head) == 64


def test_unified_span_sheaf_bridge_invariants() -> None:
    res = verify_unified_span_sheaf_bridge()
    assert res["span_graph_prefix_hash_invariant"] is True
    assert res["oespan_suffix_drift"] == 0.0
    assert res["mapped_byte_anchor_tokens"] == res["total_visible_tokens"]
    assert res["sheaf_1v1_tie_regime"] == "ABSTAIN_TIE"
    assert res["sheaf_2v1_fault_regime"] == "RECOVERABLE_FAULT"
    assert res["sheaf_2v1_isolated_fault"] == ["src_corrupt"]
    assert res["derived_A_final"] == "11"
    assert res["derived_B_final"] == "14"
    assert res["derived_Total"] == "25"
    assert res["semantic_replay_verified"] is True


def test_verify_t42_exec_bridge_and_cli() -> None:
    doc = json.loads(T42_JSON.read_text(encoding="utf-8"))
    ok, criteria, advisories = verify_t42_exec_bridge(doc)
    assert ok is True
    assert len(criteria) == 4
    assert all(c.passed for c in criteria)
    assert any("COMMITMENT_COMPLEXITY_LAW" in a for a in advisories)

    broken = copy.deepcopy(doc)
    broken["part3_learned_ood_policy"]["mean_ood_accuracy_hypergraph_sheaf"] = 0.40
    ok_bad, crit_bad, _ = verify_t42_exec_bridge(broken)
    assert ok_bad is False
    assert any(c.id == "T42_LEARNED_OOD_COMPOSITION_WIN" and not c.passed for c in crit_bad)

    rc = cli_main(["campaign", "--mode", "t42-exec-bridge", "--json", "--path", str(T42_JSON)])
    assert rc == 0
