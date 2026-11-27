"""Failure branches of the checkers, the statistics, and store verification."""

import copy
import json
import math
from pathlib import Path

import pytest

from sheaf_verify.checks import (
    bpb_criteria,
    causality_criteria,
    claim_criteria,
    code_tree_criteria,
    corpus_criteria,
    environment_criteria,
    flop_and_byte_criteria,
    hypergraph_errors,
    kernel_criteria,
    leakage_criteria,
    loop_criterion,
    random_control_criteria,
    seed_criteria,
    vocab_criteria,
)
from sheaf_verify.hashing import SheafError, canonical_bytes, sha256_hex
from sheaf_verify.spec_gates import (
    g0_criteria,
    g1_criteria,
    g2_criteria,
    g3_criteria,
    g4_criteria,
    g5_criteria,
    g6_criteria,
    h1_criteria,
    h7_criteria,
    p0_fairness,
    p2_criteria,
    p3_criteria,
    p4_criteria,
    p5_criteria,
    p6_criteria,
    stage_criteria,
)
from sheaf_verify.stats import (
    TTest,
    bootstrap_mean_difference,
    hedges_g,
    mean,
    one_sided_less,
    paired_t,
    sample_variance,
    slope_interval,
    student_t_critical,
    student_t_two_tailed,
    welch,
)
from sheaf_verify.synthetic import TEST_KEY, build, default_parts
from sheaf_verify.verify import verify_store


def _failed(rows, identifier):
    if not isinstance(rows, list):
        rows = [rows]
    found = {item.id: item for item in rows}
    assert found[identifier].passed is False
    return found


def test_broken_hypergraph_and_corpus_are_rejected():
    assert hypergraph_errors("no")
    errors = hypergraph_errors({
        "document_ids": ["", "doc", "doc"],
        "format": "sheaf-verify.hypergraph.v1",
        "hyperedges": ["nope", {"documents": ["other"], "id": "", "positions": [0], "vertices": ["missing"]}],
        "vertices": [
            "nope",
            {"document": "nope", "id": "", "position": True},
            {"document": "doc", "id": "v0", "position": -1},
        ],
    })
    assert len(errors) > 3
    graph = copy.deepcopy(default_parts()["hypergraph"])
    graph["hyperedges"][0]["positions"] = [9, 9]
    assert any("positions" in item for item in hypergraph_errors(graph))
    rows = {
        "test": {"document_ids": {"a"}, "order": 0, "payload_sha256": "aa", "size_bytes": 0},
        "train": {"document_ids": {"a"}, "order": True, "payload_sha256": "aa", "size_bytes": 1},
        "valid": {"document_ids": {"a"}, "order": 1, "payload_sha256": "aa", "size_bytes": 1},
    }
    failed = _failed(corpus_criteria(rows), "corpus.distinct")
    assert failed["corpus.overlap.train_valid"].passed is False
    assert failed["corpus.chronological"].passed is False


def test_leakage_vocab_flops_and_causality_fail_closed():
    parts = default_parts()
    graph = parts["hypergraph"]
    _failed(leakage_criteria(
        {"format": "no"},
        train_ids=set(),
        valid_ids=set(),
        test_ids=set(),
        inputs=[],
        train_manifest="t",
        valid_manifest="v",
        test_manifest="s",
        label="hypergraph",
    ), "hypergraph.structure")
    leaked = copy.deepcopy(graph)
    leaked["built_from_split"] = "test"
    failed = _failed(leakage_criteria(
        leaked,
        train_ids={"doc-train"},
        valid_ids={"doc-valid"},
        test_ids={"doc-test"},
        inputs=["v"],
        train_manifest="t",
        valid_manifest="v",
        test_manifest="s",
        label="hypergraph",
    ), "hypergraph.split")
    assert failed["hypergraph.inputs"].passed is False
    _failed(vocab_criteria({"format": "no"}, {}, [], "t", set()), "vocabulary.format")
    metrics = copy.deepcopy(parts["metrics"])
    _failed(vocab_criteria(
        {"format": "sheaf-verify.vocabulary.v1", "units": ["aa", "aa"]},
        metrics,
        ["held"],
        "train",
        {"held"},
    ), "vocabulary.units")
    metrics["flop_relative_tolerance"] = 2
    metrics["arms"]["hypergraph"]["flops"] = {"graph_build": 0, "total": 1, "train": 1}
    metrics["arms"]["superword_bpe"]["flops"] = "no"
    failed = _failed(flop_and_byte_criteria(metrics, 16), "flops.tolerance")
    assert failed["flops.graph.hypergraph"].passed is False
    assert failed["flops.identity.superword_bpe"].passed is False
    _failed(causality_criteria("no", graph), "causality.format")
    trace = copy.deepcopy(parts["trace"])
    trace["horizon"] = 99
    _failed(causality_criteria(trace, graph), "causality.horizon")
    trace = copy.deepcopy(parts["trace"])
    trace["edges_consulted"][1]["edge_ids"] = ["missing"]
    _failed(causality_criteria(trace, graph), "causality.edges")


def test_controls_seeds_environment_and_claims_fail_closed():
    parts = default_parts()
    graph = copy.deepcopy(parts["hypergraph"])
    graph["randomized"] = True
    _failed(random_control_criteria(graph, parts["random"], parts["metrics"]), "random.flag")
    _failed(seed_criteria({"format": "no"}, {}, minimum=5), "seeds.format")
    seeds = copy.deepcopy(parts["seeds"])
    seeds["arms"]["hypergraph"] = [1, 1, 1, 1, 1]
    _failed(seed_criteria(seeds, parts["metrics"]), "seeds.unique.hypergraph")
    paired = copy.deepcopy(parts["seeds"])
    paired["design"] = "paired"
    _failed(seed_criteria(paired, parts["metrics"]), "seeds.paired")
    _failed(environment_criteria({"format": "sheaf-verify.environment.v1", "python": "unknown"}), "environment.python")
    _failed(bpb_criteria({"arms": {}}, 16), "bpb.arms")
    _failed(claim_criteria(b"short"), "claim.text")
    _failed(claim_criteria(b"\x00" + b"x" * 30), "claim.text")
    _failed(claim_criteria(b"ab" * 32), "claim.frozen")
    _failed(kernel_criteria({"backend": "cuda", "entry": ""}, "aa" * 32, 0, set()), "kernel.entry")
    _failed(loop_criterion({
        "loops": [{"path_a": ["missing"], "path_b": ["op"]}],
        "operations": [{"id": "op", "morphism": {"a": "b"}, "source": "s", "target": "s"}],
        "states": ["s"],
    }), "sheaf.loop_closure")
    listing = [{"mode": "100644", "path": "model.py", "sha256": "nope", "size": 1}]
    _failed(code_tree_criteria({"metadata": {}}, listing, [("model.py", "100644", b"x")]), "code.bytes")


def test_spec_records_reject_incomplete_documents():
    assert g0_criteria(None)[0].passed is False
    record = {
        "build_cpu_hours": 3,
        "byte_fallback_percent": 4,
        "comparison": "other",
        "format": "sheaf-verify.g0.v1",
        "gpt2_bpe_valid_units": 100,
        "not_data": True,
        "peak_ram_gb": 20,
        "record_class": "synthetic",
        "split": "train",
        "valid_units": 100,
        "vocab_size": 70000,
    }
    failed = _failed(g0_criteria(record), "g0.split")
    assert failed["g0.fallback"].passed is False
    assert failed["g0.build_cpu"].passed is False
    assert failed["g0.vocab_cap"].passed is False
    assert g1_criteria({}, 1, 1)[0].passed is False
    assert g2_criteria({"comparisons": []})[0].passed is False
    assert g3_criteria({"dtype": "fp32", "rel_err": 1, "sheafheat_step_fraction": 1})[0].id == "g3.dtype"
    assert g4_criteria({"baseline": {}, "components": [], "headline": {}})[0].passed is False
    assert g5_criteria({"sizes": [{"gain": -1, "params": 15000000}]})[0].passed is False
    assert g6_criteria({"a7_all_in_flops": 10, "a7_bpb": 1, "all_in_flops": 10, "arm": "other", "bpb": 2})[0].passed is False
    assert h1_criteria({"arms": {}})[0].passed is False
    assert h7_criteria({"headline": True, "tasks": []})[0].passed is False
    assert p0_fairness({"baseline_wall_interpolated": True, "bytes_per_s": 0, "flops": 0, "mfu": 2, "schedule_complete": False})[0].passed is False
    assert p2_criteria({}, 1)[0].passed is False
    assert p3_criteria({"fineweb_edu_val_bpb": 0, "not_data": False, "record_class": "raw", "wt103_full_epoch": False})[0].passed is False
    assert p4_criteria({"sizes": [{"params": 1, "seeds": 0}]})[0].passed is False
    assert p5_criteria({"cuda_graph_all_arms": False, "time_to_target_bpb_s": 0})[0].passed is False
    assert p6_criteria({"replication_note": "short", "replication_recorded": False})[0].passed is False
    assert stage_criteria({}, "H0")[0].passed is False


def test_statistics_cover_degenerate_samples():
    with pytest.raises(SheafError):
        mean([])
    with pytest.raises(SheafError):
        sample_variance([1.0])
    same = welch([1.0, 1.0], [1.0, 1.0])
    assert same.p == 1.0
    apart = welch([1.0, 1.0], [2.0, 2.0])
    assert math.isinf(apart.t)
    assert paired_t([1.0, 1.0], [1.0, 1.0]).p == 1.0
    assert math.isinf(paired_t([2.0, 2.0], [1.0, 1.0]).t)
    assert hedges_g([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert math.isinf(hedges_g([2.0, 2.0], [1.0, 1.0]))
    with pytest.raises(SheafError):
        bootstrap_mean_difference([1.0], [1.0], seed=1, draws=0, paired=False)
    low, high = bootstrap_mean_difference([1.0, 2.0], [0.0, 1.0], seed=1, draws=1, paired=True)
    assert low == high
    assert student_t_two_tailed(math.inf, 4) == 0.0
    assert one_sided_less(TTest(math.inf, 4, 0.0)) == 1.0
    with pytest.raises(SheafError):
        student_t_critical(0.0, 4)
    with pytest.raises(SheafError):
        slope_interval([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    slope, low, high = slope_interval([0.0, 1.0, 2.0], [0.0, 1.0, 3.0])
    assert low < slope < high


def test_verify_reports_a_tampered_store(tmp_path: Path):
    store = build(tmp_path / "store")
    (store.root / "notes.txt").write_text("extra\n", encoding="utf-8")
    (store.root / "payloads" / "left.partial").write_bytes(b"")
    (store.root / "manifests" / "nope.txt").write_text("{}\n", encoding="utf-8")
    manifest_path = next((store.root / "manifests").glob("*.json"))
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    pretty = json.dumps(document, indent=2).encode("utf-8")
    digest = sha256_hex(pretty)
    assert pretty != canonical_bytes(document)
    (store.root / "manifests" / f"{digest}.json").write_bytes(pretty)
    payload = next(p for p in sorted((store.root / "payloads").iterdir()) if not p.name.endswith(".partial"))
    if payload.is_file():
        payload.unlink()
    (store.root / "payloads" / "orphan").symlink_to(store.root / "notes.txt")
    report = verify_store(store, key=TEST_KEY, strict=True)
    text = "\n".join(report.problems)
    assert "unexpected path" in text
    assert "partial file" in text
    assert "unexpected file" in text
    assert "not the canonical hash" in text
    assert report.ok is False
