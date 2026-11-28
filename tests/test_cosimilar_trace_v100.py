"""Comprehensive unit, mathematical law, adversarial spoof, and CLI tests for v1.0.0 Cosimilar Trace Engine."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from hadukenn_core.cosimilar_engine import run_t47_cosimilar_campaign, write_t47_artifacts
from sheaf_verify.campaign import verify_t47_cosimilar_trace
from sheaf_verify.cli import main as cli_main
from sheaf_verify.cosimilar_trace import (
    ActiveHypergraphTraceSearcher,
    CosimilarObserverSheafEngine,
    ExactRationalMarkovKernel,
    HyperedgeSpec,
    compute_cosimilar_quotient,
    evaluate_three_way_synergy,
)
from sheaf_verify.hashing import SheafError

ROOT = Path(__file__).resolve().parents[1]


def test_exact_rational_schur_trace_laws_and_edge_cases():
    mat = [
        [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(0, 1)],
        [Fraction(1, 6), Fraction(1, 2), Fraction(1, 6), Fraction(1, 6)],
        [Fraction(1, 5), Fraction(1, 5), Fraction(2, 5), Fraction(1, 5)],
        [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
    ]
    kern = ExactRationalMarkovKernel(mat, ["a", "b", "c", "d"])
    laws = kern.verify_trace_laws(outer_window=[0, 1, 2], inner_window=[0, 2])
    assert laws.stochastic_exact is True
    assert laws.transitive_exact is True
    assert laws.stationary_restricted_exact is True
    assert laws.dilation_multiplicative_exact is True
    assert laws.total_dilation > laws.outer_dilation > 1.0

    # Trivial 1-state and full-window traces
    one = ExactRationalMarkovKernel([[1]])
    assert one.stationary_distribution() == [Fraction(1, 1)]
    full_tr = kern.schur_trace([0, 1, 2, 3])
    assert full_tr.matrix == kern.matrix

    # Invalid inputs rejected cleanly
    with pytest.raises(SheafError):
        ExactRationalMarkovKernel([])
    with pytest.raises(SheafError):
        ExactRationalMarkovKernel([[-1, 2], [0, 1]])
    with pytest.raises(SheafError):
        ExactRationalMarkovKernel([[Fraction(1, 2), Fraction(1, 4)], [Fraction(1, 2), Fraction(1, 2)]])
    with pytest.raises(SheafError):
        ExactRationalMarkovKernel([[1, 0], [0, 1]], ["only_one"])
    with pytest.raises(SheafError):
        kern.schur_trace([])
    with pytest.raises(SheafError):
        kern.verify_trace_laws([0, 1], [2])


def test_cosimilar_quotient_zero_surprise_vs_coarse_lumping():
    # 4-state chain where (0,1) and (2,3) are exact cosimilar pairs
    mat = [
        [Fraction(2, 6), Fraction(2, 6), Fraction(1, 6), Fraction(1, 6)],
        [Fraction(1, 6), Fraction(3, 6), Fraction(0, 6), Fraction(2, 6)],
        [Fraction(1, 8), Fraction(1, 8), Fraction(3, 8), Fraction(3, 8)],
        [Fraction(0, 8), Fraction(2, 8), Fraction(4, 8), Fraction(2, 8)],
    ]
    kern = ExactRationalMarkovKernel(mat)
    quot = compute_cosimilar_quotient({"default": mat}, observables=["x", "x", "y", "y"])
    assert quot == [[0, 1], [2, 3]]
    assert kern.lumping_surprise_bpb(quot) == pytest.approx(0.0, abs=1e-14)

    # Non-cosimilar cross-pairing [[0, 2], [1, 3]] incurs strictly positive KL surprise
    bad_part = [[0, 2], [1, 3]]
    assert kern.lumping_surprise_bpb(bad_part) > 1e-4

    with pytest.raises(SheafError):
        kern.lumping_surprise_bpb([[0, 1], [2]])
    with pytest.raises(SheafError):
        compute_cosimilar_quotient({})
    with pytest.raises(SheafError):
        compute_cosimilar_quotient({"r": []}, observables=[])
    with pytest.raises(SheafError):
        compute_cosimilar_quotient({"r": [[1, 0], [0, 1]]}, observables=["a"])


def test_active_hypergraph_search_and_3way_synergy():
    edges = [
        HyperedgeSpec("e0", (0, 1, 2), ("src", "mid", "dst"), weight=3.0),
        HyperedgeSpec("e1", (1, 2, 3, 4), ("a", "b", "c", "d"), weight=2.0),
        HyperedgeSpec("e2", (3, 4, 5), ("x", "y", "z"), weight=1.0),
    ]
    searcher = ActiveHypergraphTraceSearcher(6, edges, [2.0, 2.0, 2.0, 0.5, 0.2, 0.1])
    assert searcher.incidence_storage_ints() == 10
    full_tr = searcher.float_schur_trace([0, 1, 2, 3, 4, 5])
    assert len(full_tr) == 6
    res = searcher.search_optimal_window(seed_node=0, budget_k=3)
    assert len(res.selected_window) == 3
    assert res.candidates_evaluated < res.total_combinatorial_subsets

    triples = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)] * 25
    syn = evaluate_three_way_synergy(triples)
    assert syn["pairwise_markov_accuracy"] == pytest.approx(0.5)
    assert syn["hypergraph_3way_accuracy"] == pytest.approx(1.0)

    with pytest.raises(SheafError):
        evaluate_three_way_synergy([])
    with pytest.raises(SheafError):
        evaluate_three_way_synergy([(0, 1, 2)])
    with pytest.raises(SheafError):
        ActiveHypergraphTraceSearcher(1, edges)
    with pytest.raises(SheafError):
        searcher.search_optimal_window(seed_node=0, budget_k=0)


def test_observer_sheaf_fork_selective_cache_spoof_and_scitt():
    edges = [
        HyperedgeSpec("c0_a", (0, 1, 2, 3), ("a", "b", "c", "d"), weight=2.0),
        HyperedgeSpec("c0_b", (1, 2, 4, 5), ("a", "b", "c", "d"), weight=1.5),
        HyperedgeSpec("c0_c", (0, 2, 3, 5), ("a", "b", "c", "d"), weight=1.2),
        HyperedgeSpec("c1_a", (6, 7, 8), ("a", "b", "c"), weight=2.5),
    ]
    observers = {
        "o1": [0, 1, 2, 3],
        "o2": [1, 2, 3, 4],
        "o3": [0, 2, 3, 5],
        "o_disjoint": [6, 7, 8],
    }
    eng = CosimilarObserverSheafEngine(9, edges, observers)
    coh0 = eng.check_overlap_coherence()
    assert coh0["coherent"] is True
    assert coh0["verdict"] == "COHERENT"

    forked = eng.fork_branch()
    upd = forked.apply_intervention("c0_a", 6.0)
    assert upd["reused"] == ["o_disjoint"]
    assert forked.state_root_sha256() != eng.state_root_sha256()
    assert forked.check_overlap_coherence()["coherent"] is True

    # 1-vs-1 tie when only two observers share an overlap vs multi-witness spoof localization
    bad_o2 = [list(r) for r in forked.cache["o2"].traced_matrix]
    bad_o2[0] = [0.85, 0.05, 0.05, 0.05]
    loc = forked.check_overlap_coherence(external_overrides={"o2": bad_o2})
    assert loc["coherent"] is False
    assert loc["verdict"] == "SPOOF_LOCALIZED"
    assert loc["flagged_observers"] == ["o2"]

    bundle = forked.emit_scitt_audit_bundle(b"unit-test-signing-key-v100")
    assert bundle["scitt_verified"] is True

    with pytest.raises(SheafError):
        forked.apply_intervention("missing_edge", 1.0)
    with pytest.raises(SheafError):
        forked.apply_intervention("c0_a", -1.0)


def test_t47_campaign_and_cli_roundtrip(tmp_path: Path):
    artifact = write_t47_artifacts(tmp_path)
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    ok_flag, criteria, advisories = verify_t47_cosimilar_trace(doc)
    assert ok_flag is True
    assert all(c.passed for c in criteria)
    assert any("COSIMILAR_TRACE_READY" in a for a in advisories)

    rc = cli_main(["campaign", "--mode", "t47-cosimilar-trace", "--path", str(artifact), "--json"])
    assert rc == 0

    # Tampered T-47 artifact must fail gate verification and return CLI exit code 1
    bad_doc = dict(doc)
    bad_doc["realm_2_hypergraph_vs_2d_markov"] = dict(doc["realm_2_hypergraph_vs_2d_markov"])
    bad_doc["realm_2_hypergraph_vs_2d_markov"]["hypergraph_3way_xor_accuracy"] = 0.5
    bad_path = tmp_path / "bad_t47.json"
    bad_path.write_text(json.dumps(bad_doc), encoding="utf-8")
    rc_bad = cli_main(["campaign", "--mode", "t47-cosimilar-trace", "--path", str(bad_path), "--json"])
    assert rc_bad == 1

    direct = run_t47_cosimilar_campaign()
    assert direct["realm_1_trace_and_cosimilarity"]["seeds_passing_all_4_trace_laws"] == 5


def test_observer_sheaf_rejects_non_finite_matrices():
    edges = [
        HyperedgeSpec("e0", (0, 1, 2), ("a", "b", "c"), weight=1.5),
        HyperedgeSpec("e1", (1, 2, 3), ("a", "b", "c"), weight=1.5),
        HyperedgeSpec("e2", (2, 3, 4), ("a", "b", "c"), weight=1.5),
    ]
    observers = {"obs_a": [0, 1, 2, 3], "obs_b": [1, 2, 3, 4], "obs_c": [0, 1, 2, 4]}
    eng = CosimilarObserverSheafEngine(5, edges, observers)

    nan_mat = [[float("nan")] * 4 for _ in range(4)]
    res_nan = eng.check_overlap_coherence(external_overrides={"obs_b": nan_mat})
    assert res_nan["coherent"] is False
    assert res_nan["verdict"] == "SPOOF_LOCALIZED"
    assert res_nan["flagged_observers"] == ["obs_b"]
    assert math.isinf(float(res_nan["max_overlap_l1_discrepancy"]))

    inf_mat = [[float("inf")] * 4 for _ in range(4)]
    res_inf = eng.check_overlap_coherence(external_overrides={"obs_a": inf_mat})
    assert res_inf["coherent"] is False
    assert res_inf["verdict"] == "SPOOF_LOCALIZED"
    assert res_inf["flagged_observers"] == ["obs_a"]

    eng2 = CosimilarObserverSheafEngine(5, edges, {"obs_a": [0, 1, 2, 3], "obs_b": [1, 2, 3, 4]})
    res_tie = eng2.check_overlap_coherence(external_overrides={"obs_b": nan_mat})
    assert res_tie["coherent"] is False
    assert res_tie["verdict"] == "AMBIGUOUS_CONFLICT"
    assert res_tie["flagged_observers"] == ["obs_a", "obs_b"]

    eng_1node = CosimilarObserverSheafEngine(5, edges, {"obs_a": [0, 1], "obs_b": [1, 4]})
    nan_2x2 = [[float("nan")] * 2 for _ in range(2)]
    res_1node = eng_1node.check_overlap_coherence(external_overrides={"obs_b": nan_2x2})
    assert res_1node["coherent"] is False
    assert res_1node["verdict"] == "AMBIGUOUS_CONFLICT"
    assert math.isinf(float(res_1node["max_overlap_l1_discrepancy"]))


def test_t47_campaign_handles_null_nested_sections_without_crash(tmp_path: Path, capsys):
    artifact = ROOT / "iterations" / "t47-cosimilar-trace" / "T47_COSIMILAR_TRACE_RESULTS.json"
    base_doc = json.loads(artifact.read_text(encoding="utf-8"))
    for key in (
        "realm_1_trace_and_cosimilarity",
        "realm_2_hypergraph_vs_2d_markov",
        "realm_3_active_window_search",
        "realm_4_observer_sheaf_and_interventions",
    ):
        null_doc = dict(base_doc)
        null_doc[key] = None
        ok_flag, criteria, _ = verify_t47_cosimilar_trace(null_doc)
        assert ok_flag is False
        assert any(not c.passed for c in criteria)

    null_path = tmp_path / "t47_null.json"
    bad_doc = dict(base_doc)
    bad_doc["realm_1_trace_and_cosimilarity"] = None
    null_path.write_text(json.dumps(bad_doc), encoding="utf-8")
    rc = cli_main(["campaign", "--mode", "t47-cosimilar-trace", "--path", str(null_path), "--json"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False


def test_t47_campaign_rejects_top_level_non_dict(tmp_path: Path, capsys):
    for bad_root in (None, [], 42, "bad", True, False):
        ok_flag, criteria, advisories = verify_t47_cosimilar_trace(bad_root)  # type: ignore[arg-type]
        assert ok_flag is False and advisories == []
        assert len(criteria) == 1 and criteria[0].id == "T47_SCHEMA_AND_SEEDS" and not criteria[0].passed

    for idx, raw in enumerate(("null", "[]", "42")):
        p = tmp_path / f"root_{idx}.json"
        p.write_text(raw, encoding="utf-8")
        for flag in ("--path", "--in"):
            assert cli_main(["campaign", "--mode", "t47-cosimilar-trace", flag, str(p), "--json"]) == 1
            assert json.loads(capsys.readouterr().out)["ok"] is False
