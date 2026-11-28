"""Tests for v0.2.0 pillars: numerics, sheaf cohomology, SAVI, attestation, anticheat, and mutation."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from tests.support import criterion, latest

from sheaf_verify.anticheat import audit_kernel_source, four_distribution_probe_criteria
from sheaf_verify.attestation import (
    build_ledger_receipt,
    dsse_pae,
    dsse_sign,
    dsse_verify,
    intoto_statement_v1,
    merkle_audit_path,
    merkle_consistency_proof,
    merkle_tree_head,
    validate_strict_json_bytes,
    verify_ledger_receipt,
    verify_merkle_inclusion,
)
from sheaf_verify.hashing import SheafError
from sheaf_verify.ledger import Ledger
from sheaf_verify.mutation import run_mutation_suite
from sheaf_verify.numerics import (
    CertifiedBall,
    certified_bpb_identity,
    drift_gate_criterion,
    hardware_wallclock_criteria,
    learner_sampler_parity_criteria,
    reduction_drift_envelope,
)
from sheaf_verify.savi import (
    ebh_discoveries,
    empirical_bayes_paired_t,
    hoeffding_bentkus_p,
    kelly_e_process,
    ltt_calibrate_thresholds,
    merge_dependent_e_values,
)
from sheaf_verify.sheaf_cohomology import (
    CellularSheaf,
    IncrementalPartitionedSheaf,
    SheafEdge,
    cellular_sheaf_criteria,
    diagnose_contradictions,
    grounded_mapping_cone_energy,
)
from sheaf_verify.synthetic import TEST_KEY, build


def test_certified_ball_arithmetic_and_predicates():
    a = CertifiedBall.from_number("0.1")
    b = CertifiedBall.from_number("0.2")
    c = CertifiedBall.from_number("0.3")
    total = a + b
    assert total.center == c.center
    assert total.radius == 0
    assert (c - b).center == a.center
    prod = a * b
    assert prod.center == Fraction(2, 100)
    quot = c / a
    assert quot.center == Fraction(3, 1)
    root = CertifiedBall.from_number(2).sqrt()
    assert root.low_exact * root.low_exact <= Fraction(2) <= root.high_exact * root.high_exact
    assert CertifiedBall.from_number(0).sqrt().center == 0
    slacked = CertifiedBall.from_number(1.25, ulp_slack=4)
    assert slacked.radius > 0
    assert slacked.radius_float > 0
    assert slacked.strictly_less_than(1.26)
    assert slacked.certified_at_most(1.26)
    assert slacked.strictly_greater_than(1.24)
    assert (-slacked).center == -slacked.center
    with pytest.raises(SheafError):
        _ = a / CertifiedBall(Fraction(0), Fraction(1, 10))
    with pytest.raises(SheafError):
        CertifiedBall.from_number(True)
    with pytest.raises(SheafError):
        CertifiedBall.from_number("not-a-fraction")
    with pytest.raises(SheafError):
        CertifiedBall.from_number(float("nan"))
    with pytest.raises(SheafError):
        CertifiedBall(Fraction(0), Fraction(-1))
    with pytest.raises(SheafError):
        CertifiedBall.from_number(-1).sqrt()


def test_bpb_ball_and_precision_envelopes():
    seeds = [1.2, 1.3, 1.4]
    assert certified_bpb_identity(130.0, 100, seeds).passed is True
    assert certified_bpb_identity(135.0, 100, seeds).passed is False
    assert certified_bpb_identity(130.0, 0, seeds).passed is False

    assert reduction_drift_envelope("fp32", 1, mode="tree") == 0.0
    tree_env = reduction_drift_envelope("bf16", 65536, mode="tree")
    seq_env = reduction_drift_envelope("bf16", 65536, mode="sequential")
    assert tree_env < 0.08
    assert seq_env == float("inf")
    with pytest.raises(SheafError):
        reduction_drift_envelope("unknown", 8)
    with pytest.raises(SheafError):
        reduction_drift_envelope("bf16", 8, mode="bad")

    assert drift_gate_criterion(0.01, "bf16", 1024).passed is True
    assert drift_gate_criterion(0.5, "bf16", 1024).passed is False

    parity_ok = learner_sampler_parity_criteria({
        "batch_invariant_reductions": True,
        "learner_dtype": "mxfp8",
        "sampler_dtype": "mxfp8",
        "unmatched_layers": [],
    })
    assert all(item.passed for item in parity_ok)
    parity_bad = learner_sampler_parity_criteria({
        "batch_invariant_reductions": False,
        "learner_dtype": "mxfp8",
        "sampler_dtype": "bf16",
        "unmatched_layers": ["up_proj"],
    })
    assert all(not item.passed for item in parity_bad)
    assert learner_sampler_parity_criteria({"learner_dtype": "bad"})[0].passed is False

    hw_5070 = hardware_wallclock_criteria(
        gpu_name="NVIDIA GeForce RTX 5070",
        sm_arch="sm_120",
        dtype="bf16",
        benchmark_role="wallclock",
        solo_device=True,
    )
    assert all(item.passed for item in hw_5070)
    hw_1660_wall = hardware_wallclock_criteria(
        gpu_name="NVIDIA GeForce GTX 1660 Ti",
        sm_arch="sm_75",
        dtype="bf16",
        benchmark_role="wallclock",
        solo_device=False,
    )
    assert any(not item.passed for item in hw_1660_wall)
    hw_1660_eval = hardware_wallclock_criteria(
        gpu_name="NVIDIA GeForce GTX 1660 Ti",
        sm_arch="sm_75",
        dtype="fp32",
        benchmark_role="eval_fp32",
    )
    assert all(item.passed for item in hw_1660_eval)
    assert hardware_wallclock_criteria(
        gpu_name="GPU", sm_arch="sm_80", dtype="fp32", benchmark_role="unknown"
    )[0].passed is False


def test_cellular_sheaf_cohomology_cone_incremental_and_ties():
    edges = [
        SheafEdge("e01", "v0", "v1", 1, [[1.0]], [[1.0]]),
        SheafEdge("e12", "v1", "v2", 1, [[1.0]], [[1.0]]),
        SheafEdge("e02", "v0", "v2", 1, [[1.0]], [[1.0]]),
    ]
    sheaf_1d = CellularSheaf({"v0": 1, "v1": 1, "v2": 1}, edges)
    coh_1d = sheaf_1d.cohomology()
    assert coh_1d.dim_h0 == 1
    assert coh_1d.dim_h1 == 1
    assert coh_1d.spectral_gap > 2.9

    sheaf_2d = CellularSheaf(
        {"v0": 1, "v1": 1, "v2": 1},
        edges,
        faces=[(("e01", 1), ("e12", 1), ("e02", -1))],
    )
    coh_2d = sheaf_2d.cohomology()
    assert coh_2d.dim_h0 == 1
    assert coh_2d.dim_h1 == 0
    assert abs(coh_2d.boundary_defect_frobenius) < 1e-12

    twisted = CellularSheaf(
        {"v0": 1, "v1": 1, "v2": 1},
        [
            SheafEdge("e01", "v0", "v1", 1, [[1.0]], [[1.0]]),
            SheafEdge("e12", "v1", "v2", 1, [[1.0]], [[1.0]]),
            SheafEdge("e02", "v0", "v2", 1, [[-1.0]], [[1.0]]),
        ],
    )
    t_coh = twisted.cohomology()
    assert t_coh.dim_h0 == 0
    assert t_coh.dim_h1 == 0

    assign_hallucinated = {"v0": [5.0], "v1": [5.0], "v2": [5.0]}
    internal, cone_e, obs = grounded_mapping_cone_energy(sheaf_1d, assign_hallucinated, {"v0": [1.0]})
    assert internal == pytest.approx(0.0)
    assert cone_e == pytest.approx(16.0)
    assert obs >= 1

    assert diagnose_contradictions(sheaf_1d, assign_hallucinated).regime == "CONSISTENT"
    fault_2v1 = diagnose_contradictions(sheaf_1d, {"v0": [1.0], "v1": [1.0], "v2": [9.0]})
    assert fault_2v1.regime == "RECOVERABLE_FAULT"
    assert fault_2v1.faulty_vertices == ("v2",)

    dyad = CellularSheaf({"a": 1, "b": 1}, [SheafEdge("eab", "a", "b", 1, [[1.0]], [[1.0]])])
    tie = diagnose_contradictions(dyad, {"a": [1.0], "b": [-1.0]})
    assert tie.regime == "ABSTAIN_TIE"

    chain_v = {f"u{i}": 1 for i in range(8)}
    chain_e = [SheafEdge(f"e{i}", f"u{i}", f"u{i+1}", 1, [[1.0]], [[1.0]]) for i in range(7)]
    chain_sheaf = CellularSheaf(chain_v, chain_e)
    parts = {
        "c0": ["u0", "u1", "u2"],
        "c1": ["u2", "u3", "u4"],
        "c2": ["u4", "u5", "u6"],
        "c3": ["u6", "u7"],
    }
    inc = IncrementalPartitionedSheaf(chain_sheaf, parts)
    assert inc.update_vertex("u0", [3.0], mode="lazy") == 0
    assert inc.flush() == 1
    recomputed = inc.update_vertex("u7", [-2.0], mode="eager")
    assert recomputed == 1
    full_assign = {f"u{i}": [0.0] for i in range(8)}
    full_assign["u0"] = [3.0]
    full_assign["u7"] = [-2.0]
    assert inc.total_energy() == pytest.approx(chain_sheaf.section_energy(full_assign))


def test_optional_cellular_sheaf_in_h4_gate(tmp_path: Path):
    def mutate_ok(parts):
        parts["hypergraph"]["cellular_sheaf"] = {
            "anchors": {"v0": [2.0]},
            "assignment": {"v0": [2.0], "v1": [2.0]},
            "edges": [{"edge_dim": 1, "id": "se0", "source": "v0", "source_map": [[1.0]], "target": "v1", "target_map": [[1.0]]}],
            "max_dim_h1": 0,
            "vertex_dims": {"v0": 1, "v1": 1},
        }

    store_ok = build(tmp_path / "cs_ok", mutate=mutate_ok)
    rows_ok = latest(store_ok)
    assert rows_ok["H4"]["status"] == "pass"
    assert criterion(rows_ok["H4"], "sheaf.mapping_cone")["passed"] is True

    crit_bad = cellular_sheaf_criteria({
        "anchors": {"v0": [0.0]},
        "assignment": {"v0": [4.0], "v1": [4.0]},
        "edges": [{"edge_dim": 1, "id": "se0", "source": "v0", "source_map": [[1.0]], "target": "v1", "target_map": [[1.0]]}],
        "max_dim_h1": 0,
        "vertex_dims": {"v0": 1, "v1": 1},
    })
    assert any(not c.passed for c in crit_bad)
    assert cellular_sheaf_criteria("bad")[0].passed is False


def test_savi_eprocess_ebh_empirical_bayes_and_ltt():
    eproc_win = kelly_e_process([0.045, 0.048, 0.046, 0.049, 0.047, 0.048, 0.049, 0.048, 0.050], scale_bound=0.05, alpha=0.05)
    assert eproc_win.rejected_null is True
    assert eproc_win.stopped_at is not None

    eproc_loss = kelly_e_process(
        [-0.045, -0.048, -0.046, -0.049, -0.047, -0.048, -0.049, -0.050],
        scale_bound=0.05,
        alpha=0.05,
        futility_threshold=5.0,
    )
    assert eproc_loss.rejected_null is False
    assert eproc_loss.futile is True
    assert eproc_loss.max_futility_e >= 5.0

    merged = merge_dependent_e_values([0.5, 25.0, 4.5])
    assert merged == pytest.approx(10.0)
    disc = ebh_discoveries([0.2, 65.0, 45.0, 0.8], alpha=0.05)
    assert disc == [1, 2]
    assert ebh_discoveries([], alpha=0.05) == []

    treat = [1.184, 1.191, 1.179]
    ctrl = [1.192, 1.199, 1.188]
    history = [
        [-0.007, -0.008, -0.009],
        [-0.005, -0.006, -0.004],
        [-0.008, -0.007, -0.008],
        [-0.006, -0.005, -0.007],
    ]
    eb = empirical_bayes_paired_t(treat, ctrl, historical_pair_diffs=history)
    assert eb.pearson_r > 0.98
    assert eb.eb_df == pytest.approx(10.0)
    assert eb.eb_p_one_sided < eb.welch_p_one_sided
    assert eb.eb_p_one_sided < 0.001

    eb_direct = empirical_bayes_paired_t(treat, ctrl, prior_df=8.0, prior_pair_var=1e-6)
    assert eb_direct.eb_df == pytest.approx(10.0)

    p_hb = hoeffding_bentkus_p([0.0] * 80, risk_bound=0.10)
    assert p_hb < 0.01
    assert hoeffding_bentkus_p([0.5] * 20, risk_bound=0.10) == 1.0
    cal = ltt_calibrate_thresholds(
        [(0.9, [0.0] * 80), (0.7, [0.01] * 80), (0.3, [0.25] * 80)],
        risk_bound=0.10,
        delta=0.05,
    )
    assert cal.certified_thresholds == (0.9, 0.7)
    assert cal.selected_threshold == 0.7


def test_attestation_strict_json_dsse_and_merkle_receipts(tmp_path: Path):
    with pytest.raises(SheafError, match="duplicate"):
        validate_strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(SheafError, match="safe range"):
        validate_strict_json_bytes(b'{"big":99999999999999999999}')
    with pytest.raises(SheafError, match="non-finite"):
        validate_strict_json_bytes(b'{"x":NaN}')
    with pytest.raises(SheafError, match="depth"):
        validate_strict_json_bytes(b"[" * 135 + b"1" + b"]" * 135)

    pae = dsse_pae("application/vnd.in-toto+json", b"{}")
    assert pae == b"DSSEv1 28 application/vnd.in-toto+json 2 {}"

    stmt = intoto_statement_v1(
        subjects=[{"name": "artifact", "digest": {"sha256": "a" * 64}}],
        predicate={"verdict": "pass"},
    )
    env = dsse_sign(stmt, key=TEST_KEY)
    verified = dsse_verify(env, key=TEST_KEY)
    assert verified == stmt
    with pytest.raises(SheafError):
        dsse_verify(env, key=b"wrong-key-material-0000")

    assert merkle_tree_head([]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    leaves = [f"leaf-{i}".encode("ascii") for i in range(7)]
    root = merkle_tree_head(leaves)
    for idx, leaf in enumerate(leaves):
        path = merkle_audit_path(idx, leaves)
        assert verify_merkle_inclusion(
            leaf_bytes=leaf,
            index=idx,
            tree_size=len(leaves),
            audit_path=path,
            expected_root_hex=root,
        )
        assert not verify_merkle_inclusion(
            leaf_bytes=b"forged",
            index=idx,
            tree_size=len(leaves),
            audit_path=path,
            expected_root_hex=root,
        )
    cproof = merkle_consistency_proof(3, leaves)
    assert len(cproof) >= 1

    store = build(tmp_path / "merkle_store")
    records = Ledger(store.root).records()
    rec = build_ledger_receipt(records, 5, key=TEST_KEY)
    ok_flag, problems = verify_ledger_receipt(rec, key=TEST_KEY)
    assert ok_flag is True and problems == []
    rec["record"]["status"] = "fail"
    ok_bad, probs_bad = verify_ledger_receipt(rec, key=TEST_KEY)
    assert ok_bad is False and probs_bad


def test_anticheat_kernel_auditor_and_probes():
    clean_kernel = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask)
"""
    rep_ok = audit_kernel_source(clean_kernel, require_triton_or_custom=True)
    assert rep_ok.ok is True
    assert rep_ok.has_triton_jit is True

    syntax_bad = audit_kernel_source("def broken(:\n    pass")
    assert syntax_bad.ok is False

    hacked_kernel = """
import subprocess
import sys
from inspect import stack
import torch.nn.functional as F
from torch.nn import Linear

def hacked(x, w):
    caller = sys._getframe(1).f_locals
    torch.cuda.Event.elapsed_time = lambda self, other: 0.001
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    eval("1+1")
    return F.linear(x, w)
"""
    rep_bad = audit_kernel_source(hacked_kernel, require_triton_or_custom=True)
    assert rep_bad.ok is False
    assert len(rep_bad.violations) >= 6

    probe_ok = four_distribution_probe_criteria({
        "distributions": {
            "D1_baseline": {"all_finite": True, "rel_err": 1e-4},
            "D2_overflow_scale": {"all_finite": True, "rel_err": 2e-4},
            "D3_underflow_eps": {"all_finite": True, "rel_err": 1e-4},
            "D4_sign_symmetry": {"all_finite": True, "rel_err": 1e-4},
        },
        "max_rel_err": 1e-3,
        "memory_ratio_vs_baseline": 1.05,
        "prompt_shapes_stripped": True,
        "tf32_baseline_enforced": True,
    })
    assert all(c.passed for c in probe_ok)
    probe_bad = four_distribution_probe_criteria({
        "distributions": {"D1_baseline": {"all_finite": False, "rel_err": 0.5}},
        "memory_ratio_vs_baseline": 3.5,
        "prompt_shapes_stripped": False,
        "tf32_baseline_enforced": False,
    })
    assert any(not c.passed for c in probe_bad)


def test_mutation_suite_achieves_100_percent_kill_rate(tmp_path: Path):
    report = run_mutation_suite(tmp_path / "mutants")
    assert report.generated == 12
    assert report.killed == 12
    assert report.survived == 0
    assert report.mutation_score == 1.0
