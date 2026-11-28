"""Unit, invariant, boundary, and sm_120 CUDA CPU-parity tests for M3.1 C0 Universal Kernel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hadukenn_core.kernel_state import (
    DIMENSIONLESS,
    GOLDILOCKS_P,
    HASH_RATE,
    MAX_TICK_ABS,
    PLANCK_TIME_S,
    SECOND,
    WATT,
    KernelState,
    QFraction,
    SIUnit,
    build_default_streams,
    compose,
    detect_execution_environment,
)
from hadukenn_core.rtx5070_frontier import (
    C0_RTX5070_CUDA_SOURCE,
    CUDA_RTX5070_KERNEL_SOURCE,
    RTX_5070_SPECS,
    NF4BlockQuantizer,
    RTX5070HardwareSpec,
    c0_rtx5070_warp_julia_and_tuple_cpu,
    compile_c0_rtx5070_cuda_kernel,
    export_cuda_kernel_source,
    plan_rtx5070_frontier_vram,
    run_c0_rtx5070_cuda_if_available,
    verify_c0_rtx5070_cpu_parity,
)
from hadukenn_core.universal_kernel import (
    CORE_OPERATORS_01_20,
    op_01,
    op_02,
    op_03,
    op_04,
    op_05,
    op_06,
    op_07,
    op_08,
    op_09,
    op_10,
    op_11,
    op_12,
    op_13,
    op_14,
    op_15,
    op_16,
    op_17,
    op_18,
    op_19,
    op_20,
)
from sheaf_verify import formal_verifiers as fv
from sheaf_verify.hashing import SheafError


def _make_state(
    n_steps: int = 128,
    m_streams: int = 4,
    *,
    check_properties: bool = True,
    pareto_scales: list[int] | None = None,
    seed: int = 20261108,
) -> KernelState:
    rng = np.random.default_rng(seed)
    t_base = np.linspace(1.0, float(n_steps), n_steps)
    s0 = np.sin(0.12 * t_base) + 0.45 * np.cos(0.31 * t_base) + rng.normal(0.0, 0.12, n_steps)
    s1 = np.cos(0.10 * t_base - 0.2) + 0.55 * np.sin(0.24 * t_base) + rng.normal(0.0, 0.14, n_steps)
    s2 = 0.65 * s1 * np.sin(0.18 * t_base) + rng.normal(0.0, 0.11, n_steps)
    s3 = 0.80 * s2 + 0.50 * (s0 * s1) + rng.normal(0.0, 0.11, n_steps)
    s4 = 0.45 * np.sin(0.22 * t_base) + 0.35 * s0 + rng.normal(0.0, 0.10, n_steps)

    streams_all = [
        ("m0_ai_power", {"t": t_base, "values": s0, "time_unit": SECOND, "val_unit": WATT}),
        ("m1_grid_freq", {"t": (t_base - 0.05) / 1.0001, "values": s1, "time_unit": SECOND, "val_unit": HASH_RATE}),
        ("m2_mempool_fee", {"t": (t_base + 0.12) / 0.9999, "values": s2, "time_unit": SECOND, "val_unit": DIMENSIONLESS}),
        ("m3_vol_skew", {"t": t_base, "values": s3, "time_unit": SECOND, "val_unit": DIMENSIONLESS}),
        ("m4_thermal_headroom", {"t": t_base, "values": s4, "time_unit": SECOND, "val_unit": DIMENSIONLESS}),
    ]
    raw_streams = dict(streams_all[:m_streams])
    clock_relations: dict[str, Any] = {}
    if m_streams >= 2:
        clock_relations["m1_grid_freq"] = {"offset_s": 0.05, "rate": 1.0001, "uncertainty_s": 1e-4}
    if m_streams >= 3:
        clock_relations["m2_mempool_fee"] = {"offset_s": -0.12, "rate": 0.9999, "uncertainty_s": 1e-4}
    obj_series = 0.55 * np.hypot(s1, s3) + 0.45 * np.abs(s0 * s2)
    need = {
        "intent": "local_to_global_regime_and_slice_mining",
        "primary_stream": "m1_grid_freq" if m_streams >= 2 else "m0_ai_power",
        "objective_series": obj_series,
    }
    mode = {"check_properties": check_properties, "pareto_scales": pareto_scales or [256, 512]}
    return KernelState(raw_streams, clock_relations, need, mode=mode)


def _independent_f32_warp_ref(c_h_in: np.ndarray, gamma_sheaf: float) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c_h_in, dtype=np.float32)
    n = int(c.shape[0])
    gamma = np.float32(gamma_sheaf)

    def qmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return np.stack(
            [
                pw * qw - px * qx - py * qy - pz * qz,
                pw * qx + px * qw + py * qz - pz * qy,
                pw * qy - px * qz + py * qw + pz * qx,
                pw * qz + px * qy - py * qx + pz * qw,
            ],
            axis=-1,
        ).astype(np.float32)

    q = c.copy()
    lanes = np.arange(n, dtype=np.int32) & 31
    for offset in (1, 2, 4, 8, 16):
        prev = np.zeros_like(q)
        if offset < n:
            prev[offset:] = q[:-offset]
        q = np.where((lanes >= offset)[:, None], qmul(prev, q), q)

    for _ in range(4):
        sq = qmul(q, q)
        lx = np.float32(2.0) * (q[..., 2] * c[..., 3] - q[..., 3] * c[..., 2])
        ly = np.float32(2.0) * (q[..., 3] * c[..., 1] - q[..., 1] * c[..., 3])
        lz = np.float32(2.0) * (q[..., 1] * c[..., 2] - q[..., 2] * c[..., 1])
        nw = sq[..., 0] + c[..., 0]
        nx = sq[..., 1] + c[..., 1] + gamma * lx
        ny = sq[..., 2] + c[..., 2] + gamma * ly
        nz = sq[..., 3] + c[..., 3] + gamma * lz
        msq = np.float32(0.25) * (nw * nw + nx * nx + ny * ny + nz * nz)
        gain = (np.float32(3.0) + msq) / (np.float32(1.0) + np.float32(3.0) * msq)
        q = np.stack([nw * gain, nx * gain, ny * gain, nz * gain], axis=-1).astype(np.float32)

    p5 = np.sqrt(q[..., 2] ** 2 + q[..., 3] ** 2).astype(np.float32)
    return q, p5


def test_kernel_state_and_reexported_formal_primitives() -> None:
    assert QFraction is fv.QFraction
    assert SIUnit is fv.SIUnit
    assert SECOND is fv.SECOND
    assert DIMENSIONLESS is fv.DIMENSIONLESS
    assert WATT is fv.WATT
    assert HASH_RATE is fv.HASH_RATE
    assert PLANCK_TIME_S == fv.PLANCK_TIME_S
    assert MAX_TICK_ABS == fv.MAX_TICK_ABS
    assert GOLDILOCKS_P == fv.GOLDILOCKS_P
    assert (WATT * SECOND).dims == (2, 1, -2, 0, 0, 0, 0)

    env = detect_execution_environment()
    for key in ("os", "arch", "python", "numpy", "cuda_device_detected", "tenstorrent_kmd_detected"):
        assert key in env

    raw_a, clk_a, need_a = build_default_streams(128)
    raw_b, clk_b, need_b = build_default_streams(128)
    assert np.array_equal(raw_a["m0_ai_power"]["values"], raw_b["m0_ai_power"]["values"])
    assert clk_a == clk_b
    assert np.array_equal(need_a["objective_series"], need_b["objective_series"])

    s_strict = KernelState(raw_a, clk_a, need_a, mode={"check_properties": True})
    assert s_strict.checks == []
    assert s_strict.value_graph_policy_alignment == {}
    with pytest.raises(AssertionError):
        s_strict.record_check("demo.fail", "proven", False, "forced failure")
    # Empirical (wall-clock) checks are recorded but never fatal, even in strict mode.
    s_strict.record_check("demo.timing", "empirical", False, "slow shared host")
    assert s_strict.checks[-1] == {
        "invariant": "demo.timing",
        "tag": "empirical",
        "passed": False,
        "detail": "slow shared host",
    }

    s_soft = KernelState(raw_a, clk_a, need_a, mode={"check_properties": False})
    s_soft.record_check("demo.fail", "proven", False, "soft failure")
    assert len(s_soft.checks) == 1
    assert s_soft.checks[0]["passed"] is False


def test_op01_to_op07_sync_carriers_ticks_sheaf_dirichlet_and_haar() -> None:
    s = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_07)(_make_state(n_steps=128, m_streams=4))

    t0 = s.synchronized_streams["m0_ai_power"]["t"]
    t1 = s.synchronized_streams["m1_grid_freq"]["t"]
    t2 = s.synchronized_streams["m2_mempool_fee"]["t"]
    assert t0.flags.writeable is False
    assert s.synchronized_streams["m0_ai_power"]["x"].flags.writeable is False
    assert float(np.max(np.abs(t1 - t0))) < 1e-9
    assert float(np.max(np.abs(t2 - t0))) < 1e-9

    assert s.endogenous_params["tau_star"] >= 1
    assert s.endogenous_params["m_star"] >= 2
    assert s.endogenous_params["sigma_hat_mad"] > 0.0
    assert s.endogenous_params["W_star"] >= 8

    assert s.carriers["num_carrier_sheets"] >= 1
    assert s.carriers["sample_branches"]
    assert all(bool(b["strictly_monotone"]) for b in s.carriers["sample_branches"])

    assert s.interval_stats["num_crossing_events"] >= 2
    assert s.network_stats["rn_fiedler_lambda2"] > 0.0
    assert s.sheaf_projection["dirichlet_energy_drop_pct"] > 50.0
    assert s.dynamics_and_wavelets["linegraph_degree_sum"] > 0

    checks_by_name = {c["invariant"]: c for c in s.checks}
    assert checks_by_name["op_06.sheaf_dirichlet_energy_identity"]["passed"] is True
    assert checks_by_name["op_07.linegraph_identity_and_haar_quadratic_scaling"]["passed"] is True


@pytest.mark.parametrize(
    "bad_t,bad_vals,time_unit,clk,raises_always",
    [
        ([0.0], [1.0], SECOND, {}, True),
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], WATT, {}, True),
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], SECOND, {"s": {"offset_s": 0.0, "rate": -1.0}}, False),
        ([0.0, 1e-45, 2e-45], [1.0, 2.0, 3.0], SECOND, {}, False),
        ([0.0, float("nan"), 2.0], [1.0, 2.0, 3.0], SECOND, {}, False),
        ([0.0, 1.0, 2.0], [1.0, float("nan"), 3.0], SECOND, {}, False),
    ],
)
def test_op01_rejects_invalid_streams(
    bad_t: list[float],
    bad_vals: list[float],
    time_unit: SIUnit,
    clk: dict[str, Any],
    raises_always: bool,
) -> None:
    s_strict = KernelState(
        {"s": {"t": bad_t, "values": bad_vals, "time_unit": time_unit, "val_unit": DIMENSIONLESS}},
        clk,
        {"primary_stream": "s"},
        mode={"check_properties": True},
    )
    with pytest.raises((AssertionError, ValueError, SheafError)):
        op_01(s_strict)

    s_soft = KernelState(
        {"s": {"t": bad_t, "values": bad_vals, "time_unit": time_unit, "val_unit": DIMENSIONLESS}},
        clk,
        {"primary_stream": "s"},
        mode={"check_properties": False},
    )
    if raises_always:
        with pytest.raises((ValueError, SheafError)):
            op_01(s_soft)
    else:
        op_01(s_soft)
        assert s_soft.checks and s_soft.checks[-1]["passed"] is False


@pytest.mark.parametrize("kind", ["ones", "linear", "flat_op03", "sawtooth_op03"])
def test_op02_and_op03_reject_degenerate_signals(kind: str) -> None:
    t_arr = np.linspace(1.0, 64.0, 64)
    if kind in ("ones", "linear"):
        vals = np.ones(64) if kind == "ones" else np.linspace(0.0, 10.0, 64)
        for check_props in (True, False):
            st = KernelState(
                {"s": {"t": t_arr, "values": vals, "time_unit": SECOND, "val_unit": DIMENSIONLESS}},
                {},
                {"primary_stream": "s"},
                mode={"check_properties": check_props},
            )
            op_01(st)
            if check_props:
                with pytest.raises(AssertionError):
                    op_02(st)
            else:
                op_02(st)
                assert st.endogenous_params["sigma_hat_mad"] == 0.0
                assert st.checks[-1]["passed"] is False
    else:
        vals = np.zeros(64) if kind == "flat_op03" else np.array([(-1.0) ** i for i in range(64)], dtype=float)
        flat_tol = 1e-3 if kind == "flat_op03" else 0.05
        for check_props in (True, False):
            st = KernelState(
                {"s": {"t": t_arr, "values": vals, "time_unit": SECOND, "val_unit": DIMENSIONLESS}},
                {},
                {"primary_stream": "s"},
                mode={"check_properties": check_props},
            )
            op_01(st)
            st.endogenous_params = {"flat_tol_star": flat_tol}
            if check_props:
                with pytest.raises(AssertionError):
                    op_03(st)
            else:
                op_03(st)
                assert st.carriers["num_carrier_sheets"] == 0
                assert st.checks[-1]["passed"] is False


def test_op08_and_op09_rotors_determinism_and_m_lt_3_guard() -> None:
    pipe = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_07, op_08, op_09)
    s_a = pipe(_make_state(n_steps=128, m_streams=4, seed=20261108))
    s_b = pipe(_make_state(n_steps=128, m_streams=4, seed=20261108))

    assert s_a.spectral_circuit["latest_12neuron_witness"] == s_b.spectral_circuit["latest_12neuron_witness"]
    assert s_a.manifold_and_rotors["s3_scan_norm_error"] < 1e-12
    assert s_a.manifold_and_rotors["adaptive_so4_eigenrotor_rank_ic"] >= s_a.manifold_and_rotors["default_p0_rank_ic"]
    assert s_a.manifold_and_rotors["mean_octonion_associator"] > 0.0
    assert s_a.spectral_circuit["dipole_forward_invariance_error"] < 1e-12
    assert s_a.spectral_circuit["f12_causal_concentration_pct"] > 90.0

    s_m2 = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_07)(_make_state(n_steps=96, m_streams=2, check_properties=False))
    with pytest.raises((ValueError, SheafError)):
        op_08(s_m2)


def test_op10_to_op15_receipt_provenance_pareto_hodge_simt() -> None:
    pipe = compose(
        op_01,
        op_02,
        op_03,
        op_04,
        op_05,
        op_06,
        op_07,
        op_08,
        op_09,
        op_10,
        op_11,
        op_12,
        op_13,
        op_14,
        op_15,
    )
    s_a = pipe(_make_state(n_steps=96, m_streams=4, pareto_scales=[256, 512], seed=101))
    s_b = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_07, op_08, op_09, op_10, op_11)(
        _make_state(n_steps=96, m_streams=4, seed=202)
    )

    cm_a = s_a.crypto_mining_telemetry
    cm_b = s_b.crypto_mining_telemetry
    assert cm_a["invariant_schedule_words_in_ker_deltaW"] == 17
    assert cm_a["dead_terminal_register_slices_r58_r63"] == "1/48"
    assert cm_a["btc_sha256d_effective_rounds"] < 52.0
    assert cm_a["master_kernel_sha256_receipt"] == fv.canonical_receipt_sha256(cm_a["receipt_body"])
    assert cm_a["master_kernel_sha256_receipt"] != cm_b["master_kernel_sha256_receipt"]
    assert cm_a["telemetry_provenance"]["master_kernel_sha256_receipt"] == "live_computed"
    assert cm_a["telemetry_provenance"]["blackhole_p150a_sheaf_pouw_mh_s"] == "analytical_model"

    rtx_a = cm_a["rtx_5070_blackwell_silicon_map"]
    rtx_b = cm_b["rtx_5070_blackwell_silicon_map"]
    assert rtx_a["tuple_operator_spectral_norm"] > 0.0
    assert rtx_a["tuple_operator_spectral_norm"] != rtx_b["tuple_operator_spectral_norm"]
    assert rtx_a["rtx5070_slice_optimized_sha256d_gh_s"] > rtx_a["rtx5070_raw_sha256d_gh_s"]
    assert rtx_a["telemetry_provenance"]["l2_cache_footprint_kb"] == "live_computed"
    assert rtx_a["telemetry_provenance"]["rtx5070_active_power_w"] == "analytical_model"

    checks_by_name = {c["invariant"]: c for c in s_a.checks}
    assert checks_by_name["op_12.tig_opow_pareto_dominance"]["passed"] is True
    assert s_a.relational_decompositions["D1_2simplicial_1hodge"]["orthogonality_error"] < 1e-10
    assert s_a.simt_gpu_auditor["unpadded_tile_32x32_bank_conflicts"] == 31
    assert s_a.simt_gpu_auditor["c0_padded_tile_32x33_bank_conflicts"] == 0

    tig = s_a.tig_8challenge_portfolio
    assert tig["telemetry_provenance"]["c001_satisfiability_gain_pct"] == "live"
    for cid in ("c002", "c003", "c004", "c005", "c006", "c007", "c008"):
        key = next(k for k in tig["telemetry_provenance"] if cid in k)
        assert tig["telemetry_provenance"][key] == "analytical_reference"


def test_op16_to_op19_r1cs_cayley_betti_and_morphisms() -> None:
    pipe = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_07, op_08, op_09, op_16, op_17, op_18, op_19)
    s = pipe(_make_state(n_steps=96, m_streams=4))

    zk = s.zk_goldilocks_circuit
    assert zk["prime_field_modulus"] == str(GOLDILOCKS_P)
    assert zk["r1cs_gate_residual_mod_p"] == 0
    assert zk["dipole_sum_mod_p"] == 0
    assert "y_float" in zk
    r1cs_check = fv.verify_goldilocks_r1cs_dipole(zk["y_float"], s.spectral_circuit["latest_12neuron_witness"])
    assert r1cs_check["ok"] is True

    ck = s.cayley_koopman_propagator
    assert ck["generator_skew_symmetry_error"] < 1e-12
    for key in ("dt_1", "dt_5", "dt_25"):
        assert ck["operator_l2_norms_by_horizon"][key] <= 1.000001

    ph = s.persistent_homology_radar
    barcode = ph["filtration_betti_barcode"]
    assert len(barcode) == 6
    assert barcode[0]["beta_0_components"] >= barcode[-1]["beta_0_components"]
    assert ph["wasserstein_1_topological_drift"] >= 0.0

    sm = s.symbolic_morphism_compiler
    assert sm["total_depth3_morphisms_evaluated"] == 120
    assert sm["synthesized_morphism_rank_ic"] > sm["best_single_operator_ic"]
    assert sm["synthesized_morphism_rank_ic"] > 0.40

    s_m2 = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_19)(
        _make_state(n_steps=96, m_streams=2, check_properties=False)
    )
    assert math.isfinite(s_m2.symbolic_morphism_compiler["best_single_operator_ic"])
    assert math.isfinite(s_m2.symbolic_morphism_compiler["synthesized_morphism_rank_ic"])

    s_m3 = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_19)(_make_state(n_steps=96, m_streams=3))
    assert s_m3.symbolic_morphism_compiler["synthesized_morphism_rank_ic"] > 0.0


@pytest.mark.parametrize("m_streams", [3, 4, 5])
def test_op20_m_agnostic_and_sparse_guard(m_streams: int) -> None:
    pipe = compose(op_01, op_02, op_03, op_04, op_05, op_06, op_20)
    s = pipe(_make_state(n_steps=96, m_streams=m_streams))
    cl = s.closed_loop_value_morphism
    weights = cl["solved_network_graph_parameters_Theta"]["anisotropic_channel_weights"]
    assert len(weights) == m_streams
    assert math.isclose(sum(float(v) for v in weights.values()), 1.0, abs_tol=1e-2)
    assert cl["total_closed_loop_evaluations"] == 35
    obj = cl["user_value_objective_sought"]
    assert cl["closed_loop_optimal_scores"][obj] > cl["open_loop_baseline_scores"][obj]

    s_sparse_4 = _make_state(n_steps=4, m_streams=4, check_properties=False)
    op_01(s_sparse_4)
    s_sparse_4.endogenous_params = {"W_star": 2}
    op_06(s_sparse_4)
    with pytest.raises(ValueError, match="sparse"):
        op_20(s_sparse_4)

    s_sparse_8 = _make_state(n_steps=8, m_streams=4, check_properties=False)
    op_01(s_sparse_8)
    s_sparse_8.endogenous_params = {"W_star": 4}
    op_06(s_sparse_8)
    op_20(s_sparse_8)
    assert len(s_sparse_8.closed_loop_value_morphism["solved_network_graph_parameters_Theta"]["anisotropic_channel_weights"]) == 4


def test_core_operators_01_20_pipeline() -> None:
    s = compose(*CORE_OPERATORS_01_20)(_make_state(n_steps=96, m_streams=4, pareto_scales=[512]))
    assert len(s.checks) >= 20
    assert all(bool(c["passed"]) for c in s.checks)


def test_rtx5070_cu_file_and_source_export(tmp_path: Path) -> None:
    cu_path = Path(__file__).resolve().parents[1] / "src" / "hadukenn_core" / "csrc" / "c0_rtx5070_sheaf_tuple.cu"
    assert cu_path.is_file()
    disk_src = cu_path.read_text(encoding="utf-8")
    assert disk_src == CUDA_RTX5070_KERNEL_SOURCE
    assert disk_src == C0_RTX5070_CUDA_SOURCE
    for token in (
        "pade11_inv_rms",
        "qmul_reg",
        "__shfl_up_sync",
        "c0_rtx5070_warp_julia_and_tuple_kernel",
        "2.0f * (qy * c.w - qz * c.z)",
        "sm_120",
        "Blackwell",
    ):
        assert token in disk_src

    exported_path = tmp_path / "c0_rtx5070_sheaf_tuple.cu"
    ret = export_cuda_kernel_source(exported_path)
    assert ret == disk_src
    assert exported_path.read_text(encoding="utf-8") == disk_src

    assert RTX_5070_SPECS["l2_cache_total_kb"] == 49152
    assert len(plan_rtx5070_frontier_vram()) == 6
    assert RTX5070HardwareSpec().l2_cache_mib == 48.0
    assert NF4BlockQuantizer().block_size == 64


@pytest.mark.parametrize("n_items", [1, 16, 32, 48, 64])
def test_sm120_cpu_warp_parity_and_boundary_reset(n_items: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")

    rng = np.random.default_rng(1337 + n_items)
    c_in = (0.15 * rng.standard_normal((n_items, 4))).astype(np.float32)
    c_in[:, 0] = np.float32(0.85)

    q_out, p5_out = c0_rtx5070_warp_julia_and_tuple_cpu(c_in, 0.25)
    ref_q, ref_p5 = _independent_f32_warp_ref(c_in, 0.25)

    assert q_out.shape == (n_items, 4)
    assert p5_out.shape == (n_items,)
    assert q_out.dtype == np.float32
    assert p5_out.dtype == np.float32
    assert np.all(np.isfinite(q_out)) and np.all(np.isfinite(p5_out))
    assert float(np.max(np.abs(q_out - ref_q))) <= 1e-5
    assert float(np.max(np.abs(p5_out - ref_p5))) <= 1e-5

    parity = verify_c0_rtx5070_cpu_parity(c_in, 0.25)
    assert parity["ok"] is True
    assert parity["passed"] is True
    assert parity["max_q_err"] <= 1e-5

    if n_items > 32:
        q_tail, _ = c0_rtx5070_warp_julia_and_tuple_cpu(c_in[32:], 0.25)
        assert float(np.max(np.abs(q_out[32:] - q_tail))) <= 1e-5

        # Unreset full-array scan diverges from 32-lane warp-reset scan on warp 1.
        q_unreset = c_in.copy()
        for offset in (1, 2, 4, 8, 16):
            prev = np.zeros_like(q_unreset)
            prev[offset:] = q_unreset[:-offset]
            pw, px, py, pz = prev[..., 0], prev[..., 1], prev[..., 2], prev[..., 3]
            qw, qx, qy, qz = q_unreset[..., 0], q_unreset[..., 1], q_unreset[..., 2], q_unreset[..., 3]
            q_mul = np.stack(
                [
                    pw * qw - px * qx - py * qy - pz * qz,
                    pw * qx + px * qw + py * qz - pz * qy,
                    pw * qy - px * qz + py * qw + pz * qx,
                    pw * qz + px * qy - py * qx + pz * qw,
                ],
                axis=-1,
            ).astype(np.float32)
            q_unreset = np.where((np.arange(n_items) >= offset)[:, None], q_mul, q_unreset)
        assert float(np.max(np.abs(q_out[32:] - q_unreset[32:]))) > 1e-3


def test_sm120_cpu_parity_edge_cases_and_optional_nvcc_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    q_empty, p5_empty = c0_rtx5070_warp_julia_and_tuple_cpu(np.zeros((0, 4), dtype=np.float32), 0.25)
    assert q_empty.shape == (0, 4) and q_empty.dtype == np.float32
    assert p5_empty.shape == (0,) and p5_empty.dtype == np.float32

    with pytest.raises(ValueError):
        c0_rtx5070_warp_julia_and_tuple_cpu(np.zeros((8, 3), dtype=np.float32), 0.25)

    bad_c = np.zeros((4, 4), dtype=np.float32)
    bad_c[0, 0] = np.float32("nan")
    with pytest.raises(ValueError):
        c0_rtx5070_warp_julia_and_tuple_cpu(bad_c, 0.25)

    with pytest.raises(ValueError):
        c0_rtx5070_warp_julia_and_tuple_cpu(np.ones((4, 4), dtype=np.float32), float("nan"))

    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.delenv("NVCC", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    comp = compile_c0_rtx5070_cuda_kernel()
    assert comp["available"] is False
    assert comp["compiled"] is False

    c_in = np.tile(np.array([[0.85, 0.05, -0.04, 0.03]], dtype=np.float32), (8, 1))
    q_res, p5_res, backend = run_c0_rtx5070_cuda_if_available(c_in, 0.25)
    assert backend == "cpu_parity"
    assert q_res.shape == (8, 4) and p5_res.shape == (8,)
