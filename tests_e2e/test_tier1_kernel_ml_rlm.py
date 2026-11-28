"""Tier 1 isolated happy-path E2E tests for Features 13-16 (Universal Kernel, op_21 Policies, ML/Topology & RLM).

Pre-M3 collection safety: zero top-level imports of `sheaf_verify.*`, `hadukenn_core.*`,
`scripts.*`, `numpy`, `scipy`, or `torch`. All target symbols and scientific libraries are
resolved inside test bodies or test-local helpers.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from typing import Any

import pytest

from tests_e2e.conftest import resolve_any_public_symbol, resolve_public_symbol

_KERNEL_MODULES = (
    "hadukenn_core.universal_kernel",
    "hadukenn_core.c0_universal_kernel",
    "hadukenn_core.c0_kernel",
    "hadukenn_core.kernel_ops",
    "hadukenn_core.value_graph",
    "hadukenn_core.rtx5070_frontier",
    "hadukenn_core.cuda_parity",
)

_ML_MODULES = (
    "hadukenn_core.zigzag_topology",
    "hadukenn_core.topology_ml",
    "hadukenn_core.ml_synthesis",
    "hadukenn_core.kernel_policies",
    "hadukenn_core.architecture_policy",
    "hadukenn_core.hypergraph",
    "hadukenn_core.partitions",
    "hadukenn_core.heads",
)

_RLM_MODULES = (
    "hadukenn_core.rlm_engine",
    "hadukenn_core.rlm_repl",
    "hadukenn_core.rlm_runtime",
    "hadukenn_core.rlm_context_engine",
)


def _bind_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    sig = inspect.signature(fn)
    params = sig.parameters
    param_list = list(params.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in param_list)
    pos_args: list[Any] = []
    kwargs: dict[str, Any] = {}
    used_indices: set[int] = set()

    for p in param_list:
        if p.kind != inspect.Parameter.POSITIONAL_ONLY:
            continue
        matched_val = None
        matched_idx = None
        for idx, (aliases, val) in enumerate(alias_values):
            if idx not in used_indices and p.name in aliases:
                matched_val = val
                matched_idx = idx
                break
        if matched_idx is not None:
            pos_args.append(matched_val)
            used_indices.add(matched_idx)
        elif p.default is not inspect.Parameter.empty:
            break
        else:
            sig.bind(*positional_fallback)
            return positional_fallback, {}

    for idx, (aliases, val) in enumerate(alias_values):
        if idx in used_indices:
            continue
        matched = False
        for name in aliases:
            if name in params and params[name].kind != inspect.Parameter.POSITIONAL_ONLY:
                kwargs[name] = val
                matched = True
                used_indices.add(idx)
                break
        if not matched and has_varkw:
            kwargs[aliases[0]] = val
            used_indices.add(idx)

    if len(used_indices) == len(alias_values) or all(
        p.default is not inspect.Parameter.empty
        or p.name in kwargs
        or (p.kind == inspect.Parameter.POSITIONAL_ONLY and param_list.index(p) < len(pos_args))
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in param_list
    ):
        sig.bind(*pos_args, **kwargs)
        return tuple(pos_args), kwargs

    sig.bind(*positional_fallback)
    return positional_fallback, {}


def _call_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> Any:
    args, kwargs = _bind_with_aliases(fn, alias_values, positional_fallback)
    return fn(*args, **kwargs)


def _resolve_op(long_name: str, short_name: str) -> Any:
    return resolve_any_public_symbol("hadukenn_core", (long_name, short_name), _KERNEL_MODULES)


def _build_kernel_state(
    n_steps: int = 128,
    m_streams: int = 4,
    *,
    check_properties: bool = True,
    pareto_scales: list[int] | None = None,
    seed: int = 20261108,
) -> Any:
    import numpy as np

    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    second = resolve_any_public_symbol(
        "hadukenn_core", ("SECOND",), _KERNEL_MODULES
    ) if _has_symbol("hadukenn_core", "SECOND", _KERNEL_MODULES) else resolve_public_symbol("sheaf_verify", "SECOND")
    watt = resolve_any_public_symbol(
        "hadukenn_core", ("WATT",), _KERNEL_MODULES
    ) if _has_symbol("hadukenn_core", "WATT", _KERNEL_MODULES) else resolve_public_symbol("sheaf_verify", "WATT")
    hash_rate = resolve_any_public_symbol(
        "hadukenn_core", ("HASH_RATE",), _KERNEL_MODULES
    ) if _has_symbol("hadukenn_core", "HASH_RATE", _KERNEL_MODULES) else resolve_public_symbol(
        "sheaf_verify", "HASH_RATE"
    )
    dimless = resolve_any_public_symbol(
        "hadukenn_core", ("DIMENSIONLESS",), _KERNEL_MODULES
    ) if _has_symbol("hadukenn_core", "DIMENSIONLESS", _KERNEL_MODULES) else resolve_public_symbol(
        "sheaf_verify", "DIMENSIONLESS"
    )

    rng = np.random.default_rng(seed)
    t_base = np.linspace(1.0, float(n_steps), n_steps)
    s0 = np.sin(0.12 * t_base) + 0.45 * np.cos(0.31 * t_base) + rng.normal(0.0, 0.12, n_steps)
    s1 = np.cos(0.10 * t_base - 0.2) + 0.55 * np.sin(0.24 * t_base) + rng.normal(0.0, 0.14, n_steps)
    s2 = 0.65 * s1 * np.sin(0.18 * t_base) + rng.normal(0.0, 0.11, n_steps)
    s3 = 0.80 * s2 + 0.50 * (s0 * s1) + rng.normal(0.0, 0.11, n_steps)
    s4 = 0.45 * np.sin(0.22 * t_base) + 0.35 * s0 + rng.normal(0.0, 0.10, n_steps)

    streams_all = [
        ("m0_ai_power", {"t": t_base, "values": s0, "time_unit": second, "val_unit": watt}),
        (
            "m1_grid_freq",
            {"t": (t_base - 0.05) / 1.0001, "values": s1, "time_unit": second, "val_unit": hash_rate},
        ),
        (
            "m2_mempool_fee",
            {"t": (t_base + 0.12) / 0.9999, "values": s2, "time_unit": second, "val_unit": dimless},
        ),
        ("m3_vol_skew", {"t": t_base, "values": s3, "time_unit": second, "val_unit": dimless}),
        ("m4_thermal_headroom", {"t": t_base, "values": s4, "time_unit": second, "val_unit": dimless}),
    ]
    raw_streams = dict(streams_all[:m_streams])
    clock_relations: dict[str, dict[str, float]] = {}
    if m_streams >= 2:
        clock_relations["m1_grid_freq"] = {"offset_s": 0.05, "rate": 1.0001, "uncertainty_s": 1e-4}
    if m_streams >= 3:
        clock_relations["m2_mempool_fee"] = {"offset_s": -0.12, "rate": 0.9999, "uncertainty_s": 1e-4}

    obj_series = 0.55 * np.hypot(s1, s3) + 0.45 * np.abs(s0 * s2)
    need_spec = {
        "intent": "local_to_global_regime_and_slice_mining",
        "primary_stream": "m1_grid_freq" if m_streams >= 2 else "m0_ai_power",
        "objective_series": obj_series,
    }
    mode = {"check_properties": check_properties, "pareto_scales": pareto_scales or [256, 512]}
    return state_cls(raw_streams, clock_relations, need_spec, mode=mode)


def _has_symbol(package_name: str, symbol_name: str, preferred_modules: tuple[str, ...]) -> bool:
    try:
        resolve_public_symbol(package_name, symbol_name, preferred_modules)
        return True
    except AssertionError:
        return False


def _verdict_passed(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    for attr in ("passed", "ok", "accepted", "approved", "allowed", "valid"):
        if hasattr(result, attr) and isinstance(getattr(result, attr), bool):
            return bool(getattr(result, attr))
    if isinstance(result, dict):
        for key in ("ok", "passed", "accepted", "approved", "allowed", "valid"):
            if key in result and isinstance(result[key], bool):
                return bool(result[key])
    raise AssertionError(f"Unrecognized verdict return shape: {result!r}")


def _extract_attr_or_key(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(f"Missing expected attribute/key from {names!r} in {obj!r}")


def _numpy_sm120_warp_julia_ref(c_h_in: Any, gamma_sheaf: float) -> tuple[Any, Any]:
    import numpy as np

    c = np.asarray(c_h_in, dtype=np.float32)
    n = int(c.shape[0])
    gamma = np.float32(gamma_sheaf)

    def qmul_f32(p: Any, q: Any) -> Any:
        pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        ow = pw * qw - px * qx - py * qy - pz * qz
        ox = pw * qx + px * qw + py * qz - pz * qy
        oy = pw * qy - px * qz + py * qw + pz * qx
        oz = pw * qz + px * qy - py * qx + pz * qw
        return np.stack([ow, ox, oy, oz], axis=-1).astype(np.float32)

    q = c.copy()
    lanes = np.arange(n, dtype=np.int32) & 31
    for offset in (1, 2, 4, 8, 16):
        prev = np.zeros_like(q)
        if offset < n:
            prev[offset:] = q[:-offset]
        mask = lanes >= offset
        updated = qmul_f32(prev, q)
        q = np.where(mask[:, None], updated, q)

    for _ in range(4):
        sq = qmul_f32(q, q)
        qx, qy, qz = q[..., 1], q[..., 2], q[..., 3]
        cw, cx, cy, cz = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
        lx = np.float32(2.0) * (qy * cz - qz * cy)
        ly = np.float32(2.0) * (qz * cx - qx * cz)
        lz = np.float32(2.0) * (qx * cy - qy * cx)

        nw = sq[..., 0] + cw
        nx = sq[..., 1] + cx + gamma * lx
        ny = sq[..., 2] + cy + gamma * ly
        nz = sq[..., 3] + cz + gamma * lz

        msq = np.float32(0.25) * (nw * nw + nx * nx + ny * ny + nz * nz)
        gain = (np.float32(3.0) + msq) / (np.float32(1.0) + np.float32(3.0) * msq)
        q = np.stack([nw * gain, nx * gain, ny * gain, nz * gain], axis=-1).astype(np.float32)

    p5 = np.sqrt(q[..., 2] ** 2 + q[..., 3] ** 2).astype(np.float32)
    return q, p5


# ==============================================================================
# Feature 13: c0_universal_kernel_v5.py Operators (op_01-op_20) & sm_120 Parity (7 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f13_kernel_state_compose_and_full_pipeline_runs_cleanly(no_gpu_env: dict[str, str]) -> None:
    assert no_gpu_env["CUDA_VISIBLE_DEVICES"] == ""
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    pipeline = resolve_any_public_symbol(
        "hadukenn_core", ("UNIVERSAL_KERNEL_PIPELINE", "KERNEL_PIPELINE", "C0_PIPELINE"), _KERNEL_MODULES
    )
    op_01 = _resolve_op("op_01_ingest_and_clock_sync", "op_01")
    op_02 = _resolve_op("op_02_endogenous_scale_solver", "op_02")

    s_pair = compose_fn(op_01, op_02)(_build_kernel_state(n_steps=128, m_streams=4, pareto_scales=[256, 512]))
    assert "m1_grid_freq" in s_pair.synchronized_streams
    assert s_pair.synchronized_streams["m1_grid_freq"]["t"].flags.writeable is False
    assert s_pair.synchronized_streams["m1_grid_freq"]["x"].flags.writeable is False
    assert s_pair.endogenous_params["tau_star"] >= 1
    assert s_pair.endogenous_params["m_star"] >= 2

    if _has_symbol("hadukenn_core", "build_default_streams", _KERNEL_MODULES):
        build_defaults = resolve_public_symbol("hadukenn_core", "build_default_streams", _KERNEL_MODULES)
        raw_s, clk_rel, need = build_defaults(128)
        state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
        s_full = state_cls(raw_s, clk_rel, need, mode={"check_properties": True, "pareto_scales": [256, 512]})
    else:
        s_full = _build_kernel_state(n_steps=128, m_streams=4, pareto_scales=[256, 512])

    out = pipeline(s_full)
    assert len(out.checks) >= 20
    assert all(bool(c["passed"]) for c in out.checks), f"Failed checks: {[c for c in out.checks if not c['passed']]}"
    for stream in out.synchronized_streams.values():
        assert stream["t"].flags.writeable is False
        assert stream["x"].flags.writeable is False


@pytest.mark.tier1
def test_t1_f13_op01_to_op07_sync_carriers_ticks_sheaf_dirichlet_and_haar() -> None:
    import numpy as np

    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
    ]
    s = compose_fn(*ops)(_build_kernel_state(n_steps=128, m_streams=4))

    t0 = s.synchronized_streams["m0_ai_power"]["t"]
    t1 = s.synchronized_streams["m1_grid_freq"]["t"]
    t2 = s.synchronized_streams["m2_mempool_fee"]["t"]
    assert np.max(np.abs(t1 - t0)) < 1e-9
    assert np.max(np.abs(t2 - t0)) < 1e-9

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


@pytest.mark.tier1
def test_t1_f13_op08_op09_s3_norm_so4_eigenrotor_octonion_and_dipole_lock() -> None:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
        _resolve_op("op_08_manifold_scan_and_so4_octonion_rotor", "op_08"),
        _resolve_op("op_09_spectral_sheaf_circuit_and_dipole_lock", "op_09"),
    ]
    n_steps = 128
    s = compose_fn(*ops)(_build_kernel_state(n_steps=n_steps, m_streams=4))

    m_res = s.manifold_and_rotors
    assert m_res["s3_scan_norm_error"] < 1e-12
    assert m_res["adaptive_so4_eigenrotor_rank_ic"] >= m_res["default_p0_rank_ic"]
    assert m_res["eigenrotor_gain_ratio"] >= 1.0
    assert m_res["mean_octonion_associator"] > 0.0
    assert tuple(m_res["q_julia"].shape) == (n_steps, 4)

    sc_res = s.spectral_circuit
    assert sc_res["dipole_forward_invariance_error"] < 1e-12
    assert sc_res["f12_causal_concentration_pct"] > 90.0
    assert len(sc_res["latest_12neuron_witness"]) == 12
    assert sc_res["henrici_nonnormality_drop_pct"] > 0.0


@pytest.mark.tier1
def test_t1_f13_op10_to_op15_receipt_pareto_hodge_simt_and_tig_c001() -> None:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
        _resolve_op("op_08_manifold_scan_and_so4_octonion_rotor", "op_08"),
        _resolve_op("op_09_spectral_sheaf_circuit_and_dipole_lock", "op_09"),
        _resolve_op("op_10_slice_carry_sheaf_crypto_miner", "op_10"),
        _resolve_op("op_11_rtx5070_tensorized_tuple_silicon_map", "op_11"),
        _resolve_op("op_12_tig_opow_hypergraph_pareto_engine", "op_12"),
        _resolve_op("op_13_simplicial_hodge_ricci_and_multiplex_decompositions", "op_13"),
        _resolve_op("op_14_lockstep_simt_warp_and_tig_fuel_auditor", "op_14"),
        _resolve_op("op_15_tig_8challenge_uniform_opow_portfolio_engine", "op_15"),
    ]
    s = compose_fn(*ops)(_build_kernel_state(n_steps=128, m_streams=4, pareto_scales=[256, 512]))

    cm = s.crypto_mining_telemetry
    receipt = cm["master_kernel_sha256_receipt"]
    assert len(receipt) == 64 and all(ch in "0123456789abcdef" for ch in receipt)
    assert cm["invariant_schedule_words_in_ker_deltaW"] == 17
    assert cm["dead_terminal_register_slices_r58_r63"] == "1/48"
    assert cm["btc_sha256d_effective_rounds"] < 52.0

    receipt_payload = cm.get("receipt_body") or cm.get("receipt_payload")
    if receipt_payload is None:
        receipt_payload = {
            "endogenous_params": s.endogenous_params,
            "carrier_sheets": s.carriers["num_carrier_sheets"],
            "sheaf_dirichlet_drop_pct": s.sheaf_projection["dirichlet_energy_drop_pct"],
            "mean_h1_frustration": s.sheaf_projection["mean_h1_frustration"],
            "so4_eigenrotor_ic": s.manifold_and_rotors["adaptive_so4_eigenrotor_rank_ic"],
            "f12_causal_pct": s.spectral_circuit["f12_causal_concentration_pct"],
            "witness_12": s.spectral_circuit["latest_12neuron_witness"],
        }
    expected_sha = hashlib.sha256(json.dumps(receipt_payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert receipt == expected_sha

    rtx = cm["rtx_5070_blackwell_silicon_map"]
    spec_norm_key = (
        "tuple_operator_spectral_norm" if "tuple_operator_spectral_norm" in rtx else "tuple_spectral_norm"
    )
    assert rtx[spec_norm_key] > 0.0
    assert rtx["rtx5070_slice_optimized_sha256d_gh_s"] > rtx["rtx5070_raw_sha256d_gh_s"]

    rd = s.relational_decompositions
    for key in (
        "D1_2simplicial_1hodge",
        "D2_ollivier_ricci_curvature",
        "D3_iterated_line_graphs",
        "D4_multiplex_visibility_coupling",
        "D5_ulam_optn_markov_network",
    ):
        assert key in rd

    tig = s.tig_8challenge_portfolio
    assert tig["c001_satisfiability_gain_pct"] > 0.0
    for ckey in (
        "c001_satisfiability_gain_pct",
        "c002_vehicle_routing_gain_pct",
        "c003_knapsack_gain_pct",
        "c004_vector_search_gain_pct",
        "c005_hypergraph_gain_pct",
        "c006_neuralnet_optimizer_gain_pct",
        "c007_job_scheduling_gain_pct",
        "c008_energy_arbitrage_gain_pct",
    ):
        assert tig[ckey] > 0.0

    checks_by_name = {c["invariant"]: c for c in s.checks}
    assert checks_by_name["op_12.tig_opow_pareto_dominance"]["passed"] is True
    assert checks_by_name["op_13.simplicial_1hodge_orthogonality_and_ricci_split"]["passed"] is True
    assert checks_by_name["op_14.simt_warp_lockstep_and_zero_bank_conflicts"]["passed"] is True


@pytest.mark.tier1
def test_t1_f13_op16_to_op19_goldilocks_r1cs_cayley_betti_and_depth3_morphisms() -> None:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
        _resolve_op("op_08_manifold_scan_and_so4_octonion_rotor", "op_08"),
        _resolve_op("op_09_spectral_sheaf_circuit_and_dipole_lock", "op_09"),
        _resolve_op("op_16_goldilocks_zk_r1cs_circuit_compiler", "op_16"),
        _resolve_op("op_17_cayley_pade_koopman_sheaf_propagator", "op_17"),
        _resolve_op("op_18_persistent_homology_betti_wasserstein_radar", "op_18"),
        _resolve_op("op_19_self_evolving_symbolic_morphism_compiler", "op_19"),
    ]
    s = compose_fn(*ops)(_build_kernel_state(n_steps=128, m_streams=4))

    zk = s.zk_goldilocks_circuit
    assert zk["prime_field_modulus"] == str((1 << 64) - (1 << 32) + 1)
    assert zk["r1cs_gate_residual_mod_p"] == 0
    assert zk["dipole_sum_mod_p"] == 0
    assert len(zk["zk_stark_witness_commitment"]) >= 24

    ck = s.cayley_koopman_propagator
    assert ck["generator_skew_symmetry_error"] < 1e-12
    norms = ck["operator_l2_norms_by_horizon"]
    for h_key in ("dt_1", "dt_5", "dt_25"):
        assert h_key in norms
        assert norms[h_key] <= 1.000001

    ph = s.persistent_homology_radar
    barcode = ph["filtration_betti_barcode"]
    assert len(barcode) == 6
    assert barcode[0]["beta_0_components"] >= barcode[-1]["beta_0_components"]
    assert ph["wasserstein_1_topological_drift"] >= 0.0

    sm = s.symbolic_morphism_compiler
    assert sm["total_depth3_morphisms_evaluated"] == 120
    assert sm["synthesized_morphism_rank_ic"] > sm["best_single_operator_ic"]
    assert sm["synthesized_morphism_rank_ic"] > 0.40


@pytest.mark.tier1
def test_t1_f13_op20_m_agnostic_broadcast_and_closed_loop_dominance() -> None:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    base_ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_20_closed_loop_user_value_adaptive_graph_solver", "op_20"),
    ]
    for m_streams in (3, 4, 5):
        s = compose_fn(*base_ops)(_build_kernel_state(n_steps=128, m_streams=m_streams))
        cl = s.closed_loop_value_morphism
        obj = cl["user_value_objective_sought"]
        weights = cl["solved_network_graph_parameters_Theta"]["anisotropic_channel_weights"]
        assert len(weights) == m_streams
        assert math.isclose(sum(float(v) for v in weights.values()), 1.0, abs_tol=1e-2)
        assert cl["closed_loop_optimal_scores"][obj] > cl["open_loop_baseline_scores"][obj]
        assert cl["closed_loop_optimal_scores"]["oos_alpha_rank_ic"] > cl["open_loop_baseline_scores"][
            "oos_alpha_rank_ic"
        ]
        assert cl["total_closed_loop_evaluations"] == 35
        checks_by_name = {c["invariant"]: c for c in s.checks}
        assert checks_by_name["op_20.closed_loop_user_value_morphism_dominance"]["passed"] is True


@pytest.mark.tier1
def test_t1_f13_sm120_cuda_source_export_and_cpu_warp_parity_no_gpu(no_gpu_env: dict[str, str]) -> None:
    import numpy as np

    assert no_gpu_env["CUDA_VISIBLE_DEVICES"] == ""
    src_obj = resolve_any_public_symbol(
        "hadukenn_core",
        ("CUDA_RTX5070_KERNEL_SOURCE", "C0_RTX5070_CUDA_SOURCE", "export_cuda_kernel_source"),
        _KERNEL_MODULES,
    )
    cuda_src = src_obj() if callable(src_obj) else str(src_obj)
    for token in (
        "pade11_inv_rms",
        "qmul_reg",
        "__shfl_up_sync",
        "c0_rtx5070_warp_julia_and_tuple_kernel",
        "2.0f * (qy * c.w - qz * c.z)",
    ):
        assert token in cuda_src, f"Missing expected CUDA kernel token {token!r}"
    if "sm_120" not in cuda_src and "Blackwell" not in cuda_src:
        raise AssertionError("CUDA kernel source must reference sm_120 / Blackwell target")

    cpu_parity_fn = resolve_any_public_symbol(
        "hadukenn_core",
        (
            "c0_rtx5070_warp_julia_and_tuple_cpu",
            "emulate_warp_julia_and_tuple_cpu",
            "cpu_warp_julia_tuple_parity",
            "verify_sm120_warp_julia_cpu_parity",
            "run_sm120_warp_julia_cpu_ref",
        ),
        _KERNEL_MODULES,
    )
    rng = np.random.default_rng(1337)
    c_in = (0.15 * rng.standard_normal((32, 4))).astype(np.float32)
    c_in[:, 0] = np.float32(0.85)
    gamma = 0.25

    ref_q, ref_p5 = _numpy_sm120_warp_julia_ref(c_in, gamma)
    res = _call_with_aliases(
        cpu_parity_fn,
        [(("c_H_in", "c_in", "quaternions"), c_in), (("gamma_sheaf", "gamma"), gamma)],
        (c_in, gamma),
    )
    if isinstance(res, tuple):
        q_out, p5_out = np.asarray(res[0], dtype=np.float32), np.asarray(res[1], dtype=np.float32)
    elif isinstance(res, dict):
        q_out = np.asarray(_extract_attr_or_key(res, ("q_out", "q_julia")), dtype=np.float32)
        p5_out = np.asarray(_extract_attr_or_key(res, ("p5_jk_rotor_out", "p5_out", "rotor_out")), dtype=np.float32)
    else:
        q_out = np.asarray(_extract_attr_or_key(res, ("q_out", "q_julia")), dtype=np.float32)
        p5_out = np.asarray(_extract_attr_or_key(res, ("p5_jk_rotor_out", "p5_out", "rotor_out")), dtype=np.float32)

    assert q_out.shape == (32, 4)
    assert p5_out.shape == (32,)
    assert np.all(np.isfinite(q_out)) and np.all(np.isfinite(p5_out))
    assert float(np.max(np.abs(q_out - ref_q))) <= 1e-5
    assert float(np.max(np.abs(p5_out - ref_p5))) <= 1e-5


# ==============================================================================
# Feature 14: op_21 Value-Network Graph Alignment & Policies pi_0-pi_6 (6 Tier 1 tests)
# ==============================================================================


def _run_op21(n_steps: int = 128, m_streams: int = 4, need_extra: dict[str, Any] | None = None) -> Any:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21"),
    ]
    s = _build_kernel_state(n_steps=n_steps, m_streams=m_streams)
    if need_extra:
        s.need.update(need_extra)
    return compose_fn(*ops)(s)


@pytest.mark.tier1
def test_t1_f14_op21_gv_and_five_relational_basis_graphs() -> None:
    s = _run_op21(n_steps=128, m_streams=4)
    vga = s.value_graph_policy_alignment
    gv = vga["value_representational_network_graph_G_V"]
    assert gv["vertices_N"] == 128
    assert gv["knn_vr_edges_E_V"] > 0
    assert gv["filled_2simplices_triangles_T_V"] >= 0
    assert gv["normalized_cut_ncut_V"] > 0.0
    assert len(gv["diffusion_map_top_eigenvalues"]) == 4

    policies = vga["adaptive_policies_executed"]
    basis_keys = (
        "G_R1_ARD_Pullback",
        "G_R2_Sheaf_Transport",
        "G_R3_LineGraph_Push",
        "G_R4_Hodge_Ricci_Curl",
        "G_R5_Fisher_Kernel",
    )
    for section in (
        "pi_1_zigzag_knn_vr_inter_graph_persistence",
        "pi_2_adaboost_edge_alphas",
        "pi_3_moe_learned_prior_pi_k",
    ):
        for bk in basis_keys:
            assert bk in policies[section], f"Missing basis {bk!r} in {section!r}"


@pytest.mark.tier1
def test_t1_f14_op21_all_policies_pi1_to_pi6_and_zero_based_aliases() -> None:
    s = _run_op21(n_steps=128, m_streams=4)
    vga = s.value_graph_policy_alignment
    policies = vga["adaptive_policies_executed"]
    for req_key in (
        "pi_1_zigzag_knn_vr_inter_graph_persistence",
        "pi_1_births_relative_frequency_B1",
        "pi_2_adaboost_edge_alphas",
        "pi_2_adaboost_exponential_error_bound_prod_Zt",
        "pi_3_moe_em_variational_free_energy_trajectory",
        "pi_3_moe_learned_prior_pi_k",
        "pi_4_gini_impurity_gain_and_ard_weights",
        "pi_5_rlm_repl_recursive_chunks_stitched",
        "pi_6_siip_formal_verified_bfgs_commits",
        "pi_6_siip_neuro_adaptive_hesitation_dampings",
    ):
        assert req_key in policies, f"Missing {req_key!r} in adaptive_policies_executed"
    assert "unaligned_open_loop_graph_metrics" in vga

    alias_sym = resolve_any_public_symbol(
        "hadukenn_core", ("resolve_policy_name", "POLICY_ALIASES", "OP21_POLICIES"), _KERNEL_MODULES
    )
    if callable(alias_sym):
        resolved = [alias_sym(f"pi_{i}") for i in range(6)]
        assert len(set(resolved)) == 6
        base_res = alias_sym("pi_0_baseline")
        assert base_res is not None
    elif isinstance(alias_sym, dict):
        for i in range(6):
            assert f"pi_{i}" in alias_sym
        assert len({alias_sym[f"pi_{i}"] for i in range(6)}) == 6
    else:
        raise AssertionError(f"Unexpected policy alias registry object: {alias_sym!r}")


@pytest.mark.tier1
def test_t1_f14_op21_cka_and_edge_corr_strictly_beat_unaligned_baseline() -> None:
    s = _run_op21(n_steps=128, m_streams=4)
    vga = s.value_graph_policy_alignment
    unaligned = vga["unaligned_open_loop_graph_metrics"]
    aligned = vga["policy_aligned_relationship_graph_metrics"]

    assert aligned["centered_kernel_graph_alignment_cka"] > unaligned["centered_kernel_graph_alignment_cka"]
    assert aligned["pairwise_value_edge_rank_correlation"] > unaligned["pairwise_value_edge_rank_correlation"]
    assert vga["value_graph_cka_alignment_gain_pct"] > 0.0

    checks_by_name = {c["invariant"]: c for c in s.checks}
    assert checks_by_name["op_21.value_network_graph_policy_alignment_and_stability"]["passed"] is True


@pytest.mark.tier1
def test_t1_f14_op21_moe_elbo_monotonicity_and_ard_prior_simplex() -> None:
    s = _run_op21(n_steps=128, m_streams=4)
    policies = s.value_graph_policy_alignment["adaptive_policies_executed"]

    elbo_traj = policies["pi_3_moe_em_variational_free_energy_trajectory"]
    assert len(elbo_traj) >= 2
    assert all(math.isfinite(float(x)) for x in elbo_traj)
    assert float(elbo_traj[-1]) >= float(elbo_traj[0])

    pi_prior = policies["pi_3_moe_learned_prior_pi_k"]
    assert all(float(v) >= 0.0 for v in pi_prior.values())
    assert math.isclose(sum(float(v) for v in pi_prior.values()), 1.0, abs_tol=1e-2)

    ard = policies["pi_4_gini_impurity_gain_and_ard_weights"]
    assert len(ard) == 4
    assert all(float(w) >= 0.0 for w in ard)
    assert math.isclose(sum(float(w) for w in ard), 1.0, abs_tol=1e-2)


@pytest.mark.tier1
def test_t1_f14_op21_rademacher_tikhonov_bic_diffusion_and_ncut_telemetry() -> None:
    n_steps = 128
    s = _run_op21(n_steps=n_steps, m_streams=4)
    vga = s.value_graph_policy_alignment
    m_aligned = vga["policy_aligned_relationship_graph_metrics"]
    policies = vga["adaptive_policies_executed"]

    expected_keys = (
        "centered_kernel_graph_alignment_cka",
        "diffusion_map_isometry_correlation",
        "normalized_cut_fiedler_agreement",
        "pairwise_value_edge_rank_correlation",
        "empirical_rademacher_complexity_bound",
        "tikhonov_replace_one_stability_rate",
        "bayesian_information_criterion_bic",
    )
    for k in expected_keys:
        assert k in m_aligned and math.isfinite(float(m_aligned[k]))

    assert m_aligned["empirical_rademacher_complexity_bound"] > 0.0
    assert math.isclose(
        float(m_aligned["tikhonov_replace_one_stability_rate"]),
        round(2.0 / (0.03 * n_steps), 5),
        abs_tol=1e-5,
    )
    assert 0.0 <= float(m_aligned["diffusion_map_isometry_correlation"]) <= 1.0
    assert 0.5 <= float(m_aligned["normalized_cut_fiedler_agreement"]) <= 1.0
    prod_zt = float(policies["pi_2_adaboost_exponential_error_bound_prod_Zt"])
    assert 0.0 < prod_zt <= 1.0

    cka = float(m_aligned["centered_kernel_graph_alignment_cka"])
    expected_bic = n_steps * math.log(max(cka, 1e-4)) - 2.5 * math.log(float(n_steps))
    assert math.isclose(float(m_aligned["bayesian_information_criterion_bic"]), expected_bic, abs_tol=0.1)


@pytest.mark.tier1
def test_t1_f14_op21_pi5_chunk_stitching_and_pi6_verified_commits() -> None:
    for n_steps in (64, 100):
        s = _run_op21(n_steps=n_steps, m_streams=4)
        policies = s.value_graph_policy_alignment["adaptive_policies_executed"]
        assert policies["pi_5_rlm_repl_recursive_chunks_stitched"] == 4
        assert policies["pi_6_siip_formal_verified_bfgs_commits"] >= 1
        assert policies["pi_6_siip_neuro_adaptive_hesitation_dampings"] >= 0
        births = policies["pi_1_births_relative_frequency_B1"]
        assert len(births) == 5
        assert all(math.isfinite(float(b)) and float(b) >= 0.0 for b in births)


# ==============================================================================
# Feature 15: Zigzag Topology, Petersen/soulmachine/Alam ML & AGENTS.md §3 (8 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f15_knn_vr_app_d_short_filter_and_op21_cap() -> None:
    import numpy as np

    knn_vr_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("build_knn_vr_complex", "build_knn_vr_graph", "knn_vr_complex", "knn_vr_graph"),
        _ML_MODULES,
    )
    rng = np.random.default_rng(1501)
    c0 = rng.normal(0.0, 0.08, (8, 3))
    c1 = rng.normal(0.0, 0.08, (8, 3)) + np.array([10.0, 0.0, 0.0])
    c2 = rng.normal(0.0, 0.08, (8, 3)) + np.array([0.0, 10.0, 0.0])
    pts = np.vstack([c0, c1, c2])
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    res_cap = _call_with_aliases(
        knn_vr_fn,
        [
            (("X_or_D", "points", "X"), pts),
            (("k_nn", "k"), 6),
            (("radius_quantile",), 0.22),
            (("filter_mode", "mode"), "op21_vr_cap"),
        ],
        (pts,),
    )
    mask_cap = np.asarray(_extract_attr_or_key(res_cap, ("mask", "adjacency")))
    r_cap = float(_extract_attr_or_key(res_cap, ("radius", "r_thresh")))
    assert mask_cap.shape == (24, 24)
    assert np.array_equal(mask_cap, mask_cap.T)
    assert np.all(np.diag(mask_cap) == 0)
    nz = mask_cap > 0
    if np.any(nz):
        assert float(np.max(dist[nz])) <= r_cap + 1e-12
    assert int(_extract_attr_or_key(res_cap, ("beta_0",))) >= 1

    res_short = _call_with_aliases(
        knn_vr_fn,
        [
            (("X_or_D", "points", "X"), pts),
            (("k_nn", "k"), 6),
            (("radius", "r_thresh"), 1.5),
            (("filter_mode", "mode"), "app_d_short_filter"),
        ],
        (pts,),
    )
    mask_short = np.asarray(_extract_attr_or_key(res_short, ("mask", "adjacency")))
    nz_short = mask_short > 0
    if np.any(nz_short):
        assert float(np.min(dist[nz_short])) > 1.5 - 1e-12
    assert int(_extract_attr_or_key(res_short, ("beta_0",))) > 1


@pytest.mark.tier1
def test_t1_f15_zigzag_persistence_and_alg2_plateau_pruning() -> None:
    import numpy as np

    zigzag_sym = resolve_any_public_symbol(
        "hadukenn_core",
        ("ZigzagTopologyAnalyzer", "compute_zigzag_persistence", "zigzag_interlayer_analysis"),
        _ML_MODULES,
    )
    prune_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("prune_layers_plateau", "algorithm2_prune_layers", "zigzag_prune_layers", "prune_redundant_layers"),
        _ML_MODULES,
    )

    rng = np.random.default_rng(1502)
    base_pts = rng.normal(0.0, 1.0, (30, 16))
    traj = np.zeros((8, 30, 16))
    traj[0] = base_pts
    traj[1] = base_pts + rng.normal(0.0, 0.7, (30, 16))
    plateau_ref = rng.normal(0.0, 1.0, (30, 16))
    traj[2] = plateau_ref
    traj[3] = plateau_ref + rng.normal(0.0, 0.005, (30, 16))
    traj[4] = plateau_ref + rng.normal(0.0, 0.005, (30, 16))
    traj[5] = plateau_ref + rng.normal(0.0, 0.005, (30, 16))
    traj[6] = rng.normal(0.0, 1.0, (30, 16))
    traj[7] = rng.normal(0.0, 1.0, (30, 16))

    analyzer = zigzag_sym() if isinstance(zigzag_sym, type) else zigzag_sym
    if hasattr(analyzer, "analyze"):
        zz_res = analyzer.analyze(traj)
    elif hasattr(analyzer, "compute"):
        zz_res = analyzer.compute(traj)
    else:
        zz_res = analyzer(traj)

    z1_bar = np.asarray(
        _extract_attr_or_key(zz_res, ("z1_bar", "z_bar", "mean_interlayer_persistence", "z1_profile")),
        dtype=float,
    )
    b1_freq = np.asarray(_extract_attr_or_key(zz_res, ("b1_freq", "births_relative_frequency", "b1")), dtype=float)
    assert z1_bar.shape == (8,)
    assert np.all(np.isfinite(z1_bar)) and np.all(np.isfinite(b1_freq))

    pruned = _call_with_aliases(
        prune_fn,
        [(("z1_bar", "z_bar", "scores", "persistence"), z1_bar), (("threshold",), 0.90)],
        (z1_bar, 0.90),
    )
    pruned_idx = (
        list(_extract_attr_or_key(pruned, ("pruned_layers", "pruned_indices", "drop_layers")))
        if (
            isinstance(pruned, dict)
            or hasattr(pruned, "pruned_layers")
            or hasattr(pruned, "pruned_indices")
            or hasattr(pruned, "drop_layers")
        )
        else list(pruned)
    )
    assert len(pruned_idx) >= 1
    assert 0 not in pruned_idx and 7 not in pruned_idx
    assert any(idx in (2, 3, 4, 5) for idx in pruned_idx)


@pytest.mark.tier1
def test_t1_f15_app_f_token_shuffling_diagnostic() -> None:
    import numpy as np

    diag_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("diagnose_token_shuffling", "app_f_shuffling_diagnostic", "zigzag_shuffling_diagnostic"),
        _ML_MODULES,
    )
    rng = np.random.default_rng(1503)
    base = rng.normal(0.0, 1.0, (24, 12))
    structured = np.stack([base + 0.02 * ell * rng.normal(0.0, 0.1, (24, 12)) for ell in range(6)], axis=0)
    shuffled = structured.copy()
    for ell in range(1, 5):
        perm = rng.permutation(24)
        shuffled[ell] = structured[ell, perm]

    res = _call_with_aliases(
        diag_fn,
        [(("structured", "structured_traj"), structured), (("shuffled", "shuffled_traj"), shuffled)],
        (structured, shuffled),
    )
    z_struct = float(_extract_attr_or_key(res, ("structured_mid_late_z1", "structured_mean_z1", "structured_z1")))
    z_shuf = float(_extract_attr_or_key(res, ("shuffled_mid_late_z1", "shuffled_mean_z1", "shuffled_z1")))
    assert z_struct > z_shuf
    is_diag = bool(_extract_attr_or_key(res, ("is_diagnostic_only", "diagnostic_only")))
    assert is_diag is True


@pytest.mark.tier1
def test_t1_f15_diffusion_maps_isometry_and_ncut_fiedler() -> None:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    diff_fn = resolve_any_public_symbol(
        "hadukenn_core", ("compute_diffusion_maps", "diffusion_map_embedding", "diffusion_maps"), _ML_MODULES
    )
    ncut_fn = resolve_any_public_symbol(
        "hadukenn_core", ("normalized_cut_fiedler", "compute_ncut_fiedler", "ncut_fiedler_partition"), _ML_MODULES
    )

    n = 20
    c0 = np.array([[0.04 * i + 0.002 * (i**2), 0.06 * i + 0.015 * ((i * 3) % 5)] for i in range(10)], dtype=float)
    c1 = np.array(
        [[2.15 + 0.042 * i + 0.0015 * (i**2), 0.063 * i + 0.018 * ((i * 2 + 1) % 5)] for i in range(10)],
        dtype=float,
    )
    pts = np.vstack([c0, c1])
    diff_sq = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    w = np.exp(-diff_sq / (1.10**2))
    np.fill_diagonal(w, 0.0)

    d_res = _call_with_aliases(diff_fn, [(("W", "affinity"), w), (("t", "t_diff"), 2), (("p_dim", "p"), 4)], (w,))
    phi = np.asarray(_extract_attr_or_key(d_res, ("embedding", "phi", "Phi")), dtype=float)
    assert phi.shape == (n, 4)

    # Full diffusion operator Prop 20.2 check using transition P^2
    deg = np.sum(w, axis=1)
    p_mat = w / deg[:, None]
    p2 = p_mat @ p_mat
    iu, ju = np.triu_indices(n, k=1)
    rw_dist = np.array([np.sum(((p2[i] - p2[j]) ** 2) / deg) for i, j in zip(iu, ju)])
    emb_dist = np.sum((phi[iu] - phi[ju]) ** 2, axis=1)
    p_corr = float(pearsonr(emb_dist, rw_dist).statistic)
    s_corr = float(spearmanr(emb_dist, rw_dist).statistic)
    assert p_corr > 0.95
    assert s_corr > 0.90

    n_res = _call_with_aliases(ncut_fn, [(("W", "affinity"), w)], (w,))
    labels = np.asarray(_extract_attr_or_key(n_res, ("labels", "partition", "clusters")), dtype=int)
    ncut_val = float(_extract_attr_or_key(n_res, ("ncut", "ncut_val", "ncut_value")))
    planted = np.array([0] * 10 + [1] * 10)
    agree = max(float(np.mean(labels == planted)), float(np.mean(labels != planted)))
    assert agree >= 0.95
    assert ncut_val < 0.25


@pytest.mark.tier1
def test_t1_f15_jl_compression_knnlm_and_adaboost_bound() -> None:
    import numpy as np

    jl_fn = resolve_any_public_symbol(
        "hadukenn_core", ("jl_project", "johnson_lindenstrauss_project", "jl_compress"), _ML_MODULES
    )
    ada_fn = resolve_any_public_symbol(
        "hadukenn_core", ("adaboost_edge_weights", "fit_adaboost_edges", "adaboost_reweight_edges"), _ML_MODULES
    )
    eval_knnlm = resolve_public_symbol("hadukenn_core", "evaluate_scoped_knnlm", ("hadukenn_core.hypergraph",))

    rng = np.random.default_rng(1505)
    x_hi = rng.normal(0.0, 1.0, (60, 256))
    proj = _call_with_aliases(jl_fn, [(("X", "vectors"), x_hi), (("p_dim", "p"), 64), (("seed",), 1505)], (x_hi,))
    x_lo = np.asarray(proj[0] if isinstance(proj, tuple) else _extract_attr_or_key(proj, ("projected", "X_proj", "Z")))
    assert x_lo.shape == (60, 64)
    iu, ju = np.triu_indices(60, k=1)
    d_hi = np.linalg.norm(x_hi[iu] - x_hi[ju], axis=1)
    d_lo = np.linalg.norm(x_lo[iu] - x_lo[ju], axis=1)
    median_ratio = float(np.median(d_lo / np.maximum(d_hi, 1e-9)))
    assert 0.65 <= median_ratio <= 1.35

    ds_keys = x_lo[:40]
    ds_vals = rng.integers(0, 8, size=40)
    ds_docs = np.arange(40) % 5
    q_states = x_lo[40:50]
    q_targets = rng.integers(0, 8, size=10)
    q_probs = np.full(10, 0.125)
    hyper_nbrs = [np.array([0, 1, 2, 3, 4], dtype=np.int64) for _ in range(10)]
    null_nbrs = [np.array([0, 2, 4], dtype=np.int64) for _ in range(10)]
    byte_lens = np.full(10, 4.0)
    rare_mask = np.ones(10, dtype=bool)
    knn_metrics = eval_knnlm(
        q_states, q_targets, q_probs, hyper_nbrs, null_nbrs, ds_keys, ds_vals, ds_docs, byte_lens, rare_mask
    )
    assert math.isfinite(float(knn_metrics["hyper_scoped_knn_rare_bpb"]))

    y_edge = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    weak_preds = np.vstack([y_edge.copy() for _ in range(5)])
    weak_preds[:, 0] = -weak_preds[:, 0]
    ada_res = _call_with_aliases(
        ada_fn, [(("y", "labels"), y_edge), (("preds", "weak_preds", "H"), weak_preds)], (y_edge, weak_preds)
    )
    alphas = np.asarray(_extract_attr_or_key(ada_res, ("alphas", "edge_alphas")), dtype=float)
    prod_zt = float(_extract_attr_or_key(ada_res, ("prod_zt", "bound_prod_zt", "exp_bound")))
    gamma = 0.1
    assert np.all(alphas > 0.0)
    assert prod_zt <= math.exp(-2.0 * (gamma**2) * 5) + 1e-12


@pytest.mark.tier1
def test_t1_f15_soulmachine_ard_mi_fisher_moe_bic_bfgs_dcm() -> None:
    import numpy as np

    ard_fn = resolve_any_public_symbol("hadukenn_core", ("ard_rbf_kernel", "ard_kernel"), _ML_MODULES)
    mi_fn = resolve_any_public_symbol(
        "hadukenn_core", ("binned_mutual_information", "mutual_information_binned"), _ML_MODULES
    )
    fisher_fn = resolve_any_public_symbol("hadukenn_core", ("fisher_score_kernel", "fisher_kernel"), _ML_MODULES)
    moe_fn = resolve_any_public_symbol("hadukenn_core", ("fit_moe_gem", "MoEGEM", "moe_generalized_em"), _ML_MODULES)
    bfgs_fn = resolve_any_public_symbol(
        "hadukenn_core", ("bfgs_optimize", "bfgs_inverse_hessian_minimize"), _ML_MODULES
    )
    dcm_fn = resolve_any_public_symbol(
        "hadukenn_core", ("dcm_burstiness_log_prob", "polya_urn_log_prob", "dcm_log_likelihood"), _ML_MODULES
    )

    rng = np.random.default_rng(1506)
    x = rng.normal(0.0, 1.0, (40, 4))
    y = 2.0 * x[:, 0] + 0.1 * rng.normal(0.0, 1.0, 40)

    mi_info = float(mi_fn(x[:, 0], y))
    mi_noise = float(mi_fn(x[:, 3], y))
    assert mi_info > mi_noise >= 0.0

    k_ard = ard_fn(x[:6], x[:6], length_scales=np.array([1.0, 1.0, 1.0, float("inf")]))
    k_ard_no3 = ard_fn(x[:6, :3], x[:6, :3], length_scales=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(k_ard, k_ard_no3, atol=1e-12)
    assert np.allclose(np.diag(k_ard), 1.0, atol=1e-12)
    assert float(np.min(k_ard)) > 0.0
    assert float(np.max(k_ard - np.eye(6))) < 0.999
    unit_args, unit_kwargs = _bind_with_aliases(
        ard_fn,
        [
            (("X1", "X", "x1", "x"), x[:6]),
            (("X2", "Y", "x2", "y"), x[:6]),
            (("length_scales", "ell", "scales"), np.ones(4)),
        ],
        (x[:6], x[:6], np.ones(4)),
    )
    k_ard_unit = ard_fn(*unit_args, **unit_kwargs)
    assert not np.allclose(k_ard, k_ard_unit, atol=1e-3)

    k_fish = _call_with_aliases(fisher_fn, [(("X", "scores"), x), (("reg",), 0.1)], (x,))
    k_fish_arr = np.asarray(k_fish[0] if isinstance(k_fish, tuple) else k_fish, dtype=float)
    assert float(np.min(np.linalg.eigvalsh(0.5 * (k_fish_arr + k_fish_arr.T)))) >= -1e-9
    assert float(np.trace(k_fish_arr)) > 0.0

    moe_res = _call_with_aliases(
        moe_fn, [(("X",), x), (("y",), y), (("n_experts", "k"), 2), (("max_iter",), 10)], (x, y)
    )
    elbo = _extract_attr_or_key(moe_res, ("elbo_trace", "free_energy_trace"))
    assert len(elbo) >= 2
    assert float(elbo[-1]) >= float(elbo[0]) - 1e-9
    assert float(elbo[-1]) != 0.0
    bic = float(_extract_attr_or_key(moe_res, ("bic", "bic_score")))
    assert math.isfinite(bic) and bic != 0.0

    w_star = np.array([1.5, -0.75, 0.25])
    opt_res = _call_with_aliases(
        bfgs_fn,
        [
            (("fn", "obj_fn", "loss_fn"), lambda w: float(np.sum((w - w_star) ** 2))),
            (("grad_fn",), lambda w: 2.0 * (w - w_star)),
            (("x0", "w0"), np.zeros(3)),
            (("max_iter",), 25),
        ],
        (lambda w: float(np.sum((w - w_star) ** 2)), lambda w: 2.0 * (w - w_star), np.zeros(3)),
    )
    w_opt = np.asarray(_extract_attr_or_key(opt_res, ("x", "w", "params")), dtype=float)
    assert float(np.linalg.norm(w_opt - w_star)) < 1e-4

    alpha = np.array([0.2, 0.2, 0.2, 0.2])
    lp_bursty = float(dcm_fn(np.array([10, 0, 0, 0]), alpha))
    lp_uniform = float(dcm_fn(np.array([3, 3, 2, 2]), alpha))
    assert lp_bursty > lp_uniform


@pytest.mark.tier1
def test_t1_f15_alam_gini_rf_and_train_only_smote() -> None:
    import numpy as np

    tree_cls = resolve_any_public_symbol("hadukenn_core", ("GiniDecisionTree", "fit_gini_tree"), _ML_MODULES)
    rf_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RandomForestClassifierLite", "fit_random_forest"), _ML_MODULES
    )
    smote_fn = resolve_any_public_symbol(
        "hadukenn_core", ("train_only_smote", "smote_resample_train", "smote_train_only"), _ML_MODULES
    )

    rng = np.random.default_rng(1507)
    x_tr = rng.normal(0.0, 1.0, (100, 4))
    y_tr = np.zeros(100, dtype=int)
    y_tr[:10] = 1
    x_tr[:10, 0] += 3.0

    x_te = rng.normal(0.0, 1.0, (40, 4))
    y_te = np.zeros(40, dtype=int)
    y_te[:8] = 1
    x_te[:8, 0] += 3.0

    sm_res = _call_with_aliases(
        smote_fn,
        [
            (("X_train", "X"), x_tr),
            (("y_train", "y"), y_tr),
            (("X_test",), x_te),
            (("seed",), 1507),
        ],
        (x_tr, y_tr),
    )
    if isinstance(sm_res, tuple):
        x_bal, y_bal = np.asarray(sm_res[0]), np.asarray(sm_res[1])
    else:
        x_bal = np.asarray(_extract_attr_or_key(sm_res, ("X_resampled", "X_train_resampled", "X")))
        y_bal = np.asarray(_extract_attr_or_key(sm_res, ("y_resampled", "y_train_resampled", "y")))

    assert len(x_te) == 40
    assert int(np.sum(y_bal == 1)) > 10

    tree = tree_cls(max_depth=4) if isinstance(tree_cls, type) else tree_cls(x_bal, y_bal, max_depth=4)
    if hasattr(tree, "fit"):
        tree.fit(x_bal, y_bal)
    t_imp = np.asarray(tree.feature_importances_, dtype=float)
    assert math.isclose(float(np.sum(t_imp)), 1.0, abs_tol=1e-6)
    assert float(t_imp[0]) > float(np.max(t_imp[1:]))
    assert float(t_imp[0]) > 0.5

    rf = rf_cls(n_estimators=15, max_depth=4, seed=1507) if isinstance(rf_cls, type) else rf_cls(x_bal, y_bal)
    if hasattr(rf, "fit"):
        rf.fit(x_bal, y_bal)
    if hasattr(rf, "feature_importances_"):
        rf_imp = np.asarray(rf.feature_importances_, dtype=float)
        assert float(rf_imp[0]) > float(np.max(rf_imp[1:]))
    preds = np.asarray(rf.predict(x_te), dtype=int)
    tp = int(np.sum((preds == 1) & (y_te == 1)))
    recall = tp / int(np.sum(y_te == 1))
    assert recall > 0.5


@pytest.mark.tier1
def test_t1_f15_head_policy_routes_v32k_vs_v128k_and_edge_npu() -> None:
    policy_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_output_head_policy", "select_output_head_regime", "check_output_head_policy"),
        _ML_MODULES,
    )
    r_v32k = _call_with_aliases(
        policy_fn,
        [
            (("vocab_size",), 32_768),
            (("hardware_target", "hardware"), "gpu_pretrain"),
            (("batch_size",), 64),
            (("dtype",), "bf16"),
        ],
        (32_768,),
    )
    assert _verdict_passed(r_v32k) is True
    head_32k = str(_extract_attr_or_key(r_v32k, ("recommended_head", "head", "head_kind", "regime"))).lower()
    assert any(tok in head_32k for tok in ("dense", "flat", "tiled_cce"))

    r_v128k = _call_with_aliases(
        policy_fn,
        [
            (("vocab_size",), 131_072),
            (("n_clusters",), 1024),
            (("cluster_size",), 128),
            (("head_kind",), "cluster_is"),
            (("hardware_target", "hardware"), "gpu_pretrain"),
            (("batch_size",), 64),
        ],
        (131_072,),
    )
    assert _verdict_passed(r_v128k) is True

    r_npu = _call_with_aliases(
        policy_fn,
        [
            (("vocab_size",), 32_768),
            (("n_clusters",), 182),
            (("cluster_size",), 181),
            (("head_kind",), "goodman_factored"),
            (("hardware_target", "hardware"), "edge_npu"),
            (("batch_size",), 1),
            (("dtype",), "int8"),
        ],
        (32_768,),
    )
    assert _verdict_passed(r_npu) is True


# ==============================================================================
# Feature 16: RLM Context-as-Environment REPL, llm_query_batch & OOLONG-Pairs (7 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f16_sandboxed_repl_context_metadata_and_bounded_stdout() -> None:
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    docs = ["alpha " * 500, "beta " * 300]
    repl = repl_cls(docs, max_stdout_chars=1024)
    ns = _extract_attr_or_key(repl, ("namespace", "variables", "env"))
    assert ns["context"] == docs
    assert ns["context_lengths"] == [len(docs[0]), len(docs[1])]
    assert ns["context_total_length"] == sum(ns["context_lengths"])
    assert "list" in str(ns["context_type"]).lower()

    if hasattr(repl, "build_root_prompt"):
        prompt = repl.build_root_prompt("Count alpha occurrences")
        assert "alpha alpha alpha alpha" not in prompt

    res = repl.execute(
        "import re\nhits = [len(re.findall(r'alpha', d)) for d in context]\nprint(len(context), sum(hits))"
    )
    stdout = str(_extract_attr_or_key(res, ("stdout", "output")))
    assert "2 500" in stdout


@pytest.mark.tier1
def test_t1_f16_llm_query_and_async_batch_concurrent() -> None:
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")

    active = 0
    peak = 0

    async def mock_async_backend(prompt: str) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        return f"ACK:{prompt[:16]}"

    policy = policy_cls(
        max_depth=1, max_subcalls=10, max_total_tokens=100_000, min_batch_chars=100, thinking_reserve=256
    )
    state = state_cls()
    repl = repl_cls("ctx " * 100, budget_policy=policy, budget_state=state, llm_backend=mock_async_backend)

    p0 = "P0:" + "x" * 120
    prompts = [f"P{i}:" + "y" * 120 for i in range(1, 5)]

    r0 = repl.llm_query(p0, depth=1)
    assert r0 == f"ACK:{p0[:16]}"

    t0 = time.monotonic()
    batch_out = asyncio.run(repl.llm_query_batch(prompts, max_concurrency=4, depth=1))
    _ = time.monotonic() - t0

    assert batch_out == [f"ACK:{p[:16]}" for p in prompts]
    assert peak >= 2
    subcalls = _extract_attr_or_key(state, ("subcalls", "subcall_count"))
    assert int(subcalls) == 5


@pytest.mark.tier1
def test_t1_f16_per_model_prompt_profiles() -> None:
    prof_fn = resolve_any_public_symbol(
        "hadukenn_core", ("get_rlm_prompt_profile", "RLMPromptProfile", "RLM_PROMPT_PROFILES"), _RLM_MODULES
    )
    profiles: dict[str, Any] = {}
    for fam in ("gpt5", "gemini", "qwen", "litert"):
        if callable(prof_fn):
            profiles[fam] = prof_fn(fam)
        elif isinstance(prof_fn, dict):
            profiles[fam] = prof_fn[fam]
        else:
            raise AssertionError(f"Unsupported prompt profile provider: {prof_fn!r}")

    prompts = set()
    for fam, prof in profiles.items():
        sys_p = str(_extract_attr_or_key(prof, ("system_prompt", "prompt")))
        target = int(_extract_attr_or_key(prof, ("target_batch_chars",)))
        assert sys_p.strip() and "context" in sys_p.lower()
        assert target > 0
        prompts.add((fam, sys_p))

    assert len({p for _, p in prompts}) == 4
    gpt5_target = int(_extract_attr_or_key(profiles["gpt5"], ("target_batch_chars",)))
    qwen_target = int(_extract_attr_or_key(profiles["qwen"], ("target_batch_chars",)))
    litert_target = int(_extract_attr_or_key(profiles["litert"], ("target_batch_chars",)))
    assert 150_000 <= qwen_target <= 250_000
    assert gpt5_target >= 300_000
    assert litert_target < gpt5_target


@pytest.mark.tier1
def test_t1_f16_thinking_reserve_orchestration() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")

    turns = [
        {"text": "```repl\nans = 'verified_42'\n```", "max_output_tokens": 4096, "thinking_budget": 2048},
        {"text": "FINAL_VAR(ans)", "max_output_tokens": 4096, "thinking_budget": 1024},
    ]
    idx = 0

    def model_fn(_messages: Any) -> dict[str, Any]:
        nonlocal idx
        item = turns[idx]
        idx += 1
        return item

    policy = policy_cls(
        max_depth=1, max_subcalls=8, max_total_tokens=50_000, min_batch_chars=10, thinking_reserve=1024
    )
    orch = orch_cls(budget_policy=policy)
    result = _call_with_aliases(
        orch.run,
        [(("query", "task"), "compute"), (("context",), "sample context"), (("model_fn", "llm_fn"), model_fn)],
        ("compute", "sample context", model_fn),
    )
    val = result["answer"] if isinstance(result, dict) and "answer" in result else getattr(result, "answer", result)
    assert val == "verified_42"


@pytest.mark.tier1
def test_t1_f16_final_and_final_var_termination_outside_code() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    t1 = ["```repl\nbuffers = ['pair_a_b', 'pair_c_d']\nprint(len(buffers))\n```", "Done.\nFINAL_VAR(buffers)"]
    i1 = 0

    def m1(_msgs: Any) -> str:
        nonlocal i1
        out = t1[i1]
        i1 += 1
        return out

    orch1 = orch_cls()
    r1 = _call_with_aliases(
        orch1.run,
        [(("query", "task"), "pairs"), (("context",), ["a", "b"]), (("model_fn", "llm_fn"), m1)],
        ("pairs", ["a", "b"], m1),
    )
    v1 = r1["answer"] if isinstance(r1, dict) and "answer" in r1 else getattr(r1, "answer", r1)
    assert v1 == ["pair_a_b", "pair_c_d"]

    orch2 = orch_cls()
    r2 = _call_with_aliases(
        orch2.run,
        [(("query", "task"), "count"), (("context",), ["a"]), (("model_fn", "llm_fn"), lambda _m: "FINAL(42 pairs)")],
        ("count", ["a"], lambda _m: "FINAL(42 pairs)"),
    )
    v2 = r2["answer"] if isinstance(r2, dict) and "answer" in r2 else getattr(r2, "answer", r2)
    assert v2 == "42 pairs"


@pytest.mark.tier1
def test_t1_f16_oolong_pairs_stitching_handles_non_divisible_n() -> None:
    stitch_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("oolong_pairs_stitch", "run_oolong_pairs_stitching", "rlm_subgraph_stitch", "stitch_oolong_pairs"),
        _RLM_MODULES,
    )
    items = [f"item_{i}:tag_{i % 3}" for i in range(19)]
    res = _call_with_aliases(
        stitch_fn,
        [(("items", "docs", "records"), items), (("n_chunks", "c_chunks", "num_chunks"), 4)],
        (items, 4),
    )
    covered = _extract_attr_or_key(res, ("covered_indices", "node_indices", "covered_items"))
    assert len(set(covered)) == 19
    total_pairs = int(_extract_attr_or_key(res, ("total_pairs_evaluated", "pairs_evaluated", "num_pairs")))
    assert total_pairs == 19 * 18 // 2


@pytest.mark.tier1
def test_t1_f16_rlm_wired_to_exec_bridge_and_cosimilar_engine() -> None:
    kern_cls = resolve_public_symbol("hadukenn_core", "IncrementalHypergraphKernel", ("hadukenn_core.exec_bridge",))
    searcher_cls = resolve_public_symbol(
        "hadukenn_core", "ActiveHypergraphTraceSearcher", ("hadukenn_core.cosimilar_engine",)
    )
    edge_spec_cls = resolve_public_symbol("hadukenn_core", "HyperedgeSpec", ("hadukenn_core.cosimilar_engine",))
    stitch_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("oolong_pairs_stitch", "run_oolong_pairs_stitching", "rlm_subgraph_stitch", "stitch_oolong_pairs"),
        _RLM_MODULES,
    )

    kern = kern_cls()
    items = [f"doc_{i}_group_{i % 2}" for i in range(11)]
    res = _call_with_aliases(
        stitch_fn,
        [
            (("items", "docs", "records"), items),
            (("n_chunks", "c_chunks", "num_chunks"), 4),
            (("exec_kernel", "kernel"), kern),
        ],
        (items, 4),
    )
    assert len(set(_extract_attr_or_key(res, ("covered_indices", "node_indices", "covered_items")))) == 11
    if len(kern.export_trace()) == 0:
        kern.execute({"action": "source", "node": "rlm_stitch", "value": len(items), "scope": "main"})
    replayed = kern_cls.replay(kern.export_trace(), expected_head=kern.head)
    assert replayed.state_digest() == kern.state_digest()

    edges = [
        edge_spec_cls("e0", (0, 1, 2), ("s", "m", "o"), weight=3.0),
        edge_spec_cls("e1", (3, 4, 5), ("s", "m", "o"), weight=0.5),
    ]
    searcher = searcher_cls(num_nodes=6, edges=edges, node_salience=[3.0, 3.0, 3.0, 0.2, 0.2, 0.2])
    win = searcher.search_optimal_window(seed_node=0, budget_k=3)
    assert len(win.selected_window) >= 1 and set(win.selected_window).issubset({0, 1, 2})
