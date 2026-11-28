"""Tier 2 boundary, negative, and exclusion E2E tests for Features 13-16.

Pre-M3 collection safety: zero top-level imports of `sheaf_verify.*`, `hadukenn_core.*`,
`scripts.*`, `numpy`, `scipy`, or `torch`. All target symbols and scientific libraries are
resolved inside test bodies or test-local helpers.
"""

from __future__ import annotations

import inspect
import math
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


def _has_symbol(package_name: str, symbol_name: str, preferred_modules: tuple[str, ...]) -> bool:
    try:
        resolve_public_symbol(package_name, symbol_name, preferred_modules)
        return True
    except AssertionError:
        return False


def _resolve_op(long_name: str, short_name: str) -> Any:
    return resolve_any_public_symbol("hadukenn_core", (long_name, short_name), _KERNEL_MODULES)


def _resolve_unit(name: str) -> Any:
    if _has_symbol("hadukenn_core", name, _KERNEL_MODULES):
        return resolve_public_symbol("hadukenn_core", name, _KERNEL_MODULES)
    return resolve_public_symbol("sheaf_verify", name)


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


def _assert_rejected(
    fn: Any,
    *args: Any,
    extra_excs: tuple[type[BaseException], ...] = (),
    **kwargs: Any,
) -> None:
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    caught = (sheaf_err, ValueError, AssertionError, *extra_excs)
    try:
        res = fn(*args, **kwargs)
    except caught:
        return
    assert _verdict_passed(res) is False, f"Expected rejection, got passing result {res!r}"


def _extract_attr_or_key(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(f"Missing expected attribute/key from {names!r} in {obj!r}")


def _build_kernel_state(
    n_steps: int = 96,
    m_streams: int = 4,
    *,
    check_properties: bool = True,
    pareto_scales: list[int] | None = None,
    seed: int = 20261108,
) -> Any:
    import numpy as np

    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    second = _resolve_unit("SECOND")
    watt = _resolve_unit("WATT")
    hash_rate = _resolve_unit("HASH_RATE")
    dimless = _resolve_unit("DIMENSIONLESS")

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


def _numpy_sm120_warp_julia_ref(c_h_in: Any, gamma_sheaf: float, *, per_warp: bool = True) -> tuple[Any, Any]:
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
    lanes = np.arange(n, dtype=np.int32) & 31 if per_warp else np.arange(n, dtype=np.int32)
    offset = 1
    limit = 32 if per_warp else n
    while offset < limit:
        prev = np.zeros_like(q)
        if offset < n:
            prev[offset:] = q[:-offset]
        mask = lanes >= offset
        q = np.where(mask[:, None], qmul_f32(prev, q), q)
        offset <<= 1

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
# Feature 13: Boundary, Rejection & CUDA Parity Divergence Tests (7 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f13_op01_rejects_short_stream_non_second_unit_negative_rate_and_planck() -> None:
    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    op_01 = _resolve_op("op_01_ingest_and_clock_sync", "op_01")
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    second = _resolve_unit("SECOND")
    watt = _resolve_unit("WATT")
    dimless = _resolve_unit("DIMENSIONLESS")

    s_short = state_cls(
        {"s": {"t": [0.0], "values": [1.0], "time_unit": second, "val_unit": dimless}},
        {},
        {"primary_stream": "s"},
        mode={"check_properties": True},
    )
    with pytest.raises((ValueError, sheaf_err)):
        op_01(s_short)

    for bad_unit in (watt, dimless):
        s_unit = state_cls(
            {"s": {"t": [0.0, 1.0, 2.0], "values": [1.0, 2.0, 3.0], "time_unit": bad_unit, "val_unit": dimless}},
            {},
            {"primary_stream": "s"},
            mode={"check_properties": True},
        )
        with pytest.raises((ValueError, sheaf_err)):
            op_01(s_unit)

    for bad_t, clk in (
        ([0.0, 1.0, 2.0], {"s": {"offset_s": 0.0, "rate": -1.0}}),
        ([0.0, 1e-45, 2e-45], {}),
        ([0.0, float("nan"), 2.0], {}),
    ):
        s_strict = state_cls(
            {"s": {"t": bad_t, "values": [1.0, 2.0, 3.0], "time_unit": second, "val_unit": dimless}},
            clk,
            {"primary_stream": "s"},
            mode={"check_properties": True},
        )
        with pytest.raises((AssertionError, ValueError, sheaf_err)):
            op_01(s_strict)

        s_soft = state_cls(
            {"s": {"t": bad_t, "values": [1.0, 2.0, 3.0], "time_unit": second, "val_unit": dimless}},
            clk,
            {"primary_stream": "s"},
            mode={"check_properties": False},
        )
        op_01(s_soft)
        assert s_soft.checks and s_soft.checks[-1]["passed"] is False


@pytest.mark.tier2
def test_t2_f13_op02_rejects_constant_and_linear_zero_noise_signal() -> None:
    import numpy as np

    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    op_01 = _resolve_op("op_01_ingest_and_clock_sync", "op_01")
    op_02 = _resolve_op("op_02_endogenous_scale_solver", "op_02")
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    second = _resolve_unit("SECOND")
    dimless = _resolve_unit("DIMENSIONLESS")
    t_arr = np.linspace(1.0, 64.0, 64)

    for vals in (np.ones(64), np.linspace(0.0, 10.0, 64)):
        s_strict = state_cls(
            {"s": {"t": t_arr, "values": vals, "time_unit": second, "val_unit": dimless}},
            {},
            {"primary_stream": "s"},
            mode={"check_properties": True},
        )
        op_01(s_strict)
        with pytest.raises((AssertionError, ValueError, sheaf_err)):
            op_02(s_strict)

        s_soft = state_cls(
            {"s": {"t": t_arr, "values": vals, "time_unit": second, "val_unit": dimless}},
            {},
            {"primary_stream": "s"},
            mode={"check_properties": False},
        )
        op_01(s_soft)
        op_02(s_soft)
        assert s_soft.endogenous_params["sigma_hat_mad"] == 0.0
        assert s_soft.checks[-1]["passed"] is False


@pytest.mark.tier2
def test_t2_f13_op03_rejects_flat_and_alternating_sawtooth_zero_carriers() -> None:
    import numpy as np

    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    op_01 = _resolve_op("op_01_ingest_and_clock_sync", "op_01")
    op_03 = _resolve_op("op_03_selected_index_pullback", "op_03")
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    second = _resolve_unit("SECOND")
    dimless = _resolve_unit("DIMENSIONLESS")
    t_arr = np.linspace(1.0, 64.0, 64)
    sawtooth = np.array([(-1.0) ** i for i in range(64)], dtype=float)

    for vals, flat_tol in ((np.zeros(64), 1e-3), (sawtooth, 0.05)):
        s_strict = state_cls(
            {"s": {"t": t_arr, "values": vals, "time_unit": second, "val_unit": dimless}},
            {},
            {"primary_stream": "s"},
            mode={"check_properties": True},
        )
        op_01(s_strict)
        s_strict.endogenous_params = {"flat_tol_star": flat_tol}
        with pytest.raises((AssertionError, ValueError, sheaf_err)):
            op_03(s_strict)

        s_soft = state_cls(
            {"s": {"t": t_arr, "values": vals, "time_unit": second, "val_unit": dimless}},
            {},
            {"primary_stream": "s"},
            mode={"check_properties": False},
        )
        op_01(s_soft)
        s_soft.endogenous_params = {"flat_tol_star": flat_tol}
        op_03(s_soft)
        assert s_soft.carriers["num_carrier_sheets"] == 0
        assert s_soft.checks[-1]["passed"] is False


@pytest.mark.tier2
def test_t2_f13_op15_and_op10_op11_separate_live_vs_analytical_telemetry() -> None:
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
        _resolve_op("op_15_tig_8challenge_uniform_opow_portfolio_engine", "op_15"),
    ]
    s_a = compose_fn(*ops)(_build_kernel_state(n_steps=96, m_streams=4, seed=101))
    s_b = compose_fn(*ops)(_build_kernel_state(n_steps=96, m_streams=4, seed=202))

    cm_a, cm_b = s_a.crypto_mining_telemetry, s_b.crypto_mining_telemetry
    assert cm_a["master_kernel_sha256_receipt"] != cm_b["master_kernel_sha256_receipt"]

    rtx_a = cm_a["rtx_5070_blackwell_silicon_map"]
    rtx_b = cm_b["rtx_5070_blackwell_silicon_map"]
    spec_key = "tuple_operator_spectral_norm" if "tuple_operator_spectral_norm" in rtx_a else "tuple_spectral_norm"
    assert rtx_a[spec_key] != rtx_b[spec_key]

    # Verify explicit separation of live vs analytical/modeled provenance in op_15 (and op_10/op_11)
    tig = s_a.tig_8challenge_portfolio
    prov = tig.get("challenge_provenance") or tig.get("telemetry_provenance") or tig.get("telemetry_kind")
    live_list = tig.get("live_challenges")
    ana_list = tig.get("analytical_challenges")
    ana_cids = ("c002", "c003", "c004", "c005", "c006", "c007", "c008")
    ana_words = ("analytical", "static", "reference", "modeled")
    if isinstance(prov, dict):
        assert any("c001" in str(k).lower() for k in prov)
        c1_tag = str(next(v for k, v in prov.items() if "c001" in str(k).lower())).lower()
        assert "live" in c1_tag
        for cid in ana_cids:
            matches = [v for k, v in prov.items() if cid in str(k).lower()]
            assert matches, f"Missing analytical provenance entry for {cid}"
            cid_tag = str(matches[0]).lower()
            assert any(w in cid_tag for w in ana_words), f"{cid} must be analytical/modeled, got {cid_tag!r}"
            assert "live" not in cid_tag, f"{cid} must not be tagged live, got {cid_tag!r}"
        if live_list is not None:
            for cid in ana_cids:
                assert not any(cid in str(x).lower() for x in live_list)
        if ana_list is not None:
            for cid in ana_cids:
                assert any(cid in str(x).lower() for x in ana_list)
    elif live_list is not None and ana_list is not None:
        assert any("c001" in str(x).lower() for x in live_list)
        for cid in ana_cids:
            assert not any(cid in str(x).lower() for x in live_list)
            assert any(cid in str(x).lower() for x in ana_list)
    else:
        raise AssertionError(
            "op_15 must explicitly separate live c001 from analytical c002..c008 via provenance metadata"
        )

    op10_prov = cm_a.get("telemetry_provenance") or cm_a.get("analytical_fields") or rtx_a.get("telemetry_provenance")
    assert op10_prov is not None, "op_10/op_11 must expose telemetry provenance separating live vs modeled metrics"


@pytest.mark.tier2
def test_t2_f13_op19_m2_degenerate_lietwist_handled_without_crash() -> None:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    base_ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_19_self_evolving_symbolic_morphism_compiler", "op_19"),
    ]
    s_m2 = _build_kernel_state(n_steps=96, m_streams=2, check_properties=False)
    try:
        out_m2 = compose_fn(*base_ops)(s_m2)
    except (ValueError, sheaf_err):
        pass
    else:
        sm2 = out_m2.symbolic_morphism_compiler
        assert math.isfinite(float(sm2["best_single_operator_ic"]))
        assert math.isfinite(float(sm2["synthesized_morphism_rank_ic"]))

    s_m3 = _build_kernel_state(n_steps=96, m_streams=3, check_properties=True)
    out_m3 = compose_fn(*base_ops)(s_m3)
    sm3 = out_m3.symbolic_morphism_compiler
    assert math.isfinite(float(sm3["synthesized_morphism_rank_ic"]))
    assert float(sm3["synthesized_morphism_rank_ic"]) > 0.0
    assert len(sm3["top_3_discovered_morphisms"]) == 3


@pytest.mark.tier2
def test_t2_f13_op20_m_ne_4_and_sparse_e_cnt_lt_4_shape_integrity() -> None:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_20_closed_loop_user_value_adaptive_graph_solver", "op_20"),
    ]
    for m_streams in (3, 5):
        s_m = compose_fn(*ops)(_build_kernel_state(n_steps=96, m_streams=m_streams, check_properties=False))
        w = s_m.closed_loop_value_morphism["solved_network_graph_parameters_Theta"]["anisotropic_channel_weights"]
        assert len(w) == m_streams

    # Sparse E_cnt < 4 regime: N=8 gives 28 upper-tri pairs -> q_eps=0.06 yields < 4 edges
    s_sparse = _build_kernel_state(n_steps=8, m_streams=4, check_properties=False)
    op_01 = _resolve_op("op_01_ingest_and_clock_sync", "op_01")
    op_06 = _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06")
    op_20 = _resolve_op("op_20_closed_loop_user_value_adaptive_graph_solver", "op_20")
    op_01(s_sparse)
    s_sparse.endogenous_params = {"W_star": 4}
    op_06(s_sparse)
    try:
        op_20(s_sparse)
    except (ValueError, sheaf_err) as exc:
        msg = str(exc).lower()
        assert not any(tok in msg for tok in ("broadcast", "shapes", "aligned", "concatenate")), (
            f"Unexpected shape/broadcast crash on E_cnt < 4: {exc!r}"
        )


@pytest.mark.tier2
def test_t2_f13_sm120_cuda_warp_boundary_n_gt_32_divergence_vs_full_scan(no_gpu_env: dict[str, str]) -> None:
    import numpy as np

    assert no_gpu_env["CUDA_VISIBLE_DEVICES"] == ""
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
    rng = np.random.default_rng(1307)
    c_in = (0.20 * rng.standard_normal((48, 4))).astype(np.float32)
    c_in[:, 0] = np.float32(0.80)
    gamma = 0.25

    ref_warp_q, ref_warp_p5 = _numpy_sm120_warp_julia_ref(c_in, gamma, per_warp=True)
    ref_full_q, _ = _numpy_sm120_warp_julia_ref(c_in, gamma, per_warp=False)
    assert float(np.max(np.abs(ref_warp_q[32:] - ref_full_q[32:]))) > 1e-3

    res = _call_with_aliases(
        cpu_parity_fn,
        [(("c_H_in", "c_in", "quaternions"), c_in), (("gamma_sheaf", "gamma"), gamma)],
        (c_in, gamma),
    )
    if isinstance(res, tuple):
        q_out, p5_out = np.asarray(res[0], dtype=np.float32), np.asarray(res[1], dtype=np.float32)
    else:
        q_out = np.asarray(_extract_attr_or_key(res, ("q_out", "q_julia")), dtype=np.float32)
        p5_out = np.asarray(_extract_attr_or_key(res, ("p5_jk_rotor_out", "p5_out", "rotor_out")), dtype=np.float32)

    assert q_out.shape == (48, 4)
    assert float(np.max(np.abs(q_out - ref_warp_q))) <= 1e-5
    assert float(np.max(np.abs(p5_out - ref_warp_p5))) <= 1e-5
    assert float(np.max(np.abs(q_out[32:] - ref_full_q[32:]))) > 1e-3


# ==============================================================================
# Feature 14: op_21 Boundary, Rejection & Remainder Tests (6 Tier 2 tests)
# ==============================================================================


def _run_op21_from_state(state: Any) -> Any:
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
    return compose_fn(*ops)(state)


def _assert_pi5_node_coverage(policies: dict[str, Any], expected_n: int) -> None:
    coverage_keys = (
        "pi_5_covered_node_count",
        "pi_5_node_policy_weights",
        "pi_5_chunk_sizes",
        "pi_5_chunk_slices",
    )
    present = [k for k in coverage_keys if k in policies]
    assert present, (
        f"Missing explicit pi_5 node coverage telemetry from {coverage_keys!r} in {sorted(policies)!r}"
    )
    if "pi_5_covered_node_count" in policies:
        assert int(policies["pi_5_covered_node_count"]) == expected_n
    if "pi_5_node_policy_weights" in policies:
        assert len(policies["pi_5_node_policy_weights"]) == expected_n
    if "pi_5_chunk_sizes" in policies:
        sizes = [int(x) for x in policies["pi_5_chunk_sizes"]]
        assert sum(sizes) == expected_n and all(sz > 0 for sz in sizes)
    if "pi_5_chunk_slices" in policies:
        slices = policies["pi_5_chunk_slices"]
        covered: list[int] = []
        if all(isinstance(s, slice) for s in slices):
            for sl in slices:
                covered.extend(range(*sl.indices(expected_n)))
        elif (
            slices
            and all(isinstance(s, (tuple, list)) and len(s) == 2 and all(isinstance(v, int) for v in s) for s in slices)
            and slices[0][0] == 0
            and slices[-1][1] == expected_n
            and all(slices[i][1] == slices[i + 1][0] for i in range(len(slices) - 1))
        ):
            for start, stop in slices:
                covered.extend(range(start, stop))
        else:
            for chunk in slices:
                covered.extend(int(idx) for idx in chunk)
        assert len(covered) == expected_n and set(covered) == set(range(expected_n))


@pytest.mark.tier2
def test_t2_f14_op21_pi5_n17_remainder_nodes_not_silently_dropped() -> None:
    s17_base = _build_kernel_state(n_steps=17, m_streams=4, check_properties=False, seed=1401)
    out17_base = _run_op21_from_state(s17_base)
    vga17 = out17_base.value_graph_policy_alignment
    assert vga17["value_representational_network_graph_G_V"]["vertices_N"] == 17

    policies17 = vga17["adaptive_policies_executed"]
    _assert_pi5_node_coverage(policies17, 17)

    s17_pert = _build_kernel_state(n_steps=17, m_streams=4, check_properties=False, seed=1401)
    s17_pert.need["objective_series"][-1] += 15.0
    out17_pert = _run_op21_from_state(s17_pert)
    vga17_p = out17_pert.value_graph_policy_alignment
    assert vga17_p["policy_aligned_relationship_graph_metrics"] != vga17["policy_aligned_relationship_graph_metrics"]

    s33 = _build_kernel_state(n_steps=33, m_streams=4, check_properties=False, seed=1401)
    out33 = _run_op21_from_state(s33)
    vga33 = out33.value_graph_policy_alignment
    assert vga33["value_representational_network_graph_G_V"]["vertices_N"] == 33
    _assert_pi5_node_coverage(vga33["adaptive_policies_executed"], 33)


@pytest.mark.tier2
def test_t2_f14_op21_rejects_n_lt_5_insufficient_vertices() -> None:
    import numpy as np

    op_21 = _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21")
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    rng = np.random.default_rng(1402)

    for n_small in (4, 3):
        s = state_cls({}, {}, {"objective_series": rng.normal(0.0, 1.0, n_small)}, mode={"check_properties": True})
        s.sheaf_projection = {"std_mat": rng.normal(0.0, 1.0, (n_small, 4))}
        with pytest.raises((sheaf_err, ValueError, AssertionError)):
            op_21(s)


@pytest.mark.tier2
def test_t2_f14_op21_pi1_beta_p_zero_acyclic_graph_guarded() -> None:
    import numpy as np

    op_21 = _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21")
    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)

    t = np.linspace(0.0, 1.0, 12)
    std_mat = np.column_stack([t, 2.0 * t, 3.0 * t, 4.0 * t])
    obj = np.linspace(1.0, 12.0, 12)
    s = state_cls({}, {}, {"objective_series": obj}, mode={"check_properties": False})
    s.sheaf_projection = {"std_mat": std_mat}

    out = op_21(s)
    policies = out.value_graph_policy_alignment["adaptive_policies_executed"]
    pi1 = policies["pi_1_zigzag_knn_vr_inter_graph_persistence"]
    b1 = policies["pi_1_births_relative_frequency_B1"]
    assert all(math.isfinite(float(v)) for v in pi1.values())
    assert all(math.isfinite(float(v)) for v in b1)

    zp_fn = resolve_any_public_symbol(
        "sheaf_verify", ("zigzag_persistence_ratio", "zigzag_interlayer_persistence", "compute_zigzag_zp")
    )
    pi_grid = [[0.0, 1.0], [1.0, 2.0]]
    z_zero = _call_with_aliases(zp_fn, [(("pi_hat", "pi_eff"), pi_grid), (("beta_p",), 0)], (pi_grid, 0))
    assert float(z_zero) == 0.0


@pytest.mark.tier2
def test_t2_f14_op21_rejects_invalid_or_unknown_policy_name() -> None:
    op_21 = _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21")
    resolve_policy = resolve_any_public_symbol(
        "hadukenn_core", ("resolve_policy_name", "validate_policy_names", "OP21_POLICIES"), _KERNEL_MODULES
    )
    for bad_name in ("pi_99", "unknown_policy", ""):
        if callable(resolve_policy):
            _assert_rejected(resolve_policy, bad_name, extra_excs=(KeyError,))
        elif isinstance(resolve_policy, dict):
            assert bad_name not in resolve_policy

        s = _build_kernel_state(n_steps=32, m_streams=4, check_properties=False, seed=1404)
        s.need["policies"] = [bad_name]
        s.mode["policies"] = [bad_name]
        op_01 = _resolve_op("op_01_ingest_and_clock_sync", "op_01")
        op_06 = _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06")
        op_01(s)
        s.endogenous_params = {"W_star": 6}
        op_06(s)
        _assert_rejected(op_21, s, extra_excs=(KeyError,))


@pytest.mark.tier2
def test_t2_f14_op21_pi6_siip_hesitation_damping_threshold_and_trigger() -> None:
    import numpy as np

    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    gate = gate_cls()
    damp_fn = getattr(gate, "damp_direction", getattr(gate, "apply_hesitation_damping", None))
    assert damp_fn is not None

    p_exact = [-0.25, math.sqrt(1.0 - 0.25**2)]
    out_exact = damp_fn(p_exact, [1.0, 0.0])
    v_exact = out_exact[0] if isinstance(out_exact, tuple) and isinstance(out_exact[0], (list, tuple)) else out_exact
    assert math.isclose(float(v_exact[0]), -0.25, abs_tol=1e-12)

    p_below = [-0.30, math.sqrt(1.0 - 0.30**2)]
    out_below = damp_fn(p_below, [1.0, 0.0])
    v_below = out_below[0] if isinstance(out_below, tuple) and isinstance(out_below[0], (list, tuple)) else out_below
    assert math.isclose(float(v_below[0]), -0.15, abs_tol=1e-12)

    op_21 = _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21")
    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    rng = np.random.default_rng(1405)
    n = 48
    t = np.arange(n, dtype=float)
    osc = np.sin(1.9 * t) + 0.1 * rng.normal(0.0, 1.0, n)
    std_mat = np.column_stack([osc, np.cos(1.7 * t), (-1.0) ** t * 0.8, rng.normal(0.0, 1.0, n)])
    s = state_cls({}, {}, {"objective_series": np.abs(osc)}, mode={"check_properties": False})
    s.sheaf_projection = {"std_mat": std_mat}
    out = op_21(s)
    pol = out.value_graph_policy_alignment["adaptive_policies_executed"]
    commits = int(pol["pi_6_siip_formal_verified_bfgs_commits"])
    damping_cnt = int(pol["pi_6_siip_neuro_adaptive_hesitation_dampings"])
    assert commits >= 1
    assert damping_cnt >= 1


@pytest.mark.tier2
def test_t2_f14_op21_rejects_nan_inf_inputs_and_non_decreasing_siip_commits() -> None:
    import numpy as np

    op_21 = _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21")
    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    rng = np.random.default_rng(1406)

    for bad_obj_val in (float("nan"), float("inf")):
        obj = rng.normal(0.0, 1.0, 24)
        obj[5] = bad_obj_val
        s = state_cls({}, {}, {"objective_series": obj}, mode={"check_properties": True})
        s.sheaf_projection = {"std_mat": rng.normal(0.0, 1.0, (24, 4))}
        with pytest.raises((sheaf_err, ValueError, AssertionError)):
            op_21(s)

    bad_mat = rng.normal(0.0, 1.0, (24, 4))
    bad_mat[3, 1] = float("nan")
    s_mat = state_cls({}, {}, {"objective_series": rng.normal(0.0, 1.0, 24)}, mode={"check_properties": True})
    s_mat.sheaf_projection = {"std_mat": bad_mat}
    with pytest.raises((sheaf_err, ValueError, AssertionError)):
        op_21(s_mat)

    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    gate = gate_cls()
    verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate", None))
    assert verify_fn is not None
    _assert_rejected(verify_fn, f_k=1.0, f_cand=1.0, x_cand=[0.5, 0.5])
    _assert_rejected(verify_fn, f_k=1.0, f_cand=float("nan"), x_cand=[0.5, 0.5])


# ==============================================================================
# Feature 15: Empirical Exclusions, Leakage & Numerical Guards (8 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f15_rejects_182x181_goodman_on_v32k_gpu_pretrain() -> None:
    policy_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_output_head_policy", "select_output_head_regime", "check_output_head_policy"),
        _ML_MODULES,
    )
    for head_kind in ("fp2", "cluster_is", "goodman_factored"):
        args, kwargs = _bind_with_aliases(
            policy_fn,
            [
                (("vocab_size",), 32_768),
                (("n_clusters",), 182),
                (("cluster_size",), 181),
                (("head_kind",), head_kind),
                (("hardware_target", "hardware"), "gpu_pretrain"),
                (("batch_size",), 64),
                (("dtype",), "bf16"),
            ],
            (32_768, 182, 181, "gpu_pretrain", 64, "bf16"),
        )
        _assert_rejected(policy_fn, *args, **kwargs)

    ok_128k = _call_with_aliases(
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
    assert _verdict_passed(ok_128k) is True

    ok_npu = _call_with_aliases(
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
    assert _verdict_passed(ok_npu) is True


@pytest.mark.tier2
def test_t2_f15_hypergraph_claim_requires_chodrow_null() -> None:
    import numpy as np

    doc_cls = resolve_public_symbol("hadukenn_core", "CorpusDocument", ("hadukenn_core.hypergraph",))
    build_hg = resolve_public_symbol("hadukenn_core", "build_decontaminated_hypergraph", ("hadukenn_core.hypergraph",))
    build_null = resolve_public_symbol("hadukenn_core", "build_chodrow_null", ("hadukenn_core.hypergraph",))
    claim_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_hypergraph_claim", "audit_chodrow_null_control"), _ML_MODULES
    )

    train_docs = [
        doc_cls(f"tr_{i}", f"dom_{i % 3}", f"token_{i} shared_{i % 4} anchor_{i % 2}", f"sha_tr_{i}")
        for i in range(18)
    ]
    eval_docs = [
        doc_cls(f"ev_{i}", f"dom_{i % 3}", f"eval_{i} shared_{i % 4} anchor_{i % 2}", f"sha_ev_{i}")
        for i in range(6)
    ]
    _, hg = build_hg(train_docs, eval_docs, min_edge_size=2, max_edge_size=12, max_doc_degree=12, seed=13)
    null_hg = build_null(hg, n_swaps_factor=4, seed=13)

    deg_orig = np.asarray(hg.incidence.sum(axis=1)).ravel()
    deg_null = np.asarray(null_hg.incidence.sum(axis=1)).ravel()
    card_orig = sorted(int(x) for x in np.asarray(hg.incidence.sum(axis=0)).ravel())
    card_null = sorted(int(x) for x in np.asarray(null_hg.incidence.sum(axis=0)).ravel())
    assert np.array_equal(deg_orig, deg_null)
    assert card_orig == card_null

    args_missing, kwargs_missing = _bind_with_aliases(
        claim_fn,
        [(("hypergraph", "hg"), hg), (("chodrow_null", "null_hg"), None), (("metric", "hyper_bpb"), 1.05)],
        (hg, None),
    )
    _assert_rejected(claim_fn, *args_missing, **kwargs_missing)

    args_tie, kwargs_tie = _bind_with_aliases(
        claim_fn,
        [
            (("hypergraph", "hg"), hg),
            (("chodrow_null", "null_hg"), null_hg),
            (("hyper_bpb", "metric"), 1.10),
            (("null_bpb",), 1.09),
        ],
        (hg, null_hg, 1.10, 1.09),
    )
    _assert_rejected(claim_fn, *args_tie, **kwargs_tie)


@pytest.mark.tier2
def test_t2_f15_rejects_passive_e1_shuffling_claim() -> None:
    doc_cls = resolve_public_symbol("hadukenn_core", "CorpusDocument", ("hadukenn_core.hypergraph",))
    build_hg = resolve_public_symbol("hadukenn_core", "build_decontaminated_hypergraph", ("hadukenn_core.hypergraph",))
    walk_fn = resolve_public_symbol("hadukenn_core", "prefix_causal_walk_order", ("hadukenn_core.hypergraph",))
    e1_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_e1_ordering_policy", "check_e1_ordering_policy"), _ML_MODULES
    )

    args, kwargs = _bind_with_aliases(
        e1_fn,
        [(("mode", "ordering_mode"), "passive_e1_shuffle"), (("e1_passive_shuffle",), True)],
        ("passive_e1_shuffle",),
    )
    _assert_rejected(e1_fn, *args, **kwargs)

    train_docs = [doc_cls(f"d{i}", "dom", f"alpha_{i % 3} beta_{i}", f"s{i}") for i in range(10)]
    eval_docs = [doc_cls("e0", "dom", "gamma_0", "se0")]
    _, hg = build_hg(train_docs, eval_docs, min_edge_size=2, max_edge_size=8, seed=13)
    with pytest.raises(ValueError):
        walk_fn(hg, mode="unknown")


@pytest.mark.tier2
def test_t2_f15_smote_leakage_into_test_split_rejected() -> None:
    import numpy as np

    smote_fn = resolve_any_public_symbol(
        "hadukenn_core", ("train_only_smote", "smote_resample_train", "smote_train_only"), _ML_MODULES
    )
    split_fn = resolve_public_symbol("sheaf_verify", "validate_split_disjointness")

    rng = np.random.default_rng(1504)
    x_tr = rng.normal(0.0, 1.0, (20, 3))
    y_tr = np.array([1, 1, 1] + [0] * 17)
    x_te = x_tr[:6].copy()

    args, kwargs = _bind_with_aliases(
        smote_fn,
        [
            (("X_train", "X"), x_tr),
            (("y_train", "y"), y_tr),
            (("X_test",), x_te),
            (("train_ids",), {"id_0", "id_1", "id_2"}),
            (("test_ids",), {"id_2", "id_99"}),
        ],
        (x_tr, y_tr, x_te),
    )
    _assert_rejected(smote_fn, *args, **kwargs)

    s_args, s_kwargs = _bind_with_aliases(
        split_fn,
        [
            (("train_ids", "train"), {"tr_0"}),
            (("test_ids", "test"), {"te_0", "smote_1"}),
            (("synthetic_ids", "synthetic"), {"smote_1"}),
        ],
        ({"tr_0"}, {"te_0", "smote_1"}, {"smote_1"}),
    )
    _assert_rejected(split_fn, *s_args, **s_kwargs)

    d_args, d_kwargs = _bind_with_aliases(
        split_fn,
        [
            (("train_ids", "train"), {"tr_0", "shared"}),
            (("test_ids", "test"), {"te_0", "shared"}),
            (("synthetic_ids", "synthetic"), set()),
        ],
        ({"tr_0", "shared"}, {"te_0", "shared"}, set()),
    )
    _assert_rejected(split_fn, *d_args, **d_kwargs)


@pytest.mark.tier2
def test_t2_f15_diffusion_maps_and_ncut_isolated_node_dii_zero() -> None:
    import numpy as np

    diff_fn = resolve_any_public_symbol(
        "hadukenn_core", ("compute_diffusion_maps", "diffusion_map_embedding", "diffusion_maps"), _ML_MODULES
    )
    ncut_fn = resolve_any_public_symbol(
        "hadukenn_core", ("normalized_cut_fiedler", "compute_ncut_fiedler", "ncut_fiedler_partition"), _ML_MODULES
    )

    w = np.full((10, 10), 0.05, dtype=float)
    w[:5, :5] = 1.0
    w[5:9, 5:9] = 1.0
    w[9, :] = 0.0
    w[:, 9] = 0.0
    np.fill_diagonal(w, 0.0)
    assert float(np.sum(w[9])) == 0.0

    d_res = _call_with_aliases(diff_fn, [(("W", "affinity"), w), (("t", "t_diff"), 2), (("p_dim", "p"), 4)], (w,))
    phi = np.asarray(_extract_attr_or_key(d_res, ("embedding", "phi", "Phi")), dtype=float)
    assert phi.shape == (10, 4)
    assert np.all(np.isfinite(phi))

    n_res = _call_with_aliases(ncut_fn, [(("W", "affinity"), w)], (w,))
    ncut_val = float(_extract_attr_or_key(n_res, ("ncut", "ncut_val", "ncut_value")))
    assert math.isfinite(ncut_val)


@pytest.mark.tier2
def test_t2_f15_singular_covariance_n_lt_d_guard() -> None:
    import numpy as np

    fisher_fn = resolve_any_public_symbol("hadukenn_core", ("fisher_score_kernel", "fisher_kernel"), _ML_MODULES)
    rng = np.random.default_rng(1506)
    x_under = rng.normal(0.0, 1.0, (4, 12))

    args_bad, kwargs_bad = _bind_with_aliases(fisher_fn, [(("X", "scores"), x_under), (("reg",), 0.0)], (x_under, 0.0))
    _assert_rejected(fisher_fn, *args_bad, **kwargs_bad)

    k_reg = _call_with_aliases(fisher_fn, [(("X", "scores"), x_under), (("reg",), 0.25)], (x_under, 0.25))
    k_arr = np.asarray(k_reg[0] if isinstance(k_reg, tuple) else k_reg, dtype=float)
    assert k_arr.shape == (4, 4)
    assert np.all(np.isfinite(k_arr))
    assert float(np.min(np.linalg.eigvalsh(0.5 * (k_arr + k_arr.T)))) >= -1e-9
    assert float(np.trace(k_arr)) > 0.0


@pytest.mark.tier2
def test_t2_f15_bfgs_non_positive_secant_curvature_sy_le_0() -> None:
    import numpy as np

    bfgs_step_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("bfgs_step", "bfgs_inverse_hessian_update", "bfgs_update_inverse_hessian", "bfgs_optimize"),
        _ML_MODULES,
    )
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    if "optimize" in getattr(bfgs_step_fn, "__name__", "").lower():
        # Force non-convex negative curvature s^T y <= 0 through gradient function
        def obj_neg(w: Any) -> float:
            return float(-np.sum(w * w))

        def grad_neg(w: Any) -> Any:
            return -2.0 * w

        res = _call_with_aliases(
            bfgs_step_fn,
            [
                (("fn", "obj_fn", "loss_fn"), obj_neg),
                (("grad_fn",), grad_neg),
                (("x0", "w0"), np.array([1.0, 0.5])),
                (("max_iter",), 5),
            ],
            (obj_neg, grad_neg, np.array([1.0, 0.5])),
        )
        b_inv = _extract_attr_or_key(res, ("B_inv", "H_inv", "hess_inv"))
        b_arr = np.asarray(b_inv, dtype=float)
        assert np.all(np.isfinite(b_arr))
        assert float(np.min(np.linalg.eigvalsh(0.5 * (b_arr + b_arr.T)))) > 0.0
    else:
        b_init = np.eye(2, dtype=float)
        s_vec = np.array([1.0, 0.0])
        for y_vec in (np.array([-0.5, 0.0]), np.array([0.0, 0.0])):
            try:
                b_next = _call_with_aliases(
                    bfgs_step_fn,
                    [(("B_inv", "H_inv"), b_init), (("s", "delta_k"), s_vec), (("y", "y_k"), y_vec)],
                    (b_init, s_vec, y_vec),
                )
            except (ValueError, sheaf_err):
                continue
            b_arr = np.asarray(b_next, dtype=float)
            assert np.all(np.isfinite(b_arr))
            assert float(np.min(np.linalg.eigvalsh(0.5 * (b_arr + b_arr.T)))) > 0.0


@pytest.mark.tier2
def test_t2_f15_zigzag_beta0_zero_and_small_n_guard() -> None:
    import numpy as np

    zigzag_sym = resolve_any_public_symbol(
        "hadukenn_core",
        ("ZigzagTopologyAnalyzer", "compute_zigzag_persistence", "zigzag_interlayer_analysis"),
        _ML_MODULES,
    )
    diff_fn = resolve_any_public_symbol(
        "hadukenn_core", ("compute_diffusion_maps", "diffusion_map_embedding", "diffusion_maps"), _ML_MODULES
    )
    gini_fn = resolve_public_symbol("sheaf_verify", "gini_impurity")
    smote_fn = resolve_any_public_symbol(
        "hadukenn_core", ("train_only_smote", "smote_resample_train", "smote_train_only"), _ML_MODULES
    )

    line_pts = np.column_stack([np.linspace(0.0, 10.0, 12), np.zeros((12, 3))])
    traj = np.stack([line_pts, line_pts.copy(), line_pts.copy()], axis=0)
    analyzer = zigzag_sym() if isinstance(zigzag_sym, type) else zigzag_sym
    zz_res = analyzer.analyze(traj) if hasattr(analyzer, "analyze") else analyzer(traj)
    z1 = np.asarray(
        _extract_attr_or_key(zz_res, ("z1_bar", "z_bar", "mean_interlayer_persistence", "z1_profile")),
        dtype=float,
    )
    assert np.all(np.isfinite(z1)) and np.allclose(z1, 0.0, atol=1e-12)

    w_small = np.ones((4, 4)) - np.eye(4)
    args_d, kwargs_d = _bind_with_aliases(
        diff_fn, [(("W", "affinity"), w_small), (("t", "t_diff"), 2), (("p_dim", "p"), 4)], (w_small, 2, 4)
    )
    _assert_rejected(diff_fn, *args_d, **kwargs_d)

    _assert_rejected(gini_fn, [])
    x_single = np.ones((8, 3))
    y_single = np.zeros(8, dtype=int)
    args_s, kwargs_s = _bind_with_aliases(
        smote_fn, [(("X_train", "X"), x_single), (("y_train", "y"), y_single)], (x_single, y_single)
    )
    _assert_rejected(smote_fn, *args_s, **kwargs_s)


# ==============================================================================
# Feature 16: RLM Appendix A Pitfalls & Sandbox Security Tests (9 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f16_blocks_tiny_1line_llm_query_spam_and_max_subcalls() -> None:
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")

    calls: list[str] = []

    def backend(prompt: str) -> str:
        calls.append(prompt)
        return "ok"

    policy = policy_cls(
        max_depth=1, max_subcalls=2, max_total_tokens=100_000, min_batch_chars=10_000, thinking_reserve=512
    )
    repl = repl_cls("context", budget_policy=policy, budget_state=state_cls(), llm_backend=backend)

    _assert_rejected(repl.llm_query, "line 1: check?", depth=1)
    assert len(calls) == 0

    big_prompt = "x" * 10_500
    assert repl.llm_query(big_prompt, depth=1) == "ok"
    assert repl.llm_query(big_prompt, depth=1) == "ok"
    _assert_rejected(repl.llm_query, big_prompt, depth=1)
    assert len(calls) == 2


@pytest.mark.tier2
def test_t2_f16_blocks_max_depth_exceeded() -> None:
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")

    invoked = 0

    def backend(_prompt: str) -> str:
        nonlocal invoked
        invoked += 1
        return "ok"

    policy = policy_cls(
        max_depth=1, max_subcalls=8, max_total_tokens=50_000, min_batch_chars=10, thinking_reserve=256
    )
    repl = repl_cls("context", budget_policy=policy, budget_state=state_cls(), llm_backend=backend)
    _assert_rejected(repl.llm_query, "valid sized prompt", depth=2)
    assert invoked == 0


@pytest.mark.tier2
def test_t2_f16_thinking_reserve_and_total_token_exhaustion() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    policy_think = policy_cls(
        max_depth=1, max_subcalls=8, max_total_tokens=100_000, min_batch_chars=10, thinking_reserve=1024
    )
    orch = orch_cls(budget_policy=policy_think)

    def exhausted_model(_msgs: Any) -> dict[str, Any]:
        return {"text": "", "max_output_tokens": 4096, "thinking_budget": 3900}

    try:
        res = _call_with_aliases(
            orch.run,
            [(("query", "task"), "q"), (("context",), "ctx"), (("model_fn", "llm_fn"), exhausted_model)],
            ("q", "ctx", exhausted_model),
        )
    except (ValueError, sheaf_err, AssertionError):
        pass
    else:
        assert _verdict_passed(res) is False

    policy_tok = policy_cls(
        max_depth=1, max_subcalls=10, max_total_tokens=500, min_batch_chars=10, thinking_reserve=100
    )
    repl = repl_cls("ctx", budget_policy=policy_tok, budget_state=state_cls(), llm_backend=lambda _p: "ok")
    _assert_rejected(repl.llm_query, "x" * 100, depth=1, tokens=600)


@pytest.mark.tier2
def test_t2_f16_ignores_final_inside_fenced_code_block() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    turns = [
        "```repl\nfake = 'bad'\nprint('FINAL(fake)')\n# FINAL_VAR(fake)\n```",
        "Still inspecting context...",
    ]
    idx = 0

    def model_fn(_msgs: Any) -> str:
        nonlocal idx
        out = turns[min(idx, len(turns) - 1)]
        idx += 1
        return out

    orch = orch_cls(max_turns=2)
    try:
        res = _call_with_aliases(
            orch.run,
            [(("query", "task"), "find"), (("context",), "ctx"), (("model_fn", "llm_fn"), model_fn)],
            ("find", "ctx", model_fn),
        )
    except (ValueError, sheaf_err, AssertionError):
        return
    ans = res.get("answer") if isinstance(res, dict) else getattr(res, "answer", None)
    assert ans != "fake" and ans != "bad"
    assert _verdict_passed(res) is False


@pytest.mark.tier2
def test_t2_f16_rejects_undefined_and_empty_final_var() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    for setup_code, final_text in (
        ("x = 'nonempty'", "FINAL_VAR(nonexistent_buf)"),
        ("empty_str = ''", "FINAL_VAR(empty_str)"),
        ("empty_list = []", "FINAL_VAR(empty_list)"),
    ):
        seq = [f"```repl\n{setup_code}\n```", final_text]
        pos = 0

        def model_fn(_msgs: Any, _seq: list[str] = seq) -> str:
            nonlocal pos
            val = _seq[min(pos, len(_seq) - 1)]
            pos += 1
            return val

        orch = orch_cls(max_turns=2)
        try:
            res = _call_with_aliases(
                orch.run,
                [(("query", "task"), "q"), (("context",), "ctx"), (("model_fn", "llm_fn"), model_fn)],
                ("q", "ctx", model_fn),
            )
        except (ValueError, sheaf_err, AssertionError):
            continue
        assert _verdict_passed(res) is False


@pytest.mark.tier2
def test_t2_f16_rejects_plan_text_in_final() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    plan = "FINAL(I will now write a Python script to inspect the context and compute pairs.)"
    orch = orch_cls(max_turns=2)
    try:
        res = _call_with_aliases(
            orch.run,
            [(("query", "task"), "q"), (("context",), "ctx"), (("model_fn", "llm_fn"), lambda _m: plan)],
            ("q", "ctx", lambda _m: plan),
        )
    except (ValueError, sheaf_err, AssertionError):
        return
    assert _verdict_passed(res) is False


@pytest.mark.tier2
def test_t2_f16_anti_repetition_loop_preserves_best_verified_buffer() -> None:
    orch_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES
    )
    turns = [
        "```repl\nbest_pairs = [('n1', 'n2')]\nprint(best_pairs)\n```",
        "Verifying best_pairs again without changes.",
        "Verifying best_pairs again without changes.",
        "Verifying best_pairs again without changes.",
        "FINAL(wrong fallback free text)",
    ]
    idx = 0

    def model_fn(_msgs: Any) -> str:
        nonlocal idx
        out = turns[min(idx, len(turns) - 1)]
        idx += 1
        return out

    orch = orch_cls(max_turns=6, max_consec_no_new_var=3)
    res = _call_with_aliases(
        orch.run,
        [(("query", "task"), "pairs"), (("context",), "n1 n2"), (("model_fn", "llm_fn"), model_fn)],
        ("pairs", "n1 n2", model_fn),
    )
    ans = res["answer"] if isinstance(res, dict) and "answer" in res else getattr(res, "answer", res)
    assert ans == [("n1", "n2")]
    assert idx <= 4


@pytest.mark.tier2
def test_t2_f16_enforces_stdout_truncation_cap() -> None:
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    repl = repl_cls("context", max_stdout_chars=512)
    res = repl.execute("print('A' * 50000)")
    stdout = str(_extract_attr_or_key(res, ("stdout", "output")))
    assert len(stdout) < 1024
    assert "trunc" in stdout.lower() or bool(_extract_attr_or_key(res, ("truncated",))) is True


@pytest.mark.tier2
def test_t2_f16_sandbox_blocks_unsafe_os_subprocess_open_import() -> None:
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"),
        _RLM_MODULES,
    )
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    repl = repl_cls("context")

    unsafe_snippets = (
        "import os\nos.system('echo x')",
        "import subprocess",
        "open('/etc/passwd', 'r')",
        "__import__('os')",
        "().__class__.__base__.__subclasses__()",
    )
    for snippet in unsafe_snippets:
        _assert_rejected(
            repl.execute,
            snippet,
            extra_excs=(PermissionError, ImportError, AttributeError, NameError, sheaf_err),
        )
