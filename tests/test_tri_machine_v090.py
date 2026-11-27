"""Unit, SIMD parity, arena-cap, NPU-loop, watchdog, and CLI tests for v0.9.0 Tri-Machine HAL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hadukenn_core.tri_machine import (
    GIB,
    LAPBEAST_PROFILE,
    PROFILES,
    WBEAST_PROFILE,
    ZENBOOK_PROFILE,
    LapBeastPowerGuard,
    StaticSlabArena,
    ZenbookLoopedNPUEngine,
    apply_thread_affinity,
    detect_host_isa,
    detect_machine_profile,
    get_tri_machine_lib,
    hal_qgemm_int8,
    hal_scan_causal_spans,
    hal_sheaf_kv_prefix_gate,
    hal_tiled_cce_mup,
    quantize_symmetric_int8,
    run_tri_machine_verification_suite,
)
from sheaf_verify.campaign import verify_t46_tri_machine_hal
from sheaf_verify.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]


def test_c11_hal_compiles_and_detects_isa():
    lib = get_tri_machine_lib()
    assert lib is not None
    isa = detect_host_isa()
    assert isa["backend"] != "python_numpy_fallback"
    assert "SCALAR" in isa["flags"]
    assert set(PROFILES.keys()) == {"wbeast", "lapbeast", "zenbook"}
    assert ZENBOOK_PROFILE.ram_arena_cap_bytes == 13 * GIB


def test_machine_profile_overrides_and_affinity():
    assert detect_machine_profile("wbeast") == WBEAST_PROFILE
    assert detect_machine_profile("lapbeast") == LAPBEAST_PROFILE
    assert detect_machine_profile("zenbook") == ZENBOOK_PROFILE
    aff_p = apply_thread_affinity(LAPBEAST_PROFILE, role="compute")
    aff_e = apply_thread_affinity(LAPBEAST_PROFILE, role="audit_io")
    assert aff_p["requested_cores"] == list(LAPBEAST_PROFILE.p_cores)
    assert aff_e["requested_cores"] == list(LAPBEAST_PROFILE.e_cores)


def test_static_slab_arena_64byte_alignment_and_ceiling():
    with StaticSlabArena(8192) as arena:
        p1 = arena.alloc_bytes(100)
        p2 = arena.alloc_bytes(333)
        assert p1 % 64 == 0
        assert p2 % 64 == 0
        arr = arena.alloc_f32_view((16, 16))
        assert arr.shape == (16, 16)
        assert arena.stats()["alloc_count"] == 3
        with pytest.raises(MemoryError):
            arena.alloc_bytes(16384)
        assert arena.stats()["overflow_count"] == 1
        arena.reset()
        assert arena.stats()["offset_bytes"] == 0


def test_int8_sdot_vnni_qgemm_matches_exact_int32_reference():
    rng = np.random.default_rng(77)
    x = rng.standard_normal((19, 137)).astype(np.float32)  # non-multiple of 64 tests tail loop
    w = rng.standard_normal((29, 137)).astype(np.float32)
    xq, xs = quantize_symmetric_int8(x)
    wq, ws = quantize_symmetric_int8(w)
    out_c = hal_qgemm_int8(xq, xs, wq, ws)
    ref_i32 = xq.astype(np.int32) @ wq.astype(np.int32).T
    out_ref = ref_i32.astype(np.float32) * (xs[:, None] * ws[None, :])
    assert np.max(np.abs(out_c - out_ref)) < 1e-4


def test_int8_qgemm_extreme_values_do_not_saturate_int16():
    """vpmaddubsw saturates after the unsigned-bias transform; 127*127 vectors must stay exact."""
    xq = np.full((4, 64), 127, dtype=np.int8)
    wq = np.full((5, 64), 127, dtype=np.int8)
    xs = np.ones(4, dtype=np.float32)
    ws = np.ones(5, dtype=np.float32)
    out_c = hal_qgemm_int8(xq, xs, wq, ws)
    out_ref = (xq.astype(np.int32) @ wq.astype(np.int32).T).astype(np.float32)
    assert np.max(np.abs(out_c - out_ref)) < 1e-4
    assert int(out_c[0, 0]) == 127 * 127 * 64


def test_mup_cce_and_sheaf_gate_and_causal_spans():
    rng = np.random.default_rng(91)
    H = rng.standard_normal((33, 192)).astype(np.float32)
    E = (rng.standard_normal((521, 192)) * 0.08).astype(np.float32)
    y = rng.integers(0, 521, size=33, dtype=np.int32)
    loss, dH, dE = hal_tiled_cce_mup(H, E, y, tile_V=128, mup_base_d=384, compute_grad=True)
    assert loss.shape == (33,)
    assert dH is not None and dH.shape == H.shape
    assert dE is not None and dE.shape == E.shape

    q = rng.standard_normal((17, 48)).astype(np.float32)
    ns = rng.standard_normal((17, 6, 48)).astype(np.float32)
    nv = rng.standard_normal((17, 6, 96)).astype(np.float32)
    pref, energy = hal_sheaf_kv_prefix_gate(q, ns, nv, beta=0.75)
    assert pref.shape == (17, 96)
    assert np.all(energy >= 0.0)

    p_bytes = b"sheaf-verify causal byte span test prefix 0123456789"
    s1 = [t for t in hal_scan_causal_spans(p_bytes + b" suffix_one") if t[1] <= len(p_bytes)]
    s2 = [t for t in hal_scan_causal_spans(p_bytes + b" radically_different_suffix_two") if t[1] <= len(p_bytes)]
    assert s1 == s2 and len(s1) > 0


def test_zenbook_looped_npu_zero_alloc_and_lapbeast_watchdog():
    with pytest.raises(MemoryError):
        ZenbookLoopedNPUEngine(d_model=64, d_ff=128, arena_cap_bytes=14 * GIB)

    eng = ZenbookLoopedNPUEngine(d_model=128, d_ff=384, loop_steps=4, arena_cap_bytes=8 * 1024 * 1024)
    try:
        x = np.ones((8, 128), dtype=np.float32) * 0.25
        out_q, tel = eng.run_loop(x, use_fp32_ref=False)
        out_f, _ = eng.run_loop(x, use_fp32_ref=True)
        assert tel["dynamic_slab_allocs_in_loop"] == 0
        cos = float(np.sum(out_q * out_f) / (np.linalg.norm(out_q) * np.linalg.norm(out_f)))
        assert cos > 0.995
    finally:
        eng.close()

    guard = LapBeastPowerGuard(LAPBEAST_PROFILE)
    assert guard.evaluate(19, False, 60, True).exit_code == 143
    assert guard.evaluate(50, False, 60, False).exit_code == 75
    assert guard.evaluate(50, True, 60, False).allow_launch is True
    assert LapBeastPowerGuard(WBEAST_PROFILE).evaluate(0, False, 65, True).continue_running is True


def test_t46_campaign_verifier_and_cli_roundtrip(tmp_path: Path, capsys):
    iter_file = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"
    if iter_file.is_file():
        doc = json.loads(iter_file.read_text(encoding="utf-8"))
    else:
        doc = run_tri_machine_verification_suite()
        doc["regulatory_compliance"].pop("bundle", None)

    ok_flag, criteria, advisories = verify_t46_tri_machine_hal(doc)
    assert ok_flag is True
    assert all(c.passed for c in criteria)
    assert any("TRI_MACHINE_HAL_READY" in a for a in advisories)

    out_json = tmp_path / "t46_ok.json"
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rc = cli_main(["campaign", "--json", "--mode", "t46-tri-machine-hal", "--path", str(out_json)])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True

    # Negative check: corrupt Zenbook RAM ceiling and dynamic alloc count
    bad_doc = json.loads(json.dumps(doc))
    bad_doc["profiles"]["zenbook"]["ram_arena_cap_gib"] = 16.0
    bad_doc["benchmarks_and_parity"]["zenbook_npu_loop_dynamic_allocs"] = 5
    bad_json = tmp_path / "t46_bad.json"
    bad_json.write_text(json.dumps(bad_doc, indent=2) + "\n", encoding="utf-8")
    rc_bad = cli_main(["campaign", "--json", "--mode", "t46-tri-machine-hal", "--path", str(bad_json)])
    assert rc_bad == 1
    _ = capsys.readouterr()
