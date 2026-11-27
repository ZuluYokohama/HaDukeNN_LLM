"""Integration tests for Milestone M2 Snapdragon X Plus & Hexagon HTP v73 hardware paths."""

from __future__ import annotations

import json
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hadukenn_core import heads, learned_action_bridge, local_e2e_runner, npu_qnn_backend, tri_machine
from hadukenn_core.fast_ops import detect_oryon_capabilities, get_fast_head_lib
from hadukenn_core.hst_foundation_lm import (
    STAGED_FOUNDATION_LADDER,
    _eval_exact_target_byte_bpb,
    build_foundation_corpus,
    run_mup_width_transfer_probe,
    verify_oryon_neon_tiled_cce,
)
from hadukenn_core.learned_action_bridge import (
    ENTITIES,
    OPS,
    QUANTITIES,
    LearnedSheafHypergraphBridge,
    QuantizedLoopedBridgeNPU,
    execute_predicted_episode,
    generate_relational_episodes,
)
from hadukenn_core.local_e2e_runner import run_local_e2e
from hadukenn_core.snapdragon_local import SnapdragonXPlusSpec, plan_hst_foundation_snapdragon
from hadukenn_core.tokenization import ByteBPETokenizer
from hadukenn_core.transformer_lm import CausalTransformerLM, TransformerConfig
from sheaf_verify.campaign import verify_t46_snapdragon_local
from sheaf_verify.hst_standard import HSTTokenizer


def _train_small_bridge(seed: int = 13) -> tuple[ByteBPETokenizer, LearnedSheafHypergraphBridge, list[Any]]:
    rng = np.random.default_rng(seed)
    tr_eps = generate_relational_episodes(rng, n_episodes=32, ood_long_horizon=False)
    ood_eps = generate_relational_episodes(rng, n_episodes=8, ood_long_horizon=True)
    tok = ByteBPETokenizer.train([st.text for ep in tr_eps for st in ep.steps], vocab_size=280, min_pair_count=2)
    bridge = LearnedSheafHypergraphBridge(tok.vocab_size, d_model=64, loop_steps=2, seed=seed)
    bridge.train_policy(tok, tr_eps, epochs=95, lr=0.045)
    return tok, bridge, ood_eps


def _measure_action_disagreement(
    fp_model: LearnedSheafHypergraphBridge,
    npu: QuantizedLoopedBridgeNPU,
    tok: ByteBPETokenizer,
    ood_eps: list[Any],
) -> float:
    total = 0
    agree = 0
    for ep in ood_eps:
        for st in ep.steps:
            tids = tok.encode(st.text)
            x_feat = fp_model._featurize(st.text, tids, st.sheaf_hint_vec)[None, :]
            _, (lo, ls, ld, lq) = fp_model.forward_batch(x_feat)
            fp_pred = (int(np.argmax(lo)), int(np.argmax(ls)), int(np.argmax(ld)), int(np.argmax(lq)))
            q_pred = npu.predict_step(st.text, tids, st.sheaf_hint_vec)
            total += 1
            if fp_pred == q_pred:
                agree += 1
    return 1.0 - float(agree) / float(max(1, total))


def test_quantized_looped_bridge_npu_backends_and_teardown() -> None:
    tok, bridge, ood_eps = _train_small_bridge(seed=13)
    has_i8mm = bool(detect_oryon_capabilities().get("has_i8mm"))
    qnn_test_backend = "oryon_i8mm" if has_i8mm else "auto"

    for bad_backend in ("invalid_tier", "", 42):
        with pytest.raises(ValueError, match="Unsupported backend"):
            QuantizedLoopedBridgeNPU(bridge, backend=bad_backend)  # type: ignore[arg-type]
    for bad_dcvs in ("turbo", "", None):
        with pytest.raises(ValueError, match="Unsupported dcvs_mode"):
            QuantizedLoopedBridgeNPU(bridge, backend="numpy", dcvs_mode=bad_dcvs)  # type: ignore[arg-type]

    loop_backends = ("numpy", "hal_c11", "oryon_i8mm", "auto") if has_i8mm else ("numpy", "hal_c11", "auto")
    for backend in loop_backends:
        dcvs = "power_saver" if backend in ("oryon_i8mm", "auto") else "burst"
        with QuantizedLoopedBridgeNPU(bridge, backend=backend, dcvs_mode=dcvs) as npu:
            assert npu.closed is False
            expected_bytes = npu.w_in_q.nbytes + npu.w_loop_q.nbytes + npu.w_heads_q.nbytes + 4096
            assert npu.arena_bytes() == expected_bytes
            assert npu.arena_bytes() < 64 * 1024 * 1024
            assert npu.w_heads_q.shape == (
                bridge.d_model,
                len(OPS) + 2 * len(ENTITIES) + len(QUANTITIES),
            )
            assert npu.w_op_q.shape == (bridge.d_model, len(OPS))
            assert npu.w_src_q.shape == (bridge.d_model, len(ENTITIES))
            assert npu.w_dst_q.shape == (bridge.d_model, len(ENTITIES))
            assert npu.w_qty_q.shape == (bridge.d_model, len(QUANTITIES))

            drift = _measure_action_disagreement(bridge, npu, tok, ood_eps)
            assert drift <= 0.02, f"backend={backend} drift={drift}"

            ok_state, ok_spoof, ok_replay = execute_predicted_episode(npu, tok, ood_eps[0])
            assert isinstance(ok_state, bool)
            assert ok_spoof is True
            assert ok_replay is True

            zero_pred = npu.predict_step("", np.zeros((0,), dtype=np.int64), (0.0,) * 8)
            assert len(zero_pred) == 4
            assert npu.last_canary_ok is True

        assert npu.closed is True
        npu.close()
        assert npu.closed is True
        with pytest.raises(RuntimeError, match="closed"):
            npu.predict_step(ood_eps[0].steps[0].text, tok.encode(ood_eps[0].steps[0].text), (0.0,) * 8)

    # Exception-safe context-manager teardown on user error
    leaked_ref: QuantizedLoopedBridgeNPU | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with QuantizedLoopedBridgeNPU(bridge, backend=qnn_test_backend) as npu_err:
            leaked_ref = npu_err
            assert npu_err.closed is False
            raise RuntimeError("boom")
    assert leaked_ref is not None and leaked_ref.closed is True

    # Exception-safe __init__ cleanup if weight packing fails after HexagonQNNBackend allocation
    close_calls = 0
    orig_qnn_close = npu_qnn_backend.HexagonQNNBackend.close
    orig_hstack = np.hstack

    def _spy_qnn_close(self_qnn: Any) -> None:
        nonlocal close_calls
        close_calls += 1
        orig_qnn_close(self_qnn)

    try:
        npu_qnn_backend.HexagonQNNBackend.close = _spy_qnn_close  # type: ignore[method-assign]
        np.hstack = lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("pack_fail"))  # type: ignore[assignment]
        with pytest.raises(ValueError, match="pack_fail"):
            QuantizedLoopedBridgeNPU(bridge, backend=qnn_test_backend)
        assert close_calls >= 1
    finally:
        np.hstack = orig_hstack
        npu_qnn_backend.HexagonQNNBackend.close = orig_qnn_close  # type: ignore[method-assign]

    # Verify 4-GEMM per-step dispatch (1 input + 2 loop + 1 fused 4-head = 4 GEMMs)
    hal_calls = 0
    orig_hal = tri_machine.hal_qgemm_int8

    def _count_hal(x_q: np.ndarray, x_s: np.ndarray, w_q: np.ndarray, w_s: np.ndarray) -> np.ndarray:
        nonlocal hal_calls
        hal_calls += 1
        return orig_hal(x_q, x_s, w_q, w_s)

    orig_bridge_hal = learned_action_bridge.hal_qgemm_int8
    try:
        learned_action_bridge.hal_qgemm_int8 = _count_hal
        with QuantizedLoopedBridgeNPU(bridge, backend="hal_c11") as npu_hal:
            step0 = ood_eps[0].steps[0]
            npu_hal.predict_step(step0.text, tok.encode(step0.text), step0.sheaf_hint_vec)
        assert hal_calls == 2 + bridge.loop_steps
    finally:
        learned_action_bridge.hal_qgemm_int8 = orig_bridge_hal

    qnn_calls = 0
    orig_qgemm = npu_qnn_backend.HexagonQNNBackend.qgemm_int8_f32

    def _count_qnn(self_qnn: Any, x_q: np.ndarray, x_s: np.ndarray, w_qt: np.ndarray, w_s: np.ndarray) -> np.ndarray:
        nonlocal qnn_calls
        qnn_calls += 1
        return orig_qgemm(self_qnn, x_q, x_s, w_qt, w_s)

    try:
        npu_qnn_backend.HexagonQNNBackend.qgemm_int8_f32 = _count_qnn  # type: ignore[method-assign]
        with QuantizedLoopedBridgeNPU(bridge, backend=qnn_test_backend) as npu_qnn:
            step0 = ood_eps[0].steps[0]
            npu_qnn.predict_step(step0.text, tok.encode(step0.text), step0.sheaf_hint_vec)
        assert qnn_calls == 2 + bridge.loop_steps
    finally:
        npu_qnn_backend.HexagonQNNBackend.qgemm_int8_f32 = orig_qgemm  # type: ignore[method-assign]

    # Deterministic mocked htp_qnn verification (exercise HTP branch regardless of host)
    def _mock_htp(
        a_int8: np.ndarray,
        b_int8: np.ndarray,
        *,
        force_tier: str = "auto",
        counter: Any = None,
        dcvs_mode: str = "burst",
        canary_byte: int = 0x5A,
    ) -> dict[str, Any]:
        del force_tier, counter, dcvs_mode, canary_byte
        out = a_int8.astype(np.int32) @ b_int8.astype(np.int32)
        return {
            "output": out,
            "tier_used": "htp_qnn",
            "canary_ok": True,
            "speedup_vs_numpy": 1.25,
            "elapsed_ms": 0.1,
            "numpy_ref_ms": 0.125,
            "compile_ms": 0.0,
            "run_ms": 0.1,
            "total_wall_ms": 0.1,
            "providers": ["QNNExecutionProvider", "CPUExecutionProvider"],
        }

    orig_run_htp = npu_qnn_backend.run_int8_matmul_htp
    try:
        npu_qnn_backend.run_int8_matmul_htp = _mock_htp
        with QuantizedLoopedBridgeNPU(bridge, backend="htp_qnn", dcvs_mode="power_saver") as npu_mock_htp:
            drift_mock = _measure_action_disagreement(bridge, npu_mock_htp, tok, ood_eps)
            assert drift_mock <= 0.02
            assert npu_mock_htp.last_tier_used == "htp_qnn"
            assert npu_mock_htp.last_canary_ok is True
            assert "QNNExecutionProvider" in npu_mock_htp.last_providers
    finally:
        npu_qnn_backend.run_int8_matmul_htp = orig_run_htp

    # Missing Windows Python error propagation for strict htp_qnn
    orig_discover = npu_qnn_backend.discover_windows_python
    orig_platform = sys.platform
    try:
        npu_qnn_backend.discover_windows_python = lambda: None
        sys.platform = "linux"
        with QuantizedLoopedBridgeNPU(bridge, backend="htp_qnn") as npu_no_win:
            step0 = ood_eps[0].steps[0]
            with pytest.raises(RuntimeError, match="Windows ARM64 Python"):
                npu_no_win.predict_step(step0.text, tok.encode(step0.text), step0.sheaf_hint_vec)
    finally:
        npu_qnn_backend.discover_windows_python = orig_discover
        sys.platform = orig_platform

    # Live Windows ARM64 QNN HTP v73 verification when available
    if npu_qnn_backend.discover_windows_python() is not None:
        with QuantizedLoopedBridgeNPU(bridge, backend="htp_qnn") as npu_live_htp:
            live_drift = _measure_action_disagreement(bridge, npu_live_htp, tok, ood_eps[:2])
            assert live_drift <= 0.02
            assert npu_live_htp.last_tier_used == "htp_qnn"
            assert npu_live_htp.last_canary_ok is True
            assert "QNNExecutionProvider" in npu_live_htp.last_providers
        assert npu_live_htp.closed is True

    assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0


def test_plan_hst_foundation_snapdragon_ladder_and_boundaries() -> None:
    plan = plan_hst_foundation_snapdragon()
    assert plan["schema"] == "sheaf-verify.hst-snapdragon-plan.v1"
    assert plan["wsl2_cap_gib"] == 7.5
    assert plan["host_slab_gib"] == 13.0
    tiers = plan["tiers"]
    assert len(tiers) == len(STAGED_FOUNDATION_LADDER) == 5
    assert [t["tier_id"] for t in tiers] == [t.tier_id for t in STAGED_FOUNDATION_LADDER]

    prev_params = 0
    prev_int8 = 0.0
    prev_bf16 = 0.0
    for src_t, item in zip(STAGED_FOUNDATION_LADDER, tiers, strict=True):
        d = src_t.d_model
        exp_backbone = src_t.n_layers * (4 * d * d + 3 * d * src_t.d_ff + 2 * d) + d
        exp_embed = src_t.vocab_size * d
        assert item["backbone_params"] == exp_backbone
        assert item["embed_params"] == exp_embed
        assert item["total_params"] == exp_backbone + exp_embed
        assert 0.0 < item["nf4_cpu_gib"] < item["int8_npu_gib"] < item["bf16_train_gib"]
        assert item["total_params"] > prev_params
        assert item["int8_npu_gib"] > prev_int8
        assert item["bf16_train_gib"] > prev_bf16
        prev_params = item["total_params"]
        prev_int8 = item["int8_npu_gib"]
        prev_bf16 = item["bf16_train_gib"]

    by_id = {t["tier_id"]: t for t in tiers}
    assert by_id["HST-Core-125M"]["backbone_params"] == 84_953_856
    assert by_id["HST-Wide-3.2B"]["head_regime"] == "cluster_is_v128k"
    for small_id in ("HST-Proxy-15M", "HST-Core-125M", "HST-Edge-360M", "HST-Base-1.3B"):
        assert by_id[small_id]["head_regime"] == "tiled_cce"

    assert by_id["HST-Proxy-15M"]["fits_wsl2_cap"] is True
    assert by_id["HST-Core-125M"]["fits_wsl2_cap"] is True
    assert by_id["HST-Edge-360M"]["fits_wsl2_cap"] is False
    assert by_id["HST-Edge-360M"]["fits_host_slab"] is True
    assert by_id["HST-Base-1.3B"]["fits_wsl2_cap"] is False
    assert by_id["HST-Base-1.3B"]["fits_host_slab"] is True
    assert by_id["HST-Wide-3.2B"]["fits_wsl2_cap"] is False
    assert by_id["HST-Wide-3.2B"]["fits_host_slab"] is False
    assert by_id["HST-Wide-3.2B"]["int8_npu_fits_host_slab"] is True
    assert by_id["HST-Wide-3.2B"]["nf4_cpu_fits_host_slab"] is True

    # Boundary flip checks
    p_exact_125 = {t["tier_id"]: t for t in plan_hst_foundation_snapdragon(wsl2_cap_gib=4.8)["tiers"]}
    p_below_125 = {t["tier_id"]: t for t in plan_hst_foundation_snapdragon(wsl2_cap_gib=4.79)["tiers"]}
    assert p_exact_125["HST-Core-125M"]["fits_wsl2_cap"] is True
    assert p_below_125["HST-Core-125M"]["fits_wsl2_cap"] is False

    p_exact_13b = {t["tier_id"]: t for t in plan_hst_foundation_snapdragon(host_slab_gib=11.4)["tiers"]}
    p_below_13b = {t["tier_id"]: t for t in plan_hst_foundation_snapdragon(host_slab_gib=11.39)["tiers"]}
    assert p_exact_13b["HST-Base-1.3B"]["fits_host_slab"] is True
    assert p_below_13b["HST-Base-1.3B"]["fits_host_slab"] is False

    p_32b_fit = {t["tier_id"]: t for t in plan_hst_foundation_snapdragon(host_slab_gib=19.8)["tiers"]}
    assert p_32b_fit["HST-Wide-3.2B"]["fits_host_slab"] is True

    low_avail_spec = SnapdragonXPlusSpec(wsl2_mem_available_gib=0.1)
    p_low = plan_hst_foundation_snapdragon(spec=low_avail_spec)
    assert all(t["fits_wsl2_mem_available"] is False for t in p_low["tiers"])
    assert all(t["int8_npu_fits_wsl2_mem_available"] is False for t in p_low["tiers"])

    for bad_cap in (0.0, -2.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            plan_hst_foundation_snapdragon(wsl2_cap_gib=bad_cap)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            plan_hst_foundation_snapdragon(host_slab_gib=bad_cap)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        plan_hst_foundation_snapdragon(spec="not_a_spec")  # type: ignore[arg-type]


def test_hst_foundation_lm_oryon_neon_tiled_cce_dispatch(monkeypatch: Any) -> None:
    is_arm64 = platform.machine().lower() in ("aarch64", "arm64")
    caps = detect_oryon_capabilities()
    assert caps["native_loaded"] is True
    if is_arm64:
        assert caps["has_neon"] is True
    lib = get_fast_head_lib()
    assert lib is not None and hasattr(lib, "tiled_cce_fwd_bwd_f32")

    calls: list[tuple[str, str, bool, bool, bool]] = []
    orig_native_cce = heads.run_tiled_cce_avx512

    def _spy_native_cce(
        H: np.ndarray,
        E: np.ndarray,
        targets: np.ndarray,
        tile_V: int = 512,
        compute_grad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        calls.append((str(H.dtype), str(E.dtype), bool(H.flags.c_contiguous), bool(E.flags.c_contiguous), compute_grad))
        return orig_native_cce(H, E, targets, tile_V=tile_V, compute_grad=compute_grad)

    monkeypatch.setattr(heads, "run_tiled_cce_avx512", _spy_native_cce)

    res = verify_oryon_neon_tiled_cce(seed=13)
    assert res["native_loaded"] is True
    if is_arm64:
        assert res["has_neon"] is True
    assert res["tiled_cce_native_symbol"] == "tiled_cce_fwd_bwd_f32"
    assert res["tiled_cce_parity_ok"] is True
    assert res["loss_max_abs_err"] < 1e-4
    assert res["dh_max_abs_err"] < 1e-4
    assert res["de_max_abs_err"] < 1e-4
    assert len(calls) >= 3
    assert any(c[4] is True for c in calls)
    assert any(c[4] is False for c in calls)
    assert all(c[0] == "float32" and c[1] == "float32" and c[2] and c[3] for c in calls)

    # Stress-test float64 bottleneck_proj in _eval_exact_target_byte_bpb: must coerce to float32 & hit native CCE
    calls.clear()
    cfg = TransformerConfig(
        vocab_size=256, d_model=32, n_heads=4, n_layers=1, d_ff=64, window_len=12, head_kind="tiled_cce"
    )
    lm = CausalTransformerLM(cfg, seed=17)
    rng = np.random.default_rng(17)
    proj_f64 = rng.normal(0.0, 0.2, size=(32, 8)).astype(np.float64)
    eval_ids = rng.integers(0, 256, size=(28,), dtype=np.int64).tolist()
    bpb = _eval_exact_target_byte_bpb(lm, eval_ids, np.ones(256, dtype=np.int32), bottleneck_proj=proj_f64)
    assert np.isfinite(bpb) and bpb > 0.0
    assert len(calls) == 1
    assert calls[0][:4] == ("float32", "float32", True, True)

    # Verify muP width transfer probe drives native Oryon NEON tiled CCE
    calls.clear()
    tr_docs, te_docs, _, _ = build_foundation_corpus(seed=13)
    tok = HSTTokenizer.train(tr_docs[:24], vocab_size=320, stage_a_fraction=0.65, min_pair_count=2)
    probe = run_mup_width_transfer_probe(tok, "".join(tr_docs[:24]), "".join(te_docs[:8]), seed=13)
    assert probe["base_width_d"] == 32
    assert len(calls) > 0
    assert all(c[0] == "float32" and c[1] == "float32" for c in calls)


def test_local_e2e_runner_hst_foundation_and_schema_invariance(tmp_path: Path) -> None:
    if not detect_oryon_capabilities().get("has_i8mm"):
        pytest.skip("requires ARM64 Oryon i8mm CPU")
    rep_default = run_local_e2e()
    assert "hst_foundation" not in rep_default
    ok_def, crit_def, _ = verify_t46_snapdragon_local(rep_default)
    assert ok_def is True, [c.as_dict() for c in crit_def]
    default_keys = set(rep_default.keys())

    out_py = tmp_path / "t46_hst_py.json"
    rep_hst = run_local_e2e(json_out=out_py, include_hst_foundation=True)
    assert out_py.is_file()
    assert set(rep_hst.keys()) == default_keys | {"hst_foundation"}
    hst_sec = rep_hst["hst_foundation"]
    assert hst_sec["schema"] == "sheaf-verify.hst-snapdragon-integration.v1"
    assert [t["tier_id"] for t in hst_sec["ladder_plan"]["tiers"]] == [
        t.tier_id for t in STAGED_FOUNDATION_LADDER
    ]
    assert hst_sec["oryon_neon_tiled_cce"]["tiled_cce_parity_ok"] is True
    ok_hst, crit_hst, _ = verify_t46_snapdragon_local(rep_hst)
    assert ok_hst is True, [c.as_dict() for c in crit_hst]

    # Tampering with hst_foundation must be caught by manifest_sha256
    tampered = json.loads(out_py.read_text(encoding="utf-8"))
    tampered["hst_foundation"]["dcvs_mode"] = "tampered"
    ok_tamper, crit_tamper, _ = verify_t46_snapdragon_local(tampered)
    assert ok_tamper is False
    assert any(c.id == "T46_MANIFEST_SHA256_AND_HYGIENE" and not c.passed for c in crit_tamper)

    out_cli = tmp_path / "t46_hst_cli.json"
    rc = local_e2e_runner.main(["--include-hst-foundation", "--json-out", str(out_cli)])
    assert rc == 0
    cli_doc = json.loads(out_cli.read_text(encoding="utf-8"))
    assert "hst_foundation" in cli_doc
    assert verify_t46_snapdragon_local(cli_doc)[0] is True
    assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0


def test_quantized_looped_bridge_small_regime_drift_and_gemm_count() -> None:
    rng = np.random.default_rng(21)
    tr_eps = generate_relational_episodes(rng, n_episodes=20, ood_long_horizon=False)
    ood_eps = generate_relational_episodes(rng, n_episodes=6, ood_long_horizon=True)
    tok = ByteBPETokenizer.train([st.text for ep in tr_eps for st in ep.steps], vocab_size=280, min_pair_count=2)
    bridge = LearnedSheafHypergraphBridge(tok.vocab_size, d_model=64, loop_steps=2, seed=21)
    bridge.train_policy(tok, tr_eps, epochs=80, lr=0.045)

    has_i8mm = bool(detect_oryon_capabilities().get("has_i8mm"))
    small_backends = ("numpy", "hal_c11", "oryon_i8mm", "auto") if has_i8mm else ("numpy", "hal_c11", "auto")
    for backend in small_backends:
        with QuantizedLoopedBridgeNPU(bridge, backend=backend) as npu:
            drift = _measure_action_disagreement(bridge, npu, tok, ood_eps)
            assert drift <= 0.02, f"backend={backend} drift={drift}"
            ok_state, ok_spoof, ok_replay = execute_predicted_episode(npu, tok, ood_eps[0])
            assert isinstance(ok_state, bool)
            assert ok_spoof is True
            assert ok_replay is True
        assert npu.closed is True

    hal_calls = 0
    orig_hal = tri_machine.hal_qgemm_int8
    orig_bridge_hal = learned_action_bridge.hal_qgemm_int8

    def _count_hal(x_q: np.ndarray, x_s: np.ndarray, w_q: np.ndarray, w_s: np.ndarray) -> np.ndarray:
        nonlocal hal_calls
        hal_calls += 1
        return orig_hal(x_q, x_s, w_q, w_s)

    try:
        learned_action_bridge.hal_qgemm_int8 = _count_hal
        with QuantizedLoopedBridgeNPU(bridge, backend="hal_c11") as npu_hal:
            step0 = ood_eps[0].steps[0]
            npu_hal.predict_step(step0.text, tok.encode(step0.text), step0.sheaf_hint_vec)
        assert hal_calls == 2 + bridge.loop_steps
        assert hal_calls == 4
    finally:
        learned_action_bridge.hal_qgemm_int8 = orig_bridge_hal

    qnn_calls = 0
    orig_qgemm = npu_qnn_backend.HexagonQNNBackend.qgemm_int8_f32

    def _count_qnn(self_qnn: Any, x_q: np.ndarray, x_s: np.ndarray, w_qt: np.ndarray, w_s: np.ndarray) -> np.ndarray:
        nonlocal qnn_calls
        qnn_calls += 1
        return orig_qgemm(self_qnn, x_q, x_s, w_qt, w_s)

    try:
        npu_qnn_backend.HexagonQNNBackend.qgemm_int8_f32 = _count_qnn  # type: ignore[method-assign]
        qnn_pair = ("oryon_i8mm", "auto") if has_i8mm else ("auto",)
        for qnn_backend in qnn_pair:
            qnn_calls = 0
            with QuantizedLoopedBridgeNPU(bridge, backend=qnn_backend) as npu_qnn:
                step0 = ood_eps[0].steps[0]
                npu_qnn.predict_step(step0.text, tok.encode(step0.text), step0.sheaf_hint_vec)
            assert qnn_calls == 2 + bridge.loop_steps
            assert qnn_calls == 4
    finally:
        npu_qnn_backend.HexagonQNNBackend.qgemm_int8_f32 = orig_qgemm  # type: ignore[method-assign]

    assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0


def test_quantized_looped_bridge_concurrent_htp_session_no_worker_leak(monkeypatch: Any) -> None:
    bridge = LearnedSheafHypergraphBridge(vocab_size=260, d_model=16, loop_steps=2, seed=13)
    tok_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    sheaf_vec = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)

    # 1. Live Win32 HTP QNN concurrency check when Windows ARM64 Python is available
    if npu_qnn_backend.discover_windows_python() is not None:
        live_workers_created = 0
        orig_worker_init = npu_qnn_backend.PersistentWinWorker.__init__

        def _counting_worker_init(self_w: Any, win_python: str, *, startup_timeout_s: float = 30.0) -> None:
            nonlocal live_workers_created
            live_workers_created += 1
            orig_worker_init(self_w, win_python, startup_timeout_s=startup_timeout_s)

        try:
            npu_qnn_backend.PersistentWinWorker.__init__ = _counting_worker_init  # type: ignore[method-assign]
            with QuantizedLoopedBridgeNPU(bridge, backend="htp_qnn", dcvs_mode="burst") as n_live:
                barrier = threading.Barrier(4)

                def _run_live(_idx: int) -> tuple[int, int, int, int]:
                    barrier.wait(timeout=15.0)
                    return n_live.predict_step("move 5 litres from vessel_a to vessel_b", tok_ids, sheaf_vec)

                with ThreadPoolExecutor(max_workers=4) as pool:
                    live_preds = list(pool.map(_run_live, range(4)))
                assert len(set(live_preds)) == 1
                assert live_workers_created == 1
                assert len(npu_qnn_backend._ACTIVE_WORKERS) == 1
            assert n_live.closed is True
            assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0
        finally:
            npu_qnn_backend.PersistentWinWorker.__init__ = orig_worker_init  # type: ignore[method-assign]

    # 2. Deterministic mocked PersistentWinWorker race check (exercises _session_lock + _WORKER_LOCK)
    mock_workers_created = 0
    count_lock = threading.Lock()

    class _FakeProc:
        returncode: int | None = None

        @staticmethod
        def poll() -> int | None:
            return None

    class _FakePersistentWorker:
        def __init__(self, win_python: str, *, startup_timeout_s: float = 30.0) -> None:
            nonlocal mock_workers_created
            del startup_timeout_s
            self.win_python = win_python
            self._io_lock = threading.Lock()
            self._closed = False
            self._proc = _FakeProc()
            time.sleep(0.02)
            with count_lock:
                mock_workers_created += 1
            with npu_qnn_backend._WORKER_LOCK:
                npu_qnn_backend._ACTIVE_WORKERS.append(self)  # type: ignore[arg-type]

        def call_int8_matmul(
            self,
            a_int8: np.ndarray,
            b_int8: np.ndarray,
            *,
            dcvs_mode: str = "burst",
            canary_byte: int = 0x5A,
            timeout_s: float = 120.0,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            del dcvs_mode, canary_byte, timeout_s
            with self._io_lock:
                if self._closed:
                    raise RuntimeError("PersistentWinWorker is closed")
                out = a_int8.astype(np.int32) @ b_int8.astype(np.int32)
                return out, {
                    "canary_ok": True,
                    "compile_ms": 0.0,
                    "run_ms": 0.1,
                    "providers": ["QNNExecutionProvider"],
                    "tile_passes": 1,
                }

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            with npu_qnn_backend._WORKER_LOCK:
                try:
                    npu_qnn_backend._ACTIVE_WORKERS.remove(self)  # type: ignore[arg-type]
                except ValueError:
                    pass

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(npu_qnn_backend, "discover_windows_python", lambda: "/fake/win/python.exe")
    monkeypatch.setattr(npu_qnn_backend, "PersistentWinWorker", _FakePersistentWorker)

    with QuantizedLoopedBridgeNPU(bridge, backend="htp_qnn", dcvs_mode="burst") as n_shared:
        barrier = threading.Barrier(4)

        def _run_mock(_idx: int) -> tuple[int, int, int, int]:
            barrier.wait(timeout=5.0)
            return n_shared.predict_step("move 5 litres from vessel_a to vessel_b", tok_ids, sheaf_vec)

        with ThreadPoolExecutor(max_workers=4) as pool:
            mock_preds = list(pool.map(_run_mock, range(4)))
        assert len(set(mock_preds)) == 1
        assert mock_workers_created == 1
        assert len(npu_qnn_backend._ACTIVE_WORKERS) == 1

    assert n_shared.closed is True
    assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0
    with pytest.raises(RuntimeError, match="closed"):
        n_shared.predict_step("move 5 litres from vessel_a to vessel_b", tok_ids, sheaf_vec)
    assert mock_workers_created == 1
    assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0


def test_quantized_looped_bridge_rejects_nonfinite_inputs_and_weights() -> None:
    bridge = LearnedSheafHypergraphBridge(vocab_size=260, d_model=16, loop_steps=1, seed=13)
    tok_ids = np.array([1, 2, 3], dtype=np.int64)
    bad_vecs = (
        (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (float("-inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1e39, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    has_i8mm = bool(detect_oryon_capabilities().get("has_i8mm"))
    nonfinite_backends = ("numpy", "hal_c11", "oryon_i8mm") if has_i8mm else ("numpy", "hal_c11", "auto")
    for backend in nonfinite_backends:
        with QuantizedLoopedBridgeNPU(bridge, backend=backend) as npu:
            for bad_vec in bad_vecs:
                with pytest.raises(ValueError, match="non-finite"):
                    npu.predict_step("init vessel_a holds 5 litres", tok_ids, bad_vec)

    for attr_name in ("w_in", "w_loop", "w_op", "w_src", "w_dst", "w_qty"):
        for bad_val in (float("nan"), float("inf")):
            bad_bridge = LearnedSheafHypergraphBridge(vocab_size=260, d_model=16, loop_steps=1, seed=13)
            arr = getattr(bad_bridge, attr_name)
            arr[0, 0] = bad_val
            with pytest.raises(ValueError, match="non-finite"):
                QuantizedLoopedBridgeNPU(bad_bridge, backend="numpy")


def test_quantized_looped_bridge_concurrent_close_race_raises_only_closed(monkeypatch: Any) -> None:
    bridge = LearnedSheafHypergraphBridge(vocab_size=260, d_model=16, loop_steps=2, seed=13)
    tok_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    sheaf_vec = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)

    class _FastFakeQNN:
        def __init__(self, force_tier: str | None = "auto", dcvs_mode: str = "burst") -> None:
            del force_tier, dcvs_mode
            self._closed = False
            self.last_tier_used = "htp_qnn"
            self.last_canary_ok = True
            self.last_providers = ["QNNExecutionProvider"]

        def qgemm_int8_f32(
            self, x_q: np.ndarray, x_s: np.ndarray, w_q_t: np.ndarray, w_s: np.ndarray
        ) -> np.ndarray:
            if self._closed:
                raise RuntimeError("HexagonQNNBackend is closed")
            out = x_q.astype(np.int32) @ w_q_t.astype(np.int32)
            return out.astype(np.float32) * (x_s[:, None] * w_s[None, :])

        def close(self) -> None:
            self._closed = True

    monkeypatch.setattr(learned_action_bridge, "HexagonQNNBackend", _FastFakeQNN)

    for _round in range(25):
        npu = QuantizedLoopedBridgeNPU(bridge, backend="htp_qnn")
        barrier = threading.Barrier(4)
        unexpected: list[str] = []

        def _worker(idx: int, bridge_inst: QuantizedLoopedBridgeNPU = npu, b: threading.Barrier = barrier) -> None:
            try:
                b.wait(timeout=5.0)
                if idx == 0:
                    bridge_inst.close()
                else:
                    bridge_inst.predict_step("move 5 litres from vessel_a to vessel_b", tok_ids, sheaf_vec)
            except RuntimeError as exc:
                if "closed" not in str(exc):
                    unexpected.append(f"RuntimeError: {exc}")
            except Exception as exc:  # noqa: BLE001
                unexpected.append(f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_worker, range(4)))
        assert unexpected == []
        assert npu.closed is True
