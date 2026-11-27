"""Unit, parity, concurrency, and speedup tests for M2 Oryon SIMD kernels (F4-F9)."""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pytest

from hadukenn_core.fast_ops import (
    NF4_CODEBOOK,
    OpCounter,
    benchmark_kernel_parity,
    detect_oryon_capabilities,
    get_fast_head_lib,
    run_axpy_f32,
    run_cluster_is_avx512,
    run_dot_f32,
    run_int8_gemm,
    run_nf4_matvec,
    run_tiled_cce_avx512,
)
from hadukenn_core.heads import flat_tied_fwd_bwd, tiled_cce_fwd_bwd
from hadukenn_core.rtx5070_frontier import NF4BlockQuantizer


def _pack_nf4_rows(
    weights: np.ndarray,
    group_size: int,
    lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack `[M, K]` float32 weights into row-wise high-nibble-first NF4 + per-group scales."""
    m, k = weights.shape
    packed_cols = (k + 1) // 2
    n_groups = (k + group_size - 1) // group_size
    thresholds = 0.5 * (lut[:-1] + lut[1:])

    packed = np.zeros((m, packed_cols), dtype=np.uint8)
    scales = np.zeros((m, n_groups), dtype=np.float32)

    for g in range(n_groups):
        c0 = g * group_size
        c1 = min(c0 + group_size, k)
        blk = weights[:, c0:c1]
        absmax = np.maximum(np.max(np.abs(blk), axis=1), 1e-12).astype(np.float32)
        scales[:, g] = absmax
        normed = blk / absmax[:, None]
        codes = np.searchsorted(thresholds, normed).astype(np.uint8)
        for offset in range(c1 - c0):
            col = c0 + offset
            p = col // 2
            if col % 2 == 0:
                packed[:, p] |= (codes[:, offset] & 0x0F) << 4
            else:
                packed[:, p] |= codes[:, offset] & 0x0F
    return packed, scales


def _reference_nf4_matvec(
    x: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    lut: np.ndarray,
) -> np.ndarray:
    m = packed.shape[0]
    k = x.shape[0]
    n_groups = scales.shape[1]
    out = np.zeros(m, dtype=np.float64)
    for row in range(m):
        acc = 0.0
        for g in range(n_groups):
            c0 = g * group_size
            c1 = min(c0 + group_size, k)
            g_acc = 0.0
            for col in range(c0, c1):
                b = int(packed[row, col // 2])
                code = (b >> 4) & 0x0F if col % 2 == 0 else b & 0x0F
                g_acc += float(lut[code]) * float(x[col])
            acc += float(scales[row, g]) * g_acc
        out[row] = acc
    return out.astype(np.float32)


def _reference_cluster_is(
    H: np.ndarray,
    E_cl: np.ndarray,
    C_root: np.ndarray,
    tc: np.ndarray,
    tm: np.ndarray,
    neg_c: np.ndarray,
    compute_grad: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    n, d = H.shape
    nc, s, _ = E_cl.shape
    k_neg = neg_c.shape[1]
    out_loss = np.zeros(n, dtype=np.float64)
    dH = np.zeros((n, d), dtype=np.float64) if compute_grad else None
    dE = np.zeros((nc, s, d), dtype=np.float64) if compute_grad else None

    for i in range(n):
        h = H[i].astype(np.float64)
        r = C_root.astype(np.float64) @ h
        r_exp = np.exp(r - np.max(r))
        r_prob = r_exp / np.sum(r_exp)

        act_c = [int(tc[i])] + [int(v) for v in neg_c[i]]
        logits = []
        for blk, c_idx in enumerate(act_c):
            log_w = 0.0
            if blk > 0:
                q_tc = r_prob[int(tc[i])]
                q_c = r_prob[c_idx]
                q_cond = q_c / max(1.0 - q_tc, 1e-7)
                denom = max(float(k_neg) * q_cond, 1e-7)
                log_w = -np.log(denom)
            blk_logits = E_cl[c_idx].astype(np.float64) @ h + log_w
            logits.append(blk_logits)
        flat_logits = np.concatenate(logits)
        e_exp = np.exp(flat_logits - np.max(flat_logits))
        sum_o = float(np.sum(e_exp))
        out_loss[i] = -np.log(max(float(e_exp[int(tm[i])]) / sum_o, 1e-30))

        if compute_grad and dH is not None and dE is not None:
            probs = e_exp / sum_o
            probs[int(tm[i])] -= 1.0
            coeffs = probs / float(n)
            for blk, c_idx in enumerate(act_c):
                blk_c = coeffs[blk * s : (blk + 1) * s]
                dH[i] += blk_c @ E_cl[c_idx].astype(np.float64)
                dE[c_idx] += blk_c[:, None] * h[None, :]

    return (
        out_loss.astype(np.float32),
        None if dH is None else dH.astype(np.float32),
        None if dE is None else dE.astype(np.float32),
    )


def test_oryon_capability_discovery_and_cc_override(monkeypatch, tmp_path: Path):
    is_arm64 = platform.machine().lower() in ("aarch64", "arm64")
    caps = detect_oryon_capabilities()
    if is_arm64:
        assert caps["arch"] == "aarch64"
        assert caps["has_neon"] is True
        assert caps["has_dotprod"] is True
        assert caps["has_i8mm"] is True
        assert "-mcpu=oryon-1" in caps["flags"]
        assert 1 <= int(caps["num_threads"]) <= 8
    else:
        assert int(caps["num_threads"]) >= 1
    assert caps["native_loaded"] is True
    assert caps["pool_ready"] is True
    assert caps["library_path"] is not None
    so_path = Path(caps["library_path"])
    assert so_path.is_file()
    assert ".cache/hadukenn_core" in so_path.as_posix()
    assert get_fast_head_lib() is not None

    # Verify CC override to a missing compiler cleanly disables native lib & exercises NumPy fallback
    monkeypatch.setenv("CC", "/nonexistent_oryon_cc_xyz")
    caps_none = detect_oryon_capabilities()
    assert caps_none["native_loaded"] is False
    assert get_fast_head_lib() is None

    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([4.0, 0.5, -1.0], dtype=np.float32)
    assert run_dot_f32(a, b) == pytest.approx(-0.5, abs=1e-5)
    assert np.allclose(run_axpy_f32(2.0, a, b), np.array([6.0, -3.5, 6.0], dtype=np.float32))
    c_auto = run_int8_gemm(np.array([[2, -3]], dtype=np.int8), np.array([[4], [5]], dtype=np.int8), mode="auto")
    assert np.array_equal(c_auto, np.array([[-7]], dtype=np.int32))
    with pytest.raises(RuntimeError):
        run_int8_gemm(np.array([[1]], dtype=np.int8), np.array([[1]], dtype=np.int8), mode="i8mm")

    # Verify a distinct wrapper CC produces a distinct cache_key / .so entry
    real_cc = caps["compiler"]
    assert real_cc is not None
    fake_bin = tmp_path / "cc_wrap.sh"
    fake_bin.write_text(f"#!/usr/bin/env sh\nexec {real_cc} \"$@\"\n", encoding="utf-8")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("CC", str(fake_bin))
    caps_wrap = detect_oryon_capabilities()
    assert caps_wrap["native_loaded"] is True
    assert caps_wrap["cache_key"] != caps["cache_key"]


def test_neon_dot_f32_and_axpy_f32_parity_and_parallel():
    rng = np.random.default_rng(101)
    ctr = OpCounter()
    lengths = (0, 1, 3, 7, 15, 16, 17, 33, 64, 576, 4099, 65537, 131075)
    for n in lengths:
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        alpha = -1.375

        dot_got = run_dot_f32(a, b, counter=ctr)
        dot_ref = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        tol = max(1e-5, 2e-5 * abs(dot_ref))
        assert abs(dot_got - dot_ref) <= tol, f"dot mismatch at n={n}: got={dot_got}, ref={dot_ref}"

        b_orig = b.copy()
        axpy_got = run_axpy_f32(alpha, a, b, counter=ctr)
        assert np.array_equal(b, b_orig), "run_axpy_f32 must not mutate input y"
        axpy_ref = (b.astype(np.float64) + alpha * a.astype(np.float64)).astype(np.float32)
        if n > 0:
            assert np.allclose(axpy_got, axpy_ref, rtol=1e-5, atol=1e-6)
        else:
            assert axpy_got.shape == (0,)

    assert ctr.calls == 2 * len(lengths)
    assert ctr.mac_ops == 2 * sum(lengths)
    assert ctr.threads_used >= 1
    assert ctr.ops_per_byte() > 0.0

    with pytest.raises(ValueError):
        run_dot_f32(np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        run_axpy_f32(1.0, np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))


def test_int8_gemm_all_modes_and_unaligned_shapes():
    rng = np.random.default_rng(202)
    shapes = (
        (0, 4, 3),
        (4, 0, 3),
        (3, 4, 0),
        (1, 1, 1),
        (1, 1, 17),
        (7, 1, 9),
        (65, 1, 33),
        (1, 17, 7),
        (3, 5, 3),
        (7, 11, 5),
        (17, 19, 13),
        (33, 65, 17),
        (64, 128, 64),
        (128, 96, 128),
    )
    caps = detect_oryon_capabilities()
    modes_list = ["auto"]
    if caps.get("has_i8mm"):
        modes_list.append("i8mm")
    if caps.get("has_dotprod"):
        modes_list.append("sdot")
    modes_list.extend(["scalar", "numpy"])
    modes = tuple(modes_list)
    ctr = OpCounter()

    for m, n, k in shapes:
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
        ref = a.astype(np.int32) @ b.astype(np.int32)
        for mode in modes:
            got = run_int8_gemm(a, b, mode=mode, counter=ctr)
            assert got.dtype == np.int32
            assert got.shape == (m, n)
            assert np.array_equal(got, ref), f"INT8 GEMM mismatch for shape={(m, n, k)}, mode={mode}"

    # Extreme signed boundaries (-128 * -128 = 16384)
    a_ext = np.full((5, 9), -128, dtype=np.int8)
    b_ext = np.full((9, 7), -128, dtype=np.int8)
    ref_ext = a_ext.astype(np.int32) @ b_ext.astype(np.int32)
    ext_modes = tuple(m for m in modes if m not in ("auto", "numpy"))
    for mode in ext_modes:
        assert np.array_equal(run_int8_gemm(a_ext, b_ext, mode=mode), ref_ext)

    assert ctr.calls == len(shapes) * len(modes)
    with pytest.raises(ValueError):
        run_int8_gemm(a_ext, b_ext, mode="invalid_mode")
    with pytest.raises(ValueError):
        run_int8_gemm(np.zeros((4, 5), dtype=np.int8), np.zeros((6, 3), dtype=np.int8))


def test_nf4_matvec_parity_and_quantizer_roundtrip():
    rng = np.random.default_rng(303)
    ctr = OpCounter()

    cases = (
        (0, 8, 4),
        (4, 0, 64),
        (1, 1, 1),
        (9, 65, 13),
        (17, 131, 32),
        (64, 256, 64),
        (128, 512, 64),
    )
    for m, k, gs in cases:
        w = (rng.standard_normal((m, k)) * 0.25).astype(np.float32)
        x = (rng.standard_normal(k) * 0.5).astype(np.float32)
        packed, scales = _pack_nf4_rows(w, gs, NF4_CODEBOOK)
        got = run_nf4_matvec(x, packed, scales, group_size=gs, counter=ctr)
        ref = _reference_nf4_matvec(x, packed, scales, gs, NF4_CODEBOOK)
        assert got.shape == (m,)
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-4), f"NF4 mismatch at {(m, k, gs)}"

    # Custom LUT verification
    custom_lut = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    w_c = (rng.standard_normal((11, 37)) * 0.8).astype(np.float32)
    x_c = rng.standard_normal(37).astype(np.float32)
    p_c, s_c = _pack_nf4_rows(w_c, 16, custom_lut)
    got_c = run_nf4_matvec(x_c, p_c, s_c, group_size=16, lut_f32=custom_lut)
    ref_c = _reference_nf4_matvec(x_c, p_c, s_c, 16, custom_lut)
    assert np.allclose(got_c, ref_c, rtol=1e-4, atol=1e-4)

    # Roundtrip parity against NF4BlockQuantizer (when K % block_size == 0)
    nf4 = NF4BlockQuantizer(block_size=64)
    w_q = (rng.standard_normal((16, 128)) * 0.2).astype(np.float32)
    x_q = rng.standard_normal(128).astype(np.float32)
    qstate = nf4.quantize(w_q)
    w_deq = nf4.dequantize(qstate)
    packed_q = qstate["packed"].reshape(16, 64)  # type: ignore[union-attr]
    q_abs = qstate["q_absmax"].reshape(16, 2)  # type: ignore[union-attr]
    scales_q = (q_abs.astype(np.float32) / 255.0) * float(qstate["super_max"])  # type: ignore[arg-type]
    got_q = run_nf4_matvec(x_q, packed_q, scales_q, group_size=64)
    ref_q = (w_deq @ x_q).astype(np.float32)
    assert np.allclose(got_q, ref_q, rtol=1e-4, atol=1e-4)

    with pytest.raises(ValueError):
        run_nf4_matvec(x_q, packed_q, scales_q, group_size=0)
    with pytest.raises(ValueError):
        run_nf4_matvec(x_q, packed_q, scales_q, group_size=64, lut_f32=np.zeros(8, dtype=np.float32))


def test_head_kernels_backwards_compat_and_parallel_grad_race_safety():
    rng = np.random.default_rng(404)

    # 1. Tiled CCE forward/backward at N=130 (>= 128 parallel path) and unaligned V=513
    n, d, v = 130, 48, 513
    H = (rng.standard_normal((n, d)) * 0.2).astype(np.float32)
    E = (rng.standard_normal((v, d)) * 0.2).astype(np.float32)
    y = rng.integers(0, v, size=n, dtype=np.int32)

    l_flat, dH_flat, dE_flat = flat_tied_fwd_bwd(H, E, y, compute_grad=True)
    l_np, _, _ = tiled_cce_fwd_bwd(H, E, y, tile_V=128, compute_grad=False, use_native=False)
    l_c_fwd, dH_none, dE_none = run_tiled_cce_avx512(H, E, y, tile_V=128, compute_grad=False)
    assert dH_none is None and dE_none is None
    assert np.max(np.abs(l_c_fwd - l_np)) < 1e-5

    runs = [run_tiled_cce_avx512(H, E, y, tile_V=128, compute_grad=True) for _ in range(3)]
    for l_c, dH_c, dE_c in runs:
        assert dH_c is not None and dE_c is not None
        assert np.max(np.abs(l_c - l_flat)) < 1e-5
        assert np.max(np.abs(dH_c - dH_flat)) < 1e-5
        assert np.max(np.abs(dE_c - dE_flat)) < 1e-5
    # Verify deterministic parallel reduction across runs (race-free)
    assert np.allclose(runs[0][2], runs[1][2], atol=1e-7)
    assert np.allclose(runs[0][2], runs[2][2], atol=1e-7)

    # 2. Cluster-IS forward/backward at N=130 and dynamic heap sizes (NC=4200 > 4096)
    nc, s = 32, 16
    E_cl = (rng.standard_normal((nc, s, d)) * 0.2).astype(np.float32)
    C_rt = E_cl.mean(axis=1).copy()
    tc = rng.integers(0, nc, size=n, dtype=np.int32)
    tm = rng.integers(0, s, size=n, dtype=np.int32)
    neg_c = np.mod(tc[:, None] + np.array([[1, 2, 3]], dtype=np.int32), nc)

    l_ref, dH_ref, dE_ref = _reference_cluster_is(H, E_cl, C_rt, tc, tm, neg_c, compute_grad=True)
    is_runs = [run_cluster_is_avx512(H, E_cl, C_rt, tc, tm, neg_c, compute_grad=True) for _ in range(3)]
    for l_is, dH_is, dE_is in is_runs:
        assert dH_is is not None and dE_is is not None
        assert np.max(np.abs(l_is - l_ref)) < 1e-5
        assert np.max(np.abs(dH_is - dH_ref)) < 1e-5
        assert np.max(np.abs(dE_is - dE_ref)) < 1e-5
    assert np.array_equal(is_runs[0][2], is_runs[1][2])
    assert np.array_equal(is_runs[0][2], is_runs[2][2])

    # Dynamic heap fallback path: NC > 4096 and 1+K_neg > 16
    nc_big, s_big, k_big = 4200, 4, 17
    H_b = (rng.standard_normal((8, 16)) * 0.1).astype(np.float32)
    E_b = (rng.standard_normal((nc_big, s_big, 16)) * 0.1).astype(np.float32)
    C_b = E_b.mean(axis=1).copy()
    tc_b = rng.integers(0, nc_big, size=8, dtype=np.int32)
    tm_b = rng.integers(0, s_big, size=8, dtype=np.int32)
    neg_b = np.mod(tc_b[:, None] + np.arange(1, k_big + 1, dtype=np.int32)[None, :], nc_big)
    l_big, dH_big, dE_big = run_cluster_is_avx512(H_b, E_b, C_b, tc_b, tm_b, neg_b, compute_grad=True)
    assert l_big.shape == (8,) and np.all(np.isfinite(l_big))
    assert dH_big is not None and np.all(np.isfinite(dH_big))
    assert dE_big is not None and np.all(np.isfinite(dE_big))


def test_op_counter_and_benchmark_kernel_parity_speedup():
    rng = np.random.default_rng(505)

    # Intentional mismatch must fail parity check BEFORE reporting speedup
    with pytest.raises(AssertionError):
        benchmark_kernel_parity(
            "bad_mismatch",
            lambda: np.ones(4, dtype=np.float32),
            lambda: np.zeros(4, dtype=np.float32),
            assert_parity=True,
        )

    bad_rep = benchmark_kernel_parity(
        "soft_mismatch",
        lambda: np.ones(4, dtype=np.float32),
        lambda: np.zeros(4, dtype=np.float32),
        assert_parity=False,
    )
    assert bad_rep["parity_ok"] is False

    caps = detect_oryon_capabilities()

    # Live INT8 i8mm GEMM vs C scalar baseline on (256, 256) @ (256, 256)
    a_i8 = rng.integers(-64, 64, size=(256, 256), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(256, 256), dtype=np.int8)
    gemm_ctr = OpCounter()
    if caps.get("has_i8mm"):
        rep_gemm = benchmark_kernel_parity(
            "oryon_i8mm_vs_scalar",
            lambda: run_int8_gemm(a_i8, b_i8, mode="i8mm", counter=gemm_ctr),
            lambda: run_int8_gemm(a_i8, b_i8, mode="scalar"),
            warmup=1,
            iters=3,
            rtol=0.0,
            atol=0.0,
            counter=gemm_ctr,
        )
        assert rep_gemm["speedup_x"] > 1.0
    else:
        rep_gemm = benchmark_kernel_parity(
            "auto_int8_vs_numpy",
            lambda: run_int8_gemm(a_i8, b_i8, mode="auto", counter=gemm_ctr),
            lambda: run_int8_gemm(a_i8, b_i8, mode="numpy"),
            warmup=1,
            iters=3,
            rtol=0.0,
            atol=0.0,
            counter=gemm_ctr,
        )
    assert rep_gemm["parity_ok"] is True
    assert rep_gemm["max_abs_err"] == 0.0
    assert gemm_ctr.mac_ops > 0

    # Live fused NF4 MatVec vs pure NumPy dequantize+matmul on (512, 1024)
    w_f32 = (rng.standard_normal((512, 1024)) * 0.25).astype(np.float32)
    x_f32 = rng.standard_normal(1024).astype(np.float32)
    packed, scales = _pack_nf4_rows(w_f32, 64, NF4_CODEBOOK)

    def _numpy_nf4_ref() -> np.ndarray:
        hi = (packed >> 4) & 0x0F
        lo = packed & 0x0F
        codes = np.empty((512, 1024), dtype=np.uint8)
        codes[:, 0::2] = hi
        codes[:, 1::2] = lo
        w_deq = NF4_CODEBOOK[codes].reshape(512, 16, 64) * scales[:, :, None]
        return (w_deq.reshape(512, 1024) @ x_f32).astype(np.float32)

    nf4_ctr = OpCounter()
    rep_nf4 = benchmark_kernel_parity(
        "oryon_nf4_matvec_vs_numpy",
        lambda: run_nf4_matvec(x_f32, packed, scales, group_size=64, counter=nf4_ctr),
        _numpy_nf4_ref,
        warmup=1,
        iters=5,
        rtol=1e-4,
        atol=1e-4,
        counter=nf4_ctr,
    )
    assert rep_nf4["parity_ok"] is True
    if caps.get("has_neon"):
        assert rep_nf4["speedup_x"] > 1.0
    assert nf4_ctr.lut_lookups > 0


def test_zero_k_and_head_kernel_validation_regressions(monkeypatch):
    # 1. K == 0 with M > 0, N > 0 returns exact zeros across repeated calls for INT8 GEMM & NF4 MatVec
    caps = detect_oryon_capabilities()
    a0 = np.empty((4, 0), dtype=np.int8)
    b0 = np.empty((0, 5), dtype=np.int8)
    expected_gemm = np.zeros((4, 5), dtype=np.int32)
    zero_modes = ["auto"]
    if caps.get("has_i8mm"):
        zero_modes.append("i8mm")
    if caps.get("has_dotprod"):
        zero_modes.append("sdot")
    zero_modes.extend(["scalar", "numpy"])
    for mode in zero_modes:
        for _ in range(3):
            got = run_int8_gemm(a0, b0, mode=mode)
            assert got.dtype == np.int32
            assert got.shape == (4, 5)
            assert np.array_equal(got, expected_gemm)

    x0 = np.empty(0, dtype=np.float32)
    qw0 = np.empty((4, 0), dtype=np.uint8)
    sc0 = np.empty((4, 0), dtype=np.float32)
    expected_nf4 = np.zeros(4, dtype=np.float32)
    for _ in range(3):
        got_nf4 = run_nf4_matvec(x0, qw0, sc0, group_size=64)
        assert got_nf4.dtype == np.float32
        assert got_nf4.shape == (4,)
        assert np.all(np.isfinite(got_nf4))
        assert np.array_equal(got_nf4, expected_nf4)

    # 2. run_tiled_cce_avx512 validation & N == 0 handling
    H_ok = np.zeros((4, 8), dtype=np.float32)
    E_ok = np.zeros((10, 8), dtype=np.float32)
    y_ok = np.array([0, 1, 2, 3], dtype=np.int32)

    # Mismatched ndims / shapes
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(np.zeros((4,), dtype=np.float32), E_ok, y_ok)
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, np.zeros((10, 8, 1), dtype=np.float32), y_ok)
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, E_ok, np.zeros((4, 1), dtype=np.int32))
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, np.zeros((10, 12), dtype=np.float32), y_ok)
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, E_ok, np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, np.zeros((0, 8), dtype=np.float32), y_ok)

    # Out-of-bounds target indices (including negative, >= V, and int64 overflow)
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, E_ok, np.array([0, 1, -1, 2], dtype=np.int32))
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, E_ok, np.array([0, 1, 10, 2], dtype=np.int32))
    with pytest.raises(ValueError):
        run_tiled_cce_avx512(H_ok, E_ok, np.array([0, 1, 2**32, 2], dtype=np.int64))

    # Clean N == 0 returns
    l0, dh0, de0 = run_tiled_cce_avx512(
        np.zeros((0, 8), dtype=np.float32),
        E_ok,
        np.zeros((0,), dtype=np.int32),
        compute_grad=False,
    )
    assert l0.shape == (0,) and dh0 is None and de0 is None
    l0g, dh0g, de0g = run_tiled_cce_avx512(
        np.zeros((0, 8), dtype=np.float32),
        E_ok,
        np.zeros((0,), dtype=np.int32),
        compute_grad=True,
    )
    assert l0g.shape == (0,) and dh0g is not None and dh0g.shape == (0, 8)
    assert de0g is not None and de0g.shape == (10, 8) and np.array_equal(de0g, np.zeros((10, 8), dtype=np.float32))

    # 3. run_cluster_is_avx512 validation & N == 0 / K_neg == 0 handling
    NC, S, d, K_neg = 5, 4, 8, 2
    E_cl = np.zeros((NC, S, d), dtype=np.float32)
    C_rt = np.zeros((NC, d), dtype=np.float32)
    tc_ok = np.array([0, 1, 2, 3], dtype=np.int32)
    tm_ok = np.array([0, 1, 2, 3], dtype=np.int32)
    neg_ok = np.array([[1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int32)

    # Mismatched ndims / shapes
    with pytest.raises(ValueError):
        run_cluster_is_avx512(np.zeros((4,), dtype=np.float32), E_cl, C_rt, tc_ok, tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, np.zeros((NC, S), dtype=np.float32), C_rt, tc_ok, tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, np.zeros((NC,), dtype=np.float32), tc_ok, tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, np.zeros((2, d), dtype=np.float32), tc_ok, tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, np.zeros((NC, S, 12), dtype=np.float32), C_rt, tc_ok, tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, np.zeros((4, 1), dtype=np.int32), tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, np.zeros(3, dtype=np.int32), neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, tm_ok, np.zeros((3, K_neg), dtype=np.int32))
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, tm_ok, np.zeros((4,), dtype=np.int32))

    # Out-of-bounds target_c, target_m, neg_clusters
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, np.array([0, -1, 1, 2], dtype=np.int32), tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, np.array([0, NC, 1, 2], dtype=np.int32), tm_ok, neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, np.array([0, -1, 1, 2], dtype=np.int32), neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, np.array([0, S, 1, 2], dtype=np.int32), neg_ok)
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, tm_ok, np.array([[1, -1], [2, 3], [3, 4], [4, 0]], dtype=np.int32))
    with pytest.raises(ValueError):
        run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, tm_ok, np.array([[1, NC], [2, 3], [3, 4], [4, 0]], dtype=np.int32))

    # Clean K_neg == 0 and N == 0
    l_k0, dh_k0, de_k0 = run_cluster_is_avx512(
        H_ok, E_cl, C_rt, tc_ok, tm_ok, np.zeros((4, 0), dtype=np.int32), compute_grad=True
    )
    assert l_k0.shape == (4,) and np.all(np.isfinite(l_k0))
    assert dh_k0 is not None and dh_k0.shape == (4, d)
    assert de_k0 is not None and de_k0.shape == (NC, S, d)

    l_n0, dh_n0, de_n0 = run_cluster_is_avx512(
        np.zeros((0, d), dtype=np.float32),
        E_cl,
        C_rt,
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0, K_neg), dtype=np.int32),
        compute_grad=True,
    )
    assert l_n0.shape == (0,)
    assert dh_n0 is not None and dh_n0.shape == (0, d)
    assert de_n0 is not None and de_n0.shape == (NC, S, d)

    # 4. Unavailable ISA modes still raise RuntimeError even when K == 0
    monkeypatch.setenv("CC", "/nonexistent_oryon_cc_xyz")
    for bad_mode in ("i8mm", "sdot"):
        with pytest.raises(RuntimeError):
            run_int8_gemm(a0, b0, mode=bad_mode)
    assert np.array_equal(run_int8_gemm(a0, b0, mode="auto"), expected_gemm)
    assert np.array_equal(run_nf4_matvec(x0, qw0, sc0, group_size=64), expected_nf4)


def test_head_kernels_reject_non_integer_index_dtypes():
    H_ok = np.zeros((4, 8), dtype=np.float32)
    E_ok = np.zeros((10, 8), dtype=np.float32)
    NC, S, d, K_neg = 5, 4, 8, 2
    E_cl = np.zeros((NC, S, d), dtype=np.float32)
    C_rt = np.zeros((NC, d), dtype=np.float32)
    tc_ok = np.array([0, 1, 2, 3], dtype=np.int32)
    tm_ok = np.array([0, 1, 2, 3], dtype=np.int32)
    neg_ok = np.array([[1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int32)

    bad_1d_cases = (
        np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        np.array([0.0, np.inf, 1.0, 2.0], dtype=np.float64),
        np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float64),
        np.array([True, False, True, False], dtype=np.bool_),
        np.array([0 + 0j, 1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex128),
        np.array(["0", "1", "2", "3"], dtype=str),
    )
    bad_2d_cases = (
        np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 0.5]], dtype=np.float32),
        np.array([[0.0, np.inf], [1.0, 2.0], [2.0, 3.0], [3.0, 0.0]], dtype=np.float64),
        np.array([[0.0, np.nan], [1.0, 2.0], [2.0, 3.0], [3.0, 0.0]], dtype=np.float64),
        np.array([[True, False], [False, True], [True, False], [False, True]], dtype=np.bool_),
        np.array(
            [[0 + 0j, 1 + 0j], [1 + 0j, 2 + 0j], [2 + 0j, 3 + 0j], [3 + 0j, 0 + 0j]],
            dtype=np.complex128,
        ),
        np.array([["0", "1"], ["1", "2"], ["2", "3"], ["3", "0"]], dtype=str),
    )

    # N > 0: all non-integer dtypes must raise ValueError
    for bad_1d, bad_2d in zip(bad_1d_cases, bad_2d_cases):
        with pytest.raises(ValueError):
            run_tiled_cce_avx512(H_ok, E_ok, bad_1d)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(H_ok, E_cl, C_rt, bad_1d, tm_ok, neg_ok)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, bad_1d, neg_ok)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(H_ok, E_cl, C_rt, tc_ok, tm_ok, bad_2d)

    # N == 0: non-integer dtypes must still raise ValueError before N == 0 early return
    H_0 = np.zeros((0, d), dtype=np.float32)
    tc_0 = np.zeros((0,), dtype=np.int32)
    tm_0 = np.zeros((0,), dtype=np.int32)
    neg_0 = np.zeros((0, K_neg), dtype=np.int32)
    for bad_dt in (np.float32, np.float64, np.bool_, np.complex128, np.dtype("<U1")):
        bad_1d_0 = np.zeros((0,), dtype=bad_dt)
        bad_2d_0 = np.zeros((0, K_neg), dtype=bad_dt)
        with pytest.raises(ValueError):
            run_tiled_cce_avx512(H_0, E_ok, bad_1d_0)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(H_0, E_cl, C_rt, bad_1d_0, tm_0, neg_0)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(H_0, E_cl, C_rt, tc_0, bad_1d_0, neg_0)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(H_0, E_cl, C_rt, tc_0, tm_0, bad_2d_0)

    # All standard signed/unsigned NumPy integer dtypes are accepted for both N > 0 and N == 0
    for int_dt in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        loss_cce, _, _ = run_tiled_cce_avx512(H_ok, E_ok, np.array([0, 1, 2, 3], dtype=int_dt))
        assert loss_cce.shape == (4,)
        loss_is, _, _ = run_cluster_is_avx512(
            H_ok,
            E_cl,
            C_rt,
            tc_ok.astype(int_dt),
            tm_ok.astype(int_dt),
            neg_ok.astype(int_dt),
        )
        assert loss_is.shape == (4,)
        loss_cce_0, _, _ = run_tiled_cce_avx512(H_0, E_ok, np.zeros((0,), dtype=int_dt))
        assert loss_cce_0.shape == (0,)
        loss_is_0, _, _ = run_cluster_is_avx512(
            H_0,
            E_cl,
            C_rt,
            np.zeros((0,), dtype=int_dt),
            np.zeros((0,), dtype=int_dt),
            np.zeros((0, K_neg), dtype=int_dt),
        )
        assert loss_is_0.shape == (0,)
