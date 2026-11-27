"""Opaque-box 4-tier E2E test suite for HaDukeNN_LLM v0.9.0 Snapdragon X Plus local deployment.

Parked outside ``tests/test_*.py`` until M5 merges so concurrent M2-M5 ``selftest`` runs
remain isolated. Compatible 100% with both ``pytest`` and the zero-dependency
``sheaf_verify.selftest`` minimal pytest shim:
  - Plain top-level ``def test_*`` functions only (no classes, no decorators, no fixtures).
  - Only ``tmp_path``, ``monkeypatch`` (``setenv``/``delenv``/``setattr``/``chdir``), and ``capsys``.
  - Only ``pytest.raises(..., match=...)`` and scalar ``pytest.approx(...)``.
  - Zero ``from tests.*`` imports (immune to external ``.pth`` package shadowing).

Gating:
  - Set ``HADUKENN_E2E_ALLOW_PENDING=1`` during pre-M5 dry runs to allow unmerged M2-M5
    symbols to report ``[PENDING]`` cleanly while baseline/invariant tests ALWAYS run.
  - Leave ``HADUKENN_E2E_ALLOW_PENDING`` unset (default strict mode) post-M5 so any missing
    contracted module or symbol fails closed.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hadukenn_core import torch_bridge
from hadukenn_core.all_realms_suite import QuantizedINT8NPULoopEngine
from hadukenn_core.benchmark_suite import load_clean_technical_corpus
from hadukenn_core.fast_ops import (
    get_fast_head_lib,
    run_cluster_is_avx512,
    run_tiled_cce_avx512,
)
from hadukenn_core.heads import flat_tied_fwd_bwd, tiled_cce_fwd_bwd
from hadukenn_core.partitions import build_balanced_partition
from hadukenn_core.rtx5070_frontier import NF4_CODEBOOK, NF4BlockQuantizer
from hadukenn_core.sheaf_bridge import (
    build_sheaf_bridge_operator,
    compute_kruskal_mst_laplacian,
    compute_lle_barycentric_weights,
    quantize_dequantize_symmetric_int8,
    run_t45_sheaf_bridge_suite,
    whitened_spectral_warmstart,
)
from sheaf_verify.anticheat import audit_kernel_source
from sheaf_verify.campaign import verify_t45_sheaf_bridge
from sheaf_verify.cli import main as cli_main
from sheaf_verify.compliance import emit_regulatory_bundle, verify_regulatory_bundle
from sheaf_verify.mutation import run_mutation_suite
from sheaf_verify.seal import load_key
from sheaf_verify.selftest import CANONICAL_ANCHOR, audit_python_style_stdlib
from sheaf_verify.store import Store
from sheaf_verify.verify import verify_store

ROOT = Path(__file__).resolve().parents[1]
_BANNED_DRIVE_PREFIXES = (
    "C:" + "\\",
    "D:" + "\\",
    "E:" + "\\",
    "H:" + "\\",
    "c:" + "\\",
    "d:" + "\\",
    "e:" + "\\",
    "h:" + "\\",
)
_BANNED_SUBSTRINGS = (
    "Permission is hereby " + "granted",
    "well" + "bore",
    "drill" + "ing",
    "DEVn" + "-1",
    "DESKTOP" + "-",
    "train" + "_paper",
)
_DRIVE_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9_])[CcDdEeHh]:" + re.escape("\\"))


def _is_arm64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def _has_oryon_i8mm() -> bool:
    detect = _import_optional("hadukenn_core.fast_ops", "detect_oryon_capabilities")
    if detect is None:
        return False
    return bool(detect().get("has_i8mm"))


def _require_oryon_i8mm(reason: str) -> None:
    if not _has_oryon_i8mm():
        pytest.skip(reason)


def _allow_pending() -> bool:
    return os.environ.get("HADUKENN_E2E_ALLOW_PENDING") == "1"


def _import_optional(module_name: str, attr_name: str | None = None) -> Any:
    """Import a contracted M2-M5 module/symbol; defer only when HADUKENN_E2E_ALLOW_PENDING=1."""
    try:
        mod = importlib.import_module(module_name)
    except (ModuleNotFoundError, ImportError) as exc:
        target = f"{module_name}.{attr_name}" if attr_name else module_name
        if _allow_pending():
            print(f"[PENDING] {target} ({exc})")
            return None
        raise AssertionError(
            f"Contracted module {target!r} missing in strict E2E mode: {exc}"
        ) from exc

    if attr_name is None:
        return mod
    if not hasattr(mod, attr_name):
        target = f"{module_name}.{attr_name}"
        if _allow_pending():
            print(f"[PENDING] {target} not yet merged")
            return None
        raise AssertionError(f"Contracted symbol {target!r} missing in strict E2E mode")
    return getattr(mod, attr_name)


def _load_docs_budget_module() -> Any:
    script = ROOT / ".github" / "scripts" / "docs_budget.py"
    spec = importlib.util.spec_from_file_location("docs_budget_e2e", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_proc_meminfo_avail_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return float(int(parts[1])) / (1024.0 * 1024.0)
    return 4.0


def _pack_nf4_weights(weight_fp: np.ndarray, group_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Pack [M, K] float32 matrix into uint8 NF4 nibbles [M, K//2] and per-group scales [M, K//group_size]."""
    m_rows, k_cols = weight_fp.shape
    thresholds = 0.5 * (NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:])
    n_groups = k_cols // group_size
    blocks = weight_fp.astype(np.float32).reshape(m_rows, n_groups, group_size)
    scales = np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12).astype(np.float32)
    normed = blocks / scales[:, :, None]
    codes = np.searchsorted(thresholds, normed).astype(np.uint8).reshape(m_rows, k_cols)
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).astype(np.uint8)
    return packed, scales


def _dequant_nf4_weights(
    qweights_u8: np.ndarray,
    scales_f32: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    m_rows = qweights_u8.shape[0]
    k_cols = qweights_u8.shape[1] * 2
    hi = (qweights_u8 >> 4) & 0x0F
    lo = qweights_u8 & 0x0F
    codes = np.empty((m_rows, k_cols), dtype=np.uint8)
    codes[:, 0::2] = hi
    codes[:, 1::2] = lo
    vals = NF4_CODEBOOK[codes].reshape(m_rows, k_cols // group_size, group_size)
    return (vals * scales_f32[:, :, None]).reshape(m_rows, k_cols).astype(np.float32)


def _assert_torch_bridge_ast_contracts() -> None:
    src = (ROOT / "src" / "hadukenn_core" / "torch_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    expected = {
        "ParadigmaticClusterISHeadTorch",
        "NF4FrozenLinearTorch",
        "OESpanEmbeddingTorch",
        "AsymmetricSheafKVPrefixTorch",
    }
    assert expected.issubset(class_names), f"Missing torch_bridge classes: {expected - class_names}"
    rep = audit_kernel_source(src)
    assert rep.ok is True and rep.violations == (), rep.violations


def _cli_safe(argv: list[str]) -> int:
    try:
        return int(cli_main(argv))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1


def _run_benchmark_parity_helper(fn: Any, a_int8: np.ndarray, b_int8: np.ndarray) -> dict[str, Any]:
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    op_counter_cls = _import_optional("hadukenn_core.fast_ops", "OpCounter")
    counter = op_counter_cls() if op_counter_cls is not None else None
    res = fn(
        "int8_gemm_e2e",
        lambda: run_int8_gemm(a_int8, b_int8, mode="auto", counter=counter),
        lambda: run_int8_gemm(a_int8, b_int8, mode="numpy"),
        counter=counter,
    )
    assert isinstance(res, dict), f"Expected dict from benchmark_kernel_parity, got {type(res)}"
    return res


def _get_tier_field(tier: Any, name: str, default: Any = None) -> Any:
    if isinstance(tier, dict):
        return tier.get(name, default)
    return getattr(tier, name, default)


# ===========================================================================
# Tier 1: Feature Coverage (39 tests: E1=6, E2=6, E3=5, E4=6, E5=5, E6=6, E7=5)
# ===========================================================================


# --- E1: ARM64 Oryon SIMD Kernels & avx512 Compatibility (6 tests) ---


def test_t1_e1_detect_oryon_capabilities() -> None:
    detect = _import_optional("hadukenn_core.fast_ops", "detect_oryon_capabilities")
    if detect is None:
        return
    cap = detect()
    assert isinstance(cap, dict)
    if _is_arm64():
        assert cap.get("has_neon") is True
    assert int(cap.get("num_threads", 0)) >= 1
    assert isinstance(cap.get("compiler"), str) and len(cap["compiler"]) > 0
    lib_path = cap.get("library_path")
    assert lib_path is not None and Path(lib_path).is_file()


def test_t1_e1_dot_and_axpy_f32_parity() -> None:
    run_dot_f32 = _import_optional("hadukenn_core.fast_ops", "run_dot_f32")
    run_axpy_f32 = _import_optional("hadukenn_core.fast_ops", "run_axpy_f32")
    if run_dot_f32 is None or run_axpy_f32 is None:
        return
    rng = np.random.default_rng(11)
    for n_vec in (137, 65536):
        a = rng.normal(0.0, 1.0, size=(n_vec,)).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=(n_vec,)).astype(np.float32)
        x = rng.normal(0.0, 1.0, size=(n_vec,)).astype(np.float32)
        y = rng.normal(0.0, 1.0, size=(n_vec,)).astype(np.float32)
        alpha = 1.75

        dot_val = float(run_dot_f32(a, b))
        ref_dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        assert math.isclose(dot_val, ref_dot, rel_tol=1e-4, abs_tol=1e-3), (n_vec, dot_val, ref_dot)

        axpy_out = run_axpy_f32(alpha, x, y.copy())
        ref_axpy = (np.float32(alpha) * x + y).astype(np.float32)
        assert axpy_out.shape == ref_axpy.shape
        assert np.allclose(axpy_out, ref_axpy, atol=1e-5)


def test_t1_e1_int8_gemm_all_modes_exact_i32() -> None:
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    if run_int8_gemm is None:
        return
    rng = np.random.default_rng(17)
    a_i8 = rng.integers(-128, 128, size=(17, 65), dtype=np.int8)
    b_i8 = rng.integers(-128, 128, size=(65, 19), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    modes = ("auto", "i8mm", "sdot", "scalar", "numpy") if _has_oryon_i8mm() else ("auto", "scalar", "numpy")
    for mode in modes:
        res = run_int8_gemm(a_i8, b_i8, mode=mode)
        assert res.dtype == np.int32
        assert res.shape == (17, 19)
        assert np.array_equal(res, ref), f"INT8 GEMM mismatch in mode={mode}"


def test_t1_e1_nf4_matvec_parity() -> None:
    run_nf4_matvec = _import_optional("hadukenn_core.fast_ops", "run_nf4_matvec")
    if run_nf4_matvec is None:
        return
    rng = np.random.default_rng(23)
    x = rng.normal(0.0, 1.0, size=(128,)).astype(np.float32)
    w_fp = rng.normal(0.0, 0.5, size=(64, 128)).astype(np.float32)
    qweights_u8, scales_f32 = _pack_nf4_weights(w_fp, group_size=64)
    ref_w = _dequant_nf4_weights(qweights_u8, scales_f32, group_size=64)
    ref_out = ref_w @ x

    out = run_nf4_matvec(x, qweights_u8, scales_f32, group_size=64)
    assert out.shape == (64,)
    assert np.allclose(out, ref_out, atol=1e-4)


def test_t1_e1_backwards_compat_avx512_wrappers() -> None:
    assert callable(get_fast_head_lib)
    assert callable(run_cluster_is_avx512)
    assert callable(run_tiled_cce_avx512)

    rng = np.random.default_rng(31)
    h = rng.normal(0.0, 0.5, size=(16, 32)).astype(np.float32)
    e = rng.normal(0.0, 0.5, size=(64, 32)).astype(np.float32)
    targets = rng.integers(0, 64, size=(16,), dtype=np.int32)

    lib = get_fast_head_lib()
    if lib is None:
        with pytest.raises(RuntimeError):
            run_tiled_cce_avx512(h, e, targets, tile_V=64, compute_grad=True)
        detect = _import_optional("hadukenn_core.fast_ops", "detect_oryon_capabilities")
        if detect is None:
            return
        raise AssertionError("get_fast_head_lib() returned None after M2 Oryon SIMD merge")

    loss_nat, dh_nat, de_nat = run_tiled_cce_avx512(h, e, targets, tile_V=64, compute_grad=True)
    loss_ref, dh_ref, de_ref = flat_tied_fwd_bwd(h, e, targets, compute_grad=True)
    assert dh_nat is not None and de_nat is not None
    assert dh_ref is not None and de_ref is not None
    assert float(np.max(np.abs(loss_nat - loss_ref))) < 1e-5
    assert float(np.max(np.abs(dh_nat - dh_ref))) < 1e-5
    assert float(np.max(np.abs(de_nat - de_ref))) < 1e-5

    e_cl = e.reshape(8, 8, 32).copy()
    c_root = np.mean(e_cl, axis=1)
    tc = (targets // 8).astype(np.int32)
    tm = (targets % 8).astype(np.int32)
    neg = ((tc[:, None] + np.arange(1, 4, dtype=np.int32)[None, :]) % 8).astype(np.int32)
    cl_loss, cl_dh, cl_de = run_cluster_is_avx512(h, e_cl, c_root, tc, tm, neg, compute_grad=True)
    assert cl_loss.shape == (16,)
    assert np.all(np.isfinite(cl_loss))
    assert cl_dh is not None and np.all(np.isfinite(cl_dh))
    assert cl_de is not None and np.all(np.isfinite(cl_de))


def test_t1_e1_opcounter_and_benchmark_parity() -> None:
    op_counter_cls = _import_optional("hadukenn_core.fast_ops", "OpCounter")
    bench_fn = _import_optional("hadukenn_core.fast_ops", "benchmark_kernel_parity")
    if op_counter_cls is None or bench_fn is None:
        return
    counter = op_counter_cls()
    assert counter is not None
    rng = np.random.default_rng(37)
    a_i8 = rng.integers(-64, 64, size=(64, 128), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(128, 64), dtype=np.int8)
    rep = _run_benchmark_parity_helper(bench_fn, a_i8, b_i8)
    assert rep.get("parity_ok") is True
    assert float(rep.get("max_abs_err", 1.0)) == 0.0
    speedup = float(rep.get("speedup_x", rep.get("speedup_vs_numpy", 0.0)))
    assert speedup > 0.0
    assert rep.get("op_counter") is not None and int(rep["op_counter"]["mac_ops"]) > 0


# --- E2: Hexagon HTP v73 NPU & Dynamic WSL2-Win Bridge (6 tests) ---


def test_t1_e2_discover_windows_python_dynamic() -> None:
    discover = _import_optional("hadukenn_core.npu_qnn_backend", "discover_windows_python")
    if discover is None:
        return
    win_py = discover()
    assert win_py is None or (
        isinstance(win_py, str) and "WindowsApps" not in win_py and Path(win_py).is_file()
    )


def test_t1_e2_npu_tier1_htp_qnn_live_parity() -> None:
    discover = _import_optional("hadukenn_core.npu_qnn_backend", "discover_windows_python")
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if discover is None or run_htp is None:
        return
    rng = np.random.default_rng(41)
    a_i8 = rng.integers(-64, 64, size=(64, 128), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(128, 64), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    if discover() is None:
        res = run_htp(a_i8, b_i8, force_tier="auto")
        assert res["tier_used"] in {"oryon_i8mm", "numpy_ref"}
        assert np.array_equal(res["output"], ref)
        return
    res = run_htp(a_i8, b_i8, force_tier="htp_qnn")
    assert res["tier_used"] == "htp_qnn"
    assert res.get("canary_ok") is True
    assert np.array_equal(res["output"], ref)


def test_t1_e2_npu_tier2_oryon_i8mm_parity() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_htp is None:
        return
    rng = np.random.default_rng(43)
    a_i8 = rng.integers(-64, 64, size=(32, 96), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(96, 48), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    res = run_htp(a_i8, b_i8, force_tier="oryon_i8mm")
    assert res["tier_used"] == "oryon_i8mm"
    assert np.array_equal(res["output"], ref)


def test_t1_e2_npu_tier3_numpy_ref_exact() -> None:
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_htp is None:
        return
    rng = np.random.default_rng(47)
    a_i8 = rng.integers(-64, 64, size=(16, 32), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(32, 24), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    res = run_htp(a_i8, b_i8, force_tier="numpy_ref")
    assert res["tier_used"] == "numpy_ref"
    assert np.array_equal(res["output"], ref)


def test_t1_e2_npu_auto_tier_and_speedup() -> None:
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_htp is None:
        return
    rng = np.random.default_rng(53)
    a_i8 = rng.integers(-32, 32, size=(256, 512), dtype=np.int8)
    b_i8 = rng.integers(-32, 32, size=(512, 512), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    res = run_htp(a_i8, b_i8, force_tier="auto")
    assert res["tier_used"] in {"htp_qnn", "oryon_i8mm", "numpy_ref"}
    assert np.array_equal(res["output"], ref)
    if res["tier_used"] != "numpy_ref":
        assert float(res["speedup_vs_numpy"]) > 1.0


def test_t1_e2_all_realms_quantized_npu_engine_parity() -> None:
    rng = np.random.default_rng(59)
    w = rng.normal(0.0, 0.25, size=(32, 24)).astype(np.float32)
    x = rng.normal(0.0, 0.25, size=(16, 32)).astype(np.float32)
    engine = QuantizedINT8NPULoopEngine(w)
    y_q = engine.forward_npu_int8(x)
    y_ref = x @ w
    rel_err = float(np.linalg.norm(y_q - y_ref) / max(float(np.linalg.norm(y_ref)), 1e-12))
    assert y_q.shape == (16, 24)
    assert np.all(np.isfinite(y_q))
    assert rel_err < 0.03


# --- E3: Unified-Memory Planner (51M..8B), struct32k & Flat Dense Heads (5 tests) ---


def test_t1_e3_snapdragon_spec_constants_and_meminfo() -> None:
    spec_cls = _import_optional("hadukenn_core.snapdragon_local", "SnapdragonXPlusSpec")
    if spec_cls is None:
        return
    spec = spec_cls()
    assert int(spec.cores) == 8
    assert float(spec.bandwidth_gb_s) == pytest.approx(135.0)
    assert float(spec.tdp_watts) == pytest.approx(14.5)
    assert float(spec.physical_ram_gib) == pytest.approx(15.61, rel=0.05)
    assert float(spec.wsl2_mem_available_gib) > 0.0


def test_t1_e3_plan_all_five_tiers_struct32k() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    if plan_fn is None:
        return
    tiers = plan_fn()
    names = [_get_tier_field(t, "name") or _get_tier_field(t, "tier") for t in tiers]
    assert names == ["51M", "125M", "1.5B", "3.2B", "8B"]
    for tier in tiers:
        assert int(_get_tier_field(tier, "vocab_size")) == 32768
        assert _get_tier_field(tier, "head_kind") == "flat_dense_struct32k"
        assert float(_get_tier_field(tier, "bandwidth_tok_s")) > 0.0
        assert float(_get_tier_field(tier, "power_w")) <= 14.5


def test_t1_e3_memory_gating_fits_wsl2_vs_physical(monkeypatch: Any) -> None:
    spec_cls = _import_optional("hadukenn_core.snapdragon_local", "SnapdragonXPlusSpec")
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    if spec_cls is None or plan_fn is None:
        return
    orig_read_text = Path.read_text

    def _fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self) == "/proc/meminfo":
            return "MemTotal:        8388608 kB\nMemAvailable:    4194304 kB\n"
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    spec = spec_cls()
    tiers = plan_fn(spec)
    by_name = {(_get_tier_field(t, "name") or _get_tier_field(t, "tier")): t for t in tiers}
    assert _get_tier_field(by_name["51M"], "fits_wsl2_mem_available") is True
    assert _get_tier_field(by_name["125M"], "fits_wsl2_mem_available") is True
    assert _get_tier_field(by_name["8B"], "fits_wsl2_mem_available") is False


def test_t1_e3_run_t46_suite_live_synthetic_fixtures() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    if suite_fn is None:
        return
    rep = suite_fn()
    assert isinstance(rep, dict)
    encoded = json.dumps(rep, sort_keys=True)
    assert len(encoded) > 100
    tiers = rep.get("tiers", [])
    assert len(tiers) == 5
    for tier in tiers:
        if tier.get("live_executed"):
            assert tier.get("fits_wsl2_mem_available") is True


def test_t1_e3_no_wslconfig_or_network_downloads(monkeypatch: Any) -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    if plan_fn is None or suite_fn is None:
        return

    def _deny_socket(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Network socket calls are forbidden in local Snapdragon E2E")

    orig_write_text = Path.write_text
    orig_write_bytes = Path.write_bytes
    orig_path_open = Path.open
    orig_builtins_open = builtins.open
    write_targets: list[str] = []

    def _is_write_mode(mode: str) -> bool:
        return any(ch in mode for ch in ("w", "a", "x", "+"))

    def _spy_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        write_targets.append(str(self))
        return orig_write_text(self, *args, **kwargs)

    def _spy_write_bytes(self: Path, *args: Any, **kwargs: Any) -> int:
        write_targets.append(str(self))
        return orig_write_bytes(self, *args, **kwargs)

    def _spy_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        mode = str(kwargs.get("mode", args[0] if args else "r"))
        if _is_write_mode(mode):
            write_targets.append(str(self))
        return orig_path_open(self, *args, **kwargs)

    def _spy_builtins_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        mode = str(kwargs.get("mode", args[0] if args else "r"))
        if _is_write_mode(mode) and isinstance(file, (str, os.PathLike)):
            write_targets.append(os.fspath(file))
        return orig_builtins_open(file, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", _deny_socket)
    monkeypatch.setattr(Path, "write_text", _spy_write_text)
    monkeypatch.setattr(Path, "write_bytes", _spy_write_bytes)
    monkeypatch.setattr(Path, "open", _spy_path_open)
    monkeypatch.setattr(builtins, "open", _spy_builtins_open)
    tiers = plan_fn()
    assert len(tiers) == 5
    rep = suite_fn()
    assert isinstance(rep, dict)
    assert all(".wslconfig" not in target.lower() for target in write_targets), write_targets


# --- E4: Chunked Sparse CSR Sheaf-Bridge & Dense Small-V Parity (6 tests) ---


def test_t1_e4_dense_small_v_apis_preserved() -> None:
    rng = np.random.default_rng(13)
    oracle = rng.normal(0.0, 1.0, size=(32, 64))
    w_init, isotropy = whitened_spectral_warmstart(oracle, d_student=8)
    assert w_init.shape == (32, 8)
    assert 0.25 <= isotropy <= 1.0

    w_lle = compute_lle_barycentric_weights(oracle, k=4)
    assert w_lle.shape == (32, 32)
    assert np.allclose(np.sum(w_lle, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.diag(w_lle), 0.0)

    l_mst, edges = compute_kruskal_mst_laplacian(oracle)
    assert len(edges) == 31
    assert np.allclose(l_mst, l_mst.T)
    assert np.allclose(np.sum(l_mst, axis=1), 0.0, atol=1e-9)

    l_bridge, stats = build_sheaf_bridge_operator(oracle, k=4)
    assert l_bridge.shape == (32, 32)
    assert np.allclose(l_bridge, l_bridge.T, atol=1e-9)
    assert stats["mst_edges"] == 31
    assert stats["lle_nnz"] == 32 * 4

    q_w = quantize_dequantize_symmetric_int8(w_init)
    assert q_w.shape == w_init.shape
    assert float(np.max(np.abs(q_w - w_init))) < 0.02


def test_t1_e4_sparse_csr_lle_row_sum_and_symmetry() -> None:
    csr_cls = _import_optional("hadukenn_core.sheaf_bridge", "SparseCSRMatrix")
    lle_csr = _import_optional("hadukenn_core.sheaf_bridge", "compute_lle_barycentric_weights_csr")
    if csr_cls is None or lle_csr is None:
        return
    rng = np.random.default_rng(61)
    oracle = rng.normal(0.0, 1.0, size=(128, 32)).astype(np.float32)
    csr = lle_csr(oracle, k=8, chunk_size=32)
    assert isinstance(csr, csr_cls)
    assert csr.shape == (128, 128)
    assert np.allclose(csr.matvec(np.ones(128, dtype=np.float32)), 1.0, atol=1e-3)


def test_t1_e4_sparse_csr_mst_laplacian_properties() -> None:
    mst_csr = _import_optional("hadukenn_core.sheaf_bridge", "compute_mst_laplacian_csr")
    if mst_csr is None:
        return
    rng = np.random.default_rng(67)
    oracle = rng.normal(0.0, 1.0, size=(128, 32)).astype(np.float32)
    csr = mst_csr(oracle, k_cand=16, chunk_size=32)
    assert csr.shape == (128, 128)
    assert np.allclose(csr.matvec(np.ones(128, dtype=np.float32)), 0.0, atol=1e-4)
    x = rng.normal(0.0, 1.0, size=(128,)).astype(np.float32)
    quad = float(x @ csr.matvec(x))
    assert quad >= -1e-5


def test_t1_e4_build_sheaf_bridge_csr_32k_budget() -> None:
    build_csr = _import_optional("hadukenn_core.sheaf_bridge", "build_sheaf_bridge_csr")
    if build_csr is None:
        return
    rng = np.random.default_rng(71)
    oracle = rng.normal(0.0, 1.0, size=(32768, 64)).astype(np.float32)
    res = build_csr(oracle, k=8, chunk_size=1024, dtype="float16")
    assert int(res["csr_bytes"]) <= 8 * 1024 * 1024
    assert int(res["peak_ram_bytes"]) < 64 * 1024 * 1024
    assert res["within_budget"] is True
    assert res["operator"].shape == (32768, 32768)


def test_t1_e4_sparse_csr_matvec_parity_vs_dense() -> None:
    build_csr = _import_optional("hadukenn_core.sheaf_bridge", "build_sheaf_bridge_csr")
    if build_csr is None:
        return
    rng = np.random.default_rng(73)
    oracle = rng.normal(0.0, 1.0, size=(96, 32)).astype(np.float32)
    l_dense, _ = build_sheaf_bridge_operator(oracle.astype(np.float64), k=8, reg=1e-3, mst_weight=0.15)
    csr_res = build_csr(oracle, k=8, reg=1e-3, mst_weight=0.15, chunk_size=32, dtype="float32")
    v = rng.normal(0.0, 1.0, size=(96,)).astype(np.float32)
    y_csr = csr_res["operator"].matvec(v)
    y_dense = (l_dense @ v.astype(np.float64)).astype(np.float32)
    assert np.all(np.isfinite(y_csr))
    assert np.allclose(y_csr, y_dense, rtol=5e-2, atol=5e-2)


def test_t1_e4_t45_suite_still_passes() -> None:
    results, bundle = run_t45_sheaf_bridge_suite()
    ok_flag, criteria, _ = verify_t45_sheaf_bridge(results)
    assert ok_flag is True, [c.as_dict() for c in criteria]
    assert bundle["bundle_schema"] == "sheaf-verify.regulatory-bundle.v1"


# --- E5: ARM64-Clean torch_bridge.py & audit-kernel (5 tests) ---


def test_t1_e5_audit_kernel_zero_violations(capsys: Any) -> None:
    tb_path = ROOT / "src" / "hadukenn_core" / "torch_bridge.py"
    rep = audit_kernel_source(tb_path.read_text(encoding="utf-8"))
    assert rep.ok is True
    assert rep.violations == ()

    capsys.readouterr()
    rc = _cli_safe(["audit-kernel", "--path", str(tb_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["violations"] == []


def test_t1_e5_nf4_frozen_linear_cpu_forward_backward() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    torch.manual_seed(7)
    mod = torch_bridge.NF4FrozenLinearTorch(torch.randn(16, 64), block_size=64, lora_rank=4)
    x = torch.randn(4, 64, requires_grad=True)
    out = mod(x)
    assert tuple(out.shape) == (4, 16)
    out.sum().backward()
    assert mod.lora_a is not None and mod.lora_a.grad is not None
    assert torch.all(torch.isfinite(mod.lora_a.grad)).item()


def test_t1_e5_paradigmatic_cluster_is_head_cpu() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    torch.manual_seed(11)
    cu = torch.arange(128, dtype=torch.long).view(8, 16)
    head = torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=16, k_neg=3, chunk_tokens=8)
    h = torch.randn(2, 4, 16, requires_grad=True)
    targets = torch.randint(0, 128, (2, 4), dtype=torch.long)
    base_w = torch.randn(128, 16, requires_grad=True)
    lm_loss, root_nll = head(h, targets, base_w)
    assert torch.isfinite(lm_loss).item() and torch.isfinite(root_nll).item()
    (lm_loss + root_nll).backward()
    assert h.grad is not None and torch.all(torch.isfinite(h.grad)).item()


def test_t1_e5_oe_span_and_sheaf_kv_prefix_cpu() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    torch.manual_seed(13)
    oe = torch_bridge.OESpanEmbeddingTorch(n_spans=32, d_span=8, d_model=16)
    base = torch.randn(2, 6, 16)
    span_ids = torch.tensor([[0, -1, 3, 4, -1, 7], [-1, 1, 2, -1, 5, 6]], dtype=torch.long)
    out_oe = oe(base, span_ids)
    assert tuple(out_oe.shape) == (2, 6, 16)

    kv_prefix = torch_bridge.AsymmetricSheafKVPrefixTorch(d_model=16, d_stalk=8, n_heads=2)
    q_ctx = torch.randn(2, 16)
    stalks = torch.randn(2, 3, 8)
    k_mem, v_mem = kv_prefix(q_ctx, stalks)
    assert tuple(k_mem.shape) == (2, 2, 3, 8)
    assert tuple(v_mem.shape) == (2, 2, 3, 8)
    q_unrot = torch.randn(2, 2, 6, 8)
    q_rot = torch.randn(2, 2, 6, 8)
    k_rot = torch.randn(2, 2, 6, 8)
    scores = kv_prefix.compute_hybrid_scores(q_unrot, q_rot, k_mem, k_rot)
    assert tuple(scores.shape) == (2, 2, 6, 9)


def test_t1_e5_nf4_values_constant_preserved() -> None:
    vals = torch_bridge.NF4_VALUES
    assert len(vals) == 16
    assert vals[0] == -1.0
    assert vals[-1] == 1.0
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# --- E6: Single-Command local_e2e_runner.py & verify_t46_snapdragon_local (6 tests) ---


def test_t1_e6_run_local_e2e_inprocess() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    if run_local_e2e is None:
        return
    rep = run_local_e2e()
    assert isinstance(rep, dict)
    assert rep.get("torch_bridge_audit_ok") is True
    assert rep.get("int8_parity_ok") is True
    assert "tiers" in rep and len(rep["tiers"]) == 5


def test_t1_e6_local_e2e_cli_json_out(tmp_path: Path) -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    if run_local_e2e is None:
        return
    out_json = tmp_path / "e2e.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    proc = subprocess.run(
        [sys.executable, "-m", "hadukenn_core.local_e2e_runner", "--json-out", str(out_json)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_json.is_file()
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)


def test_t1_e6_verify_t46_snapdragon_local_pass() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if suite_fn is None or verify_t46 is None:
        return
    rep = suite_fn()
    ok_flag, criteria, advisories = verify_t46(rep)
    assert ok_flag is True, [c.as_dict() for c in criteria]
    assert len(criteria) >= 1 and all(c.passed for c in criteria)
    assert isinstance(advisories, list)


def test_t1_e6_cli_campaign_t46_pass(tmp_path: Path, capsys: Any) -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if suite_fn is None or verify_t46 is None:
        return
    t46_path = tmp_path / "t46.json"
    t46_path.write_text(json.dumps(suite_fn()), encoding="utf-8")
    capsys.readouterr()
    rc = _cli_safe(["campaign", "--mode", "t46-snapdragon-local", "--path", str(t46_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_t1_e6_speedup_measured_over_tier3_numpy() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    if run_local_e2e is None:
        return
    rep = run_local_e2e()
    speedup = float(rep.get("oryon_simd_speedup_vs_numpy", rep.get("simd_speedup_vs_numpy", 0.0)))
    assert speedup > 1.0


def test_t1_e6_report_deterministic_schema_keys() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    if suite_fn is None:
        return
    rep = suite_fn()
    schema = rep.get("schema")
    assert isinstance(schema, str) and ("t46" in schema or "snapdragon" in schema)
    tier_names = [t.get("name") or t.get("tier") for t in rep.get("tiers", [])]
    assert tier_names == ["51M", "125M", "1.5B", "3.2B", "8B"]
    assert all(int(t.get("vocab_size", 0)) == 32768 for t in rep["tiers"])
    canonical = json.dumps(rep, sort_keys=True)
    assert json.loads(canonical) == rep


# --- E7: Repo, Anchor, Stdlib & Export Hygiene Invariants (5 tests) ---


def test_t1_e7_canonical_anchor_unchanged() -> None:
    store = Store(ROOT / "fixtures" / "passing")
    key = load_key(ROOT / "fixtures" / "test_seal.key")
    vrep = verify_store(store, key=key)
    assert vrep.ok is True
    assert vrep.head == CANONICAL_ANCHOR
    assert len(vrep.gate_rows) == 22
    assert all(row["status"] == "pass" for row in vrep.gate_rows)


def test_t1_e7_sheaf_verify_100_pct_pure_stdlib() -> None:
    issues = audit_python_style_stdlib(ROOT)
    assert issues == [], issues


def test_t1_e7_zero_drive_letters_and_binaries() -> None:
    skip = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
    scanned = {".py", ".md", ".toml", ".yml", ".json", ".html", ".css", ""}
    bad_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skip or part.endswith(".egg-info") for part in path.parts):
            continue
        assert path.suffix.lower() not in bad_suffixes, f"Forbidden binary file: {path}"
        if "fixtures/passing" in path.as_posix():
            continue
        if path.suffix in scanned or path.name == "NOTICE":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for phrase in _BANNED_SUBSTRINGS:
                assert phrase not in text, f"Banned phrase {phrase!r} found in {path}"
            assert _DRIVE_PREFIX_RE.search(text) is None, f"Hardcoded drive prefix found in {path}"


def test_t1_e7_docs_budget_passes() -> None:
    script = ROOT / ".github" / "scripts" / "docs_budget.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_t1_e7_immutables_and_license_preserved() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("# PolyForm Noncommercial License 1.0.0\n")
    notice_text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "@ZuluYokohama" in notice_text
    assert (ROOT / "iterations" / "exec-hypergraph-v01" / "README.md").is_file()


# ===========================================================================
# Tier 2: Boundary, Edge & Negative Cases (37 tests: E1=6, E2=5, E3=5, E4=6, E5=5, E6=5, E7=5)
# ===========================================================================


# --- E1 Boundary & Negative (6 tests) ---


def test_t2_e1_odd_tail_dimensions_i8mm_sdot() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm/sdot CPU")
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    if run_int8_gemm is None:
        return
    rng = np.random.default_rng(101)
    for m_dim, k_dim, n_dim in ((1, 1, 1), (3, 7, 5), (5, 15, 9), (9, 63, 17)):
        a_i8 = rng.integers(-128, 128, size=(m_dim, k_dim), dtype=np.int8)
        b_i8 = rng.integers(-128, 128, size=(k_dim, n_dim), dtype=np.int8)
        ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
        for mode in ("i8mm", "sdot"):
            out = run_int8_gemm(a_i8, b_i8, mode=mode)
            assert np.array_equal(out, ref), f"Tail mismatch for {(m_dim, k_dim, n_dim)} mode={mode}"


def test_t2_e1_int8_extreme_saturation_no_overflow() -> None:
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    if run_int8_gemm is None:
        return
    a_i8 = np.full((8, 256), -128, dtype=np.int8)
    b_i8 = np.full((256, 8), -128, dtype=np.int8)
    out = run_int8_gemm(a_i8, b_i8, mode="auto")
    assert out.dtype == np.int32
    assert np.all(out == 4194304)


def test_t2_e1_dot_axpy_zero_length_and_shape_mismatch() -> None:
    run_dot_f32 = _import_optional("hadukenn_core.fast_ops", "run_dot_f32")
    run_axpy_f32 = _import_optional("hadukenn_core.fast_ops", "run_axpy_f32")
    if run_dot_f32 is None or run_axpy_f32 is None:
        return
    assert float(run_dot_f32(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))) == 0.0
    with pytest.raises(ValueError):
        run_dot_f32(np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        run_axpy_f32(1.0, np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))


def test_t2_e1_int8_gemm_invalid_mode_and_dim_mismatch() -> None:
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    if run_int8_gemm is None:
        return
    a_ok = np.ones((4, 8), dtype=np.int8)
    b_ok = np.ones((8, 4), dtype=np.int8)
    with pytest.raises((ValueError, TypeError)):
        run_int8_gemm(a_ok, b_ok, mode="avx999")
    with pytest.raises((ValueError, TypeError)):
        run_int8_gemm(a_ok, np.ones((9, 4), dtype=np.int8))
    with pytest.raises((ValueError, TypeError)):
        run_int8_gemm(np.ones((8,), dtype=np.int8), b_ok)
    coerced = run_int8_gemm(a_ok.astype(np.float32), b_ok)
    assert coerced.dtype == np.int32 and np.all(coerced == 8)


def test_t2_e1_nf4_invalid_group_or_shape() -> None:
    run_nf4_matvec = _import_optional("hadukenn_core.fast_ops", "run_nf4_matvec")
    if run_nf4_matvec is None:
        return
    x = np.ones(64, dtype=np.float32)
    qw = np.zeros((8, 32), dtype=np.uint8)
    sc = np.ones((8, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        run_nf4_matvec(x, qw, sc, group_size=0)
    with pytest.raises(ValueError):
        run_nf4_matvec(x, qw, sc, group_size=32)
    with pytest.raises(ValueError):
        run_nf4_matvec(np.ones(32, dtype=np.float32), qw, sc, group_size=64)


def test_t2_e1_pthreads_thread_safety_repeated_calls() -> None:
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    if run_int8_gemm is None:
        return
    rng = np.random.default_rng(109)
    a_small = rng.integers(-64, 64, size=(1, 64), dtype=np.int8)
    b_small = rng.integers(-64, 64, size=(64, 16), dtype=np.int8)
    ref_small = a_small.astype(np.int32) @ b_small.astype(np.int32)

    a_big = rng.integers(-64, 64, size=(128, 64), dtype=np.int8)
    b_big = rng.integers(-64, 64, size=(64, 64), dtype=np.int8)
    ref_big = a_big.astype(np.int32) @ b_big.astype(np.int32)

    for step in range(20):
        if step % 2 == 0:
            assert np.array_equal(run_int8_gemm(a_small, b_small, mode="auto"), ref_small)
        else:
            assert np.array_equal(run_int8_gemm(a_big, b_big, mode="auto"), ref_big)


# --- E2 Boundary & Negative (5 tests) ---


def test_t2_e2_canary_0x5a_corruption_detected(monkeypatch: Any) -> None:
    npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
    if npu_mod is None:
        return
    a_i8 = np.ones((16, 32), dtype=np.int8)
    b_i8 = np.ones((32, 16), dtype=np.int8)

    for fn_name in ("verify_canary_buffer", "check_canary_buffer", "_verify_canary", "_check_canary"):
        fn = getattr(npu_mod, fn_name, None)
        if callable(fn):
            corrupted = bytearray([0x00] * 16)
            assert fn(corrupted) is False
            return

    session_cls = getattr(npu_mod, "HTPBackendSession", None)
    canary_attr = next(
        (
            name
            for name in ("_check_canary", "check_canary", "_verify_canary", "verify_canary")
            if session_cls is not None and hasattr(session_cls, name)
        ),
        None,
    )
    if canary_attr is not None:
        monkeypatch.setattr(session_cls, canary_attr, staticmethod(lambda *_a, **_kw: False))
        htp_avail = callable(getattr(npu_mod, "discover_windows_python", None)) and (
            npu_mod.discover_windows_python() is not None
        )
        if htp_avail:
            try:
                res = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="htp_qnn")
            except (RuntimeError, ValueError, OSError, AssertionError):
                return
            assert res.get("canary_ok") is False or res.get("tier_used") != "htp_qnn"
            return
        res = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
        assert res["tier_used"] != "htp_qnn" or res.get("canary_ok") is False
        return

    raise AssertionError("M4 npu_qnn_backend must expose or enforce 0x5A canary verification")


def test_t2_e2_windows_apps_stub_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    discover = _import_optional("hadukenn_core.npu_qnn_backend", "discover_windows_python")
    if discover is None:
        return

    real_win_py = discover()
    if real_win_py is not None:
        assert isinstance(real_win_py, str)
        assert "WindowsApps" not in real_win_py
        assert Path(real_win_py).is_file()

    fake_stub_dir = tmp_path / "Microsoft" / "WindowsApps"
    fake_stub_dir.mkdir(parents=True, exist_ok=True)
    fake_stub = fake_stub_dir / "python.exe"
    fake_stub.write_text("", encoding="utf-8")
    assert fake_stub.is_file()

    monkeypatch.setenv("HADUKENN_WIN_PYTHON", str(fake_stub))
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess(args=["cmd.exe"], returncode=1, stdout="", stderr=""),
    )
    res_env = discover()
    assert res_env != str(fake_stub)
    assert res_env is None or "WindowsApps" not in res_env

    monkeypatch.delenv("HADUKENN_WIN_PYTHON", raising=False)
    monkeypatch.setattr(shutil, "which", lambda cmd: str(fake_stub) if "python" in cmd else None)
    win_stub = "C:" + "\\Users\\user\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe\r\n"

    def _fake_run(cmd: Any, *_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd_str = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "wslpath" in cmd_str:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=f"{fake_stub}\n", stderr="")
        if "cmd.exe" in cmd_str or "where" in cmd_str:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=win_stub, stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    res_which = discover()
    assert res_which != str(fake_stub)
    assert res_which is None or "WindowsApps" not in res_which

    if real_win_py is not None:
        monkeypatch.setenv("HADUKENN_WIN_PYTHON", real_win_py)
        assert discover() == real_win_py
    else:
        valid_dir = tmp_path / "Python314"
        valid_dir.mkdir(parents=True, exist_ok=True)
        valid_exe = valid_dir / "python.exe"
        valid_exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        valid_exe.chmod(0o755)
        monkeypatch.setenv("HADUKENN_WIN_PYTHON", str(valid_exe))
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, *_a, **_kw: subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr=""),
        )
        assert discover() == str(valid_exe)


def test_t2_e2_npu_invalid_force_tier_raises() -> None:
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_htp is None:
        return
    a_i8 = np.ones((8, 8), dtype=np.int8)
    b_i8 = np.ones((8, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        run_htp(a_i8, b_i8, force_tier="cuda_tensor")


def test_t2_e2_npu_shape_and_dtype_validation() -> None:
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_htp is None:
        return
    with pytest.raises((ValueError, TypeError)):
        run_htp(np.ones((4, 5), dtype=np.int8), np.ones((6, 4), dtype=np.int8), force_tier="numpy_ref")
    with pytest.raises((ValueError, TypeError)):
        run_htp(np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.int8), force_tier="numpy_ref")


def test_t2_e2_htp_unavailable_falls_back_gracefully(monkeypatch: Any) -> None:
    npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
    if npu_mod is None:
        return
    monkeypatch.setattr(npu_mod, "discover_windows_python", lambda: None)
    rng = np.random.default_rng(113)
    a_i8 = rng.integers(-32, 32, size=(16, 32), dtype=np.int8)
    b_i8 = rng.integers(-32, 32, size=(32, 24), dtype=np.int8)
    res = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    assert res["tier_used"] in {"oryon_i8mm", "numpy_ref"}
    assert np.array_equal(res["output"], a_i8.astype(np.int32) @ b_i8.astype(np.int32))


# --- E3 Boundary & Negative (5 tests) ---


def test_t2_e3_meminfo_low_ram_boundary(monkeypatch: Any) -> None:
    spec_cls = _import_optional("hadukenn_core.snapdragon_local", "SnapdragonXPlusSpec")
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    if spec_cls is None or plan_fn is None:
        return
    orig_read_text = Path.read_text

    def _low_meminfo(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self) == "/proc/meminfo":
            return "MemTotal:        8388608 kB\nMemAvailable:      262144 kB\n"
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _low_meminfo)
    spec = spec_cls()
    tiers = plan_fn(spec)
    assert len(tiers) == 5
    by_name = {(_get_tier_field(t, "name") or _get_tier_field(t, "tier")): t for t in tiers}
    assert _get_tier_field(by_name["8B"], "fits_wsl2_mem_available") is False
    assert _get_tier_field(by_name["3.2B"], "fits_wsl2_mem_available") is False


def test_t2_e3_meminfo_missing_fallback(monkeypatch: Any) -> None:
    spec_cls = _import_optional("hadukenn_core.snapdragon_local", "SnapdragonXPlusSpec")
    if spec_cls is None:
        return
    orig_read_text = Path.read_text
    orig_is_file = Path.is_file

    def _missing_meminfo(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self) == "/proc/meminfo":
            raise FileNotFoundError("simulated missing /proc/meminfo")
        return orig_read_text(self, *args, **kwargs)

    def _missing_is_file(self: Path) -> bool:
        if str(self) == "/proc/meminfo":
            return False
        return orig_is_file(self)

    monkeypatch.setattr(Path, "read_text", _missing_meminfo)
    monkeypatch.setattr(Path, "is_file", _missing_is_file)
    spec = spec_cls()
    assert float(spec.wsl2_mem_available_gib) >= 0.0


def test_t2_e3_no_factored_182x181_regression() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    if plan_fn is None:
        return
    tiers = plan_fn()
    for tier in tiers:
        head_kind = str(_get_tier_field(tier, "head_kind"))
        assert "182x181" not in head_kind
        assert "factored" not in head_kind
        assert head_kind == "flat_dense_struct32k"
        assert int(_get_tier_field(tier, "vocab_size")) == 32768


def test_t2_e3_planner_invalid_vocab_or_bandwidth() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    spec_cls = _import_optional("hadukenn_core.snapdragon_local", "SnapdragonXPlusSpec")
    if plan_fn is None or spec_cls is None:
        return
    with pytest.raises(ValueError):
        plan_fn(vocab_size=0)
    with pytest.raises(ValueError):
        spec_bad = spec_cls(bandwidth_gb_s=-1.0)
        plan_fn(spec_bad)


def test_t2_e3_tier_monotonicity_ram_and_tok_s() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    if plan_fn is None:
        return
    tiers = plan_fn()
    rams = [float(_get_tier_field(t, "required_ram_gib")) for t in tiers]
    tok_s = [float(_get_tier_field(t, "bandwidth_tok_s")) for t in tiers]
    params = [float(_get_tier_field(t, "total_params_m")) for t in tiers]
    assert all(params[i] < params[i + 1] for i in range(4))
    assert all(rams[i] < rams[i + 1] for i in range(4))
    assert all(tok_s[i] >= tok_s[i + 1] for i in range(4))


# --- E4 Boundary & Negative (6 tests) ---


def test_t2_e4_csr_chunk_size_boundary_values() -> None:
    lle_csr = _import_optional("hadukenn_core.sheaf_bridge", "compute_lle_barycentric_weights_csr")
    if lle_csr is None:
        return
    rng = np.random.default_rng(127)
    oracle = rng.normal(0.0, 1.0, size=(50, 16)).astype(np.float32)
    ones = np.ones(50, dtype=np.float32)
    for chunk_size in (1, 17, 128):
        csr = lle_csr(oracle, k=6, chunk_size=chunk_size)
        assert csr.shape == (50, 50)
        assert np.allclose(csr.matvec(ones), 1.0, atol=1e-3)


def test_t2_e4_csr_k_boundary_and_invalid() -> None:
    lle_csr = _import_optional("hadukenn_core.sheaf_bridge", "compute_lle_barycentric_weights_csr")
    if lle_csr is None:
        return
    rng = np.random.default_rng(131)
    oracle = rng.normal(0.0, 1.0, size=(24, 16)).astype(np.float32)
    csr_k1 = lle_csr(oracle, k=1, chunk_size=8)
    assert csr_k1.shape == (24, 24)
    with pytest.raises(ValueError):
        lle_csr(oracle, k=0, chunk_size=8)
    with pytest.raises(ValueError):
        lle_csr(oracle, k=24, chunk_size=8)


def test_t2_e4_csr_collinear_or_zero_norm_embeddings() -> None:
    build_csr = _import_optional("hadukenn_core.sheaf_bridge", "build_sheaf_bridge_csr")
    if build_csr is None:
        return
    rng = np.random.default_rng(137)
    oracle = rng.normal(0.0, 1.0, size=(32, 16)).astype(np.float32)
    oracle[0] = 0.0
    oracle[1] = 0.0
    oracle[2] = oracle[3]
    res = build_csr(oracle, k=4, chunk_size=16, dtype="float32")
    op = res["operator"]
    if hasattr(op, "data"):
        assert np.all(np.isfinite(op.data))
    else:
        assert np.all(np.isfinite(op.w_lle.data))
        assert np.all(np.isfinite(op.l_mst.data))
    out = op.matvec(np.ones(32, dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_t2_e4_csr_matvec_wrong_vector_shape() -> None:
    lle_csr = _import_optional("hadukenn_core.sheaf_bridge", "compute_lle_barycentric_weights_csr")
    if lle_csr is None:
        return
    rng = np.random.default_rng(139)
    oracle = rng.normal(0.0, 1.0, size=(32, 16)).astype(np.float32)
    csr = lle_csr(oracle, k=4, chunk_size=16)
    with pytest.raises(ValueError):
        csr.matvec(np.ones(33, dtype=np.float32))


def test_t2_e4_mst_disconnected_knn_fallback_connects() -> None:
    mst_csr = _import_optional("hadukenn_core.sheaf_bridge", "compute_mst_laplacian_csr")
    if mst_csr is None:
        return
    rng = np.random.default_rng(149)
    c1 = rng.normal(0.0, 0.05, size=(32, 16)).astype(np.float32) + 5.0
    c2 = rng.normal(0.0, 0.05, size=(32, 16)).astype(np.float32) - 5.0
    oracle = np.vstack([c1, c2])
    csr = mst_csr(oracle, k_cand=4, chunk_size=16)
    assert csr.indices.size > 0
    assert np.allclose(csr.matvec(np.ones(64, dtype=np.float32)), 0.0, atol=1e-4)
    # Cross-cluster indicator vector must have strictly positive Laplacian quadratic energy
    indicator = np.concatenate([np.ones(32, dtype=np.float32), -np.ones(32, dtype=np.float32)])
    assert float(indicator @ csr.matvec(indicator)) > 0.1


def test_t2_e4_warmstart_invalid_d_student() -> None:
    rng = np.random.default_rng(151)
    oracle = rng.normal(0.0, 1.0, size=(16, 8))
    with pytest.raises(ValueError, match="d_student"):
        whitened_spectral_warmstart(oracle, d_student=1)
    with pytest.raises(ValueError, match="d_student"):
        whitened_spectral_warmstart(oracle, d_student=8, drop_rogue_top=2)


# --- E5 Boundary & Negative (5 tests) ---


def test_t2_e5_audit_kernel_catches_forbidden_constructs() -> None:
    bad_snippets = [
        "import ctypes\nx = 1\n",
        "import subprocess\nx = 1\n",
        "import torch.nn as nn\nm = nn.Linear(4, 4)\n",
        "import torch.nn.functional as F\ndef f(a, b): return F.cross_entropy(a, b)\n",
        "import sys\ndef f(): return sys._getframe()\n",
        "import time\ntime.perf_counter = lambda: 0\n",
    ]
    for snippet in bad_snippets:
        rep = audit_kernel_source(snippet)
        assert rep.ok is False, f"audit_kernel_source missed forbidden snippet: {snippet!r}"
        assert len(rep.violations) >= 1


def test_t2_e5_torch_unavailable_graceful_import() -> None:
    had_torch = "torch" in sys.modules
    orig_torch = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        reloaded = importlib.reload(torch_bridge)
        assert reloaded.TORCH_AVAILABLE is False
        assert len(reloaded.NF4_VALUES) == 16
    finally:
        if had_torch:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)
        importlib.reload(torch_bridge)


def test_t2_e5_oe_span_all_negative_one_mask() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    torch.manual_seed(17)
    oe = torch_bridge.OESpanEmbeddingTorch(n_spans=16, d_span=8, d_model=12)
    base = torch.randn(2, 5, 12)
    ids = torch.full((2, 5), -1, dtype=torch.long)
    out = oe(base, ids)
    assert torch.equal(out, base)


def test_t2_e5_nf4_non_multiple_block_size() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    torch.manual_seed(19)
    mod = torch_bridge.NF4FrozenLinearTorch(torch.randn(17, 35), block_size=64, lora_rank=0)
    out = mod(torch.randn(3, 35))
    assert tuple(out.shape) == (3, 17)
    assert torch.all(torch.isfinite(out)).item()


def test_t2_e5_cluster_is_chunk_tokens_larger_than_n_tok() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    torch.manual_seed(23)
    cu = torch.arange(64, dtype=torch.long).view(8, 8)
    head = torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=12, k_neg=2, chunk_tokens=256)
    h = torch.randn(5, 12)
    targets = torch.randint(0, 64, (5,), dtype=torch.long)
    base_w = torch.randn(64, 12)
    lm_loss, root_nll = head(h, targets, base_w)
    assert torch.isfinite(lm_loss).item() and torch.isfinite(root_nll).item()


# --- E6 Boundary & Negative (5 tests) ---


def test_t2_e6_verify_t46_rejects_missing_or_empty_doc() -> None:
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if verify_t46 is None:
        return
    ok_empty, crit_empty, _ = verify_t46({})
    assert ok_empty is False and any(not c.passed for c in crit_empty)
    ok_wrong, crit_wrong, _ = verify_t46({"schema": "wrong.schema.v0"})
    assert ok_wrong is False and any(not c.passed for c in crit_wrong)


def test_t2_e6_verify_t46_rejects_sheaf_budget_breach() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if suite_fn is None or verify_t46 is None:
        return
    base = suite_fn()

    bad_csr = json.loads(json.dumps(base))
    if isinstance(bad_csr.get("sheaf_bridge_csr"), dict):
        bad_csr["sheaf_bridge_csr"]["csr_bytes"] = 9 * 1024 * 1024
        bad_csr["sheaf_bridge_csr"]["within_budget"] = False
    bad_csr["csr_bytes"] = 9 * 1024 * 1024
    ok_1, _, _ = verify_t46(bad_csr)
    assert ok_1 is False

    bad_ram = json.loads(json.dumps(base))
    if isinstance(bad_ram.get("sheaf_bridge_csr"), dict):
        bad_ram["sheaf_bridge_csr"]["peak_ram_bytes"] = 65 * 1024 * 1024
        bad_ram["sheaf_bridge_csr"]["within_budget"] = False
    bad_ram["peak_ram_bytes"] = 65 * 1024 * 1024
    ok_2, _, _ = verify_t46(bad_ram)
    assert ok_2 is False


def test_t2_e6_verify_t46_rejects_torch_audit_failure_or_parity_drift() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if suite_fn is None or verify_t46 is None:
        return
    base = suite_fn()

    bad_torch = json.loads(json.dumps(base))
    bad_torch["torch_bridge_audit_ok"] = False
    ok_1, _, _ = verify_t46(bad_torch)
    assert ok_1 is False

    bad_parity = json.loads(json.dumps(base))
    bad_parity["int8_parity_ok"] = False
    ok_2, _, _ = verify_t46(bad_parity)
    assert ok_2 is False


def test_t2_e6_cli_campaign_t46_missing_path_errors(tmp_path: Path) -> None:
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if verify_t46 is None:
        return
    rc_missing_arg = _cli_safe(["campaign", "--mode", "t46-snapdragon-local"])
    assert rc_missing_arg != 0
    rc_missing_file = _cli_safe(
        ["campaign", "--mode", "t46-snapdragon-local", "--path", str(tmp_path / "missing.json")]
    )
    assert rc_missing_file != 0


def test_t2_e6_runner_json_out_unwritable_dir_graceful(tmp_path: Path) -> None:
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    if run_local_e2e is None:
        return
    blocker = tmp_path / "regular_file"
    blocker.write_text("occupied", encoding="utf-8")
    bad_target = blocker / "out.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    proc = subprocess.run(
        [sys.executable, "-m", "hadukenn_core.local_e2e_runner", "--json-out", str(bad_target)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


# --- E7 Boundary & Negative (5 tests) ---


def test_t2_e7_stdlib_audit_detects_non_stdlib_import(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "sheaf_verify"
    pkg.mkdir(parents=True)
    (pkg / "bad.py").write_text("import numpy\n", encoding="utf-8")
    issues = audit_python_style_stdlib(tmp_path)
    assert len(issues) == 1
    assert "numpy" in issues[0]


def test_t2_e7_stdlib_audit_detects_tabs_and_syntax_error(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "sheaf_verify"
    pkg.mkdir(parents=True)
    (pkg / "tab.py").write_text("\tx = 1\n", encoding="utf-8")
    (pkg / "syn.py").write_text("def broken(:\n", encoding="utf-8")
    issues = audit_python_style_stdlib(tmp_path)
    joined = "\n".join(issues)
    assert "E101" in joined
    assert "E999" in joined


def test_t2_e7_docs_budget_detects_external_script() -> None:
    budget = _load_docs_budget_module()
    assert budget.is_external_script("https://cdn.example/a.js") is True
    assert budget.is_external_script("//cdn.example/a.js") is True
    assert budget.is_external_script("local.js") is False


def test_t2_e7_mutation_score_kills_12_of_12(tmp_path: Path) -> None:
    rep = run_mutation_suite(tmp_path)
    assert rep.generated == 12
    assert rep.killed == 12
    assert rep.survived == 0
    assert rep.mutation_score == pytest.approx(1.0)


def test_t2_e7_no_teamwork_dir_in_repo() -> None:
    assert not (ROOT / ".agents" / "teamwork").exists()


# ===========================================================================
# Tier 3: Cross-Feature Pairwise Combinations (9 tests)
# ===========================================================================


def test_t3_e1_e2_oryon_i8mm_vs_htp_npu_exact_parity() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_int8_gemm is None or run_htp is None:
        return
    rng = np.random.default_rng(211)
    a_i8 = rng.integers(-64, 64, size=(64, 128), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(128, 64), dtype=np.int8)
    y_oryon = run_int8_gemm(a_i8, b_i8, mode="i8mm")
    y_auto = run_htp(a_i8, b_i8, force_tier="auto")["output"]
    y_np = run_htp(a_i8, b_i8, force_tier="numpy_ref")["output"]
    assert np.array_equal(y_oryon, y_auto)
    assert np.array_equal(y_oryon, y_np)


def test_t3_e1_e5_oryon_nf4_matvec_vs_torch_nf4_linear() -> None:
    run_nf4_matvec = _import_optional("hadukenn_core.fast_ops", "run_nf4_matvec")
    if run_nf4_matvec is None:
        return
    rng = np.random.default_rng(223)
    w_fp = rng.normal(0.0, 0.5, size=(32, 64)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(64,)).astype(np.float32)
    quantizer = NF4BlockQuantizer(block_size=64)
    qstate = quantizer.quantize(w_fp)
    qw = qstate["packed"]
    scales_dq = (
        (qstate["q_absmax"].astype(np.float32) / np.float32(255.0)) * np.float32(qstate["super_max"])
    ).reshape(32, 1)
    y_oryon = run_nf4_matvec(x, qw, scales_dq, group_size=64)

    _assert_torch_bridge_ast_contracts()
    if torch_bridge.TORCH_AVAILABLE:
        import torch

        mod = torch_bridge.NF4FrozenLinearTorch(torch.from_numpy(w_fp), block_size=64, lora_rank=0)
        y_torch = mod(torch.from_numpy(x).unsqueeze(0)).squeeze(0).detach().numpy()
        assert np.allclose(y_oryon, y_torch, atol=1e-4)
    else:
        y_quant = quantizer.dequantize(qstate) @ x
        assert np.allclose(y_oryon, y_quant, atol=1e-4)


def test_t3_e2_e3_all_realms_realm_c_with_snapdragon_planner() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    if plan_fn is None:
        return
    tiers = plan_fn()
    by_name = {(_get_tier_field(t, "name") or _get_tier_field(t, "tier")): t for t in tiers}
    avail = _read_proc_meminfo_avail_gib()
    if avail >= 1.0:
        assert _get_tier_field(by_name["51M"], "fits_wsl2_mem_available") is True
    rng = np.random.default_rng(227)
    w = rng.normal(0.0, 0.2, size=(32, 48)).astype(np.float32)
    x = rng.normal(0.0, 0.2, size=(12, 32)).astype(np.float32)
    y_npu = QuantizedINT8NPULoopEngine(w).forward_npu_int8(x)
    y_ref = x @ w
    drift = float(np.linalg.norm(y_npu - y_ref) / max(float(np.linalg.norm(y_ref)), 1e-12))
    assert drift < 0.03


def test_t3_e3_e4_planner_struct32k_with_sparse_sheaf_bridge() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    build_csr = _import_optional("hadukenn_core.sheaf_bridge", "build_sheaf_bridge_csr")
    if plan_fn is None or build_csr is None:
        return
    tiers = plan_fn()
    by_name = {(_get_tier_field(t, "name") or _get_tier_field(t, "tier")): t for t in tiers}
    t51 = by_name["51M"]
    assert int(_get_tier_field(t51, "vocab_size")) == 32768
    assert _get_tier_field(t51, "head_kind") == "flat_dense_struct32k"

    rng = np.random.default_rng(229)
    oracle = rng.normal(0.0, 1.0, size=(32768, 64)).astype(np.float32)
    csr_res = build_csr(oracle, k=8, chunk_size=1024, dtype="float16")
    assert int(csr_res["csr_bytes"]) <= 8 * 1024 * 1024
    assert int(csr_res["peak_ram_bytes"]) < 64 * 1024 * 1024
    combined_gib = float(_get_tier_field(t51, "required_ram_gib")) + float(csr_res["peak_ram_bytes"]) / (1024**3)
    assert combined_gib < 15.61


def test_t3_e4_e5_sheaf_bridge_warmstart_into_torch_bridge() -> None:
    build_csr = _import_optional("hadukenn_core.sheaf_bridge", "build_sheaf_bridge_csr")
    if build_csr is None:
        return
    rng = np.random.default_rng(233)
    oracle = rng.normal(0.0, 1.0, size=(128, 32)).astype(np.float32)
    w_warm, _ = whitened_spectral_warmstart(oracle.astype(np.float64), d_student=16)
    csr_res = build_csr(oracle, k=8, chunk_size=32, dtype="float32")
    smoothed = csr_res["operator"].matvec(w_warm[:, 0].astype(np.float32))
    assert np.all(np.isfinite(smoothed))
    _assert_torch_bridge_ast_contracts()

    if torch_bridge.TORCH_AVAILABLE:
        import torch

        cu = torch.arange(128, dtype=torch.long).view(16, 8)
        head = torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=16, k_neg=3, chunk_tokens=16)
        oe = torch_bridge.OESpanEmbeddingTorch(n_spans=16, d_span=8, d_model=16)
        h_t = oe(torch.randn(2, 4, 16), torch.full((2, 4), -1, dtype=torch.long))
        targets = torch.randint(0, 128, (2, 4), dtype=torch.long)
        lm_loss, root_nll = head(h_t, targets, torch.from_numpy(w_warm.astype(np.float32)))
        assert torch.isfinite(lm_loss).item() and torch.isfinite(root_nll).item()


def test_t3_e5_e6_torch_bridge_audit_and_e2e_runner_gate() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if run_local_e2e is None or verify_t46 is None:
        return
    rep = run_local_e2e()
    assert rep["torch_bridge_audit_ok"] is True
    ok_pass, _, _ = verify_t46(rep)
    assert ok_pass is True

    tampered = json.loads(json.dumps(rep))
    tampered["torch_bridge_audit_ok"] = False
    ok_fail, _, _ = verify_t46(tampered)
    assert ok_fail is False


def test_t3_e6_e7_local_e2e_runner_preserves_hygiene_and_anchor(tmp_path: Path) -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    if run_local_e2e is None:
        return
    out_json = tmp_path / "r.json"
    sig = inspect.signature(run_local_e2e)
    rep = run_local_e2e(json_out=out_json) if "json_out" in sig.parameters else run_local_e2e()
    serialized = json.dumps(rep, sort_keys=True)
    for drive in _BANNED_DRIVE_PREFIXES:
        assert drive not in serialized
    if out_json.is_file():
        disk_text = out_json.read_text(encoding="utf-8")
        for drive in _BANNED_DRIVE_PREFIXES:
            assert drive not in disk_text
    vrep = verify_store(Store(ROOT / "fixtures" / "passing"), key=load_key(ROOT / "fixtures" / "test_seal.key"))
    assert vrep.ok is True and vrep.head == CANONICAL_ANCHOR
    assert audit_python_style_stdlib(ROOT) == []


def test_t3_e1_e6_opcounter_telemetry_in_t46_campaign() -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if run_local_e2e is None or verify_t46 is None:
        return
    rep = run_local_e2e()
    assert float(rep.get("oryon_simd_speedup_vs_numpy", rep.get("simd_speedup_vs_numpy", 0.0))) > 1.0
    ok_flag, criteria, _ = verify_t46(rep)
    assert ok_flag is True, [c.as_dict() for c in criteria]


def test_t3_e2_e4_npu_int8_quant_with_sheaf_bridge_embeddings() -> None:
    run_htp = _import_optional("hadukenn_core.npu_qnn_backend", "run_int8_matmul_htp")
    if run_htp is None:
        return
    rng = np.random.default_rng(239)
    oracle = rng.normal(0.0, 1.0, size=(64, 48))
    w_warm, _ = whitened_spectral_warmstart(oracle, d_student=32)
    w_q_float = quantize_dequantize_symmetric_int8(w_warm)
    scale = np.maximum(np.max(np.abs(w_q_float), axis=1, keepdims=True) / 127.0, 1e-8)
    w_i8 = np.clip(np.round(w_q_float / scale), -127, 127).astype(np.int8)
    x_i8 = rng.integers(-64, 64, size=(16, 64), dtype=np.int8)
    ref = x_i8.astype(np.int32) @ w_i8.astype(np.int32)
    tiers = ("auto", "oryon_i8mm", "numpy_ref") if _has_oryon_i8mm() else ("auto", "numpy_ref")
    for tier in tiers:
        res = run_htp(x_i8, w_i8, force_tier=tier)
        assert np.array_equal(res["output"], ref)


# ===========================================================================
# Tier 4: Real-World Application Scenarios (6 tests)
# ===========================================================================


def test_t4_local_51m_inference_loop_within_snapdragon_budget() -> None:
    plan_fn = _import_optional("hadukenn_core.snapdragon_local", "plan_snapdragon_local_tiers")
    run_dot_f32 = _import_optional("hadukenn_core.fast_ops", "run_dot_f32")
    run_axpy_f32 = _import_optional("hadukenn_core.fast_ops", "run_axpy_f32")
    run_int8_gemm = _import_optional("hadukenn_core.fast_ops", "run_int8_gemm")
    run_nf4_matvec = _import_optional("hadukenn_core.fast_ops", "run_nf4_matvec")
    if None in (plan_fn, run_dot_f32, run_axpy_f32, run_int8_gemm, run_nf4_matvec):
        return

    tiers = plan_fn()
    by_name = {(_get_tier_field(t, "name") or _get_tier_field(t, "tier")): t for t in tiers}
    t51 = by_name["51M"]
    assert int(_get_tier_field(t51, "vocab_size")) == 32768
    assert _get_tier_field(t51, "head_kind") == "flat_dense_struct32k"

    rng = np.random.default_rng(307)
    d_model = 576
    batch_n = 8
    h = rng.normal(0.0, 0.2, size=(batch_n, d_model)).astype(np.float32)
    w_proj = rng.normal(0.0, 0.2, size=(d_model, d_model)).astype(np.float32)
    qw, scales = _pack_nf4_weights(w_proj, group_size=64)

    h_nf4 = np.stack([run_nf4_matvec(h[i], qw, scales, group_size=64) for i in range(batch_n)], axis=0)
    h_res = np.stack([run_axpy_f32(0.5, h_nf4[i], h[i].copy()) for i in range(batch_n)], axis=0)
    norm_val = float(run_dot_f32(h_res[0], h_res[0]))
    assert math.isfinite(norm_val) and norm_val > 0.0

    a_i8 = np.clip(np.round(h_res * 20.0), -127, 127).astype(np.int8)
    b_i8 = rng.integers(-32, 32, size=(d_model, 64), dtype=np.int8)
    gemm_out = run_int8_gemm(a_i8, b_i8, mode="auto")
    assert np.array_equal(gemm_out, a_i8.astype(np.int32) @ b_i8.astype(np.int32))

    # Flat dense struct32k head check on synthetic 32,768 vocabulary
    emb_32k = rng.normal(0.0, 0.05, size=(32768, d_model)).astype(np.float32)
    targets = rng.integers(0, 32768, size=(batch_n,), dtype=np.int32)
    loss_nat, _, _ = run_tiled_cce_avx512(h_res, emb_32k, targets, tile_V=512, compute_grad=False)
    assert loss_nat.shape == (batch_n,) and np.all(np.isfinite(loss_nat))


def test_t4_wsl2_to_win_htp_npu_offload_and_tier_cascade_resilience(monkeypatch: Any) -> None:
    npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
    if npu_mod is None:
        return
    rng = np.random.default_rng(311)
    a_i8 = rng.integers(-64, 64, size=(64, 128), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(128, 64), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)

    res_auto = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    assert np.array_equal(res_auto["output"], ref)

    monkeypatch.setattr(npu_mod, "discover_windows_python", lambda: None)
    res_t2 = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    assert res_t2["tier_used"] in {"oryon_i8mm", "numpy_ref"}
    assert np.array_equal(res_t2["output"], ref)

    res_t3 = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="numpy_ref")
    assert res_t3["tier_used"] == "numpy_ref"
    assert np.array_equal(res_t3["output"], ref)


def test_t4_v32k_sparse_sheaf_bridge_and_flat_dense_head_pipeline() -> None:
    build_csr = _import_optional("hadukenn_core.sheaf_bridge", "build_sheaf_bridge_csr")
    if build_csr is None:
        return
    rng = np.random.default_rng(313)
    oracle = rng.normal(0.0, 1.0, size=(32768, 64)).astype(np.float32)
    csr_res = build_csr(oracle, k=8, chunk_size=1024, dtype="float16")
    assert int(csr_res["csr_bytes"]) <= 8 * 1024 * 1024
    assert int(csr_res["peak_ram_bytes"]) < 64 * 1024 * 1024
    assert csr_res["within_budget"] is True

    # Student 64d flat dense head loss parity
    emb = (oracle / np.maximum(np.linalg.norm(oracle, axis=1, keepdims=True), 1e-6)).astype(np.float32)
    h = rng.normal(0.0, 0.25, size=(4, 64)).astype(np.float32)
    targets = rng.integers(0, 32768, size=(4,), dtype=np.int32)
    loss_tiled, dh_tiled, _ = tiled_cce_fwd_bwd(h, emb, targets, tile_V=512, compute_grad=True)
    loss_flat, dh_flat, _ = flat_tied_fwd_bwd(h, emb, targets, compute_grad=True)
    assert np.allclose(loss_tiled, loss_flat, atol=1e-4)
    assert dh_tiled is not None and dh_flat is not None
    assert np.allclose(dh_tiled, dh_flat, atol=1e-4)


def test_t4_single_command_local_e2e_to_t46_campaign_and_hygiene(tmp_path: Path) -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    run_local_e2e = _import_optional("hadukenn_core.local_e2e_runner", "run_local_e2e")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if run_local_e2e is None or verify_t46 is None:
        return

    out_json = tmp_path / "t46_e2e.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    p1 = subprocess.run(
        [sys.executable, "-m", "hadukenn_core.local_e2e_runner", "--json-out", str(out_json)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p1.returncode == 0, p1.stderr

    p2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "sheaf_verify",
            "campaign",
            "--mode",
            "t46-snapdragon-local",
            "--path",
            str(out_json),
            "--json",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p2.returncode == 0, p2.stderr
    assert json.loads(p2.stdout)["ok"] is True

    text_json = out_json.read_text(encoding="utf-8")
    for drive in _BANNED_DRIVE_PREFIXES:
        assert drive not in text_json
    vrep = verify_store(Store(ROOT / "fixtures" / "passing"), key=load_key(ROOT / "fixtures" / "test_seal.key"))
    assert vrep.ok is True and vrep.head == CANONICAL_ANCHOR
    p_docs = subprocess.run(
        [sys.executable, str(ROOT / ".github" / "scripts" / "docs_budget.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_docs.returncode == 0


def test_t4_all_realms_realm_c_end_to_end_with_scitt_compliance() -> None:
    docs = load_clean_technical_corpus()
    assert len(docs) >= 4
    rng = np.random.default_rng(317)
    w = rng.normal(0.0, 0.2, size=(32, 24)).astype(np.float32)
    x = rng.normal(0.0, 0.2, size=(16, 32)).astype(np.float32)
    engine = QuantizedINT8NPULoopEngine(w)
    y_q = engine.forward_npu_int8(x)
    rel_err = float(np.linalg.norm(y_q - (x @ w)) / max(float(np.linalg.norm(x @ w)), 1e-12))
    assert rel_err < 0.03

    # Also verify cluster partition + SCITT compliance bundle round-trip
    streams = [np.arange(0, 64, dtype=np.int32)]
    part = build_balanced_partition(64, 8, 8, streams, mode="paradigmatic_spectral", seed=13)
    assert part.verify_bijection()["bijection_ok"] is True

    seal_key = b"hadukenn-e2e-scitt-test-key"
    bundle = emit_regulatory_bundle(
        system_id="hadukenn-snapdragon-e2e-v0.9.0",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[
            {
                "approved": True,
                "human_escalation_required": False,
                "rel_err": round(rel_err, 6),
                "verdict": "VALID",
            }
        ],
        created_at="2026-11-26T19:00:00Z",
        signing_key=seal_key,
    )
    ok_b, errs = verify_regulatory_bundle(bundle, signing_key=seal_key)
    assert ok_b is True, errs


def test_t4_cold_start_no_gcc_no_cuda_clean_recovery(monkeypatch: Any) -> None:
    npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
    detect = _import_optional("hadukenn_core.fast_ops", "detect_oryon_capabilities")
    suite_fn = _import_optional("hadukenn_core.snapdragon_local", "run_t46_snapdragon_local_suite")
    if npu_mod is None or detect is None or suite_fn is None:
        return

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(npu_mod, "discover_windows_python", lambda: None)

    def _deny_socket(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("No network access permitted during cold-start recovery test")

    monkeypatch.setattr(socket, "socket", _deny_socket)

    cap = detect()
    if _is_arm64():
        assert cap.get("has_neon") is True
    rng = np.random.default_rng(331)
    a_i8 = rng.integers(-32, 32, size=(16, 32), dtype=np.int8)
    b_i8 = rng.integers(-32, 32, size=(32, 16), dtype=np.int8)
    res = npu_mod.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    assert res["tier_used"] in {"oryon_i8mm", "numpy_ref"}
    assert np.array_equal(res["output"], a_i8.astype(np.int32) @ b_i8.astype(np.int32))

    _assert_torch_bridge_ast_contracts()
    if _has_oryon_i8mm():
        rep = suite_fn()
        assert isinstance(rep, dict)
    assert audit_python_style_stdlib(ROOT) == []


# ===========================================================================
# Tier 5: White-Box Adversarial Hardening (9 tests)
# ===========================================================================


def test_t5_e1_head_kernels_reject_timedelta64_zero_and_nonzero_n() -> None:
    run_nf4 = _import_optional("hadukenn_core.fast_ops", "run_nf4_matvec")
    run_axpy = _import_optional("hadukenn_core.fast_ops", "run_axpy_f32")
    if run_nf4 is None or run_axpy is None:
        return

    e_cce = np.zeros((8, 4), dtype=np.float32)
    e_cl = np.zeros((4, 3, 4), dtype=np.float32)
    c_rt = np.zeros((4, 4), dtype=np.float32)
    tc_ok = np.array([0, 1], dtype=np.int32)
    tm_ok = np.array([0, 1], dtype=np.int32)
    neg_ok = np.array([[1, 2], [2, 3]], dtype=np.int32)

    for n_tok in (0, 2):
        h_mat = np.zeros((n_tok, 4), dtype=np.float32)
        tc_0 = np.zeros((n_tok,), dtype=np.int32)
        tm_0 = np.zeros((n_tok,), dtype=np.int32)
        neg_0 = np.zeros((n_tok, 2), dtype=np.int32)
        for bad_dt in ("timedelta64[s]", "datetime64[s]", "bool"):
            bad_1d = np.zeros((n_tok,), dtype=bad_dt)
            bad_2d = np.zeros((n_tok, 2), dtype=bad_dt)
            with pytest.raises(ValueError):
                run_tiled_cce_avx512(h_mat, e_cce, bad_1d)
            with pytest.raises(ValueError):
                run_cluster_is_avx512(h_mat, e_cl, c_rt, bad_1d, tm_0 if n_tok == 0 else tm_ok, neg_0 if n_tok == 0 else neg_ok)
            with pytest.raises(ValueError):
                run_cluster_is_avx512(h_mat, e_cl, c_rt, tc_0 if n_tok == 0 else tc_ok, bad_1d, neg_0 if n_tok == 0 else neg_ok)
            with pytest.raises(ValueError):
                run_cluster_is_avx512(h_mat, e_cl, c_rt, tc_0 if n_tok == 0 else tc_ok, tm_0 if n_tok == 0 else tm_ok, bad_2d)

        # Reject d == 0 at both N == 0 and N > 0
        h_d0 = np.zeros((n_tok, 0), dtype=np.float32)
        e_cce_d0 = np.zeros((8, 0), dtype=np.float32)
        e_cl_d0 = np.zeros((4, 3, 0), dtype=np.float32)
        c_rt_d0 = np.zeros((4, 0), dtype=np.float32)
        with pytest.raises(ValueError):
            run_tiled_cce_avx512(h_d0, e_cce_d0, tc_0 if n_tok == 0 else tc_ok)
        with pytest.raises(ValueError):
            run_cluster_is_avx512(
                h_d0,
                e_cl_d0,
                c_rt_d0,
                tc_0 if n_tok == 0 else tc_ok,
                tm_0 if n_tok == 0 else tm_ok,
                neg_0 if n_tok == 0 else neg_ok,
            )

    # Reject bool / float coercion on group_size and bool alpha
    x_nf4 = np.ones((4,), dtype=np.float32)
    qw_nf4 = np.zeros((2, 2), dtype=np.uint8)
    sc_nf4 = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        run_nf4(x_nf4, qw_nf4, sc_nf4, group_size=True)
    with pytest.raises(ValueError):
        run_nf4(x_nf4, qw_nf4, sc_nf4, group_size=2.5)
    with pytest.raises(ValueError):
        run_axpy(True, x_nf4, x_nf4)


def test_t5_e1_nf4_matvec_odd_group_size_and_k_parity() -> None:
    run_nf4 = _import_optional("hadukenn_core.fast_ops", "run_nf4_matvec")
    if run_nf4 is None:
        return
    rng = np.random.default_rng(501)
    thresholds = 0.5 * (NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:])

    for m_rows, k_cols, gs in ((5, 35, 1), (7, 35, 3), (9, 65, 17), (11, 65, 63)):
        w_fp = (rng.normal(0.0, 0.4, size=(m_rows, k_cols))).astype(np.float32)
        x_fp = (rng.normal(0.0, 0.5, size=(k_cols,))).astype(np.float32)
        packed_cols = (k_cols + 1) // 2
        n_groups = (k_cols + gs - 1) // gs
        packed = np.zeros((m_rows, packed_cols), dtype=np.uint8)
        scales = np.zeros((m_rows, n_groups), dtype=np.float32)
        w_deq = np.zeros((m_rows, k_cols), dtype=np.float64)

        for g_idx in range(n_groups):
            c0 = g_idx * gs
            c1 = min(c0 + gs, k_cols)
            blk = w_fp[:, c0:c1]
            absmax = np.maximum(np.max(np.abs(blk), axis=1), 1e-12).astype(np.float32)
            scales[:, g_idx] = absmax
            codes = np.searchsorted(thresholds, blk / absmax[:, None]).astype(np.uint8)
            for off in range(c1 - c0):
                col = c0 + off
                p_idx = col // 2
                if col % 2 == 0:
                    packed[:, p_idx] |= (codes[:, off] & 0x0F) << 4
                else:
                    packed[:, p_idx] |= codes[:, off] & 0x0F
                w_deq[:, col] = NF4_CODEBOOK[codes[:, off]].astype(np.float64) * absmax.astype(np.float64)

        got = run_nf4(x_fp, packed, scales, group_size=gs)
        ref = (w_deq @ x_fp.astype(np.float64)).astype(np.float32)
        assert got.shape == (m_rows,)
        assert float(np.max(np.abs(got - ref))) < 1e-4


def test_t5_e2_htp_canary_0x00_zero_and_1x1_9638_no_false_reject(monkeypatch: Any) -> None:
    npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
    worker_mod = _import_optional("hadukenn_core.npu_qnn_worker")
    if npu_mod is None or worker_mod is None:
        return

    # 1. Simulated no-op worker must ALWAYS fail closed (even with alt_byte disambiguation)
    class _FakeOrtValue:
        @staticmethod
        def ortvalue_from_numpy(arr: np.ndarray, *_args: Any) -> np.ndarray:
            return arr

    class _FakeIOBinding:
        def bind_ortvalue_input(self, *_args: Any) -> None:
            return

        def bind_ortvalue_output(self, *_args: Any) -> None:
            return

    class _NoopSess:
        def io_binding(self) -> _FakeIOBinding:
            return _FakeIOBinding()

        def run_with_iobinding(self, _iob: Any) -> None:
            return

    fake_ort = types.SimpleNamespace(OrtValue=_FakeOrtValue)
    orig_modules = sys.modules
    monkeypatch.setattr(sys, "modules", {**orig_modules, "onnxruntime": fake_ort})
    _, noop_ok, _ = worker_mod._run_session_with_canary(
        _NoopSess(),
        np.full((1, 1), 32768, dtype=np.uint16),
        (1, 1),
        0x5A,
    )
    assert noop_ok is False
    monkeypatch.setattr(sys, "modules", orig_modules)

    # 2. Live HTP v73 verification when available: 0x00 zeros @ zeros and 1x1 [[122]] @ [[-79]] == [[-9638]]
    if npu_mod.discover_windows_python() is None:
        return

    a_zero = np.zeros((4, 8), dtype=np.int8)
    b_zero = np.zeros((8, 4), dtype=np.int8)
    res_zero = npu_mod.run_int8_matmul_htp(a_zero, b_zero, force_tier="htp_qnn", canary_byte=0x00)
    assert res_zero["tier_used"] == "htp_qnn"
    assert res_zero["canary_ok"] is True
    assert np.array_equal(res_zero["output"], np.zeros((4, 4), dtype=np.int32))

    a_col = np.array([[122]], dtype=np.int8)
    b_col = np.array([[-79]], dtype=np.int8)
    res_col = npu_mod.run_int8_matmul_htp(a_col, b_col, force_tier="htp_qnn", canary_byte=0x5A)
    assert res_col["tier_used"] == "htp_qnn"
    assert res_col["canary_ok"] is True
    assert np.array_equal(res_col["output"], np.array([[-9638]], dtype=np.int32))


def test_t5_e2_zero_dim_force_tier_i8mm_and_worker_cleanup(monkeypatch: Any) -> None:
    npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
    if npu_mod is None:
        return

    npu_mod.close_all_htp_sessions()
    if sys.platform != "win32" and npu_mod.discover_windows_python() is not None:
        sess = npu_mod.HTPBackendSession()
        worker = sess._worker
        assert worker is not None and worker in npu_mod._ACTIVE_WORKERS
        sess.close()
        assert worker not in npu_mod._ACTIVE_WORKERS

    monkeypatch.setattr(npu_mod, "detect_oryon_capabilities", lambda: {"has_i8mm": False})
    with pytest.raises(RuntimeError):
        npu_mod.run_int8_matmul_htp(
            np.zeros((0, 4), dtype=np.int8),
            np.zeros((4, 4), dtype=np.int8),
            force_tier="oryon_i8mm",
        )


def test_t5_e4_csr_invalid_reg_mst_weight_and_2d_empty_rows() -> None:
    sb = _import_optional("hadukenn_core.sheaf_bridge")
    if sb is None:
        return

    rng = np.random.default_rng(505)
    emb = rng.normal(0.0, 1.0, size=(16, 8)).astype(np.float32)

    for bad_reg in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            sb.compute_lle_barycentric_weights_csr(emb, k=4, reg=bad_reg)
        with pytest.raises(ValueError):
            sb.build_sheaf_bridge_csr(emb, k=4, reg=bad_reg)

    for bad_mst in (-0.5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            sb.build_sheaf_bridge_csr(emb, k=4, mst_weight=bad_mst)

    indptr = np.array([0, 2, 2, 4, 4, 5], dtype=np.int32)
    indices = np.array([0, 3, 1, 4, 2], dtype=np.int32)
    data = np.array([1.5, -2.0, 0.5, 3.0, -1.0], dtype=np.float16)
    csr_f16 = sb.SparseCSRMatrix(shape=(5, 5), indptr=indptr, indices=indices, data=data)
    dense_f16 = csr_f16.to_dense()
    x_2d = rng.normal(0.0, 1.0, size=(5, 4)).astype(np.float32)
    assert np.allclose(csr_f16.matvec(x_2d), dense_f16 @ x_2d, atol=1e-5)
    assert np.allclose(csr_f16.rmatvec(x_2d), dense_f16.T @ x_2d, atol=1e-5)

    csr_i8 = sb._convert_csr_dtype(csr_f16, "int8")
    dense_i8 = csr_i8.to_dense()
    assert np.allclose(csr_i8.matvec(x_2d), dense_i8 @ x_2d, atol=1e-5)
    assert np.allclose(csr_i8.rmatvec(x_2d), dense_i8.T @ x_2d, atol=1e-5)


def test_t5_e5_torch_bridge_init_validation_and_zero_token_grad() -> None:
    _assert_torch_bridge_ast_contracts()
    if not torch_bridge.TORCH_AVAILABLE:
        return
    import torch

    cu = torch.arange(16, dtype=torch.long).view(4, 4)
    w_fp = torch.randn(4, 8, dtype=torch.float32)

    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=16, k_neg=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=16, k_neg=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=16, chunk_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=0)
    with pytest.raises(ValueError):
        torch_bridge.NF4FrozenLinearTorch(w_fp, block_size=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        torch_bridge.NF4FrozenLinearTorch(w_fp, block_size=4, lora_rank=-1)
    with pytest.raises(ValueError):
        torch_bridge.NF4FrozenLinearTorch(w_fp, block_size=4, lora_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        torch_bridge.AsymmetricSheafKVPrefixTorch(d_model=10, d_stalk=8, n_heads=3)
    with pytest.raises(ValueError):
        torch_bridge.AsymmetricSheafKVPrefixTorch(d_model=12, d_stalk=8, n_heads=0)
    with pytest.raises(ValueError):
        torch_bridge.AsymmetricSheafKVPrefixTorch(d_model=12, d_stalk=8, n_heads=True)  # type: ignore[arg-type]

    head = torch_bridge.ParadigmaticClusterISHeadTorch(cu, d_model=8, k_neg=2)
    base_w = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    hid_0 = torch.randn(2, 0, 8, dtype=torch.float32, requires_grad=True)
    tgt_0 = torch.empty((2, 0), dtype=torch.int64)
    lm_loss, root_nll = head(hid_0, tgt_0, base_w)
    (lm_loss + root_nll).backward()
    assert base_w.grad is not None and torch.isfinite(base_w.grad).all()
    assert head.delta_root.grad is not None and torch.isfinite(head.delta_root.grad).all()
    assert hid_0.grad is not None and hid_0.grad.shape == (2, 0, 8)


def test_t5_e2_quantized_npu_loop_engine_empty_and_0d_rejected() -> None:
    with pytest.raises(ValueError):
        QuantizedINT8NPULoopEngine(np.zeros((0, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        QuantizedINT8NPULoopEngine(np.zeros((4, 0), dtype=np.float32))
    with pytest.raises(ValueError):
        QuantizedINT8NPULoopEngine(np.zeros((4,), dtype=np.float32))

    eng = QuantizedINT8NPULoopEngine(np.ones((4, 3), dtype=np.float32), force_tier="numpy_ref")
    with pytest.raises(ValueError):
        eng.forward_npu_int8(np.array(1.0, dtype=np.float32))
    with pytest.raises(ValueError):
        eng.forward_npu_int8(np.zeros((0,), dtype=np.float32))
    with pytest.raises(ValueError):
        eng.forward_npu_int8(np.zeros((0, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        eng.forward_npu_int8(np.ones((2, 5), dtype=np.float32))


def test_t5_e6_snapdragon_manifest_neg_zero_and_campaign_tier_consistency(monkeypatch: Any) -> None:
    _require_oryon_i8mm("requires ARM64 Oryon i8mm CPU")
    sl = _import_optional("hadukenn_core.snapdragon_local")
    verify_t46 = _import_optional("sheaf_verify.campaign", "verify_t46_snapdragon_local")
    if sl is None or verify_t46 is None:
        return

    # 1. -0.0 inside report fields must be normalized before manifest_sha256
    orig_bench = sl.benchmark_kernel_parity

    def _neg_zero_bench(name: str, fn: Any, ref_fn: Any, **kwargs: Any) -> dict[str, Any]:
        rep = orig_bench(name, fn, ref_fn, **kwargs)
        rep["max_abs_err"] = -0.0
        return rep

    monkeypatch.setattr(sl, "benchmark_kernel_parity", _neg_zero_bench)
    rep = sl.run_t46_snapdragon_local_suite()
    ok_rep, crit_rep, _ = verify_t46(rep)
    assert ok_rep is True, [c for c in crit_rep if not c.passed]

    # 2. Mismatched npu_tier_used vs execution_tiers.auto_selected.tier_used must fail even if re-signed
    tamper_tier = json.loads(json.dumps(rep))
    tamper_tier["npu_tier_used"] = "numpy_ref" if tamper_tier["npu_tier_used"] != "numpy_ref" else "htp_qnn"
    tamper_tier["manifest_sha256"] = sl.compute_manifest_sha256(tamper_tier)
    assert verify_t46(tamper_tier)[0] is False

    # 3. Any drive-letter backslash path (e.g. Z:) must fail even if re-signed
    tamper_drv = json.loads(json.dumps(rep))
    tamper_drv["extra_note"] = "Z:" + chr(92) + "secret"
    tamper_drv["manifest_sha256"] = sl.compute_manifest_sha256(tamper_drv)
    assert verify_t46(tamper_drv)[0] is False


def test_t5_e7_whole_repo_ruff_zero_errors() -> None:
    assert audit_python_style_stdlib(ROOT) == []

    manifest_path = ROOT / "iterations" / "exec-hypergraph-v01" / "results" / "confirmation" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel_path, expected_sha in {**manifest["code_sha256"], **manifest["spec_sha256"]}.items():
        data = (ROOT / "iterations" / "exec-hypergraph-v01" / rel_path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected_sha, f"Locked iteration file changed: {rel_path}"

    ruff_bin = shutil.which("ruff")
    cmd: list[str] | None = [ruff_bin, "check", "."] if ruff_bin else None
    if cmd is None:
        npu_mod = _import_optional("hadukenn_core.npu_qnn_backend")
        win_py = npu_mod.discover_windows_python() if npu_mod is not None else None
        if win_py is not None:
            probe = subprocess.run(
                [win_py, "-m", "ruff", "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if probe.returncode == 0:
                cmd = [win_py, "-m", "ruff", "check", "."]

    if cmd is not None:
        res = subprocess.run(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert res.returncode == 0, f"Ruff check errors: {res.stdout} {res.stderr}"
