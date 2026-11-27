"""Integration tests for v0.9.0 Tri-Machine HAL, Hexagon QNN backend, and dual T-46 campaigns.

Designed to be 100% compatible with both pytest and ``sheaf_verify.selftest``
(no ``@pytest.mark.*``, ``@pytest.fixture``, or ``pytest.skip``).
"""

from __future__ import annotations

import copy
import ctypes
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hadukenn_core
import sheaf_verify
from hadukenn_core import local_e2e_runner, npu_qnn_backend
from hadukenn_core.fast_ops import OpCounter, detect_oryon_capabilities
from hadukenn_core.npu_qnn_backend import HexagonQNNBackend, HTPBackendSession
from hadukenn_core.tri_machine import (
    StaticSlabArena,
    ZenbookLoopedNPUEngine,
    detect_host_isa,
    get_tri_machine_lib,
    hal_qgemm_int8,
    quantize_symmetric_int8,
    run_tri_machine_verification_suite,
)
from sheaf_verify.campaign import verify_t46_snapdragon_local, verify_t46_tri_machine_hal
from sheaf_verify.cli import main as cli_main
from sheaf_verify.hashing import canonical_bytes, sha256_hex

ROOT = Path(__file__).resolve().parents[1]


def _sign_t46_snapdragon_doc(doc: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    signed = dict(body)
    signed["manifest_sha256"] = sha256_hex(canonical_bytes(body))
    return signed


def _valid_t46_snapdragon_base() -> dict[str, Any]:
    tiers_raw = [
        ("51M", 50.72, 0.18, 1200.0, 8.12, True),
        ("125M", 124.26, 0.35, 650.0, 9.86, True),
        ("1.5B", 1479.02, 1.28, 78.0, 12.18, False),
        ("3.2B", 3242.72, 3.74, 24.0, 13.92, False),
        ("8B", 8053.06, 5.48, 16.0, 14.5, False),
    ]
    tiers = [
        {
            "name": name,
            "tier": name,
            "vocab_size": 32768,
            "head_kind": "flat_dense_struct32k",
            "total_params_m": params_m,
            "required_ram_gib": ram_gib,
            "fits_physical_ram": ram_gib <= 15.61,
            "fits_wsl2_mem_available": ram_gib <= 2.0,
            "bandwidth_tok_s": tok_s,
            "power_w": power_w,
            "live_executed": want_live,
        }
        for name, params_m, ram_gib, tok_s, power_w, want_live in tiers_raw
    ]
    return {
        "schema": "sheaf-verify.t46-snapdragon-local.v1",
        "version": "0.9.0",
        "hardware": {
            "soc": "X1P-42-100",
            "cores": 8,
            "bandwidth_gb_s": 135.0,
            "tdp_watts": 14.5,
            "physical_ram_gib": 15.61,
            "wsl2_mem_available_gib": 2.0,
        },
        "tiers": tiers,
        "sheaf_bridge_csr": {
            "vocab_size": 32768,
            "csr_bytes": 2424828,
            "peak_ram_bytes": 27663136,
            "within_budget": True,
            "dense_flat_head_preserved": True,
        },
        "torch_bridge_audit_ok": True,
        "torch_bridge_violations": [],
        "int8_parity_ok": True,
        "int8_max_abs_err": 0.0,
        "oryon_simd_speedup_vs_numpy": 18.4,
        "op_counter": {"mac_ops": 16777216, "calls": 4},
        "execution_tiers": {
            "oryon_i8mm": {"parity_ok": True, "speedup_vs_numpy": 18.4},
            "numpy_ref": {"parity_ok": True},
        },
        "npu_tier_used": "htp_qnn",
        "npu_canary_ok": True,
        "no_wslconfig_modified": True,
        "no_network_downloads": True,
    }


def test_integration_r1_compiler_isa_pthreads_and_qgemm_parity(monkeypatch: Any) -> None:
    lib = get_tri_machine_lib(force_rebuild=True)
    assert lib is not None
    isa = detect_host_isa()
    assert isa["backend"] != "python_numpy_fallback"
    if platform.machine().lower() in ("aarch64", "arm64"):
        assert isa["backend"] == "arm64_neon_i8mm_sdot"
        assert "ARM_I8MM" in isa["flags"]
        assert "ARM_DOTPROD" in isa["flags"]

    c_hal = (ROOT / "src" / "hadukenn_core" / "csrc" / "tri_machine_hal.c").read_text(encoding="utf-8")
    assert "pthread_create" in c_hal
    assert "vmmlaq_s32" in c_hal
    assert "arm64_neon_i8mm_sdot" in c_hal

    rng = np.random.default_rng(2026)
    xf = rng.standard_normal((32, 128)).astype(np.float32)
    wf = rng.standard_normal((128, 128)).astype(np.float32)
    gemv_out = np.empty((32, 128), dtype=np.float32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    lib.hadukenn_gemv_f32(
        xf.ctypes.data_as(f32_p),
        wf.ctypes.data_as(f32_p),
        gemv_out.ctypes.data_as(f32_p),
        32,
        128,
        128,
        ctypes.c_float(0.75),
    )
    assert np.max(np.abs(gemv_out - (0.75 * (xf @ wf.T)))) < 1e-4

    xq, xs = quantize_symmetric_int8(xf)
    wq, ws = quantize_symmetric_int8(wf)
    q_out = hal_qgemm_int8(xq, xs, wq, ws)
    q_ref = (xq.astype(np.int32) @ wq.astype(np.int32).T).astype(np.float32) * (xs[:, None] * ws[None, :])
    assert np.max(np.abs(q_out - q_ref)) < 1e-4

    monkeypatch.setenv("CC", "/nonexistent/compiler_missing_xyz")
    assert get_tri_machine_lib() is None
    assert detect_host_isa()["backend"] == "python_numpy_fallback"
    monkeypatch.delenv("CC", raising=False)
    assert get_tri_machine_lib() is not None


def test_integration_r2_hexagon_qnn_backend_lifecycle_and_validation(monkeypatch: Any) -> None:
    with pytest.raises(ValueError):
        HexagonQNNBackend(force_tier="invalid_tier")
    with pytest.raises(ValueError):
        HexagonQNNBackend(dcvs_mode="invalid_dcvs")

    rng = np.random.default_rng(303)
    x_q = rng.integers(-64, 64, size=(8, 16), dtype=np.int8)
    w_q_t = rng.integers(-64, 64, size=(16, 12), dtype=np.int8)
    x_s = np.ones(8, dtype=np.float32) * 0.125
    w_s = np.ones(12, dtype=np.float32) * 0.25

    with HexagonQNNBackend(force_tier="numpy_ref", counter=OpCounter()) as backend:
        assert backend.closed is False
        assert backend.is_available() is True
        out = backend.qgemm_int8_f32(x_q, x_s, w_q_t, w_s)
        ref = (x_q.astype(np.int32) @ w_q_t.astype(np.int32)).astype(np.float32) * (x_s[:, None] * w_s[None, :])
        assert np.allclose(out, ref, atol=1e-5)
        assert backend.last_tier_used == "numpy_ref"
        assert backend.last_canary_ok is True
        assert backend.last_providers == ["NumPyReference"]
        assert backend.counter.calls == 1

        with pytest.raises(ValueError):
            backend.matmul_int8(x_q, w_q_t, force_tier="bad_tier")
        with pytest.raises(ValueError):
            backend.qgemm_int8_f32(x_q, np.ones(7, dtype=np.float32), w_q_t, w_s)
        with pytest.raises(ValueError):
            backend.qgemm_int8_f32(x_q, x_s, w_q_t, np.ones(11, dtype=np.float32))

    assert backend.closed is True
    assert backend.is_available() is False
    assert backend.htp_available() is False
    with pytest.raises(RuntimeError):
        backend.matmul_int8(x_q, w_q_t)
    with pytest.raises(RuntimeError):
        backend.qgemm_int8_f32(x_q.astype(np.float32), x_s, w_q_t, w_s)

    if sys.platform != "win32":
        with HexagonQNNBackend(force_tier="htp_qnn", win_python="/nonexistent/python.exe") as bad_win_htp:
            with pytest.raises(RuntimeError):
                bad_win_htp.matmul_int8(
                    np.zeros((0, 4), dtype=np.int8),
                    np.zeros((4, 3), dtype=np.int8),
                )
        with HexagonQNNBackend(force_tier="auto", win_python="/nonexistent/python.exe") as bad_win_auto:
            res_z = bad_win_auto.matmul_int8(
                np.zeros((0, 4), dtype=np.int8),
                np.zeros((4, 3), dtype=np.int8),
            )
            assert res_z["tier_used"] != "htp_qnn"
            assert res_z["output"].shape == (0, 3)

    # Monkeypatch-aware delegation test
    calls: list[str] = []

    def _mock_run_htp(
        a: np.ndarray,
        b: np.ndarray,
        force_tier: str | None = "auto",
        *,
        counter: OpCounter | None = None,
        dcvs_mode: str = "burst",
        canary_byte: int = 0x5A,
    ) -> dict[str, Any]:
        calls.append(f"{force_tier}:{dcvs_mode}:{canary_byte}")
        if counter is not None:
            counter.calls += 1
        return {
            "output": a.astype(np.int32) @ b.astype(np.int32),
            "tier_used": "htp_qnn",
            "canary_ok": True,
            "providers": ["QNNExecutionProvider"],
        }

    monkeypatch.setattr(npu_qnn_backend, "run_int8_matmul_htp", _mock_run_htp)
    with HexagonQNNBackend(force_tier="htp_qnn") as mocked_backend:
        res_m = mocked_backend.matmul_int8(x_q, w_q_t)
        assert res_m["tier_used"] == "htp_qnn"
        assert mocked_backend.last_tier_used == "htp_qnn"
        assert mocked_backend.last_providers == ["QNNExecutionProvider"]
    assert calls == ["htp_qnn:burst:90"]

    # HTPBackendSession context manager contract
    class _FakeWorker:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_sess = HTPBackendSession.__new__(HTPBackendSession)
    fake_sess.dcvs_mode = "burst"
    fake_sess.win_python = None
    fake_worker = _FakeWorker()
    fake_sess._worker = fake_worker  # type: ignore[assignment]
    with fake_sess as entered:
        assert entered is fake_sess
    assert fake_worker.closed is True
    assert fake_sess._worker is None


def test_integration_r2_zenbook_looped_npu_backends_and_fp32_parity(monkeypatch: Any) -> None:
    with pytest.raises(ValueError):
        ZenbookLoopedNPUEngine(d_model=64, d_ff=128, backend="unknown_backend")
    with pytest.raises(ValueError):
        ZenbookLoopedNPUEngine(d_model=64, d_ff=128, backend="hal_c11", dcvs_mode="bad")
    with pytest.raises(MemoryError):
        ZenbookLoopedNPUEngine(d_model=64, d_ff=128, arena_cap_bytes=4096)

    for use_c11 in (True, False):
        if not use_c11:
            monkeypatch.setenv("CC", "/nonexistent/compiler_missing_xyz")
        arena = StaticSlabArena(4096)
        assert arena.closed is False
        assert arena.alloc_bytes(64) >= 64
        arena.close()
        arena.close()
        assert arena.closed is True
        with pytest.raises(RuntimeError):
            arena.alloc_bytes(64)
        with pytest.raises(RuntimeError):
            arena.alloc_f32_view((4, 4))
        with pytest.raises(RuntimeError):
            arena.reset()
        if not use_c11:
            monkeypatch.delenv("CC", raising=False)

    x = np.ones((8, 128), dtype=np.float32) * 0.25
    backends_to_test = ["hal_c11", "numpy", "auto"]
    if detect_oryon_capabilities().get("has_i8mm"):
        backends_to_test.insert(1, "oryon_i8mm")
    if npu_qnn_backend.discover_windows_python() is not None:
        backends_to_test.append("htp_qnn")

    for backend_name in backends_to_test:
        with ZenbookLoopedNPUEngine(128, 384, loop_steps=4, backend=backend_name) as eng:
            assert eng.closed is False
            assert eng.w_up_q_t.shape == (128, 384)
            assert eng.w_dn_q_t.shape == (384, 128)
            assert eng.w_up_q_t.flags["C_CONTIGUOUS"]
            assert eng.w_dn_q_t.flags["C_CONTIGUOUS"]

            out_q, tel_q = eng.run_loop(x, use_fp32_ref=False)
            out_f, tel_f = eng.run_loop(x, use_fp32_ref=True)
            assert tel_q["dynamic_slab_allocs_in_loop"] == 0
            assert tel_f["dynamic_slab_allocs_in_loop"] == 0
            assert tel_q["backend_requested"] == backend_name
            assert tel_q["canary_ok"] is True
            assert tel_f["backend_used"] == "fp32_ref"
            cos = float(np.sum(out_q * out_f) / (np.linalg.norm(out_q) * np.linalg.norm(out_f)))
            assert cos >= 0.995, (backend_name, cos)

            with pytest.raises(ValueError):
                eng.run_loop(np.ones((8,), dtype=np.float32))
            with pytest.raises(ValueError):
                eng.run_loop(np.ones((200, 128), dtype=np.float32))

        # Idempotent close & unconditional post-close guard across all backends and fp32_ref
        assert eng.closed is True
        eng.close()
        with pytest.raises(RuntimeError):
            eng.run_loop(x, use_fp32_ref=False)
        with pytest.raises(RuntimeError):
            eng.run_loop(x, use_fp32_ref=True)

    assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0

    # Mocked htp_qnn verification (runs deterministically regardless of host HTP availability)
    monkeypatch.setattr(
        npu_qnn_backend,
        "run_int8_matmul_htp",
        lambda a, b, force_tier="auto", **_kw: {
            "output": a.astype(np.int32) @ b.astype(np.int32),
            "tier_used": "htp_qnn",
            "canary_ok": True,
            "providers": ["QNNExecutionProvider"],
        },
    )
    with ZenbookLoopedNPUEngine(64, 192, loop_steps=3, backend="htp_qnn") as mocked_eng:
        out_m, tel_m = mocked_eng.run_loop(np.ones((4, 64), dtype=np.float32) * 0.2, use_fp32_ref=False)
        out_r, _ = mocked_eng.run_loop(np.ones((4, 64), dtype=np.float32) * 0.2, use_fp32_ref=True)
        cos_m = float(np.sum(out_m * out_r) / (np.linalg.norm(out_m) * np.linalg.norm(out_r)))
        assert tel_m["dynamic_slab_allocs_in_loop"] == 0
        assert tel_m["backend_used"] == "htp_qnn"
        assert cos_m >= 0.995
    assert mocked_eng.closed is True
    with pytest.raises(RuntimeError):
        mocked_eng.run_loop(np.ones((4, 64), dtype=np.float32) * 0.2, use_fp32_ref=False)
    with pytest.raises(RuntimeError):
        mocked_eng.run_loop(np.ones((4, 64), dtype=np.float32) * 0.2, use_fp32_ref=True)


def test_integration_r3_package_metadata_and_build_scripts() -> None:
    assert hadukenn_core.__version__ == "1.2.0"
    assert sheaf_verify.__version__ == "1.2.0"

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "1.2.0"
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["tool"]["ruff"]["extend-exclude"] == ["fixtures", "iterations"]

    cmake_txt = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "hadukenn_tri_hal" in cmake_txt
    assert "Threads::Threads" in cmake_txt
    assert "hadukenn_fast_head_native" in cmake_txt

    sh_path = ROOT / "scripts" / "bootstrap_tri_machine.sh"
    ps1_path = ROOT / "scripts" / "bootstrap_tri_machine.ps1"
    py_path = ROOT / "scripts" / "build_tri_machine.py"
    assert sh_path.is_file() and os.access(sh_path, os.X_OK)
    assert ps1_path.is_file() and py_path.is_file()

    sh_txt = sh_path.read_text(encoding="utf-8")
    assert ".local/bin/clang" in sh_txt
    assert "-pthread" in sh_txt

    banned_drive_re = re.compile(r"[A-Za-z]:" + re.escape(chr(92)))
    for rel in (
        "CMakeLists.txt",
        "scripts/bootstrap_tri_machine.sh",
        "scripts/bootstrap_tri_machine.ps1",
        "scripts/build_tri_machine.py",
        "src/hadukenn_core/npu_qnn_backend.py",
        "src/hadukenn_core/tri_machine.py",
        "src/hadukenn_core/local_e2e_runner.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert banned_drive_re.search(text) is None, f"Drive-letter leak in {rel}"


def test_integration_r3_dual_t46_campaigns_and_error_branches(tmp_path: Path, capsys: Any) -> None:
    # Non-dict and empty dict rejection
    ok_none, crit_none, adv_none = verify_t46_tri_machine_hal(None)  # type: ignore[arg-type]
    assert ok_none is False and crit_none[0].id == "T46_SCHEMA_AND_THREE_MACHINES" and adv_none == []
    ok_empty, crit_empty, _ = verify_t46_tri_machine_hal({})
    assert ok_empty is False and any(not c.passed for c in crit_empty)

    base_tri = run_tri_machine_verification_suite()
    base_tri["regulatory_compliance"].pop("bundle", None)
    assert base_tri["regulatory_compliance"]["scitt_rfc9943_verified"] is True
    assert verify_t46_tri_machine_hal(base_tri)[0] is True

    # Malformed machines_configured (non-list, unhashable element, and duplicate entries)
    bad_machines = copy.deepcopy(base_tri)
    bad_machines["machines_configured"] = [{"bad": 1}]
    assert verify_t46_tri_machine_hal(bad_machines)[0] is False

    dup_machines = copy.deepcopy(base_tri)
    dup_machines["machines_configured"] = ["wbeast", "lapbeast", "zenbook", "zenbook"]
    assert verify_t46_tri_machine_hal(dup_machines)[0] is False

    bad_profiles = copy.deepcopy(base_tri)
    bad_profiles["profiles"] = "invalid"
    assert verify_t46_tri_machine_hal(bad_profiles)[0] is False

    # Bool / NaN / Inf / string / out-of-range / threshold failures in benchmarks_and_parity
    for field, bad_val in (
        ("cce_loss_max_abs_err", True),
        ("cce_loss_max_abs_err", float("nan")),
        ("cce_loss_max_abs_err", -1e-6),
        ("cce_dh_max_abs_err", float("inf")),
        ("cce_dh_max_abs_err", -1e-6),
        ("cce_de_max_abs_err", -1e-6),
        ("sheaf_kv_gate_max_abs_err", -1e-6),
        ("sheaf_coherent_neighbor_mass", "high"),
        ("sheaf_coherent_neighbor_mass", 1.01),
        ("mup_logit_std_scaled_d768", -0.1),
        ("c11_speedup_over_py_loop", 1.2),
        ("zenbook_npu_int8_cosine_sim", 0.99),
        ("zenbook_npu_int8_cosine_sim", 1.01),
        ("zenbook_npu_loop_dynamic_allocs", False),
        ("zenbook_npu_loop_dynamic_allocs", 2),
        ("lapbeast_watchdog_rc143_verified", False),
    ):
        mut = copy.deepcopy(base_tri)
        mut["benchmarks_and_parity"][field] = bad_val
        assert verify_t46_tri_machine_hal(mut)[0] is False, (field, bad_val)

    bad_scitt = copy.deepcopy(base_tri)
    bad_scitt["regulatory_compliance"] = {"scitt_rfc9943_verified": False}
    assert verify_t46_tri_machine_hal(bad_scitt)[0] is False

    # Dual CLI modes: both t46-tri-machine-hal and t46-snapdragon-local with --path and --in
    tri_json = tmp_path / "tri.json"
    tri_json.write_text(json.dumps(base_tri), encoding="utf-8")
    snap_json = tmp_path / "snap.json"
    snap_json.write_text(json.dumps(_sign_t46_snapdragon_doc(_valid_t46_snapdragon_base())), encoding="utf-8")

    capsys.readouterr()
    assert cli_main(["campaign", "--mode", "t46-tri-machine-hal", "--path", str(tri_json), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert cli_main(["campaign", "--mode", "t46-tri-machine-hal", "--in", str(tri_json), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert cli_main(["campaign", "--mode", "t46-snapdragon-local", "--path", str(snap_json), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert cli_main(["campaign", "--mode", "t46-snapdragon-local", "--in", str(snap_json), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert cli_main(["campaign", "--mode", "t46-tri-machine-hal"]) == 2
    assert cli_main(["campaign", "--mode", "t46-tri-machine-hal", "--in", str(tmp_path / "missing.json")]) == 2

    zenbook_live = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.zenbook-live.json"
    if zenbook_live.is_file():
        zen_doc = json.loads(zenbook_live.read_text(encoding="utf-8"))
        ok_zen, crit_zen, _ = verify_t46_tri_machine_hal(zen_doc)
        assert ok_zen is True, [c.as_dict() for c in crit_zen]


def test_integration_r3_local_e2e_runner_with_tri_machine_roundtrip(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    if detect_oryon_capabilities().get("has_i8mm"):
        # 1. Live roundtrip with include_tri_machine=True
        out_with_tri = tmp_path / "e2e_with_tri.json"
        rep_with_tri = local_e2e_runner.run_local_e2e(json_out=out_with_tri, include_tri_machine=True)
        assert out_with_tri.is_file()
        assert "tri_machine_hal" in rep_with_tri
        assert verify_t46_snapdragon_local(rep_with_tri)[0] is True
        assert verify_t46_tri_machine_hal(rep_with_tri["tri_machine_hal"])[0] is True

        # Ensure zero drive-letter leaks or hostname in serialized JSON
        raw_text = out_with_tri.read_text(encoding="utf-8")
        banned_drive_re = re.compile(r"[A-Za-z]:" + re.escape(chr(92)))
        assert banned_drive_re.search(raw_text) is None
        hostname = socket.gethostname().strip()
        if hostname and len(hostname) > 3:
            assert hostname not in raw_text

        # Tampering tri_machine_hal post-signing must break manifest_sha256 verification
        tampered = copy.deepcopy(rep_with_tri)
        tampered["tri_machine_hal"]["benchmarks_and_parity"]["zenbook_npu_loop_dynamic_allocs"] = 99
        assert verify_t46_snapdragon_local(tampered)[0] is False

    # 2. Verify default include_tri_machine=False omits tri_machine_hal & CLI flag works
    base_stub = _valid_t46_snapdragon_base()
    monkeypatch.setattr(
        local_e2e_runner,
        "run_t46_snapdragon_local_suite",
        lambda **_kw: copy.deepcopy(base_stub),
    )
    rep_no_tri = local_e2e_runner.run_local_e2e(include_tri_machine=False)
    assert "tri_machine_hal" not in rep_no_tri
    assert verify_t46_snapdragon_local(rep_no_tri)[0] is True

    cli_out = tmp_path / "cli_tri.json"
    capsys.readouterr()
    assert local_e2e_runner.main(["--json-out", str(cli_out), "--include-tri-machine"]) == 0
    cli_rep = json.loads(cli_out.read_text(encoding="utf-8"))
    assert "tri_machine_hal" in cli_rep
    assert verify_t46_snapdragon_local(cli_rep)[0] is True
    assert verify_t46_tri_machine_hal(cli_rep["tri_machine_hal"])[0] is True


def test_integration_r3_script_build_tri_machine_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_tri_machine.py"), "--machine", "zenbook"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(proc.stdout)
    assert rep["selected_profile"] == "zenbook"
    ok_flag, _, advisories = verify_t46_tri_machine_hal(rep)
    assert ok_flag is True
    assert any("TRI_MACHINE_HAL_READY" in a for a in advisories)
