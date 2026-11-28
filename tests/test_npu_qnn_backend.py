"""Unit, live HTP parity, fallback cascade, canary, and hygiene tests for M4 (F12-F14)."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hadukenn_core import npu_qnn_backend, npu_qnn_worker
from hadukenn_core.all_realms_suite import QuantizedINT8NPULoopEngine
from hadukenn_core.fast_ops import OpCounter


def test_discover_windows_python_rejects_windowsapps_and_honors_env(
    tmp_path: Path, monkeypatch: Any
) -> None:
    real_win_py = npu_qnn_backend.discover_windows_python()
    if real_win_py is not None:
        assert isinstance(real_win_py, str)
        assert "windowsapps" not in real_win_py.lower()
        assert Path(real_win_py).is_file()

    stub_dir = tmp_path / "Microsoft" / "WindowsApps"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_exe = stub_dir / "python.exe"
    stub_exe.write_text("", encoding="utf-8")

    # Reject WindowsApps stub via HADUKENN_WIN_PYTHON, shutil.which, and where python mock.
    monkeypatch.setenv("HADUKENN_WIN_PYTHON", str(stub_exe))
    monkeypatch.setattr(shutil, "which", lambda _name: str(stub_exe))
    win_stub_line = "C:" + chr(92) + "Users" + chr(92) + "user" + chr(92) + "Microsoft" + chr(92) + "WindowsApps" + chr(92) + "python.exe\r\n"

    def _mock_run(cmd: Any, *_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        joined = " ".join(str(x) for x in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "where python" in joined:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=win_stub_line, stderr="")
        if "wslpath" in joined:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=f"{stub_exe}\n", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _mock_run)
    assert npu_qnn_backend.discover_windows_python() is None

    # Honor valid HADUKENN_WIN_PYTHON override when probe succeeds.
    good_dir = tmp_path / "PythonArm64"
    good_dir.mkdir(parents=True, exist_ok=True)
    good_exe = good_dir / "python.exe"
    good_exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("HADUKENN_WIN_PYTHON", str(good_exe))
    assert npu_qnn_backend.discover_windows_python() == str(good_exe)

    # Probe failure must reject candidate cleanly.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *_a, **_kw: subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="err"),
    )
    assert npu_qnn_backend.discover_windows_python() is None


def test_htp_qnn_live_exact_parity_aligned_unaligned_and_saturation() -> None:
    win_py = npu_qnn_backend.discover_windows_python()
    if win_py is None:
        return

    rng = np.random.default_rng(701)
    # 1. Direct single-pass shape (K * max|A| * max|B| <= 32767)
    a_direct = rng.integers(-8, 9, size=(16, 48), dtype=np.int8)
    b_direct = rng.integers(-8, 9, size=(48, 24), dtype=np.int8)
    # 2. Unaligned M=1 and K non-multiple of 16
    a_m1 = rng.integers(-64, 64, size=(1, 37), dtype=np.int8)
    b_m1 = rng.integers(-64, 64, size=(37, 19), dtype=np.int8)
    # 3. Full [-128, 127] saturation + K > 511 (exercises 4-bit base-16 stacked decomposition)
    a_sat = rng.integers(-128, 128, size=(33, 520), dtype=np.int8)
    b_sat = rng.integers(-128, 128, size=(520, 29), dtype=np.int8)
    a_sat[0, 0] = np.int8(-128)
    b_sat[0, 0] = np.int8(-128)
    a_sat[1, 1] = np.int8(127)
    b_sat[1, 1] = np.int8(127)

    for a_i8, b_i8 in ((a_direct, b_direct), (a_m1, b_m1), (a_sat, b_sat)):
        counter = OpCounter()
        res = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="htp_qnn", counter=counter)
        ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
        assert res["tier_used"] == "htp_qnn"
        assert res["canary_ok"] is True
        assert "QNNExecutionProvider" in res["providers"]
        assert res["output"].dtype == np.int32
        assert np.array_equal(res["output"], ref)
        assert counter.calls == 1
        assert counter.mac_ops == a_i8.shape[0] * a_i8.shape[1] * b_i8.shape[1]

    # K=0 edge shape
    res_empty = npu_qnn_backend.run_int8_matmul_htp(
        np.zeros((4, 0), dtype=np.int8),
        np.zeros((0, 5), dtype=np.int8),
        force_tier="htp_qnn",
    )
    assert res_empty["tier_used"] == "htp_qnn"
    assert res_empty["canary_ok"] is True
    assert res_empty["output"].shape == (4, 5)
    assert np.array_equal(res_empty["output"], np.zeros((4, 5), dtype=np.int32))


def test_htp_qnn_live_speedup_and_oneshot_mode() -> None:
    win_py = npu_qnn_backend.discover_windows_python()
    if win_py is None:
        return

    rng = np.random.default_rng(709)
    a_i8 = rng.integers(-32, 32, size=(256, 512), dtype=np.int8)
    b_i8 = rng.integers(-32, 32, size=(512, 512), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    res = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="htp_qnn")
    assert res["tier_used"] == "htp_qnn"
    assert res["canary_ok"] is True
    assert np.array_equal(res["output"], ref)
    assert float(res["speedup_vs_numpy"]) > 1.0

    # Verify --oneshot worker mode executes cleanly with stdin=DEVNULL.
    a_small = rng.integers(-16, 16, size=(8, 16), dtype=np.int8)
    b_small = rng.integers(-16, 16, size=(16, 8), dtype=np.int8)
    out_one, resp_one = npu_qnn_backend.run_oneshot_worker(win_py, a_small, b_small)
    assert resp_one["ok"] is True
    assert resp_one["canary_ok"] is True
    assert np.array_equal(out_one, a_small.astype(np.int32) @ b_small.astype(np.int32))


def test_canary_verification_and_corruption_detection(monkeypatch: Any) -> None:
    verify_fn = npu_qnn_backend.HTPBackendSession._verify_canary
    assert verify_fn(bytearray([0x5A] * 16), 0x5A) is False
    assert verify_fn(b"", 0x5A) is False
    assert verify_fn(bytearray([0x5A, 0x01, 0x5A, 0x5A]), 0x5A) is True
    with pytest.raises(ValueError):
        verify_fn(b"\x00\x01", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        verify_fn(b"\x00\x01", 256)

    rng = np.random.default_rng(719)
    a_i8 = rng.integers(-16, 16, size=(16, 32), dtype=np.int8)
    b_i8 = rng.integers(-16, 16, size=(32, 16), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)

    monkeypatch.setattr(
        npu_qnn_backend.HTPBackendSession,
        "_verify_canary",
        staticmethod(lambda *_a, **_kw: False),
    )
    if npu_qnn_backend.discover_windows_python() is not None:
        with pytest.raises(RuntimeError, match="canary"):
            npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="htp_qnn")
        # In auto mode, canary failure cascades safely to Tier-2 / Tier-3.
        res_auto = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
        assert res_auto["tier_used"] in {"oryon_i8mm", "numpy_ref"}
        assert np.array_equal(res_auto["output"], ref)


def test_tier2_oryon_i8mm_tier3_numpy_and_auto_fallback_cascade(monkeypatch: Any) -> None:
    rng = np.random.default_rng(727)
    a_i8 = rng.integers(-64, 64, size=(48, 96), dtype=np.int8)
    b_i8 = rng.integers(-64, 64, size=(96, 40), dtype=np.int8)
    ref = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    has_i8mm = bool(npu_qnn_backend.detect_oryon_capabilities().get("has_i8mm"))

    if has_i8mm:
        res_t2 = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="oryon_i8mm")
        assert res_t2["tier_used"] == "oryon_i8mm"
        assert res_t2["canary_ok"] is True
        assert np.array_equal(res_t2["output"], ref)
    else:
        with pytest.raises(RuntimeError):
            npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="oryon_i8mm")

    res_t3 = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="numpy_ref")
    assert res_t3["tier_used"] == "numpy_ref"
    assert res_t3["canary_ok"] is True
    assert np.array_equal(res_t3["output"], ref)

    # Monkeypatching discover_windows_python mid-process must bypass any cached worker.
    monkeypatch.setattr(npu_qnn_backend, "discover_windows_python", lambda: None)
    res_auto_no_htp = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    expected_no_htp = "oryon_i8mm" if has_i8mm else "numpy_ref"
    assert res_auto_no_htp["tier_used"] == expected_no_htp
    assert np.array_equal(res_auto_no_htp["output"], ref)

    # Forced htp_qnn without Windows Python must fail closed.
    with pytest.raises(RuntimeError):
        npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="htp_qnn")

    # Simulate i8mm unavailable -> cascade to numpy_ref.
    monkeypatch.setattr(npu_qnn_backend, "detect_oryon_capabilities", lambda: {"has_i8mm": False})
    res_auto_t3 = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    assert res_auto_t3["tier_used"] == "numpy_ref"
    assert np.array_equal(res_auto_t3["output"], ref)


def test_input_validation_and_worker_json_slab_guards(tmp_path: Path) -> None:
    a_ok = np.ones((4, 8), dtype=np.int8)
    b_ok = np.ones((8, 4), dtype=np.int8)

    with pytest.raises(ValueError):
        npu_qnn_backend.run_int8_matmul_htp(a_ok, b_ok, force_tier="cuda_tensor")
    with pytest.raises(ValueError):
        npu_qnn_backend.run_int8_matmul_htp(a_ok, b_ok, force_tier="numpy_ref", dcvs_mode="turbo")
    with pytest.raises(TypeError):
        npu_qnn_backend.run_int8_matmul_htp(a_ok.astype(np.float32), b_ok, force_tier="numpy_ref")
    with pytest.raises(TypeError):
        npu_qnn_backend.run_int8_matmul_htp(a_ok, b_ok.astype(np.int32), force_tier="numpy_ref")
    with pytest.raises(ValueError):
        npu_qnn_backend.run_int8_matmul_htp(np.ones((4, 7), dtype=np.int8), b_ok, force_tier="numpy_ref")
    with pytest.raises(ValueError):
        npu_qnn_backend.run_int8_matmul_htp(np.ones((4,), dtype=np.int8), b_ok, force_tier="numpy_ref")
    with pytest.raises(ValueError):
        npu_qnn_backend.run_int8_matmul_htp(a_ok, b_ok, force_tier="numpy_ref", canary_byte=True)  # type: ignore[arg-type]

    # Worker strict JSON + slab bounds checks (bool offset / OOB / NaN rejection).
    with pytest.raises(ValueError):
        npu_qnn_worker._json_loads_strict('{"x": NaN}')
    slab_in = tmp_path / "slab_in.bin"
    slab_in.write_bytes(b"\x01\x02\x03\x04")
    with pytest.raises(ValueError):
        npu_qnn_worker._read_slab_inputs(
            {
                "_slab_in_path": str(slab_in),
                "_slab_inputs": {"a": {"offset": True, "nbytes": 4, "dtype": "int8", "shape": [2, 2]}},
            },
            tmp_path,
        )
    with pytest.raises(ValueError):
        npu_qnn_worker._read_slab_inputs(
            {
                "_slab_in_path": str(slab_in),
                "_slab_inputs": {"a": {"offset": 0, "nbytes": 8, "dtype": "int8", "shape": [2, 4]}},
            },
            tmp_path,
        )


def test_quantized_int8_npu_loop_engine_tiers_and_batch_dims() -> None:
    rng = np.random.default_rng(733)
    w = rng.normal(0.0, 0.25, size=(32, 24)).astype(np.float32)

    has_i8mm = bool(npu_qnn_backend.detect_oryon_capabilities().get("has_i8mm"))
    tiers_to_test: tuple[str | None, ...] = (None, "oryon_i8mm", "numpy_ref") if has_i8mm else (None, "numpy_ref")
    for x_shape in ((32,), (16, 32), (2, 8, 32)):
        x = rng.normal(0.0, 0.25, size=x_shape).astype(np.float32)
        y_ref = x @ w
        for tier in tiers_to_test:
            engine = QuantizedINT8NPULoopEngine(w, force_tier=tier)
            y_out = engine.forward_npu_int8(x)
            assert y_out.shape == y_ref.shape
            assert y_out.dtype == np.float32
            rel_err = float(np.linalg.norm(y_out - y_ref) / max(float(np.linalg.norm(y_ref)), 1e-12))
            assert rel_err < 0.03
            assert engine.last_canary_ok is True
            assert engine.last_tier_used in {"htp_qnn", "oryon_i8mm", "numpy_ref"}

    if npu_qnn_backend.discover_windows_python() is not None:
        x_2d = rng.normal(0.0, 0.25, size=(16, 32)).astype(np.float32)
        eng_htp = QuantizedINT8NPULoopEngine(w, force_tier="htp_qnn")
        y_htp = eng_htp.forward_npu_int8(x_2d)
        rel_err = float(np.linalg.norm(y_htp - (x_2d @ w)) / max(float(np.linalg.norm(x_2d @ w)), 1e-12))
        assert eng_htp.last_tier_used == "htp_qnn"
        assert eng_htp.last_canary_ok is True
        assert rel_err < 0.03


def test_telemetry_and_owned_files_hygiene() -> None:
    rng = np.random.default_rng(739)
    a_i8 = rng.integers(-16, 16, size=(16, 32), dtype=np.int8)
    b_i8 = rng.integers(-16, 16, size=(32, 16), dtype=np.int8)
    res = npu_qnn_backend.run_int8_matmul_htp(a_i8, b_i8, force_tier="auto")
    payload = {k: v for k, v in res.items() if k != "output"}
    serialized = json.dumps(payload, sort_keys=True)

    banned_drives = tuple(d + ":" + chr(92) for d in ("C", "D", "E", "H", "c", "d", "e", "h"))
    for drv in banned_drives:
        assert drv not in serialized

    repo_root = Path(__file__).resolve().parents[1]
    owned = (
        repo_root / "src" / "hadukenn_core" / "npu_qnn_backend.py",
        repo_root / "src" / "hadukenn_core" / "npu_qnn_worker.py",
        repo_root / "src" / "hadukenn_core" / "all_realms_suite.py",
        repo_root / "tests" / "test_npu_qnn_backend.py",
    )
    for path in owned:
        text = path.read_text(encoding="utf-8")
        for drv in banned_drives:
            assert drv not in text, f"Forbidden drive prefix in {path}"

    # Verify telemetry path sanitizer removes drive-letter backslash paths.
    dirty = "Error at " + "C:" + chr(92) + "Python314" + chr(92) + "foo.py"
    cleaned = npu_qnn_backend._sanitize_telemetry_text(dirty)
    for drv in banned_drives:
        assert drv not in cleaned

    npu_qnn_backend.close_all_htp_sessions()


def test_persistent_win_worker_post_close_raises_runtime_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    npu_qnn_backend.close_all_htp_sessions()
    a_small = np.ones((2, 4), dtype=np.int8)
    b_small = np.ones((4, 2), dtype=np.int8)

    class _FakeStdin:
        def __init__(self, out_q: queue.Queue[bytes], proc_ref: Any) -> None:
            self._out_q = out_q
            self._proc = proc_ref

        def write(self, data: bytes) -> int:
            text = data.decode("utf-8")
            if '"ping"' in text:
                self._out_q.put(b'{"ok": true}\n')
            elif '"shutdown"' in text:
                self._proc.returncode = 0
                self._out_q.put(b"")
            return len(data)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            self._out_q.put(b"")

    class _FakeStdout:
        def __init__(self, out_q: queue.Queue[bytes]) -> None:
            self._out_q = out_q

        def readline(self) -> bytes:
            return self._out_q.get()

        def close(self) -> None:
            self._out_q.put(b"")

    class _FakeStderr:
        def read(self, _n: int = 4096) -> bytes:
            return b""

        def close(self) -> None:
            return None

    class _FakePopen:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.returncode: int | None = None
            out_q: queue.Queue[bytes] = queue.Queue()
            self.stdin = _FakeStdin(out_q, self)
            self.stdout = _FakeStdout(out_q)
            self.stderr = _FakeStderr()

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    try:
        win_py_real = npu_qnn_backend.discover_windows_python()
        if win_py_real is not None:
            live_worker = npu_qnn_backend.PersistentWinWorker(win_py_real)
            live_stage = live_worker.stage_dir
            live_worker.close()
            assert not live_stage.exists()
            assert live_worker not in npu_qnn_backend._ACTIVE_WORKERS
            assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0
            with pytest.raises(RuntimeError, match="closed"):
                live_worker.ping()
            with pytest.raises(RuntimeError, match="closed"):
                live_worker.call_int8_matmul(a_small, b_small)

        monkeypatch.setattr(npu_qnn_backend, "_find_windows_temp_root", lambda: tmp_path)
        monkeypatch.setattr(npu_qnn_backend, "_wslpath_w", lambda p: str(p))
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)

        worker = npu_qnn_backend.PersistentWinWorker("/fake/win/python.exe")
        stage_dir = worker.stage_dir
        assert stage_dir.is_dir()
        worker.close()
        assert not stage_dir.exists()
        assert worker not in npu_qnn_backend._ACTIVE_WORKERS
        assert len(npu_qnn_backend._ACTIVE_WORKERS) == 0
        with pytest.raises(RuntimeError, match="closed"):
            worker.ping()
        with pytest.raises(RuntimeError, match="closed"):
            worker.call_int8_matmul(a_small, b_small)
    finally:
        npu_qnn_backend.close_all_htp_sessions()
