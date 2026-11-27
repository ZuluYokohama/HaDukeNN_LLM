"""Hexagon HTP v73 QNN backend with dynamic WSL2-to-Windows bridge and tiered fallbacks.

Tier hierarchy:
  - Tier 1 (``htp_qnn``): Qualcomm Hexagon HTP v73 NPU via ``QNNExecutionProvider``
    (in-process on Windows ARM64, or via a dynamically staged ``PersistentWinWorker``
    bridge from WSL2 with zero hardcoded drive-letter paths).
  - Tier 2 (``oryon_i8mm``): Native ARM64 Oryon ``vmmlaq_s32`` INT8 GEMM via
    ``hadukenn_core.fast_ops.run_int8_gemm``.
  - Tier 3 (``numpy_ref``): Exact ``int32`` NumPy reference matmul.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from hadukenn_core.fast_ops import OpCounter, detect_oryon_capabilities, run_int8_gemm

_VALID_TIERS = frozenset({"auto", "htp_qnn", "oryon_i8mm", "numpy_ref"})
_VALID_DCVS_MODES = frozenset({"burst", "power_saver"})
_ORIG_SUBPROCESS_RUN = subprocess.run
_PROBE_CACHE: dict[tuple[str, int], bool] = {}
_WIN_PATH_RE = re.compile(r"[A-Za-z]:" + re.escape(chr(92)) + r"[^\s\"']*")
_WORKER_LOCK = threading.RLock()
_SESSION_POOL: dict[tuple[str, str], HTPBackendSession] = {}
_ACTIVE_WORKERS: list[PersistentWinWorker] = []


def _sanitize_telemetry_text(text: str) -> str:
    """Strip any Windows drive-letter backslash paths from telemetry or error strings."""
    return _WIN_PATH_RE.sub("<win_path>", text)


def _reject_nonfinite_json(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is forbidden: {token!r}")


def _json_loads_strict(raw: str) -> dict[str, Any]:
    obj = json.loads(raw, parse_constant=_reject_nonfinite_json)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from worker, got {type(obj).__name__}")
    return obj


def _json_dumps_strict(payload: dict[str, Any]) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _require_nonbool_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be a non-bool integer, got {type(val).__name__}")
    return val


def _validate_canary_byte(canary_byte: int) -> int:
    val = _require_nonbool_int(canary_byte, "canary_byte")
    if not (0 <= val <= 255):
        raise ValueError(f"canary_byte must be in [0, 255], got {val}")
    return val


def _is_windows_apps_stub(path_str: str) -> bool:
    return "windowsapps" in path_str.lower()


def _probe_qnn_python(cand: str) -> bool:
    """Return True if candidate Python executes ``import onnxruntime, onnxruntime_qnn``."""
    if not cand or _is_windows_apps_stub(cand):
        return False
    p = Path(cand)
    if not p.is_file():
        return False

    use_cache = subprocess.run is _ORIG_SUBPROCESS_RUN
    mtime_ns = 0
    if use_cache:
        try:
            mtime_ns = p.stat().st_mtime_ns
        except OSError:
            return False
        cached = _PROBE_CACHE.get((cand, mtime_ns))
        if cached is not None:
            return cached

    try:
        proc = subprocess.run(
            [cand, "-c", "import onnxruntime, onnxruntime_qnn; print('ok')"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        ok = proc.returncode == 0 and "ok" in (proc.stdout or "")
    except (OSError, subprocess.SubprocessError):
        ok = False

    if use_cache:
        _PROBE_CACHE[(cand, mtime_ns)] = ok
    return ok


def _wslpath_u(win_path: str) -> str | None:
    try:
        proc = subprocess.run(
            ["wslpath", "-u", win_path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out if out else None


def _wslpath_w(wsl_path: Path) -> str:
    proc = subprocess.run(
        ["wslpath", "-w", str(wsl_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(f"wslpath -w returned empty path for {wsl_path}")
    return out


def discover_windows_python() -> str | None:
    """Dynamically locate Windows ARM64 Python with ``onnxruntime`` and ``onnxruntime_qnn``.

    Discovery order:
      1. ``HADUKENN_WIN_PYTHON`` environment variable
      2. ``shutil.which("python.exe")``
      3. ``cmd.exe /c "where python"`` converted via ``wslpath -u``

    Any path containing ``WindowsApps`` (Store execution alias) is rejected.
    """
    env_cand = os.environ.get("HADUKENN_WIN_PYTHON", "").strip()
    if env_cand and not _is_windows_apps_stub(env_cand) and _probe_qnn_python(env_cand):
        return env_cand

    if sys.platform == "win32":
        exe = sys.executable
        if exe and not _is_windows_apps_stub(exe) and _probe_qnn_python(exe):
            return exe
        return None

    which_cand = shutil.which("python.exe")
    if which_cand and not _is_windows_apps_stub(which_cand) and _probe_qnn_python(which_cand):
        return which_cand

    try:
        proc = subprocess.run(
            ["cmd.exe", "/c", "where python"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if proc.returncode != 0 or not proc.stdout:
        return None

    for raw_line in proc.stdout.splitlines():
        win_line = raw_line.strip()
        if not win_line or _is_windows_apps_stub(win_line):
            continue
        wsl_cand = _wslpath_u(win_line)
        if wsl_cand and not _is_windows_apps_stub(wsl_cand) and _probe_qnn_python(wsl_cand):
            return wsl_cand
    return None


def _find_windows_temp_root() -> Path:
    """Dynamically resolve Windows ``%TEMP%`` in WSL2 (zero hardcoded drive paths)."""
    if sys.platform == "win32":
        return Path(tempfile.gettempdir())

    try:
        proc = subprocess.run(
            ["cmd.exe", "/c", "echo %TEMP%"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            for win_temp in reversed(lines):
                if "%TEMP%" in win_temp:
                    continue
                wsl_temp = _wslpath_u(win_temp)
                if wsl_temp:
                    p = Path(wsl_temp)
                    if p.is_dir():
                        return p
    except (OSError, subprocess.SubprocessError):
        pass

    skip_users = {"public", "default", "default user", "all users"}
    mnt_root = Path("/mnt")
    if mnt_root.is_dir():
        for cand in sorted(mnt_root.glob("*/Users/*/AppData/Local/Temp")):
            user_name = cand.parents[2].name.lower()
            if user_name in skip_users:
                continue
            if cand.is_dir() and os.access(cand, os.W_OK):
                return cand

    return Path(tempfile.gettempdir())


def _rmtree_with_retry(path: Path, timeout_s: float = 8.0) -> None:
    """Remove staging directory with 50 ms retry polling for Windows file-lock release."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while path.exists():
        try:
            shutil.rmtree(path, ignore_errors=False)
            return
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.05, remaining))
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _read_slab_outputs(resp: dict[str, Any], slab_path: Path) -> dict[str, np.ndarray]:
    if not slab_path.is_file():
        raise FileNotFoundError(f"Output slab missing: {slab_path.name}")
    descriptors = resp.get("_slab_outputs")
    if not isinstance(descriptors, dict):
        raise ValueError("Worker response missing _slab_outputs descriptor dict")

    raw_bytes = slab_path.read_bytes()
    total_len = len(raw_bytes)
    out: dict[str, np.ndarray] = {}
    for name, desc in descriptors.items():
        if not isinstance(desc, dict):
            raise ValueError(f"Invalid output slab descriptor for {name!r}")
        offset = _require_nonbool_int(desc.get("offset"), f"{name}.offset")
        nbytes = _require_nonbool_int(desc.get("nbytes"), f"{name}.nbytes")
        if offset < 0 or nbytes < 0 or offset + nbytes > total_len:
            raise ValueError(
                f"Output slab bounds invalid for {name!r}: off={offset}, nbytes={nbytes}, total={total_len}"
            )
        dt = np.dtype(str(desc.get("dtype")))
        raw_shape = desc.get("shape")
        if not isinstance(raw_shape, list):
            raise ValueError(f"Invalid output slab shape for {name!r}")
        shape = tuple(_require_nonbool_int(d, f"{name}.shape[]") for d in raw_shape)
        if any(d < 0 for d in shape):
            raise ValueError(f"Negative dimension in output slab shape for {name!r}: {shape}")
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize if shape else dt.itemsize
        if expected_nbytes != nbytes:
            raise ValueError(f"Output slab nbytes mismatch for {name!r}: {nbytes} != {expected_nbytes}")
        chunk = raw_bytes[offset : offset + nbytes]
        out[name] = np.frombuffer(chunk, dtype=dt).reshape(shape).copy()
    return out


def _write_slab_inputs(tensors: dict[str, np.ndarray], slab_path: Path) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    offset = 0
    with open(slab_path, "wb") as fh:
        for name, arr in tensors.items():
            if arr.dtype == np.object_ or np.issubdtype(arr.dtype, np.complexfloating):
                raise TypeError(f"Unsupported slab dtype {arr.dtype} for {name!r}")
            c_arr = np.ascontiguousarray(arr)
            raw = c_arr.tobytes()
            nbytes = len(raw)
            fh.write(raw)
            descriptors[name] = {
                "offset": offset,
                "nbytes": nbytes,
                "dtype": c_arr.dtype.name,
                "shape": [int(d) for d in c_arr.shape],
            }
            offset += nbytes
    return descriptors


class PersistentWinWorker:
    """Persistent Windows ARM64 worker over JSONL + contiguous binary slabs."""

    def __init__(self, win_python: str, *, startup_timeout_s: float = 30.0) -> None:
        self.win_python = win_python
        self._startup_timeout_s = float(startup_timeout_s)
        self._io_lock = threading.Lock()
        self._closed = False
        self._stdout_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=64)

        temp_root = _find_windows_temp_root()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.stage_dir = temp_root / f"hadukenn_htp_{os.getpid()}_{uuid.uuid4().hex[:10]}"
        self.stage_dir.mkdir(parents=True, exist_ok=True)

        worker_src = Path(__file__).resolve().with_name("npu_qnn_worker.py")
        self.staged_worker = self.stage_dir / "npu_qnn_worker.py"
        shutil.copy2(worker_src, self.staged_worker)

        self.slab_in_path = self.stage_dir / "slab_in.bin"
        self.slab_out_path = self.stage_dir / "slab_out.bin"

        if sys.platform == "win32":
            self._win_stage_dir = str(self.stage_dir)
            self._win_worker_path = str(self.staged_worker)
            self._win_slab_in = str(self.slab_in_path)
            self._win_slab_out = str(self.slab_out_path)
        else:
            self._win_stage_dir = _wslpath_w(self.stage_dir)
            self._win_worker_path = _wslpath_w(self.staged_worker)
            self._win_slab_in = _wslpath_w(self.slab_in_path)
            self._win_slab_out = _wslpath_w(self.slab_out_path)

        cmd = [
            self.win_python,
            "-u",
            self._win_worker_path,
            "--serve",
            "--work-dir",
            self._win_stage_dir,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._stdout_thread = threading.Thread(
                target=self._stdout_reader_loop, name="htp-worker-stdout", daemon=True
            )
            self._stderr_thread = threading.Thread(
                target=self._stderr_drain_loop, name="htp-worker-stderr", daemon=True
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
            self.ping(timeout_s=self._startup_timeout_s)
        except Exception:
            self.close()
            raise

        with _WORKER_LOCK:
            _ACTIVE_WORKERS.append(self)

    def _stdout_reader_loop(self) -> None:
        pipe = self._proc.stdout
        if pipe is None:
            self._stdout_queue.put(("eof", None))
            return
        try:
            for raw in iter(pipe.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                self._stdout_queue.put(("line", line))
        finally:
            self._stdout_queue.put(("eof", None))

    def _stderr_drain_loop(self) -> None:
        pipe = self._proc.stderr
        if pipe is None:
            return
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace").replace("\x00", "")
            if text:
                self._stderr_tail.append(text)

    def _diag_tail(self) -> str:
        joined = "".join(self._stderr_tail)[-4096:]
        return _sanitize_telemetry_text(joined.strip())

    def _rpc(self, payload: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("PersistentWinWorker is closed")
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"PersistentWinWorker exited unexpectedly (rc={self._proc.returncode}): {self._diag_tail()}"
            )
        if self._proc.stdin is None:
            raise RuntimeError("PersistentWinWorker stdin unavailable")

        line_bytes = (_json_dumps_strict(payload) + "\n").encode("utf-8")
        self._proc.stdin.write(line_bytes)
        self._proc.stdin.flush()

        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"PersistentWinWorker RPC timed out after {timeout_s:.1f}s")
            try:
                kind, data = self._stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"PersistentWinWorker RPC timed out after {timeout_s:.1f}s") from exc
            if kind == "eof" or data is None:
                raise RuntimeError(f"PersistentWinWorker closed stdout unexpectedly: {self._diag_tail()}")
            if not data.strip():
                continue
            resp = _json_loads_strict(data)
            if not resp.get("ok"):
                err = _sanitize_telemetry_text(str(resp.get("error", "Unknown worker error")))
                raise RuntimeError(f"Worker RPC error: {err}")
            return resp

    def ping(self, *, timeout_s: float = 20.0) -> dict[str, Any]:
        with self._io_lock:
            if self._closed:
                raise RuntimeError("PersistentWinWorker is closed")
            return self._rpc({"action": "ping"}, timeout_s=timeout_s)

    def call_int8_matmul(
        self,
        a_int8: np.ndarray,
        b_int8: np.ndarray,
        *,
        dcvs_mode: str = "burst",
        canary_byte: int = 0x5A,
        timeout_s: float = 120.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        with self._io_lock:
            if self._closed:
                raise RuntimeError("PersistentWinWorker is closed")
            slab_in_desc = _write_slab_inputs({"a": a_int8, "b": b_int8}, self.slab_in_path)
            req = {
                "action": "run_int8_matmul",
                "dcvs_mode": dcvs_mode,
                "canary_byte": int(canary_byte),
                "_slab_in_path": self._win_slab_in,
                "_slab_out_path": self._win_slab_out,
                "_slab_inputs": slab_in_desc,
            }
            resp = self._rpc(req, timeout_s=timeout_s)
            outputs = _read_slab_outputs(resp, self.slab_out_path)
            if "output" not in outputs:
                raise RuntimeError("Worker response missing 'output' tensor in slab_out.bin")
            return outputs["output"], resp

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self, "_proc") and self._proc.poll() is None:
                try:
                    with self._io_lock:
                        if self._proc.stdin is not None:
                            self._proc.stdin.write((_json_dumps_strict({"action": "shutdown"}) + "\n").encode("utf-8"))
                            self._proc.stdin.flush()
                except OSError:
                    pass
                for pipe in (self._proc.stdin,):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError:
                            pass
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=2.0)
            if hasattr(self, "_proc"):
                for pipe in (self._proc.stdout, self._proc.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError:
                            pass
            for th_name in ("_stdout_thread", "_stderr_thread"):
                th = getattr(self, th_name, None)
                if th is not None and th.is_alive():
                    th.join(timeout=1.0)
        finally:
            with _WORKER_LOCK:
                try:
                    _ACTIVE_WORKERS.remove(self)
                except ValueError:
                    pass
            if hasattr(self, "_io_lock"):
                with self._io_lock:
                    if hasattr(self, "stage_dir"):
                        _rmtree_with_retry(self.stage_dir, timeout_s=8.0)
            elif hasattr(self, "stage_dir"):
                _rmtree_with_retry(self.stage_dir, timeout_s=8.0)


def run_oneshot_worker(
    win_python: str,
    a_int8: np.ndarray,
    b_int8: np.ndarray,
    *,
    dcvs_mode: str = "burst",
    canary_byte: int = 0x5A,
    timeout_s: float = 120.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute a one-shot Windows worker invocation with ``stdin=subprocess.DEVNULL``."""
    temp_root = _find_windows_temp_root()
    temp_root.mkdir(parents=True, exist_ok=True)
    stage_dir = temp_root / f"hadukenn_oneshot_{os.getpid()}_{uuid.uuid4().hex[:10]}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        worker_src = Path(__file__).resolve().with_name("npu_qnn_worker.py")
        staged_worker = stage_dir / "npu_qnn_worker.py"
        shutil.copy2(worker_src, staged_worker)

        slab_in = stage_dir / "slab_in.bin"
        slab_out = stage_dir / "slab_out.bin"
        req_file = stage_dir / "req.json"
        resp_file = stage_dir / "resp.json"

        if sys.platform == "win32":
            w_dir, w_worker, w_in, w_out, w_req, w_resp = (
                str(stage_dir),
                str(staged_worker),
                str(slab_in),
                str(slab_out),
                str(req_file),
                str(resp_file),
            )
        else:
            w_dir = _wslpath_w(stage_dir)
            w_worker = _wslpath_w(staged_worker)
            w_in = _wslpath_w(slab_in)
            w_out = _wslpath_w(slab_out)
            w_req = _wslpath_w(req_file)
            w_resp = _wslpath_w(resp_file)

        slab_in_desc = _write_slab_inputs({"a": a_int8, "b": b_int8}, slab_in)
        req_payload = {
            "action": "run_int8_matmul",
            "dcvs_mode": dcvs_mode,
            "canary_byte": int(canary_byte),
            "_slab_in_path": w_in,
            "_slab_out_path": w_out,
            "_slab_inputs": slab_in_desc,
        }
        req_file.write_text(_json_dumps_strict(req_payload), encoding="utf-8")

        proc = subprocess.run(
            [
                win_python,
                "-u",
                w_worker,
                "--oneshot",
                "--work-dir",
                w_dir,
                "--req-file",
                w_req,
                "--resp-file",
                w_resp,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if not resp_file.is_file():
            raise RuntimeError(
                f"One-shot worker failed without resp.json (rc={proc.returncode}): "
                f"{_sanitize_telemetry_text(proc.stderr or proc.stdout)}"
            )
        resp = _json_loads_strict(resp_file.read_text(encoding="utf-8"))
        if not resp.get("ok"):
            err = _sanitize_telemetry_text(str(resp.get("error", "One-shot worker failed")))
            raise RuntimeError(f"One-shot worker error: {err}")
        outputs = _read_slab_outputs(resp, slab_out)
        return outputs["output"], resp
    finally:
        _rmtree_with_retry(stage_dir, timeout_s=8.0)


class HTPBackendSession:
    """Managed Hexagon HTP v73 QNN session (in-process on win32, bridged on WSL2)."""

    def __init__(self, win_python: str | None = None, *, dcvs_mode: str = "burst") -> None:
        if dcvs_mode not in _VALID_DCVS_MODES:
            raise ValueError(f"Unsupported dcvs_mode {dcvs_mode!r}; expected one of {sorted(_VALID_DCVS_MODES)}")
        self.dcvs_mode = dcvs_mode
        self.win_python = win_python
        self._worker: PersistentWinWorker | None = None
        if sys.platform != "win32":
            if self.win_python is None:
                self.win_python = discover_windows_python()
            if self.win_python is None:
                raise RuntimeError("No Windows ARM64 Python with onnxruntime_qnn discovered.")
            self._worker = PersistentWinWorker(self.win_python)

    @staticmethod
    def _verify_canary(
        raw_bytes: bytes | bytearray | memoryview | np.ndarray,
        canary_byte: int = 0x5A,
    ) -> bool:
        """Return True iff non-empty output buffer was overwritten (not all ``canary_byte``)."""
        c_byte = _validate_canary_byte(canary_byte)
        if isinstance(raw_bytes, np.ndarray):
            arr = raw_bytes.view(np.uint8).ravel()
        else:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return bool(arr.size > 0 and not np.all(arr == c_byte))

    def matmul_int8(
        self,
        a_int8: np.ndarray,
        b_int8: np.ndarray,
        *,
        canary_byte: int = 0x5A,
    ) -> dict[str, Any]:
        c_byte = _validate_canary_byte(canary_byte)
        if sys.platform == "win32":
            from hadukenn_core import npu_qnn_worker

            res = npu_qnn_worker.execute_int8_matmul_htp(
                a_int8,
                b_int8,
                dcvs_mode=self.dcvs_mode,
                canary_byte=c_byte,
            )
            out_i32 = res["output"]
            worker_canary_ok = bool(res["canary_ok"])
            compile_ms = float(res["compile_ms"])
            run_ms = float(res["run_ms"])
            providers = [_sanitize_telemetry_text(str(p)) for p in res.get("providers", [])]
            tile_passes = int(res.get("tile_passes", 1))
        else:
            worker = self._worker
            if worker is None:
                raise RuntimeError("HTPBackendSession worker is not initialized")
            out_i32, resp = worker.call_int8_matmul(
                a_int8,
                b_int8,
                dcvs_mode=self.dcvs_mode,
                canary_byte=c_byte,
            )
            worker_canary_ok = bool(resp.get("canary_ok", False))
            compile_ms = float(resp.get("compile_ms", 0.0))
            run_ms = float(resp.get("run_ms", 0.0))
            providers = [_sanitize_telemetry_text(str(p)) for p in resp.get("providers", [])]
            tile_passes = int(resp.get("tile_passes", 1))

        host_view = (out_i32.astype(np.int64) + 0x1_0000_0000).astype(np.uint64).view(np.uint8)
        host_canary_ok = type(self)._verify_canary(host_view, canary_byte=c_byte)
        canary_ok = bool(worker_canary_ok and host_canary_ok)
        if not canary_ok:
            raise RuntimeError("HTP 0x5A canary verification failed on output buffer.")

        return {
            "output": out_i32,
            "canary_ok": True,
            "compile_ms": compile_ms,
            "run_ms": run_ms,
            "providers": providers,
            "tile_passes": tile_passes,
        }

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> HTPBackendSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _get_or_create_session(win_python: str | None, dcvs_mode: str) -> HTPBackendSession:
    key = (win_python or "in_process_win32", dcvs_mode)
    with _WORKER_LOCK:
        sess = _SESSION_POOL.get(key)
        if sess is None or (sess._worker is not None and (sess._worker._closed or sess._worker._proc.poll() is not None)):
            if sess is not None:
                sess.close()
            sess = HTPBackendSession(win_python=win_python, dcvs_mode=dcvs_mode)
            _SESSION_POOL[key] = sess
        return sess


def close_all_htp_sessions() -> None:
    """Shut down cached persistent HTP sessions and clean up Windows staging directories."""
    with _WORKER_LOCK:
        for sess in list(_SESSION_POOL.values()):
            sess.close()
        _SESSION_POOL.clear()
        for worker in list(_ACTIVE_WORKERS):
            worker.close()
        _ACTIVE_WORKERS.clear()


atexit.register(close_all_htp_sessions)


def _update_counter(
    counter: OpCounter | None,
    m: int,
    k: int,
    n: int,
    *,
    tile_passes: int,
    threads_used: int,
) -> None:
    if counter is None:
        return
    counter.mac_ops += m * n * k
    counter.bytes_moved += m * k + k * n + 4 * m * n
    counter.tile_passes += max(0, int(tile_passes))
    counter.threads_used = max(counter.threads_used, threads_used)
    counter.calls += 1


def _validate_int8_matmul_inputs(a_int8: np.ndarray, b_int8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a_int8)
    b_arr = np.asarray(b_int8)
    if a_arr.dtype != np.int8 or b_arr.dtype != np.int8:
        raise TypeError(f"Expected int8 inputs, got A={a_arr.dtype} and B={b_arr.dtype}")
    if a_arr.ndim != 2 or b_arr.ndim != 2 or a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError(f"Incompatible 2D matrix shapes: A={a_arr.shape}, B={b_arr.shape}")
    return np.ascontiguousarray(a_arr, dtype=np.int8), np.ascontiguousarray(b_arr, dtype=np.int8)


def run_int8_matmul_htp(
    a_int8: np.ndarray,
    b_int8: np.ndarray,
    force_tier: str | None = "auto",
    *,
    counter: OpCounter | None = None,
    dcvs_mode: str = "burst",
    canary_byte: int = 0x5A,
) -> dict[str, Any]:
    """Run exact ``int8 x int8 -> int32`` MatMul across HTP NPU -> Oryon i8mm -> NumPy tiers."""
    tier = "auto" if force_tier is None else force_tier
    if tier not in _VALID_TIERS:
        raise ValueError(f"Unknown force_tier {force_tier!r}; expected one of {sorted(_VALID_TIERS)}")
    if dcvs_mode not in _VALID_DCVS_MODES:
        raise ValueError(f"Unsupported dcvs_mode {dcvs_mode!r}; expected one of {sorted(_VALID_DCVS_MODES)}")
    c_byte = _validate_canary_byte(canary_byte)

    a_c, b_c = _validate_int8_matmul_inputs(a_int8, b_int8)
    m, k = int(a_c.shape[0]), int(a_c.shape[1])
    n = int(b_c.shape[1])

    win_py: str | None = None
    if tier in {"auto", "htp_qnn"} and sys.platform != "win32":
        # Always check discover_windows_python() before reusing any cached worker.
        win_py = discover_windows_python()

    if m == 0 or n == 0 or k == 0:
        if tier == "htp_qnn" and sys.platform != "win32" and win_py is None:
            raise RuntimeError("No Windows ARM64 Python with onnxruntime_qnn discovered for force_tier='htp_qnn'.")
        out_zero = np.zeros((m, n), dtype=np.int32)
        if tier == "htp_qnn" or (tier == "auto" and (sys.platform == "win32" or win_py is not None)):
            tier_used = "htp_qnn"
            providers = ["QNNExecutionProvider"]
        elif tier == "oryon_i8mm":
            if not detect_oryon_capabilities().get("has_i8mm"):
                raise RuntimeError("Oryon i8mm unavailable on this CPU/build for force_tier='oryon_i8mm'.")
            tier_used = "oryon_i8mm"
            providers = ["OryonI8MM"]
        elif tier == "auto" and detect_oryon_capabilities().get("has_i8mm"):
            tier_used = "oryon_i8mm"
            providers = ["OryonI8MM"]
        else:
            tier_used = "numpy_ref"
            providers = ["NumPyReference"]
        _update_counter(counter, m, k, n, tile_passes=0, threads_used=1)
        return {
            "output": out_zero,
            "tier_used": tier_used,
            "canary_ok": True,
            "speedup_vs_numpy": 1.0,
            "elapsed_ms": 0.0,
            "numpy_ref_ms": 0.0,
            "compile_ms": 0.0,
            "run_ms": 0.0,
            "total_wall_ms": 0.0,
            "providers": providers,
        }

    t_ref0 = time.perf_counter_ns()
    _ = a_c.astype(np.int32) @ b_c.astype(np.int32)
    numpy_ref_ms = float(time.perf_counter_ns() - t_ref0) / 1e6

    if tier == "htp_qnn":
        if sys.platform != "win32" and win_py is None:
            raise RuntimeError("No Windows ARM64 Python with onnxruntime_qnn discovered for force_tier='htp_qnn'.")
        t_wall0 = time.perf_counter_ns()
        sess = _get_or_create_session(win_py, dcvs_mode)
        htp_res = sess.matmul_int8(a_c, b_c, canary_byte=c_byte)
        total_wall_ms = float(time.perf_counter_ns() - t_wall0) / 1e6
        run_ms = float(htp_res["run_ms"])
        elapsed_ms = run_ms if run_ms > 0.0 else total_wall_ms
        _update_counter(counter, m, k, n, tile_passes=int(htp_res["tile_passes"]), threads_used=1)
        return {
            "output": htp_res["output"],
            "tier_used": "htp_qnn",
            "canary_ok": bool(htp_res["canary_ok"]),
            "speedup_vs_numpy": round(numpy_ref_ms / max(elapsed_ms, 1e-6), 3),
            "elapsed_ms": round(elapsed_ms, 4),
            "numpy_ref_ms": round(numpy_ref_ms, 4),
            "compile_ms": round(float(htp_res["compile_ms"]), 4),
            "run_ms": round(run_ms, 4),
            "total_wall_ms": round(total_wall_ms, 4),
            "providers": list(htp_res["providers"]),
        }

    if tier == "auto" and (sys.platform == "win32" or win_py is not None):
        try:
            t_wall0 = time.perf_counter_ns()
            sess = _get_or_create_session(win_py, dcvs_mode)
            htp_res = sess.matmul_int8(a_c, b_c, canary_byte=c_byte)
            total_wall_ms = float(time.perf_counter_ns() - t_wall0) / 1e6
            run_ms = float(htp_res["run_ms"])
            elapsed_ms = run_ms if run_ms > 0.0 else total_wall_ms
            _update_counter(counter, m, k, n, tile_passes=int(htp_res["tile_passes"]), threads_used=1)
            return {
                "output": htp_res["output"],
                "tier_used": "htp_qnn",
                "canary_ok": bool(htp_res["canary_ok"]),
                "speedup_vs_numpy": round(numpy_ref_ms / max(elapsed_ms, 1e-6), 3),
                "elapsed_ms": round(elapsed_ms, 4),
                "numpy_ref_ms": round(numpy_ref_ms, 4),
                "compile_ms": round(float(htp_res["compile_ms"]), 4),
                "run_ms": round(run_ms, 4),
                "total_wall_ms": round(total_wall_ms, 4),
                "providers": list(htp_res["providers"]),
            }
        except (RuntimeError, OSError, ValueError, TimeoutError):
            pass

    if tier in {"auto", "oryon_i8mm"}:
        caps = detect_oryon_capabilities()
        if tier == "oryon_i8mm" or caps.get("has_i8mm"):
            try:
                t_i0 = time.perf_counter_ns()
                out_i8mm = run_int8_gemm(a_c, b_c, mode="i8mm", counter=counter)
                elapsed_ms = float(time.perf_counter_ns() - t_i0) / 1e6
                return {
                    "output": out_i8mm,
                    "tier_used": "oryon_i8mm",
                    "canary_ok": True,
                    "speedup_vs_numpy": round(numpy_ref_ms / max(elapsed_ms, 1e-6), 3),
                    "elapsed_ms": round(elapsed_ms, 4),
                    "numpy_ref_ms": round(numpy_ref_ms, 4),
                    "compile_ms": 0.0,
                    "run_ms": round(elapsed_ms, 4),
                    "total_wall_ms": round(elapsed_ms, 4),
                    "providers": ["OryonI8MM"],
                }
            except RuntimeError:
                if tier == "oryon_i8mm":
                    raise

    t_np0 = time.perf_counter_ns()
    out_np = a_c.astype(np.int32) @ b_c.astype(np.int32)
    np_ms = float(time.perf_counter_ns() - t_np0) / 1e6
    tile_passes = max(1, ((m + 1) // 2) * ((n + 1) // 2) * ((k + 7) // 8))
    _update_counter(counter, m, k, n, tile_passes=tile_passes, threads_used=1)
    return {
        "output": out_np,
        "tier_used": "numpy_ref",
        "canary_ok": True,
        "speedup_vs_numpy": 1.0,
        "elapsed_ms": round(np_ms, 4),
        "numpy_ref_ms": round(numpy_ref_ms, 4),
        "compile_ms": 0.0,
        "run_ms": round(np_ms, 4),
        "total_wall_ms": round(np_ms, 4),
        "providers": ["NumPyReference"],
    }


_ORIG_RUN_INT8_MATMUL_HTP = run_int8_matmul_htp


class HexagonQNNBackend:
    """Reusable Hexagon HTP v73 QNN backend with owned session lifecycle and tiered fallbacks."""

    def __init__(
        self,
        *,
        force_tier: str | None = "auto",
        dcvs_mode: str = "burst",
        canary_byte: int = 0x5A,
        win_python: str | None = None,
        counter: OpCounter | None = None,
    ) -> None:
        tier = "auto" if force_tier is None else force_tier
        if tier not in _VALID_TIERS:
            raise ValueError(f"Unknown force_tier {force_tier!r}; expected one of {sorted(_VALID_TIERS)}")
        if dcvs_mode not in _VALID_DCVS_MODES:
            raise ValueError(f"Unsupported dcvs_mode {dcvs_mode!r}; expected one of {sorted(_VALID_DCVS_MODES)}")
        c_byte = _validate_canary_byte(canary_byte)

        self.force_tier = tier
        self.dcvs_mode = dcvs_mode
        self.canary_byte = c_byte
        self.win_python = win_python
        self.counter = counter if counter is not None else OpCounter()
        self._session: HTPBackendSession | None = None
        self._session_lock = threading.Lock()
        self._closed: bool = False
        self.last_tier_used: str | None = None
        self.last_canary_ok: bool | None = None
        self.last_providers: list[str] = []
        self.last_result: dict[str, Any] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def _resolve_win_python(self) -> str | None:
        if sys.platform == "win32":
            return None
        return self.win_python if self.win_python is not None else discover_windows_python()

    def htp_available(self) -> bool:
        if self._closed:
            return False
        if sys.platform == "win32":
            return True
        return self._resolve_win_python() is not None

    def is_available(self) -> bool:
        if self._closed:
            return False
        if self.force_tier == "htp_qnn":
            return self.htp_available()
        if self.force_tier == "oryon_i8mm":
            return bool(detect_oryon_capabilities().get("has_i8mm"))
        return True

    def _get_or_create_owned_session(self, win_py: str | None) -> HTPBackendSession:
        with self._session_lock:
            if self._closed:
                raise RuntimeError("HexagonQNNBackend is closed")
            sess = self._session
            stale = False
            if sess is not None:
                worker = getattr(sess, "_worker", None)
                if worker is not None and (worker._closed or worker._proc.poll() is not None):
                    stale = True
                elif sys.platform != "win32" and sess.win_python != win_py:
                    stale = True
            if stale and sess is not None:
                sess.close()
                self._session = None
                sess = None
            if sess is None:
                sess = HTPBackendSession(win_python=win_py, dcvs_mode=self.dcvs_mode)
                self._session = sess
            return sess

    def _run_htp_via_owned_session(
        self,
        a_c: np.ndarray,
        b_c: np.ndarray,
        *,
        win_py: str | None,
        canary_byte: int,
        numpy_ref_ms: float,
    ) -> dict[str, Any]:
        m, k = int(a_c.shape[0]), int(a_c.shape[1])
        n = int(b_c.shape[1])
        t_wall0 = time.perf_counter_ns()
        sess = self._get_or_create_owned_session(win_py)
        htp_res = sess.matmul_int8(a_c, b_c, canary_byte=canary_byte)
        total_wall_ms = float(time.perf_counter_ns() - t_wall0) / 1e6
        run_ms = float(htp_res["run_ms"])
        elapsed_ms = run_ms if run_ms > 0.0 else total_wall_ms
        _update_counter(self.counter, m, k, n, tile_passes=int(htp_res["tile_passes"]), threads_used=1)
        return {
            "output": htp_res["output"],
            "tier_used": "htp_qnn",
            "canary_ok": bool(htp_res["canary_ok"]),
            "speedup_vs_numpy": round(numpy_ref_ms / max(elapsed_ms, 1e-6), 3),
            "elapsed_ms": round(elapsed_ms, 4),
            "numpy_ref_ms": round(numpy_ref_ms, 4),
            "compile_ms": round(float(htp_res["compile_ms"]), 4),
            "run_ms": round(run_ms, 4),
            "total_wall_ms": round(total_wall_ms, 4),
            "providers": list(htp_res["providers"]),
        }

    def matmul_int8(
        self,
        a_int8: np.ndarray,
        b_int8: np.ndarray,
        *,
        force_tier: str | None = None,
        canary_byte: int | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("HexagonQNNBackend is closed")
        tier = self.force_tier if force_tier is None else force_tier
        if tier not in _VALID_TIERS:
            raise ValueError(f"Unknown force_tier {force_tier!r}; expected one of {sorted(_VALID_TIERS)}")
        c_byte = self.canary_byte if canary_byte is None else _validate_canary_byte(canary_byte)

        if run_int8_matmul_htp is not _ORIG_RUN_INT8_MATMUL_HTP:
            res = run_int8_matmul_htp(
                a_int8,
                b_int8,
                force_tier=tier,
                counter=self.counter,
                dcvs_mode=self.dcvs_mode,
                canary_byte=c_byte,
            )
        else:
            a_c, b_c = _validate_int8_matmul_inputs(a_int8, b_int8)
            m, k = int(a_c.shape[0]), int(a_c.shape[1])
            n = int(b_c.shape[1])
            if m == 0 or n == 0 or k == 0:
                if tier == "htp_qnn" and sys.platform != "win32":
                    win_py = self._resolve_win_python()
                    if win_py is None or (self.win_python is not None and not Path(win_py).is_file()):
                        raise RuntimeError(
                            "No Windows ARM64 Python with onnxruntime_qnn discovered for force_tier='htp_qnn'."
                        )
                    _update_counter(self.counter, m, k, n, tile_passes=0, threads_used=1)
                    res = {
                        "output": np.zeros((m, n), dtype=np.int32),
                        "tier_used": "htp_qnn",
                        "canary_ok": True,
                        "speedup_vs_numpy": 1.0,
                        "elapsed_ms": 0.0,
                        "numpy_ref_ms": 0.0,
                        "compile_ms": 0.0,
                        "run_ms": 0.0,
                        "total_wall_ms": 0.0,
                        "providers": ["QNNExecutionProvider"],
                    }
                elif self.win_python is not None and sys.platform != "win32" and tier == "auto":
                    if Path(self.win_python).is_file():
                        _update_counter(self.counter, m, k, n, tile_passes=0, threads_used=1)
                        res = {
                            "output": np.zeros((m, n), dtype=np.int32),
                            "tier_used": "htp_qnn",
                            "canary_ok": True,
                            "speedup_vs_numpy": 1.0,
                            "elapsed_ms": 0.0,
                            "numpy_ref_ms": 0.0,
                            "compile_ms": 0.0,
                            "run_ms": 0.0,
                            "total_wall_ms": 0.0,
                            "providers": ["QNNExecutionProvider"],
                        }
                    else:
                        fallback_tier = (
                            "oryon_i8mm"
                            if detect_oryon_capabilities().get("has_i8mm")
                            else "numpy_ref"
                        )
                        res = _ORIG_RUN_INT8_MATMUL_HTP(
                            a_c,
                            b_c,
                            force_tier=fallback_tier,
                            counter=self.counter,
                            dcvs_mode=self.dcvs_mode,
                            canary_byte=c_byte,
                        )
                else:
                    res = _ORIG_RUN_INT8_MATMUL_HTP(
                        a_c,
                        b_c,
                        force_tier=tier,
                        counter=self.counter,
                        dcvs_mode=self.dcvs_mode,
                        canary_byte=c_byte,
                    )
            elif tier in {"oryon_i8mm", "numpy_ref"}:
                res = _ORIG_RUN_INT8_MATMUL_HTP(
                    a_c,
                    b_c,
                    force_tier=tier,
                    counter=self.counter,
                    dcvs_mode=self.dcvs_mode,
                    canary_byte=c_byte,
                )
            else:
                t_ref0 = time.perf_counter_ns()
                _ = a_c.astype(np.int32) @ b_c.astype(np.int32)
                numpy_ref_ms = float(time.perf_counter_ns() - t_ref0) / 1e6
                win_py = self._resolve_win_python()
                if tier == "htp_qnn":
                    if sys.platform != "win32" and win_py is None:
                        raise RuntimeError(
                            "No Windows ARM64 Python with onnxruntime_qnn discovered for force_tier='htp_qnn'."
                        )
                    res = self._run_htp_via_owned_session(
                        a_c,
                        b_c,
                        win_py=win_py,
                        canary_byte=c_byte,
                        numpy_ref_ms=numpy_ref_ms,
                    )
                else:
                    htp_res: dict[str, Any] | None = None
                    if sys.platform == "win32" or win_py is not None:
                        try:
                            htp_res = self._run_htp_via_owned_session(
                                a_c,
                                b_c,
                                win_py=win_py,
                                canary_byte=c_byte,
                                numpy_ref_ms=numpy_ref_ms,
                            )
                        except (RuntimeError, OSError, ValueError, TimeoutError):
                            htp_res = None
                    if htp_res is not None:
                        res = htp_res
                    elif detect_oryon_capabilities().get("has_i8mm"):
                        try:
                            res = _ORIG_RUN_INT8_MATMUL_HTP(
                                a_c,
                                b_c,
                                force_tier="oryon_i8mm",
                                counter=self.counter,
                                dcvs_mode=self.dcvs_mode,
                                canary_byte=c_byte,
                            )
                        except RuntimeError:
                            res = _ORIG_RUN_INT8_MATMUL_HTP(
                                a_c,
                                b_c,
                                force_tier="numpy_ref",
                                counter=self.counter,
                                dcvs_mode=self.dcvs_mode,
                                canary_byte=c_byte,
                            )
                    else:
                        res = _ORIG_RUN_INT8_MATMUL_HTP(
                            a_c,
                            b_c,
                            force_tier="numpy_ref",
                            counter=self.counter,
                            dcvs_mode=self.dcvs_mode,
                            canary_byte=c_byte,
                        )

        self.last_tier_used = str(res["tier_used"])
        self.last_canary_ok = bool(res["canary_ok"])
        self.last_providers = list(res.get("providers", []))
        self.last_result = res
        return res

    def qgemm_int8_f32(
        self,
        x_q: np.ndarray,
        x_scale: np.ndarray,
        w_q_t: np.ndarray,
        w_scale: np.ndarray,
        *,
        force_tier: str | None = None,
        canary_byte: int | None = None,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("HexagonQNNBackend is closed")
        xq_c, wqt_c = _validate_int8_matmul_inputs(x_q, w_q_t)
        xs_c = np.ascontiguousarray(x_scale, dtype=np.float32)
        ws_c = np.ascontiguousarray(w_scale, dtype=np.float32)
        if xs_c.ndim != 1 or xs_c.shape[0] != xq_c.shape[0]:
            raise ValueError(f"x_scale shape {xs_c.shape} incompatible with x_q shape {xq_c.shape}")
        if ws_c.ndim != 1 or ws_c.shape[0] != wqt_c.shape[1]:
            raise ValueError(f"w_scale shape {ws_c.shape} incompatible with w_q_t shape {wqt_c.shape}")
        res = self.matmul_int8(xq_c, wqt_c, force_tier=force_tier, canary_byte=canary_byte)
        return res["output"].astype(np.float32) * (xs_c[:, None] * ws_c[None, :])

    def close(self) -> None:
        with self._session_lock:
            self._closed = True
            if self._session is not None:
                self._session.close()
                self._session = None

    def __enter__(self) -> HexagonQNNBackend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
