"""Standalone Windows ARM64 Hexagon HTP v73 QNN worker for HaDukeNN_LLM.

Runs out-of-process via ``python.exe -u npu_qnn_worker.py --serve --work-dir <dir>``
(or ``--oneshot``), and is also importable in-process when ``sys.platform == "win32"``.

Dependencies are restricted strictly to the Python standard library plus
``numpy``, ``onnx``, ``onnxruntime``, and ``onnxruntime_qnn`` (zero ``hadukenn_core``
imports).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

_ONNX_OPSET_VERSION = 21
_ONNX_IR_VERSION = 10
_MAX_SESSION_CACHE = 16
_K_STACK_TILE = 511
_M_TILE_DIRECT = 512
_M_TILE_STACKED = 256
_DIRECT_ACCUM_LIMIT = 32767

_SESSION_CACHE: OrderedDict[tuple[int, int, int, str, str], tuple[Any, list[str]]] = OrderedDict()
_QNN_REGISTERED = False


def _reject_nonfinite_json(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is forbidden: {token!r}")


def _json_loads_strict(raw: str) -> dict[str, Any]:
    obj = json.loads(raw, parse_constant=_reject_nonfinite_json)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def _json_dumps_strict(payload: dict[str, Any]) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _require_nonbool_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be a non-bool integer, got {type(val).__name__}")
    return val


def validate_canary_byte(canary_byte: int) -> int:
    val = _require_nonbool_int(canary_byte, "canary_byte")
    if not (0 <= val <= 255):
        raise ValueError(f"canary_byte must be in [0, 255], got {val}")
    return val


def verify_canary_raw(raw_canary: np.ndarray, canary_byte: int = 0x5A) -> bool:
    c_byte = validate_canary_byte(canary_byte)
    arr = np.asarray(raw_canary, dtype=np.uint8).ravel()
    return bool(arr.size > 0 and not np.all(arr == c_byte))


def _validate_static_positive_shape(shape: tuple[int, ...], name: str) -> tuple[int, ...]:
    dims: list[int] = []
    for d in shape:
        if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
            raise ValueError(f"{name} must have static positive integer dims, got {shape!r}")
        dims.append(int(d))
    return tuple(dims)


def _split_signed_nibbles(arr_i8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact 4-bit base-16 signed decomposition: x = (hi << 4) + lo with hi, lo in [-8, 8]."""
    x = np.asarray(arr_i8, dtype=np.int32)
    lo = ((x + 8) & 15) - 8
    hi = (x - lo) >> 4
    return hi.astype(np.int8), lo.astype(np.int8)


def _register_qnn_npu() -> tuple[Any, list[Any]]:
    """Register QNN EP dynamically and return NPU OrtEpDevice entries (fail closed if none)."""
    global _QNN_REGISTERED
    import onnxruntime as ort
    import onnxruntime_qnn

    if not _QNN_REGISTERED:
        ep_name = onnxruntime_qnn.get_ep_name()
        lib_path = onnxruntime_qnn.get_library_path()
        try:
            ort.register_execution_provider_library(ep_name, lib_path)
        except Exception as exc:
            msg = str(exc).lower()
            if "already" not in msg and "exist" not in msg:
                raise
        _QNN_REGISTERED = True

    npu_devs = [
        dev
        for dev in ort.get_ep_devices()
        if dev.device.type == ort.OrtHardwareDeviceType.NPU
    ]
    if not npu_devs:
        raise RuntimeError("No OrtHardwareDeviceType.NPU device discovered for QNNExecutionProvider.")
    return ort, npu_devs


def build_qdq_matmul_onnx(weight_i8: np.ndarray, m_rows: int) -> bytes:
    """Build static opset-21 / ir_version=10 QDQ MatMul ONNX graph bytes."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w_c = np.ascontiguousarray(weight_i8, dtype=np.int8)
    if w_c.ndim != 2:
        raise ValueError(f"Weight matrix must be 2D, got shape {w_c.shape}")
    m_dim, k_dim = _validate_static_positive_shape((m_rows, int(w_c.shape[0])), "A_shape")
    _, n_dim = _validate_static_positive_shape((int(w_c.shape[0]), int(w_c.shape[1])), "W_shape")

    a_scale = helper.make_tensor("a_s", TensorProto.FLOAT, [], [1.0 / 256.0])
    a_zp = helper.make_tensor("a_zp", TensorProto.UINT16, [], [32768])

    w_init = numpy_helper.from_array(w_c, name="W_i8")
    w_scale = numpy_helper.from_array(np.ones((n_dim,), dtype=np.float32), name="w_s")
    w_zp = numpy_helper.from_array(np.zeros((n_dim,), dtype=np.int8), name="w_zp")

    y_scale = helper.make_tensor("y_s", TensorProto.FLOAT, [], [1.0])
    y_zp = helper.make_tensor("y_zp", TensorProto.UINT16, [], [32768])

    nodes = [
        helper.make_node("DequantizeLinear", ["A_u16", "a_s", "a_zp"], ["A_f"]),
        helper.make_node("DequantizeLinear", ["W_i8", "w_s", "w_zp"], ["W_f"], axis=1),
        helper.make_node("MatMul", ["A_f", "W_f"], ["Y_f"]),
        helper.make_node("QuantizeLinear", ["Y_f", "y_s", "y_zp"], ["Y_u16"]),
    ]
    graph = helper.make_graph(
        nodes,
        "hadukenn_htp_qdq_matmul",
        [helper.make_tensor_value_info("A_u16", TensorProto.UINT16, [m_dim, k_dim])],
        [helper.make_tensor_value_info("Y_u16", TensorProto.UINT16, [m_dim, n_dim])],
        initializer=[a_scale, a_zp, w_init, w_scale, w_zp, y_scale, y_zp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", _ONNX_OPSET_VERSION)],
        ir_version=_ONNX_IR_VERSION,
    )
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def _get_or_compile_htp_session(
    weight_i8: np.ndarray,
    m_eff: int,
    dcvs_mode: str,
    work_dir: Path,
) -> tuple[Any, list[str], float, bool]:
    """Return cached or freshly compiled strict-HTP InferenceSession, cleaning .onnx immediately."""
    if dcvs_mode not in {"burst", "power_saver"}:
        raise ValueError(f"Unsupported dcvs_mode {dcvs_mode!r}; expected 'burst' or 'power_saver'")

    w_c = np.ascontiguousarray(weight_i8, dtype=np.int8)
    k_eff, n_eff = _validate_static_positive_shape((int(w_c.shape[0]), int(w_c.shape[1])), "weight_shape")
    _validate_static_positive_shape((m_eff, k_eff), "input_shape")
    w_sha = hashlib.sha256(w_c.tobytes()).hexdigest()
    key = (int(m_eff), k_eff, n_eff, dcvs_mode, w_sha)

    cached = _SESSION_CACHE.get(key)
    if cached is not None:
        _SESSION_CACHE.move_to_end(key)
        sess, providers = cached
        return sess, providers, 0.0, True

    ort, npu_devs = _register_qnn_npu()
    onnx_bytes = build_qdq_matmul_onnx(w_c, m_eff)

    work_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="qdq_", suffix=".onnx", dir=str(work_dir))
    tmp_path = Path(tmp_name)
    t0 = time.perf_counter_ns()
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(onnx_bytes)

        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(
            npu_devs,
            {
                "backend_type": "htp",
                "htp_performance_mode": dcvs_mode,
                "htp_graph_finalization_optimization_mode": "3",
            },
        )
        sess = ort.InferenceSession(str(tmp_path), sess_options=so)
    finally:
        tmp_path.unlink(missing_ok=True)

    compile_ms = float(time.perf_counter_ns() - t0) / 1e6
    providers = [str(p) for p in sess.get_providers()]
    if "QNNExecutionProvider" not in providers:
        raise RuntimeError(f"QNNExecutionProvider not active on session: providers={providers}")

    while len(_SESSION_CACHE) >= _MAX_SESSION_CACHE:
        _SESSION_CACHE.popitem(last=False)
    _SESSION_CACHE[key] = (sess, providers)
    return sess, providers, compile_ms, False


def _run_session_with_canary(
    sess: Any,
    a_u16: np.ndarray,
    out_shape: tuple[int, int],
    canary_byte: int,
) -> tuple[np.ndarray, bool, float]:
    import onnxruntime as ort

    m_out, n_out = out_shape
    out_nbytes = int(m_out) * int(n_out) * 2
    raw_canary = np.full(out_nbytes, canary_byte, dtype=np.uint8)
    out_view = raw_canary.view(np.uint16).reshape(out_shape)

    a_c = np.ascontiguousarray(a_u16, dtype=np.uint16)
    iob = sess.io_binding()
    iob.bind_ortvalue_input("A_u16", ort.OrtValue.ortvalue_from_numpy(a_c, "cpu", 0))
    iob.bind_ortvalue_output("Y_u16", ort.OrtValue.ortvalue_from_numpy(out_view, "cpu", 0))

    raw_canary.fill(canary_byte)
    t0 = time.perf_counter_ns()
    sess.run_with_iobinding(iob)
    run_ms = float(time.perf_counter_ns() - t0) / 1e6

    canary_ok = verify_canary_raw(raw_canary, canary_byte=canary_byte) and bool(np.all(np.isfinite(out_view)))
    if not canary_ok and raw_canary.size > 0 and bool(np.all(np.isfinite(out_view))):
        alt_byte = canary_byte ^ 0x01
        raw_alt = np.full(out_nbytes, alt_byte, dtype=np.uint8)
        out_alt = raw_alt.view(np.uint16).reshape(out_shape)
        iob_alt = sess.io_binding()
        iob_alt.bind_ortvalue_input("A_u16", ort.OrtValue.ortvalue_from_numpy(a_c, "cpu", 0))
        iob_alt.bind_ortvalue_output("Y_u16", ort.OrtValue.ortvalue_from_numpy(out_alt, "cpu", 0))
        raw_alt.fill(alt_byte)
        t1 = time.perf_counter_ns()
        sess.run_with_iobinding(iob_alt)
        run_ms += float(time.perf_counter_ns() - t1) / 1e6
        canary_ok = (
            verify_canary_raw(raw_alt, canary_byte=alt_byte)
            and bool(np.all(np.isfinite(out_alt)))
            and bool(np.array_equal(out_alt, out_view))
        )
    y_i32 = out_view.astype(np.int32) - 32768
    return y_i32, canary_ok, run_ms


def execute_int8_matmul_htp(
    a_int8: np.ndarray,
    b_int8: np.ndarray,
    *,
    dcvs_mode: str = "burst",
    canary_byte: int = 0x5A,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute exact ``int8 x int8 -> int32`` MatMul on Hexagon HTP v73."""
    c_byte = validate_canary_byte(canary_byte)
    if dcvs_mode not in {"burst", "power_saver"}:
        raise ValueError(f"Unsupported dcvs_mode {dcvs_mode!r}; expected 'burst' or 'power_saver'")

    a_arr = np.asarray(a_int8)
    b_arr = np.asarray(b_int8)
    if a_arr.dtype != np.int8 or b_arr.dtype != np.int8:
        raise TypeError(f"Expected int8 inputs, got {a_arr.dtype} and {b_arr.dtype}")
    if a_arr.ndim != 2 or b_arr.ndim != 2 or a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError(f"Incompatible shapes: A={a_arr.shape}, B={b_arr.shape}")

    a_c = np.ascontiguousarray(a_arr, dtype=np.int8)
    b_c = np.ascontiguousarray(b_arr, dtype=np.int8)
    m, k = int(a_c.shape[0]), int(a_c.shape[1])
    n = int(b_c.shape[1])

    if m == 0 or n == 0 or k == 0:
        _register_qnn_npu()
        return {
            "output": np.zeros((m, n), dtype=np.int32),
            "canary_ok": True,
            "compile_ms": 0.0,
            "run_ms": 0.0,
            "providers": ["QNNExecutionProvider"],
            "tile_passes": 0,
            "cache_hit": True,
        }

    stage_dir = work_dir if work_dir is not None else Path(tempfile.gettempdir())
    max_abs_a = int(np.max(np.abs(a_c.astype(np.int32))))
    max_abs_b = int(np.max(np.abs(b_c.astype(np.int32))))
    use_direct = (k * max_abs_a * max_abs_b) <= _DIRECT_ACCUM_LIMIT

    out_i32 = np.zeros((m, n), dtype=np.int32)
    total_compile_ms = 0.0
    total_run_ms = 0.0
    all_canary_ok = True
    providers_used: list[str] = []
    tile_passes = 0
    all_cache_hit = True

    if use_direct:
        for m0 in range(0, m, _M_TILE_DIRECT):
            m1 = min(m0 + _M_TILE_DIRECT, m)
            m_t = m1 - m0
            sess, providers, comp_ms, hit = _get_or_compile_htp_session(b_c, m_t, dcvs_mode, stage_dir)
            total_compile_ms += comp_ms
            all_cache_hit = all_cache_hit and hit
            providers_used = providers
            a_u16 = (32768 + a_c[m0:m1].astype(np.int32) * 256).astype(np.uint16)
            y_tile, c_ok, r_ms = _run_session_with_canary(sess, a_u16, (m_t, n), c_byte)
            total_run_ms += r_ms
            all_canary_ok = all_canary_ok and c_ok
            out_i32[m0:m1] = y_tile
            tile_passes += 1
    else:
        a_hi, a_lo = _split_signed_nibbles(a_c)
        b_hi, b_lo = _split_signed_nibbles(b_c)
        for k0 in range(0, k, _K_STACK_TILE):
            k1 = min(k0 + _K_STACK_TILE, k)
            b_stack = np.ascontiguousarray(np.hstack([b_hi[k0:k1], b_lo[k0:k1]]), dtype=np.int8)
            for m0 in range(0, m, _M_TILE_STACKED):
                m1 = min(m0 + _M_TILE_STACKED, m)
                m_t = m1 - m0
                sess, providers, comp_ms, hit = _get_or_compile_htp_session(
                    b_stack, 2 * m_t, dcvs_mode, stage_dir
                )
                total_compile_ms += comp_ms
                all_cache_hit = all_cache_hit and hit
                providers_used = providers
                a_stack = np.ascontiguousarray(np.vstack([a_hi[m0:m1, k0:k1], a_lo[m0:m1, k0:k1]]), dtype=np.int8)
                a_u16 = (32768 + a_stack.astype(np.int32) * 256).astype(np.uint16)

                t_rec0 = time.perf_counter_ns()
                y_stack, c_ok, r_ms = _run_session_with_canary(sess, a_u16, (2 * m_t, 2 * n), c_byte)
                hh = y_stack[:m_t, :n]
                hl = y_stack[:m_t, n:]
                lh = y_stack[m_t:, :n]
                ll = y_stack[m_t:, n:]
                out_i32[m0:m1] += (hh << 8) + ((hl + lh) << 4) + ll
                recomb_ms = float(time.perf_counter_ns() - t_rec0) / 1e6 - r_ms
                total_run_ms += r_ms + max(0.0, recomb_ms)
                all_canary_ok = all_canary_ok and c_ok
                tile_passes += 1

    return {
        "output": out_i32,
        "canary_ok": all_canary_ok,
        "compile_ms": total_compile_ms,
        "run_ms": total_run_ms,
        "providers": providers_used,
        "tile_passes": tile_passes,
        "cache_hit": all_cache_hit,
    }


def _read_slab_inputs(req: dict[str, Any], work_dir: Path) -> dict[str, np.ndarray]:
    slab_in_raw = req.get("_slab_in_path")
    slab_inputs = req.get("_slab_inputs")
    if not isinstance(slab_in_raw, str) or not isinstance(slab_inputs, dict):
        raise ValueError("Missing _slab_in_path or _slab_inputs descriptor")

    slab_path = Path(slab_in_raw)
    if not slab_path.is_absolute():
        slab_path = work_dir / slab_path
    if not slab_path.is_file():
        raise FileNotFoundError(f"Input slab file not found: {slab_path.name}")

    raw_bytes = slab_path.read_bytes()
    total_len = len(raw_bytes)
    tensors: dict[str, np.ndarray] = {}
    for name, desc in slab_inputs.items():
        if not isinstance(desc, dict):
            raise ValueError(f"Invalid slab descriptor for {name!r}")
        offset = _require_nonbool_int(desc.get("offset"), f"{name}.offset")
        nbytes = _require_nonbool_int(desc.get("nbytes"), f"{name}.nbytes")
        if offset < 0 or nbytes < 0 or offset + nbytes > total_len:
            raise ValueError(f"Slab bounds out of range for {name!r}: off={offset}, nbytes={nbytes}, total={total_len}")
        dt = np.dtype(str(desc.get("dtype")))
        raw_shape = desc.get("shape")
        if not isinstance(raw_shape, list):
            raise ValueError(f"Invalid shape for {name!r}")
        shape = tuple(_require_nonbool_int(d, f"{name}.shape[]") for d in raw_shape)
        if any(d < 0 for d in shape):
            raise ValueError(f"Negative dimension in shape for {name!r}: {shape}")
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize if shape else dt.itemsize
        if expected_nbytes != nbytes:
            raise ValueError(f"Slab nbytes mismatch for {name!r}: {nbytes} != {expected_nbytes}")
        chunk = raw_bytes[offset : offset + nbytes]
        tensors[name] = np.frombuffer(chunk, dtype=dt).reshape(shape).copy()
    return tensors


def _write_slab_outputs(tensors: dict[str, np.ndarray], slab_out_raw: Any, work_dir: Path) -> dict[str, Any]:
    if not isinstance(slab_out_raw, str) or not slab_out_raw:
        raise ValueError("Missing _slab_out_path")
    slab_path = Path(slab_out_raw)
    if not slab_path.is_absolute():
        slab_path = work_dir / slab_path
    slab_path.parent.mkdir(parents=True, exist_ok=True)

    descriptors: dict[str, Any] = {}
    offset = 0
    with open(slab_path, "wb") as fh:
        for name, arr in tensors.items():
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


def _handle_request(req: dict[str, Any], work_dir: Path) -> tuple[dict[str, Any], bool]:
    action = req.get("action")
    if action == "ping":
        return {"ok": True, "action": "ping", "pid": os.getpid()}, False
    if action == "shutdown":
        return {"ok": True, "action": "shutdown"}, True
    if action == "run_int8_matmul":
        tensors = _read_slab_inputs(req, work_dir)
        if "a" not in tensors or "b" not in tensors:
            raise ValueError("run_int8_matmul requires slab tensors 'a' and 'b'")
        dcvs_mode = str(req.get("dcvs_mode", "burst"))
        canary_byte = req.get("canary_byte", 0x5A)
        res = execute_int8_matmul_htp(
            tensors["a"],
            tensors["b"],
            dcvs_mode=dcvs_mode,
            canary_byte=canary_byte,
            work_dir=work_dir,
        )
        slab_outputs = _write_slab_outputs({"output": res["output"]}, req.get("_slab_out_path"), work_dir)
        return (
            {
                "ok": True,
                "action": "run_int8_matmul",
                "canary_ok": bool(res["canary_ok"]),
                "compile_ms": float(res["compile_ms"]),
                "run_ms": float(res["run_ms"]),
                "providers": list(res["providers"]),
                "tile_passes": int(res["tile_passes"]),
                "cache_hit": bool(res["cache_hit"]),
                "_slab_outputs": slab_outputs,
            },
            False,
        )
    raise ValueError(f"Unsupported worker action: {action!r}")


def _isolate_fd1() -> Any:
    """Duplicate OS fd 1 for JSONL protocol and route fd 1 + sys.stdout to stderr."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


def _serve_loop(work_dir: Path) -> int:
    proto_out = _isolate_fd1()
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            should_exit = False
            try:
                req = _json_loads_strict(line)
                resp, should_exit = _handle_request(req, work_dir)
            except Exception as exc:
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            proto_out.write(_json_dumps_strict(resp) + "\n")
            proto_out.flush()
            if should_exit:
                break
    finally:
        proto_out.flush()
        proto_out.close()
    return 0


def _run_oneshot(work_dir: Path, req_file: str | None, resp_file: str | None) -> int:
    proto_out = None
    if resp_file is None:
        proto_out = _isolate_fd1()
    try:
        if req_file is None:
            raise ValueError("--oneshot requires --req-file")
        req_path = Path(req_file)
        req = _json_loads_strict(req_path.read_text(encoding="utf-8"))
        resp, _ = _handle_request(req, work_dir)
    except Exception as exc:
        resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out_text = _json_dumps_strict(resp) + "\n"
    if resp_file is not None:
        Path(resp_file).write_text(out_text, encoding="utf-8")
    elif proto_out is not None:
        proto_out.write(out_text)
        proto_out.flush()
        proto_out.close()
    return 0 if resp.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HaDukeNN Hexagon HTP QNN worker")
    parser.add_argument("--serve", action="store_true", help="Run persistent JSONL IPC server")
    parser.add_argument("--oneshot", action="store_true", help="Execute a single request JSON file")
    parser.add_argument("--work-dir", default="", help="Temporary staging directory for slabs and ONNX graphs")
    parser.add_argument("--req-file", default=None, help="JSON request file path for --oneshot")
    parser.add_argument("--resp-file", default=None, help="JSON response file path for --oneshot")
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir())
    if args.serve:
        return _serve_loop(work_dir)
    if args.oneshot:
        return _run_oneshot(work_dir, args.req_file, args.resp_file)
    parser.error("Must specify either --serve or --oneshot")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
