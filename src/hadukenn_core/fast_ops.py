"""Runtime compiler, Oryon SIMD wrappers, OpCounter, and ctypes bridge for C11 kernels."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NF4_CODEBOOK = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)

_LIB_CACHE: ctypes.CDLL | None = None
_LIB_CACHE_MAP: dict[tuple[str, str, tuple[str, ...]], ctypes.CDLL | None] = {}
_LIB_META_MAP: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
_COMPILER_PROBE_CACHE: dict[tuple[str, int, str], tuple[str, list[str]]] = {}
_SRC_SHA_CACHE: dict[tuple[str, int, int], tuple[str, bytes]] = {}


@dataclass
class OpCounter:
    """Structural operation counter adapted from `btc-mining-sim` for kernel work accounting."""

    mac_ops: int = 0
    lut_lookups: int = 0
    bytes_moved: int = 0
    tile_passes: int = 0
    threads_used: int = 1
    calls: int = 0

    def ops_per_byte(self) -> float:
        if self.bytes_moved <= 0:
            return 0.0
        return float(self.mac_ops + self.lut_lookups) / float(self.bytes_moved)

    def macs_per_tile_pass(self) -> float:
        if self.tile_passes <= 0:
            return 0.0
        return float(self.mac_ops) / float(self.tile_passes)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "mac_ops": self.mac_ops,
            "lut_lookups": self.lut_lookups,
            "bytes_moved": self.bytes_moved,
            "tile_passes": self.tile_passes,
            "threads_used": self.threads_used,
            "calls": self.calls,
            "ops_per_byte": round(self.ops_per_byte(), 6),
            "macs_per_tile_pass": round(self.macs_per_tile_pass(), 6),
        }


def _discover_c_compiler() -> str | None:
    """Discover C11 compiler in order: CC -> ~/.local/bin/clang -> clang -> gcc."""
    cc_env = os.environ.get("CC", "").strip()
    if cc_env:
        cand = Path(cc_env)
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
        resolved = shutil.which(cc_env)
        if resolved is not None:
            return resolved
        return None

    local_clang = Path.home() / ".local" / "bin" / "clang"
    if local_clang.is_file() and os.access(local_clang, os.X_OK):
        return str(local_clang)

    for tool in ("clang", "gcc"):
        found = shutil.which(tool)
        if found is not None:
            return found
    return None


def _compiler_version_banner(cc: str) -> str:
    proc = subprocess.run(
        [cc, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    first = (proc.stdout or proc.stderr or "").splitlines()
    return first[0].strip() if first else "unknown"


def _probe_compiler_flags(cc: str) -> list[str]:
    """Probe target ISA flags without assuming `-fopenmp`."""
    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        candidates = [
            "-mcpu=oryon-1",
            "-march=armv8.6-a+dotprod+i8mm",
            "-march=armv8-a+dotprod",
            "-march=native",
        ]
    else:
        candidates = ["-march=native"]

    for flag in candidates:
        proc = subprocess.run(
            [cc, flag, "-x", "c", "-c", "/dev/null", "-o", "/dev/null"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return [flag]
    return []


def _bind_lib_symbols(lib: ctypes.CDLL) -> None:
    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    i8_p = ctypes.POINTER(ctypes.c_int8)
    u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.cluster_is_head_fwd_bwd_f32.argtypes = [
        f32_p,
        f32_p,
        f32_p,
        i32_p,
        i32_p,
        i32_p,
        f32_p,
        f32_p,
        f32_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cluster_is_head_fwd_bwd_f32.restype = None

    lib.tiled_cce_fwd_bwd_f32.argtypes = [
        f32_p,
        f32_p,
        i32_p,
        f32_p,
        f32_p,
        f32_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tiled_cce_fwd_bwd_f32.restype = None

    if hasattr(lib, "hadukenn_oryon_caps"):
        lib.hadukenn_oryon_caps.argtypes = []
        lib.hadukenn_oryon_caps.restype = ctypes.c_uint32

    if hasattr(lib, "hadukenn_pool_threads"):
        lib.hadukenn_pool_threads.argtypes = []
        lib.hadukenn_pool_threads.restype = ctypes.c_int

    if hasattr(lib, "hadukenn_dot_f32"):
        lib.hadukenn_dot_f32.argtypes = [f32_p, f32_p, ctypes.c_int64]
        lib.hadukenn_dot_f32.restype = ctypes.c_float

    if hasattr(lib, "hadukenn_axpy_f32"):
        lib.hadukenn_axpy_f32.argtypes = [ctypes.c_float, f32_p, f32_p, ctypes.c_int64]
        lib.hadukenn_axpy_f32.restype = None

    if hasattr(lib, "hadukenn_int8_gemm"):
        lib.hadukenn_int8_gemm.argtypes = [
            i8_p,
            i8_p,
            i32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.hadukenn_int8_gemm.restype = ctypes.c_int

    if hasattr(lib, "hadukenn_nf4_matvec_f32"):
        lib.hadukenn_nf4_matvec_f32.argtypes = [
            f32_p,
            u8_p,
            f32_p,
            f32_p,
            f32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.hadukenn_nf4_matvec_f32.restype = None


def _resolve_c_source() -> Path | None:
    csrc_dir = Path(__file__).resolve().parent / "csrc"
    native_src = csrc_dir / "fast_head_native.c"
    if native_src.is_file():
        return native_src
    fallback_src = csrc_dir / "fast_head_avx512.c"
    if fallback_src.is_file():
        return fallback_src
    return None


def _compile_and_load() -> tuple[ctypes.CDLL | None, dict[str, Any]]:
    global _LIB_CACHE
    cc = _discover_c_compiler()
    c_src = _resolve_c_source()
    if cc is None or c_src is None:
        return None, {
            "compiler": cc,
            "compiler_version": None,
            "flags": [],
            "source_path": str(c_src) if c_src else None,
            "library_path": None,
            "cache_key": None,
        }

    st_src = c_src.stat()
    src_stat_key = (str(c_src), st_src.st_mtime_ns, st_src.st_size)
    if src_stat_key in _SRC_SHA_CACHE:
        src_sha, src_bytes = _SRC_SHA_CACHE[src_stat_key]
    else:
        src_bytes = c_src.read_bytes()
        src_sha = hashlib.sha256(src_bytes).hexdigest()
        _SRC_SHA_CACHE[src_stat_key] = (src_sha, src_bytes)

    try:
        cc_mtime = Path(cc).stat().st_mtime_ns
    except OSError:
        cc_mtime = 0
    probe_key = (cc, cc_mtime, platform.machine().lower())
    if probe_key in _COMPILER_PROBE_CACHE:
        cc_ver, isa_flags = _COMPILER_PROBE_CACHE[probe_key]
    else:
        cc_ver = _compiler_version_banner(cc)
        isa_flags = _probe_compiler_flags(cc)
        _COMPILER_PROBE_CACHE[probe_key] = (cc_ver, isa_flags)

    cache_key = (src_sha, cc, tuple(isa_flags))

    if cache_key in _LIB_CACHE_MAP:
        lib = _LIB_CACHE_MAP[cache_key]
        _LIB_CACHE = lib
        return lib, _LIB_META_MAP[cache_key]

    key_material = src_bytes + f"|{cc}|{cc_ver}|{' '.join(isa_flags)}".encode()
    digest = hashlib.sha256(key_material).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "hadukenn_core"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        meta = {
            "compiler": cc,
            "compiler_version": cc_ver,
            "flags": list(isa_flags),
            "source_path": str(c_src),
            "library_path": None,
            "cache_key": digest,
        }
        _LIB_CACHE_MAP[cache_key] = None
        _LIB_META_MAP[cache_key] = meta
        return None, meta

    so_path = cache_dir / f"hadukenn_oryon_{digest}.so"
    if not so_path.is_file():
        tmp_so = cache_dir / f".tmp_{os.getpid()}_{digest}.so"
        cmd = [
            cc,
            "-std=c11",
            "-O3",
            *isa_flags,
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-Werror=implicit-function-declaration",
            str(c_src),
            "-lpthread",
            "-lm",
            "-o",
            str(tmp_so),
        ]
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            if tmp_so.exists():
                tmp_so.unlink(missing_ok=True)
            meta = {
                "compiler": cc,
                "compiler_version": cc_ver,
                "flags": list(isa_flags),
                "source_path": str(c_src),
                "library_path": None,
                "cache_key": digest,
                "compile_error": (proc.stderr or proc.stdout).strip(),
            }
            _LIB_CACHE_MAP[cache_key] = None
            _LIB_META_MAP[cache_key] = meta
            return None, meta
        os.replace(tmp_so, so_path)

    try:
        lib = ctypes.CDLL(str(so_path))
        _bind_lib_symbols(lib)
    except OSError:
        lib = None

    meta = {
        "compiler": cc,
        "compiler_version": cc_ver,
        "flags": list(isa_flags),
        "source_path": str(c_src),
        "library_path": str(so_path) if lib is not None else None,
        "cache_key": digest,
    }
    _LIB_CACHE_MAP[cache_key] = lib
    _LIB_META_MAP[cache_key] = meta
    _LIB_CACHE = lib
    return lib, meta


def get_fast_head_lib() -> ctypes.CDLL | None:
    """Compile `fast_head_native.c` into `$HOME/.cache/hadukenn_core` on first use."""
    lib, _ = _compile_and_load()
    return lib


def detect_oryon_capabilities() -> dict[str, Any]:
    """Return compiler discovery metadata, runtime HWCAP bits, and thread pool size."""
    lib, meta = _compile_and_load()
    caps = 0
    threads = 1
    if lib is not None and hasattr(lib, "hadukenn_oryon_caps"):
        caps = int(lib.hadukenn_oryon_caps())
    if lib is not None and hasattr(lib, "hadukenn_pool_threads"):
        threads = int(lib.hadukenn_pool_threads())

    return {
        "arch": platform.machine(),
        "compiler": meta.get("compiler"),
        "compiler_version": meta.get("compiler_version"),
        "flags": list(meta.get("flags", [])),
        "source_path": meta.get("source_path"),
        "library_path": meta.get("library_path"),
        "cache_key": meta.get("cache_key"),
        "native_loaded": lib is not None,
        "has_neon": bool(caps & (1 << 0)),
        "has_dotprod": bool(caps & (1 << 1)),
        "has_i8mm": bool(caps & (1 << 2)),
        "has_fcma": bool(caps & (1 << 3)),
        "has_bf16": bool(caps & (1 << 4)),
        "pool_ready": bool(caps & (1 << 8)),
        "num_threads": threads,
    }


def _ptr_f32(arr: np.ndarray | None) -> ctypes.POINTER(ctypes.c_float):
    if arr is None:
        return ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ptr_i32(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _ptr_i8(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int8):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _ptr_u8(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def run_dot_f32(
    a: np.ndarray,
    b: np.ndarray,
    counter: OpCounter | None = None,
) -> float:
    """Compute float32 dot product via NEON `vfmaq_f32` (with parallel reduction for large vectors)."""
    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    if a_c.ndim != 1 or b_c.ndim != 1 or a_c.shape[0] != b_c.shape[0]:
        raise ValueError(f"run_dot_f32 requires equal-length 1D vectors, got {a_c.shape} and {b_c.shape}")

    n = int(a_c.shape[0])
    lib = get_fast_head_lib()
    threads = 1
    if lib is not None and hasattr(lib, "hadukenn_dot_f32"):
        res = float(lib.hadukenn_dot_f32(_ptr_f32(a_c), _ptr_f32(b_c), n))
        if n >= 65536 and hasattr(lib, "hadukenn_pool_threads"):
            threads = int(lib.hadukenn_pool_threads())
    else:
        res = float(np.dot(a_c.astype(np.float64), b_c.astype(np.float64)))

    if counter is not None:
        counter.mac_ops += n
        counter.bytes_moved += 2 * n * 4
        counter.tile_passes += max(1, (n + 15) // 16) if n > 0 else 0
        counter.threads_used = max(counter.threads_used, threads)
        counter.calls += 1
    return res


def run_axpy_f32(
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    counter: OpCounter | None = None,
) -> np.ndarray:
    """Compute `y + alpha * x` as a new float32 array without mutating caller inputs."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.number)):
        raise ValueError(f"alpha must be a real scalar, got {alpha!r}")
    alpha_f = float(alpha)
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    y_out = np.array(y, dtype=np.float32, copy=True, order="C")
    if x_c.ndim != 1 or y_out.ndim != 1 or x_c.shape[0] != y_out.shape[0]:
        raise ValueError(f"run_axpy_f32 requires equal-length 1D vectors, got {x_c.shape} and {y_out.shape}")

    n = int(x_c.shape[0])
    lib = get_fast_head_lib()
    threads = 1
    if lib is not None and hasattr(lib, "hadukenn_axpy_f32"):
        lib.hadukenn_axpy_f32(alpha_f, _ptr_f32(x_c), _ptr_f32(y_out), n)
        if n >= 65536 and hasattr(lib, "hadukenn_pool_threads"):
            threads = int(lib.hadukenn_pool_threads())
    else:
        y_out = (y_out + np.float32(alpha_f) * x_c).astype(np.float32)

    if counter is not None:
        counter.mac_ops += n
        counter.bytes_moved += 3 * n * 4
        counter.tile_passes += max(1, (n + 15) // 16) if n > 0 else 0
        counter.threads_used = max(counter.threads_used, threads)
        counter.calls += 1
    return y_out


def run_int8_gemm(
    A_int8: np.ndarray,
    B_int8: np.ndarray,
    mode: str = "auto",
    counter: OpCounter | None = None,
) -> np.ndarray:
    """Execute exact `int8 x int8 -> int32` GEMM via `i8mm` / `sdot` / `scalar` / `numpy`."""
    mode_map = {"auto": 0, "i8mm": 1, "sdot": 2, "scalar": 3, "numpy": -1}
    if mode not in mode_map:
        raise ValueError(f"Unknown INT8 GEMM mode {mode!r}; expected one of {tuple(mode_map)}")

    A_c = np.ascontiguousarray(A_int8, dtype=np.int8)
    B_c = np.ascontiguousarray(B_int8, dtype=np.int8)
    if A_c.ndim != 2 or B_c.ndim != 2 or A_c.shape[1] != B_c.shape[0]:
        raise ValueError(
            f"Incompatible shapes for INT8 GEMM: A={A_c.shape}, B={B_c.shape}"
        )

    m, k = int(A_c.shape[0]), int(A_c.shape[1])
    n = int(B_c.shape[1])
    lib = None if mode == "numpy" else get_fast_head_lib()

    threads = 1
    if mode == "numpy":
        out = A_c.astype(np.int32) @ B_c.astype(np.int32)
    elif lib is not None and hasattr(lib, "hadukenn_int8_gemm"):
        out = np.zeros((m, n), dtype=np.int32) if (m == 0 or n == 0 or k == 0) else np.empty((m, n), dtype=np.int32)
        rc = int(lib.hadukenn_int8_gemm(_ptr_i8(A_c), _ptr_i8(B_c), _ptr_i32(out), m, n, k, mode_map[mode]))
        if rc != 0:
            raise RuntimeError(f"INT8 GEMM mode {mode!r} unavailable on this CPU/build (rc={rc}).")
        if m * n * k >= 32768 and hasattr(lib, "hadukenn_pool_threads"):
            threads = int(lib.hadukenn_pool_threads())
    else:
        if mode in ("i8mm", "sdot"):
            raise RuntimeError(f"Native C11 library unavailable for requested INT8 mode {mode!r}.")
        out = A_c.astype(np.int32) @ B_c.astype(np.int32)

    if counter is not None:
        counter.mac_ops += m * n * k
        counter.bytes_moved += m * k + k * n + 4 * m * n
        counter.tile_passes += max(1, ((m + 1) // 2) * ((n + 1) // 2) * ((k + 7) // 8))
        counter.threads_used = max(counter.threads_used, threads)
        counter.calls += 1
    return out


def run_nf4_matvec(
    x_f32: np.ndarray,
    qweights_u8: np.ndarray,
    scales_f32: np.ndarray,
    group_size: int = 64,
    lut_f32: np.ndarray | None = None,
    counter: OpCounter | None = None,
) -> np.ndarray:
    """Fused NF4 dequantize + MatVec (`even -> high nibble`, `odd -> low nibble`)."""
    if isinstance(group_size, (bool, np.bool_)) or not isinstance(group_size, (int, np.integer)):
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
    gs = int(group_size)
    if gs <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    x_c = np.ascontiguousarray(x_f32, dtype=np.float32)
    qw_c = np.ascontiguousarray(qweights_u8, dtype=np.uint8)
    sc_c = np.ascontiguousarray(scales_f32, dtype=np.float32)
    lut_c = NF4_CODEBOOK if lut_f32 is None else np.ascontiguousarray(lut_f32, dtype=np.float32)

    if x_c.ndim != 1:
        raise ValueError(f"x_f32 must be 1D, got shape {x_c.shape}")
    if qw_c.ndim != 2:
        raise ValueError(f"qweights_u8 must be 2D, got shape {qw_c.shape}")
    if lut_c.shape != (16,):
        raise ValueError(f"lut_f32 must have shape (16,), got {lut_c.shape}")

    k = int(x_c.shape[0])
    m = int(qw_c.shape[0])
    packed_cols = (k + 1) // 2
    n_groups = (k + gs - 1) // gs
    if qw_c.shape[1] != packed_cols:
        raise ValueError(f"qweights_u8 expected {packed_cols} packed cols for K={k}, got {qw_c.shape[1]}")
    if sc_c.shape != (m, n_groups):
        raise ValueError(f"scales_f32 expected shape {(m, n_groups)}, got {sc_c.shape}")

    out = np.zeros(m, dtype=np.float32) if (k == 0 or m == 0) else np.empty(m, dtype=np.float32)
    lib = get_fast_head_lib()
    threads = 1
    if lib is not None and hasattr(lib, "hadukenn_nf4_matvec_f32"):
        lib.hadukenn_nf4_matvec_f32(
            _ptr_f32(x_c),
            _ptr_u8(qw_c),
            _ptr_f32(sc_c),
            _ptr_f32(lut_c),
            _ptr_f32(out),
            m,
            k,
            gs,
        )
        if m * k >= 16384 and m >= 4 and hasattr(lib, "hadukenn_pool_threads"):
            threads = int(lib.hadukenn_pool_threads())
    else:
        hi = (qw_c >> 4) & 0x0F
        lo = qw_c & 0x0F
        codes = np.empty((m, packed_cols * 2), dtype=np.uint8)
        codes[:, 0::2] = hi
        codes[:, 1::2] = lo
        codes = codes[:, :k]
        w_deq = lut_c[codes]
        pad = (-k) % gs
        if pad > 0:
            w_blk = np.concatenate([w_deq, np.zeros((m, pad), dtype=np.float32)], axis=1)
        else:
            w_blk = w_deq
        w_blk = w_blk.reshape(m, n_groups, gs) * sc_c[:, :, None]
        w_flat = w_blk.reshape(m, n_groups * gs)[:, :k]
        out = (w_flat @ x_c).astype(np.float32)

    if counter is not None:
        counter.mac_ops += m * k
        counter.lut_lookups += m * k
        counter.bytes_moved += k * 4 + m * packed_cols + m * n_groups * 4 + m * 4 + 64
        counter.tile_passes += max(1, m * n_groups)
        counter.threads_used = max(counter.threads_used, threads)
        counter.calls += 1
    return out


def benchmark_kernel_parity(
    name: str,
    fn: Callable[[], np.ndarray | float],
    ref_fn: Callable[[], np.ndarray | float],
    warmup: int = 1,
    iters: int = 5,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    assert_parity: bool = True,
    counter: OpCounter | None = None,
) -> dict[str, Any]:
    """Validate numerical parity on identical fixtures BEFORE reporting latency and speedup."""
    ref_out = ref_fn()
    cand_out = fn()
    ref_arr = np.asarray(ref_out, dtype=np.float64)
    cand_arr = np.asarray(cand_out, dtype=np.float64)
    if ref_arr.shape != cand_arr.shape:
        max_abs_err = float("inf")
        parity_ok = False
    else:
        max_abs_err = float(np.max(np.abs(cand_arr - ref_arr))) if cand_arr.size > 0 else 0.0
        parity_ok = bool(np.all(np.isfinite(cand_arr)) and np.allclose(cand_arr, ref_arr, rtol=rtol, atol=atol))

    if assert_parity and not parity_ok:
        raise AssertionError(
            f"{name} parity check failed: max_abs_err={max_abs_err:.6e} (rtol={rtol}, atol={atol})"
        )

    for _ in range(max(0, warmup)):
        _ = ref_fn()
        _ = fn()

    n_iters = max(1, iters)
    t0 = time.perf_counter_ns()
    for _ in range(n_iters):
        _ = ref_fn()
    ref_ms = float(time.perf_counter_ns() - t0) / (1e6 * float(n_iters))

    t1 = time.perf_counter_ns()
    for _ in range(n_iters):
        _ = fn()
    cand_ms = float(time.perf_counter_ns() - t1) / (1e6 * float(n_iters))

    return {
        "name": name,
        "parity_ok": parity_ok,
        "max_abs_err": max_abs_err,
        "ref_ms": round(ref_ms, 4),
        "cand_ms": round(cand_ms, 4),
        "speedup_x": round(ref_ms / max(cand_ms, 1e-9), 3),
        "iters": n_iters,
        "op_counter": counter.as_dict() if counter is not None else None,
    }


def run_cluster_is_avx512(
    H: np.ndarray,
    E_clustered: np.ndarray,
    C_root: np.ndarray,
    target_c: np.ndarray,
    target_m: np.ndarray,
    neg_clusters: np.ndarray,
    compute_grad: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Execute the native C11 cluster-blocked importance-sampled cross-entropy kernel."""
    H_c = np.ascontiguousarray(H, dtype=np.float32)
    E_c = np.ascontiguousarray(E_clustered, dtype=np.float32)
    C_c = np.ascontiguousarray(C_root, dtype=np.float32)
    tc_raw = np.asarray(target_c)
    tm_raw = np.asarray(target_m)
    neg_raw = np.asarray(neg_clusters)
    if tc_raw.dtype.kind not in ("i", "u"):
        raise ValueError(f"target_c must have integer dtype, got {tc_raw.dtype}")
    if tm_raw.dtype.kind not in ("i", "u"):
        raise ValueError(f"target_m must have integer dtype, got {tm_raw.dtype}")
    if neg_raw.dtype.kind not in ("i", "u"):
        raise ValueError(f"neg_clusters must have integer dtype, got {neg_raw.dtype}")

    if H_c.ndim != 2:
        raise ValueError(f"H must be 2D, got shape {H_c.shape}")
    if E_c.ndim != 3:
        raise ValueError(f"E_clustered must be 3D, got shape {E_c.shape}")
    if C_c.ndim != 2:
        raise ValueError(f"C_root must be 2D, got shape {C_c.shape}")
    if tc_raw.ndim != 1:
        raise ValueError(f"target_c must be 1D, got shape {tc_raw.shape}")
    if tm_raw.ndim != 1:
        raise ValueError(f"target_m must be 1D, got shape {tm_raw.shape}")
    if neg_raw.ndim != 2:
        raise ValueError(f"neg_clusters must be 2D, got shape {neg_raw.shape}")

    N, d = int(H_c.shape[0]), int(H_c.shape[1])
    if d < 1:
        raise ValueError(f"embedding dim d must be >= 1, got d={d}")
    NC, S, d_e = int(E_c.shape[0]), int(E_c.shape[1]), int(E_c.shape[2])
    if d_e != d:
        raise ValueError(f"E_clustered dim mismatch: H has d={d}, E_clustered has d={d_e}")
    if C_c.shape != (NC, d):
        raise ValueError(f"C_root expected shape {(NC, d)}, got {C_c.shape}")
    if tc_raw.shape != (N,):
        raise ValueError(f"target_c expected shape {(N,)}, got {tc_raw.shape}")
    if tm_raw.shape != (N,):
        raise ValueError(f"target_m expected shape {(N,)}, got {tm_raw.shape}")
    if neg_raw.shape[0] != N:
        raise ValueError(f"neg_clusters expected {N} rows, got {neg_raw.shape}")
    K_neg = int(neg_raw.shape[1])

    if N > 0:
        if NC < 1 or S < 1:
            raise ValueError(f"NC and S must be >= 1 when N > 0, got NC={NC}, S={S}")
        tc_min, tc_max = int(tc_raw.min()), int(tc_raw.max())
        if tc_min < 0 or tc_max >= NC:
            raise ValueError(f"target_c indices out of bounds for NC={NC}: min={tc_min}, max={tc_max}")
        tm_min, tm_max = int(tm_raw.min()), int(tm_raw.max())
        if tm_min < 0 or tm_max >= S:
            raise ValueError(f"target_m indices out of bounds for S={S}: min={tm_min}, max={tm_max}")
        if K_neg > 0:
            neg_min, neg_max = int(neg_raw.min()), int(neg_raw.max())
            if neg_min < 0 or neg_max >= NC:
                raise ValueError(
                    f"neg_clusters indices out of bounds for NC={NC}: min={neg_min}, max={neg_max}"
                )

    lib = get_fast_head_lib()
    if lib is None:
        raise RuntimeError("Native C11 kernel unavailable on this machine.")

    if N == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, d), dtype=np.float32) if compute_grad else None,
            np.zeros((NC, S, d), dtype=np.float32) if compute_grad else None,
        )

    tc_c = np.ascontiguousarray(tc_raw, dtype=np.int32)
    tm_c = np.ascontiguousarray(tm_raw, dtype=np.int32)
    neg_c = np.ascontiguousarray(neg_raw, dtype=np.int32)
    out_loss = np.zeros(N, dtype=np.float32)
    dH = np.zeros_like(H_c) if compute_grad else None
    dE = np.zeros_like(E_c) if compute_grad else None

    lib.cluster_is_head_fwd_bwd_f32(
        _ptr_f32(H_c),
        _ptr_f32(E_c),
        _ptr_f32(C_c),
        _ptr_i32(tc_c),
        _ptr_i32(tm_c),
        _ptr_i32(neg_c),
        _ptr_f32(out_loss),
        _ptr_f32(dH),
        _ptr_f32(dE),
        N,
        d,
        NC,
        S,
        K_neg,
        1 if compute_grad else 0,
    )
    return out_loss, dH, dE


def run_tiled_cce_avx512(
    H: np.ndarray,
    E: np.ndarray,
    targets: np.ndarray,
    tile_V: int = 512,
    compute_grad: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Execute the native C11 L2-tiled Cut-Cross-Entropy kernel without [N, V] allocation."""
    H_c = np.ascontiguousarray(H, dtype=np.float32)
    E_c = np.ascontiguousarray(E, dtype=np.float32)
    y_raw = np.asarray(targets)
    if y_raw.dtype.kind not in ("i", "u"):
        raise ValueError(f"targets must have integer dtype, got {y_raw.dtype}")

    if H_c.ndim != 2:
        raise ValueError(f"H must be 2D, got shape {H_c.shape}")
    if E_c.ndim != 2:
        raise ValueError(f"E must be 2D, got shape {E_c.shape}")
    if y_raw.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {y_raw.shape}")

    N, d = int(H_c.shape[0]), int(H_c.shape[1])
    if d < 1:
        raise ValueError(f"embedding dim d must be >= 1, got d={d}")
    V, d_e = int(E_c.shape[0]), int(E_c.shape[1])
    if d_e != d:
        raise ValueError(f"E dim mismatch: H has d={d}, E has d={d_e}")
    if y_raw.shape[0] != N:
        raise ValueError(f"targets expected length {N}, got shape {y_raw.shape}")

    if N > 0:
        if V < 1:
            raise ValueError(f"E must have V >= 1 when N > 0, got V={V}")
        y_min, y_max = int(y_raw.min()), int(y_raw.max())
        if y_min < 0 or y_max >= V:
            raise ValueError(f"targets indices out of bounds for V={V}: min={y_min}, max={y_max}")

    lib = get_fast_head_lib()
    if lib is None:
        raise RuntimeError("Native C11 kernel unavailable on this machine.")

    if N == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, d), dtype=np.float32) if compute_grad else None,
            np.zeros((V, d), dtype=np.float32) if compute_grad else None,
        )

    y_c = np.ascontiguousarray(y_raw, dtype=np.int32)
    tile_V = min(max(int(tile_V), 64), 1024)
    out_loss = np.zeros(N, dtype=np.float32)
    dH = np.zeros_like(H_c) if compute_grad else None
    dE = np.zeros_like(E_c) if compute_grad else None

    lib.tiled_cce_fwd_bwd_f32(
        _ptr_f32(H_c),
        _ptr_f32(E_c),
        _ptr_i32(y_c),
        _ptr_f32(out_loss),
        _ptr_f32(dH),
        _ptr_f32(dE),
        N,
        d,
        V,
        tile_V,
        1 if compute_grad else 0,
    )
    return out_loss, dH, dE
