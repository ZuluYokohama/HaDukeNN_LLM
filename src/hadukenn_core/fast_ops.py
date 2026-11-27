"""Runtime compiler and ctypes bridge for the C11 AVX-512 structured-head kernels."""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

_LIB_CACHE: ctypes.CDLL | None = None
_LIB_TRIED = False


def get_fast_head_lib() -> ctypes.CDLL | None:
    """Compile `fast_head_avx512.c` into `$HOME/.cache/hadukenn_core` on first use."""
    global _LIB_CACHE, _LIB_TRIED
    if _LIB_TRIED:
        return _LIB_CACHE
    _LIB_TRIED = True
    gcc = shutil.which("gcc")
    if gcc is None:
        return None
    c_src = Path(__file__).resolve().parent / "csrc" / "fast_head_avx512.c"
    if not c_src.is_file():
        return None
    digest = hashlib.sha256(c_src.read_bytes()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "hadukenn_core"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    so_path = cache_dir / f"hadukenn_fast_head_{digest}.so"
    if not so_path.is_file():
        cmd = [
            gcc,
            "-O3",
            "-march=native",
            "-fopenmp",
            "-shared",
            "-fPIC",
            str(c_src),
            "-lm",
            "-o",
            str(so_path),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        return None

    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)

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
    _LIB_CACHE = lib
    return _LIB_CACHE


def _ptr_f32(arr: np.ndarray | None) -> ctypes.POINTER(ctypes.c_float):
    if arr is None:
        return ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ptr_i32(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def run_cluster_is_avx512(
    H: np.ndarray,
    E_clustered: np.ndarray,
    C_root: np.ndarray,
    target_c: np.ndarray,
    target_m: np.ndarray,
    neg_clusters: np.ndarray,
    compute_grad: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Execute the AVX-512 C11 cluster-blocked importance-sampled cross-entropy kernel."""
    lib = get_fast_head_lib()
    if lib is None:
        raise RuntimeError("Native C11 kernel unavailable on this machine.")
    H_c = np.ascontiguousarray(H, dtype=np.float32)
    E_c = np.ascontiguousarray(E_clustered, dtype=np.float32)
    C_c = np.ascontiguousarray(C_root, dtype=np.float32)
    tc_c = np.ascontiguousarray(target_c, dtype=np.int32)
    tm_c = np.ascontiguousarray(target_m, dtype=np.int32)
    neg_c = np.ascontiguousarray(neg_clusters, dtype=np.int32)

    N, d = H_c.shape
    NC, S, _ = E_c.shape
    K_neg = neg_c.shape[1]
    out_loss = np.empty(N, dtype=np.float32)
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
    """Execute the AVX-512 C11 L2-tiled Cut-Cross-Entropy kernel without [N, V] allocation."""
    lib = get_fast_head_lib()
    if lib is None:
        raise RuntimeError("Native C11 kernel unavailable on this machine.")
    H_c = np.ascontiguousarray(H, dtype=np.float32)
    E_c = np.ascontiguousarray(E, dtype=np.float32)
    y_c = np.ascontiguousarray(targets, dtype=np.int32)
    N, d = H_c.shape
    V = E_c.shape[0]
    tile_V = min(max(int(tile_V), 64), 1024)
    out_loss = np.empty(N, dtype=np.float32)
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
