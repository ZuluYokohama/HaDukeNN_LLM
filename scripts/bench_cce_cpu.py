#!/usr/bin/env python3
"""CPU CCE size sweep: C11 HAL vs Python row loop vs NumPy GEMM."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from hadukenn_core.tri_machine import (
    _set_openblas_threads,
    get_tri_machine_lib,
    hal_tiled_cce_mup,
    probe_host_hardware,
)


def bench(fn, repeats: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000.0


def main() -> int:
    get_tri_machine_lib(force_rebuild=False)
    rng = np.random.default_rng(20261126)
    shapes = [
        (32, 256, 1024),
        (64, 384, 2048),
        (192, 576, 4096),
        (256, 768, 8192),
        (512, 1024, 8192),
    ]
    threads = 16
    rows = []
    for N, d, V in shapes:
        H = rng.standard_normal((N, d)).astype(np.float32)
        E = (rng.standard_normal((V, d)) / math.sqrt(d)).astype(np.float32)
        y = rng.integers(0, V, size=N, dtype=np.int32)
        scale = 384.0 / d

        def py_row() -> None:
            for i in range(N):
                row = (H[i] @ E.T) * scale
                _ = float(np.max(row) + math.log(float(np.sum(np.exp(row - np.max(row))))) - row[y[i]])

        def np_gemm() -> None:
            logits = (H @ E.T) * scale
            m = np.max(logits, axis=1, keepdims=True)
            _ = (m.squeeze(-1) + np.log(np.sum(np.exp(logits - m), axis=1))) - logits[np.arange(N), y]

        def c11() -> None:
            hal_tiled_cce_mup(H, E, y, tile_V=512, mup_base_d=384, compute_grad=False, num_threads=threads)

        # Serial Python loop at OpenBLAS=1; packed GEMM at `threads`; park
        # OpenBLAS before the OpenMP kernel so the pools do not contend.
        _set_openblas_threads(1)
        py_ms = bench(py_row, repeats=3, warmup=1)
        _set_openblas_threads(threads)
        np_ms = bench(np_gemm)
        _set_openblas_threads(1)
        c_ms = bench(c11)
        flops = 2.0 * N * V * d
        row = {
            "N": N, "d": d, "V": V, "threads": threads,
            "c11_ms": round(c_ms, 3),
            "python_row_ms": round(py_ms, 3),
            "numpy_gemm_ms": round(np_ms, 3),
            "c11_gflops": round(flops / c_ms / 1e6, 1),
            "numpy_gflops": round(flops / np_ms / 1e6, 1),
            "speedup_vs_python_row": round(py_ms / c_ms, 2),
            "speedup_vs_numpy_gemm": round(np_ms / c_ms, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {"host": probe_host_hardware(), "rows": rows}
    dest = Path.home() / "hadukenn" / "logs" / "bench-cce-cpu.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
