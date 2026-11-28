#!/usr/bin/env python3
"""Live sm_89 CCE kernel bench on the RTX 4080 Laptop, vs a NumPy reference.

Stops if GPU temperature reaches 85 C. Does not overwrite sealed T-46 JSON.
"""
from __future__ import annotations

import ctypes
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

STOP_TEMP_C = 85
REPO = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "hadukenn_core"
LOGDIR = Path.home() / "hadukenn" / "logs"
SO_PATH = CACHE / "libhadukenn_cuda_cce_sm89.so"
LAUNCH_CU = REPO / "scripts" / "cuda_cce_launch.cu"


def gpu_temp() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(float(proc.stdout.strip().splitlines()[0]))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return None


def compile_so() -> Path:
    SO_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc", "-O3", "--use_fast_math", "-std=c++17",
        "-gencode", "arch=compute_89,code=sm_89",
        "-Xcompiler", "-fPIC", "-shared",
        str(LAUNCH_CU),
        "-lcudart",
        "-o", str(SO_PATH),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"nvcc failed:\n{proc.stdout}\n{proc.stderr}")
    return SO_PATH


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(SO_PATH))
    lib.hadukenn_cuda_mup_cce_run.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.hadukenn_cuda_mup_cce_run.restype = ctypes.c_int
    return lib


def numpy_cce(H, E, y, scale):
    logits = (H @ E.T) * scale
    m = np.max(logits, axis=1, keepdims=True)
    lse = (m + np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))).squeeze(-1)
    return (lse - logits[np.arange(H.shape[0]), y]).astype(np.float32)


def main() -> int:
    compile_so()
    lib = load_lib()
    rng = np.random.default_rng(20261126)
    shapes = [
        (32, 256, 1024),
        (64, 384, 2048),
        (128, 576, 4096),
        (192, 576, 4096),
        (256, 768, 8192),
        (384, 1024, 8192),
    ]
    rows = []
    stopped = None
    for N, d, V in shapes:
        temp = gpu_temp()
        if temp is not None and temp >= STOP_TEMP_C:
            stopped = f"stop at {temp} C before N={N} d={d} V={V}"
            break
        H = rng.standard_normal((N, d)).astype(np.float32)
        E = (rng.standard_normal((V, d)) / math.sqrt(d)).astype(np.float32)
        y = rng.integers(0, V, size=N, dtype=np.int32)
        scale = np.float32(384.0 / d)
        loss = np.empty(N, dtype=np.float32)
        ms = ctypes.c_float(0.0)
        t_host0 = time.perf_counter()
        rc = lib.hadukenn_cuda_mup_cce_run(
            H.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            E.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(0, ctypes.POINTER(ctypes.c_float)),
            N, d, V, scale, 0, ctypes.byref(ms),
        )
        t_host = (time.perf_counter() - t_host0) * 1000.0
        if rc != 0:
            raise SystemExit(f"cuda cce rc={rc} for N={N} d={d} V={V}")
        # warmup + 3 timed kernel-only runs
        times = []
        for _ in range(3):
            ms.value = 0.0
            rc = lib.hadukenn_cuda_mup_cce_run(
                H.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                E.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(0, ctypes.POINTER(ctypes.c_float)),
                N, d, V, scale, 0, ctypes.byref(ms),
            )
            if rc != 0:
                raise SystemExit(f"cuda cce rc={rc}")
            times.append(float(ms.value))
        t0 = time.perf_counter()
        ref = numpy_cce(H, E, y, float(scale))
        numpy_ms = (time.perf_counter() - t0) * 1000.0
        err = float(np.max(np.abs(loss - ref)))
        flops = 2.0 * N * V * d
        kms = sum(times) / len(times)
        row = {
            "N": N, "d": d, "V": V,
            "kernel_ms": round(kms, 3),
            "host_ms_incl_copy": round(t_host, 3),
            "numpy_ms": round(numpy_ms, 3),
            "max_abs_err": err,
            "kernel_gflops": round(flops / max(kms, 1e-9) / 1e6, 1),
            "numpy_gflops": round(flops / max(numpy_ms, 1e-9) / 1e6, 1),
            "speedup_vs_numpy": round(numpy_ms / max(kms, 1e-9), 2),
            "gpu_temp_c": gpu_temp(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        temp = gpu_temp()
        if temp is not None and temp >= STOP_TEMP_C:
            stopped = f"stop at {temp} C after N={N} d={d} V={V}"
            break
    out = {
        "so": str(SO_PATH),
        "arch": "sm_89",
        "stop_temp_c": STOP_TEMP_C,
        "stopped": stopped,
        "final_gpu_temp_c": gpu_temp(),
        "rows": rows,
    }
    dest = LOGDIR / "bench-cuda-sm89.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
