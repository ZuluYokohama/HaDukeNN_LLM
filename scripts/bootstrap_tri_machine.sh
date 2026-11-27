#!/usr/bin/env bash
# Unified POSIX / Linux / WSL2 Bare-Metal Bootstrap for WBeast, LapBeast (lane2), and Zenbook
set -euo pipefail

MACHINE="${1:-auto}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_OUT="${HOME}/.cache/hadukenn_core"
mkdir -p "${CACHE_OUT}"

ARCH="$(uname -m)"
if [[ "${MACHINE}" == "auto" ]]; then
    if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
        MACHINE="zenbook"
    elif [[ -d "/sys/class/power_supply/BAT0" ]] || [[ "$(nproc)" -ge 20 ]]; then
        MACHINE="lapbeast"
    else
        MACHINE="wbeast"
    fi
fi

echo "[HaDukeNN v0.9.0] Building bare-metal HAL for machine='${MACHINE}' (arch='${ARCH}')"

CC_BIN="${CC:-gcc}"
CFLAGS=(-O3 -shared -fPIC -fopenmp "-I${REPO_ROOT}/src/hadukenn_core/csrc")

case "${MACHINE}" in
    zenbook)
        CFLAGS+=(-march=armv8.6-a+dotprod+i8mm+fp16)
        export OMP_NUM_THREADS=8
        export HADUKENN_MACHINE="zenbook"
        ;;
    lapbeast)
        CFLAGS+=(-mavx2 -mfma -mbmi2)
        # Try enabling AVX-VNNI on Core Ultra 9 275HX if supported by compiler
        if "${CC_BIN}" -mavxvnni -x c -c /dev/null -o /dev/null 2>/dev/null; then
            CFLAGS+=(-mavxvnni)
        fi
        export OMP_NUM_THREADS=8
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close
        export HADUKENN_MACHINE="lapbeast"
        ;;
    wbeast|*)
        CFLAGS+=(-march=native)
        export OMP_NUM_THREADS=12
        export OMP_PROC_BIND=close
        export HADUKENN_MACHINE="wbeast"
        ;;
esac

"${CC_BIN}" "${CFLAGS[@]}" \
    "${REPO_ROOT}/src/hadukenn_core/csrc/tri_machine_hal.c" \
    -lm -o "${CACHE_OUT}/libhadukenn_tri_hal.so"

if command -v nvcc >/dev/null 2>&1 && [[ "${MACHINE}" != "zenbook" ]]; then
    echo "[HaDukeNN v0.9.0] Compiling CUDA kernels via nvcc..."
    NVCC_GENCODE=(-gencode arch=compute_120,code=sm_120)
    if [[ "${MACHINE}" == "wbeast" ]]; then
        NVCC_GENCODE+=(-gencode arch=compute_75,code=sm_75)
    fi
    nvcc -O3 --use_fast_math -std=c++17 "${NVCC_GENCODE[@]}" \
        -Xcompiler -fPIC -shared \
        "${REPO_ROOT}/src/hadukenn_core/csrc/cuda_blackwell_turing_kernels.cu" \
        -o "${CACHE_OUT}/libhadukenn_cuda_kernels.so"
fi

PYTHONPATH="${REPO_ROOT}/src" python3 "${REPO_ROOT}/scripts/build_tri_machine.py" --machine "${MACHINE}"
