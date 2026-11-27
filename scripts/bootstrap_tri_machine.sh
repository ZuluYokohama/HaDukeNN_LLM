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

nvcc_supports_arch() {
    local sm="$1"
    local tmp_cu tmp_o
    tmp_cu="$(mktemp /tmp/hadukenn_nvcc_XXXXXX.cu)"
    tmp_o="$(mktemp /tmp/hadukenn_nvcc_XXXXXX.o)"
    printf 'extern "C" __global__ void k() {}\n' > "${tmp_cu}"
    if nvcc -c "${tmp_cu}" -gencode "arch=compute_${sm},code=sm_${sm}" -o "${tmp_o}" >/dev/null 2>&1; then
        rm -f "${tmp_cu}" "${tmp_o}"
        return 0
    fi
    rm -f "${tmp_cu}" "${tmp_o}"
    return 1
}

if command -v nvcc >/dev/null 2>&1 && [[ "${MACHINE}" != "zenbook" ]]; then
    echo "[HaDukeNN v0.9.0] Compiling CUDA kernels via nvcc..."
    NVCC_GENCODE=()
    # Prefer the live GPU's compute capability (e.g. 8.9 -> 89 on an RTX 4080 Laptop).
    if command -v nvidia-smi >/dev/null 2>&1; then
        LIVE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
        LIVE_SM="${LIVE_CAP//./}"
        if [[ -n "${LIVE_SM}" ]] && nvcc_supports_arch "${LIVE_SM}"; then
            NVCC_GENCODE+=(-gencode "arch=compute_${LIVE_SM},code=sm_${LIVE_SM}")
            echo "[HaDukeNN v0.9.0] nvcc gencode sm_${LIVE_SM} from nvidia-smi compute_cap=${LIVE_CAP}"
        fi
    fi
    # Keep sm_120 when this nvcc can emit it (Blackwell toolkits).
    if nvcc_supports_arch 120; then
        if [[ " ${NVCC_GENCODE[*]} " != *"sm_120"* ]]; then
            NVCC_GENCODE+=(-gencode arch=compute_120,code=sm_120)
            echo "[HaDukeNN v0.9.0] nvcc gencode sm_120 (toolkit supports Blackwell)"
        fi
    else
        echo "[HaDukeNN v0.9.0] nvcc does not support sm_120; skipping that gencode"
    fi
    if [[ "${MACHINE}" == "wbeast" ]] && nvcc_supports_arch 75; then
        NVCC_GENCODE+=(-gencode arch=compute_75,code=sm_75)
    fi
    if [[ "${#NVCC_GENCODE[@]}" -eq 0 ]]; then
        echo "[HaDukeNN v0.9.0] ERROR: no supported nvcc gencode for this GPU/toolkit" >&2
        exit 1
    fi
    nvcc -O3 --use_fast_math -std=c++17 "${NVCC_GENCODE[@]}" \
        -Xcompiler -fPIC -shared \
        "${REPO_ROOT}/src/hadukenn_core/csrc/cuda_blackwell_turing_kernels.cu" \
        -o "${CACHE_OUT}/libhadukenn_cuda_kernels.so"
fi

PYTHONPATH="${REPO_ROOT}/src" python3 "${REPO_ROOT}/scripts/build_tri_machine.py" --machine "${MACHINE}"
