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

echo "[HaDukeNN v1.2.0] Building bare-metal HAL for machine='${MACHINE}' (arch='${ARCH}')"

discover_cc() {
    if [[ -n "${CC:-}" ]]; then
        if [[ -x "${CC}" ]]; then
            printf '%s\n' "${CC}"
            return 0
        fi
        if command -v "${CC}" >/dev/null 2>&1; then
            command -v "${CC}"
            return 0
        fi
        echo "[HaDukeNN v1.2.0] ERROR: CC='${CC}' is not executable or on PATH" >&2
        return 1
    fi
    if [[ -x "${HOME}/.local/bin/clang" ]]; then
        printf '%s\n' "${HOME}/.local/bin/clang"
        return 0
    fi
    for cand in clang cc gcc; do
        if command -v "${cand}" >/dev/null 2>&1; then
            command -v "${cand}"
            return 0
        fi
    done
    echo "[HaDukeNN v1.2.0] ERROR: No C11 compiler found (probed ~/.local/bin/clang, clang, cc, gcc)" >&2
    return 1
}

CC_BIN="$(discover_cc)"

probe_isa_flags() {
    "${CC_BIN}" "$@" -x c -c /dev/null -o /dev/null >/dev/null 2>&1
}

CFLAGS=(-std=c11 -O3 -shared -fPIC -Wall -Wextra -Werror=implicit-function-declaration "-I${REPO_ROOT}/src/hadukenn_core/csrc")

if printf '#include <omp.h>\nint f(void){return omp_get_max_threads();}\n' | "${CC_BIN}" -fopenmp -shared -fPIC -x c - -o /dev/null >/dev/null 2>&1; then
    CFLAGS+=(-fopenmp)
elif printf '#include <pthread.h>\nint f(void){return 1;}\n' | "${CC_BIN}" -pthread -shared -fPIC -x c - -o /dev/null >/dev/null 2>&1; then
    CFLAGS+=(-pthread)
fi

case "${MACHINE}" in
    zenbook)
        for isa_flag in \
            "-mcpu=oryon-1+fp16" \
            "-mcpu=oryon-1" \
            "-march=armv8.6-a+dotprod+i8mm+fp16" \
            "-march=armv8-a+dotprod" \
            "-march=native"
        do
            if probe_isa_flags "${isa_flag}"; then
                CFLAGS+=("${isa_flag}")
                break
            fi
        done
        export OMP_NUM_THREADS=8
        export HADUKENN_MACHINE="zenbook"
        ;;
    lapbeast)
        if probe_isa_flags -mavx2 -mfma -mbmi2; then
            CFLAGS+=(-mavx2 -mfma -mbmi2)
        elif probe_isa_flags -march=native; then
            CFLAGS+=(-march=native)
        fi
        # Try enabling AVX-VNNI on hybrid Intel P/E cores if supported by compiler
        if probe_isa_flags -mavxvnni; then
            CFLAGS+=(-mavxvnni)
        fi
        export OMP_NUM_THREADS=8
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close
        export HADUKENN_MACHINE="lapbeast"
        ;;
    wbeast|*)
        if probe_isa_flags -march=native; then
            CFLAGS+=(-march=native)
        elif probe_isa_flags -mavx2 -mfma -mbmi2; then
            CFLAGS+=(-mavx2 -mfma -mbmi2)
        fi
        export OMP_NUM_THREADS=12
        export OMP_PROC_BIND=close
        export HADUKENN_MACHINE="wbeast"
        ;;
esac

TMP_SO="${CACHE_OUT}/.tmp_libhadukenn_tri_hal.$$.so"
trap 'rm -f "${TMP_SO}"' EXIT
"${CC_BIN}" "${CFLAGS[@]}" \
    "${REPO_ROOT}/src/hadukenn_core/csrc/tri_machine_hal.c" \
    -lm -o "${TMP_SO}"
mv -f "${TMP_SO}" "${CACHE_OUT}/libhadukenn_tri_hal.so"

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
    echo "[HaDukeNN v1.2.0] Compiling CUDA kernels via nvcc..."
    NVCC_GENCODE=()
    # Prefer the live GPU's compute capability (e.g. 8.9 -> 89 on an RTX 4080 Laptop).
    if command -v nvidia-smi >/dev/null 2>&1; then
        LIVE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
        LIVE_SM="${LIVE_CAP//./}"
        if [[ -n "${LIVE_SM}" ]] && nvcc_supports_arch "${LIVE_SM}"; then
            NVCC_GENCODE+=(-gencode "arch=compute_${LIVE_SM},code=sm_${LIVE_SM}")
            echo "[HaDukeNN v1.2.0] nvcc gencode sm_${LIVE_SM} from nvidia-smi compute_cap=${LIVE_CAP}"
        fi
    fi
    if nvcc_supports_arch 120; then
        if [[ " ${NVCC_GENCODE[*]} " != *"sm_120"* ]]; then
            NVCC_GENCODE+=(-gencode arch=compute_120,code=sm_120)
            echo "[HaDukeNN v1.2.0] nvcc gencode sm_120 (toolkit supports Blackwell)"
        fi
    else
        echo "[HaDukeNN v1.2.0] nvcc does not support sm_120; skipping that gencode"
    fi
    if [[ "${MACHINE}" == "wbeast" ]] && nvcc_supports_arch 75; then
        NVCC_GENCODE+=(-gencode arch=compute_75,code=sm_75)
    fi
    if [[ "${#NVCC_GENCODE[@]}" -eq 0 ]]; then
        echo "[HaDukeNN v1.2.0] ERROR: no supported nvcc gencode for this GPU/toolkit" >&2
        exit 1
    fi
    nvcc -O3 --use_fast_math -std=c++17 "${NVCC_GENCODE[@]}" \
        -Xcompiler -fPIC -shared \
        "${REPO_ROOT}/src/hadukenn_core/csrc/cuda_blackwell_turing_kernels.cu" \
        -o "${CACHE_OUT}/libhadukenn_cuda_kernels.so"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_tri_machine.py" --machine "${MACHINE}"
