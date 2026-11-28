<#
.SYNOPSIS
  HaDukeNN v0.9.0 Native Windows 11 Bare-Metal Builder & Hardware Launcher
  Supports WBeast (x64 Dual-GPU sm_120 + sm_75), LapBeast (x64 Hybrid 275HX + RTX 5060),
  and Zenbook (Windows ARM64 Snapdragon X Oryon + Hexagon V73 QNN HTP + 13 GiB LPDDR5x).
#>
param(
    [ValidateSet("auto", "wbeast", "lapbeast", "zenbook")]
    [string]$Machine = "auto"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$CacheOut = Join-Path $HOME ".cache\hadukenn_core"
New-Item -ItemType Directory -Force -Path $CacheOut | Out-Null

$Arch = $env:PROCESSOR_ARCHITECTURE
if ($Machine -eq "auto") {
    if ($Arch -eq "ARM64") {
        $Machine = "zenbook"
    } elseif ([Environment]::ProcessorCount -ge 20) {
        $Machine = "lapbeast"
    } else {
        $Machine = "wbeast"
    }
}

$env:HADUKENN_MACHINE = $Machine
Write-Host "[HaDukeNN v0.9.0] Configuring bare-metal Windows build for '$Machine' ($Arch)"

$CSrc = Join-Path $RepoRoot "src\hadukenn_core\csrc\tri_machine_hal.c"
$IncDir = Join-Path $RepoRoot "src\hadukenn_core\csrc"
$OutDll = Join-Path $CacheOut "hadukenn_tri_hal.dll"

if (Get-Command clang -ErrorAction SilentlyContinue) {
    if ($Machine -eq "zenbook") {
        clang -O3 -march=armv8.6-a+dotprod+i8mm+fp16 -shared -I"$IncDir" "$CSrc" -o "$OutDll"
    } elseif ($Machine -eq "lapbeast") {
        clang -O3 -mavx2 -mfma -mavxvnni -shared -I"$IncDir" "$CSrc" -o "$OutDll"
    } else {
        clang -O3 -mavx2 -mfma -mbmi2 -shared -I"$IncDir" "$CSrc" -o "$OutDll"
    }
} elseif (Get-Command cl -ErrorAction SilentlyContinue) {
    if ($Machine -eq "zenbook") {
        cl /O2 /fp:fast /LD /I"$IncDir" "$CSrc" /Fe:"$OutDll"
    } else {
        cl /O2 /arch:AVX2 /fp:fast /openmp /LD /I"$IncDir" "$CSrc" /Fe:"$OutDll"
    }
}

if (($Machine -ne "zenbook") -and (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    $CuSrc = Join-Path $RepoRoot "src\hadukenn_core\csrc\cuda_blackwell_turing_kernels.cu"
    $CuDll = Join-Path $CacheOut "hadukenn_cuda_kernels.dll"
    if ($Machine -eq "wbeast") {
        nvcc -O3 --use_fast_math -std=c++17 `
            -gencode arch=compute_120,code=sm_120 `
            -gencode arch=compute_75,code=sm_75 `
            -shared "$CuSrc" -o "$CuDll"
    } else {
        nvcc -O3 --use_fast_math -std=c++17 `
            -gencode arch=compute_120,code=sm_120 `
            -shared "$CuSrc" -o "$CuDll"
    }
}

if ($Machine -eq "zenbook") {
    # Enforce 13.0 GiB unified LPDDR5x ceiling and sustained Hexagon HTP performance mode
    $env:QNN_HTP_PERF_MODE = "sustained_high_performance"
    $env:HADUKENN_ZENBOOK_SLAB_GIB = "13.0"
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
python (Join-Path $RepoRoot "scripts\build_tri_machine.py") --machine $Machine
