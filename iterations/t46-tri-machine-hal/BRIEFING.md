# T-46: Tri-Machine Bare-Metal Heterogeneous Build (`wbeast`, `lapbeast`, `zenbook`)

## Objective

Customize the verified HaDukeNN neuro-symbolic stack (`struct32k` causal byte-spans, `muP` L2-tiled [Cut Cross-Entropy](https://arxiv.org/abs/2411.00909), [cellular-sheaf](https://arxiv.org/abs/1808.01513) KV-prefix gating, hyperedge-scoped [kNN-LM](https://arxiv.org/abs/1911.00172), and `IncrementalHypergraphKernel` + [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/) receipts) down to register-level CPU SIMD, GPU CUDA/OpenCL, Hexagon NPU, and 64-byte cache-aligned static RAM slabs across all three target machines.

## Hardware Customization Matrix

| Machine ID | CPU & Native SIMD Path | Primary Accelerator | Co-Processor / Side-Lane | Static RAM Slab & Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **`wbeast`** | `Intel Core i9-9900K` (`8C/16T`, `256-bit AVX2 + FMA3 + vpmaddubsw`, threads `0..11` compute / `12..15` async audit) | `RTX 5070 12GB` (`sm_120` Blackwell, `48 MiB` L2, `tile_V=2048` L2-resident `muP` CCE) | `GTX 1660 Ti 6GB` (`sm_75` Turing `CUDA_VISIBLE_DEVICES=1` async `E2` kNN-LM & `e1v3`) | `26.0 GiB` aligned DDR4 slab (`6.0 GiB` OS reserve); explicit `_mm256_zeroupper()` hygiene |
| **`lapbeast`** | `Intel Core Ultra 9 275HX` (`24c = 8P+16E`, `AVX2 + FMA3 + AVX-VNNI vpdpbusd`; `0..7` P-core compute, `8..23` Skymont E-core span/Merkle) | `RTX 5060 Laptop 8GB` (`sm_120` Blackwell, `32 MiB` L2, `tile_V=1024`, `7,168 MiB` VRAM cap) | `Intel AI Boost NPU` (background INT8 stalk filter) | `24.0 GiB` DDR5-5600 slab; `LapBeastPowerGuard` (`min_start=90%`, `min_run=20%`, `83 C` thermal cap, `rc=143` graceful pause) |
| **`zenbook`** | `Qualcomm Snapdragon X Plus X1P-42-100` (`8c` Oryon `ARMv8.7-A`, `128-bit NEON + SDOT vdotq_s32 + I8MM + FP16`) | `Qualcomm Hexagon V73 NPU` (`45 TOPS` persistent looped `INT8` @ `14.5 W`, `8 MiB` VTCM, `QnnHtp.dll` static IO-binding) | `Qualcomm Adreno X1-45 iGPU` (`OpenCL cl_khr_fp16` zero-copy fused RMSNorm + RoPE) | Hard-locked **`13.0 GiB` LPDDR5x-8448** zero-copy unified slab (`0` dynamic loop allocations) |

## Quickstart Per Machine

### Linux / WSL2 (`wbeast`, `lapbeast` lane2, or `zenbook` Linux ARM64)
```bash
bash scripts/bootstrap_tri_machine.sh auto
PYTHONPATH=src python3 -m sheaf_verify.cli campaign --mode t46-tri-machine-hal --path iterations/t46-tri-machine-hal/T46_TRI_MACHINE_RESULTS.json
```

### Native Windows 11 (`wbeast`, `lapbeast`, or `zenbook` Windows ARM64)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_tri_machine.ps1 -Machine auto
```
