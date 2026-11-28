/*
 * HaDukeNN Tri-Machine Bare-Metal Hardware Abstraction Layer (HAL) Header
 * Targets:
 *   1. WBeast   : x86_64 (Intel i9-9900K AVX2+FMA3) + RTX 5070 (sm_120) + GTX 1660 Ti (sm_75) + 32 GB DDR4
 *   2. LapBeast : x86_64 Hybrid (Core Ultra 9 275HX 8P+16E AVX2+FMA3+AVX-VNNI) + RTX 5060 8GB + 32 GB DDR5
 *   3. Zenbook  : ARM64 (Snapdragon X Oryon NEON+SDOT+I8MM) + Hexagon V73 NPU (45 TOPS) + Adreno FP16 + 13 GiB LPDDR5x
 */
#ifndef HADUKENN_TRI_MACHINE_HAL_H
#define HADUKENN_TRI_MACHINE_HAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define HADUKENN_API __declspec(dllexport)
#else
  #define HADUKENN_API __attribute__((visibility("default")))
#endif

/* Capability bitflags returned by hadukenn_detect_isa_caps() */
#define HADUKENN_ISA_SCALAR      (1u << 0)
#define HADUKENN_ISA_AVX2        (1u << 1)
#define HADUKENN_ISA_FMA3        (1u << 2)
#define HADUKENN_ISA_AVX_VNNI    (1u << 3)
#define HADUKENN_ISA_AVX512F     (1u << 4)
#define HADUKENN_ISA_AVX512_VNNI (1u << 5)
#define HADUKENN_ISA_AVX512_BF16 (1u << 6)
#define HADUKENN_ISA_SHA_NI      (1u << 7)
#define HADUKENN_ISA_ARM_NEON    (1u << 8)
#define HADUKENN_ISA_ARM_DOTPROD (1u << 9)
#define HADUKENN_ISA_ARM_I8MM    (1u << 10)
#define HADUKENN_ISA_ARM_FP16    (1u << 11)

/* 64-byte (512-bit ZMM / cache-line) aligned zero-fragmentation static RAM slab */
typedef struct HadukennArena {
    uint8_t *base;
    size_t   capacity_bytes;
    size_t   offset_bytes;
    size_t   peak_bytes;
    uint32_t alloc_count;
    uint32_t overflow_count;
} HadukennArena;

HADUKENN_API uint32_t hadukenn_detect_isa_caps(void);
HADUKENN_API const char *hadukenn_active_simd_backend(void);

HADUKENN_API int hadukenn_arena_init(HadukennArena *arena, size_t capacity_bytes);
HADUKENN_API void *hadukenn_arena_alloc(HadukennArena *arena, size_t size_bytes);
HADUKENN_API void hadukenn_arena_reset(HadukennArena *arena);
HADUKENN_API void hadukenn_arena_free(HadukennArena *arena);

/* Register-blocked multi-ISA FP32 GEMV / dot / axpy */
HADUKENN_API void hadukenn_gemv_f32(
    const float *x,      /* [N, d] */
    const float *W,      /* [M, d] row-major */
    float *out,          /* [N, M] */
    int N,
    int M,
    int d,
    float scale
);

/* Quantized symmetric INT8 (w8a8 -> i32 -> f32) GEMM/GEMV for NPU fallback & VNNI/SDOT CPU */
HADUKENN_API void hadukenn_qgemm_int8_f32(
    const int8_t *x_q,       /* [N, d] quantized activations */
    const float *x_scale,    /* [N] per-token activation scale */
    const int8_t *w_q,       /* [M, d] quantized weights */
    const float *w_scale,    /* [M] per-channel weight scale */
    float *out,              /* [N, M] dequantized FP32 output */
    int N,
    int M,
    int d
);

/* L2-cache-tiled Cut-Cross-Entropy with muP (mup_scale = base_d / d) width scaling */
HADUKENN_API void hadukenn_tiled_cce_mup_fwd_bwd_f32(
    const float *H,          /* [N, d] */
    const float *E,          /* [V, d] */
    const int32_t *targets,  /* [N] */
    float *out_loss,         /* [N] */
    float *dH,               /* [N, d] or NULL */
    float *dE,               /* [V, d] or NULL */
    int N,
    int d,
    int V,
    int tile_V,
    float logit_scale,       /* 1.0f standard, or base_d / (float)d under muP */
    int compute_grad,
    int num_threads
);

/* Cellular-Sheaf KV-Prefix Coboundary Energy & Exponential Gate (U3 kvreg-sheaf) */
HADUKENN_API void hadukenn_sheaf_kv_prefix_gate_f32(
    const float *query_stalk,     /* [N, d_stalk] */
    const float *nbr_stalks,      /* [N, K_nbr, d_stalk] */
    const float *nbr_values,      /* [N, K_nbr, d_model] */
    float *out_gated_prefix,      /* [N, d_model] */
    float *out_sheaf_energy,      /* [N] mean coboundary energy */
    int N,
    int K_nbr,
    int d_stalk,
    int d_model,
    float beta
);

/* Strict Causal Byte-to-Span Rolling Rabin/FNV Window Scanner (PR #9 prefix-invariant) */
HADUKENN_API int hadukenn_scan_causal_spans_u8(
    const uint8_t *bytes,
    int len,
    int max_span_len,
    uint32_t hash_mod_mask,
    uint32_t trigger_rem,
    int32_t *out_starts,
    int32_t *out_ends,
    uint64_t *out_hashes,
    int max_spans
);

#ifdef __cplusplus
}
#endif

#endif /* HADUKENN_TRI_MACHINE_HAL_H */
