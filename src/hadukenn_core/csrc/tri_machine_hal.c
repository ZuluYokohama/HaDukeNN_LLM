/*
 * HaDukeNN Tri-Machine Bare-Metal Hardware Abstraction Layer (C11 + OpenMP + Multi-ISA SIMD)
 *
 * Custom low-level implementations for:
 *   - WBeast   (Intel i9-9900K: 256-bit AVX2 + FMA3 + vpmaddubsw INT8, 64B cache-line blocking)
 *   - LapBeast (Core Ultra 9 275HX 8P+16E: AVX2 + FMA3 + AVX-VNNI vpdpbusd, asymmetric OpenMP)
 *   - Zenbook  (Snapdragon X Oryon ARMv8.7-A: 128-bit NEON + ASIMDDP vdotq_s32 + 13 GiB slab)
 *   - Server   (Xeon / Zen4+: 512-bit AVX-512F + AVX512-VNNI vpdpbusd, clean VZEROUPPER hygiene)
 */
#define _POSIX_C_SOURCE 200809L
#include "tri_machine_hal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  #include <arm_neon.h>
#endif

/* =========================================================================
 * 1. Hardware Capability Detection (CPUID + XGETBV / ARM64 Feature Macros)
 * ========================================================================= */

#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t read_xcr0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#endif
}

static inline void cpuid_leaf(uint32_t leaf, uint32_t subleaf, uint32_t out[4]) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    for (int i = 0; i < 4; ++i) out[i] = (uint32_t)regs[i];
#else
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
#endif
}
#endif

uint32_t hadukenn_detect_isa_caps(void) {
    uint32_t caps = HADUKENN_ISA_SCALAR;
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t r0[4] = {0}, r1[4] = {0}, r7[4] = {0}, r7_1[4] = {0};
    cpuid_leaf(0, 0, r0);
    if (r0[0] >= 1) {
        cpuid_leaf(1, 0, r1);
        const int os_xsave = (r1[2] & (1u << 27)) != 0;
        const uint64_t xcr0 = os_xsave ? read_xcr0() : 0ULL;
        const int ymm_ok = (xcr0 & 0x6ULL) == 0x6ULL;
        const int zmm_ok = (xcr0 & 0xE6ULL) == 0xE6ULL;

        if (ymm_ok && (r1[2] & (1u << 12))) caps |= HADUKENN_ISA_FMA3;
        if (r0[0] >= 7) {
            cpuid_leaf(7, 0, r7);
            if (ymm_ok && (r7[1] & (1u << 5))) caps |= HADUKENN_ISA_AVX2;
            if (r7[1] & (1u << 29)) caps |= HADUKENN_ISA_SHA_NI;
            if (zmm_ok && (r7[1] & (1u << 16))) caps |= HADUKENN_ISA_AVX512F;
            if (zmm_ok && (r7[2] & (1u << 11))) caps |= HADUKENN_ISA_AVX512_VNNI;

            cpuid_leaf(7, 1, r7_1);
            if (ymm_ok && (r7_1[0] & (1u << 4))) caps |= HADUKENN_ISA_AVX_VNNI;
            if (zmm_ok && (r7_1[0] & (1u << 5))) caps |= HADUKENN_ISA_AVX512_BF16;
        }
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    caps |= HADUKENN_ISA_ARM_NEON;
  #if defined(__ARM_FEATURE_DOTPROD)
    caps |= HADUKENN_ISA_ARM_DOTPROD;
  #endif
  #if defined(__ARM_FEATURE_MATMUL_INT8)
    caps |= HADUKENN_ISA_ARM_I8MM;
  #endif
  #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    caps |= HADUKENN_ISA_ARM_FP16;
  #endif
#endif
    return caps;
}

const char *hadukenn_active_simd_backend(void) {
#if defined(__AVX512F__)
    return "x86_64_avx512f_fma";
#elif defined(__AVX2__) && defined(__FMA__)
    return "x86_64_avx2_fma3";
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return "arm64_neon_sdot";
#elif defined(__aarch64__)
    return "arm64_neon_fp32";
#else
    return "c11_scalar_portable";
#endif
}

/* =========================================================================
 * 2. 64-Byte Cache-Line Aligned Zero-Fragmentation Static Slab Arena
 * ========================================================================= */

int hadukenn_arena_init(HadukennArena *arena, size_t capacity_bytes) {
    if (!arena || capacity_bytes == 0) return -1;
    memset(arena, 0, sizeof(*arena));
    const size_t aligned_cap = (capacity_bytes + 63u) & ~(size_t)63u;
#if defined(_WIN32)
    void *ptr = _aligned_malloc(aligned_cap, 64u);
    if (!ptr) return -2;
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64u, aligned_cap) != 0 || !ptr) return -2;
#endif
    arena->base = (uint8_t *)ptr;
    arena->capacity_bytes = aligned_cap;
    return 0;
}

void *hadukenn_arena_alloc(HadukennArena *arena, size_t size_bytes) {
    if (!arena || !arena->base) return NULL;
    const size_t aligned = (size_bytes + 63u) & ~(size_t)63u;
    if (arena->offset_bytes + aligned > arena->capacity_bytes) {
        arena->overflow_count += 1u;
        return NULL;
    }
    void *out = arena->base + arena->offset_bytes;
    arena->offset_bytes += aligned;
    if (arena->offset_bytes > arena->peak_bytes) {
        arena->peak_bytes = arena->offset_bytes;
    }
    arena->alloc_count += 1u;
    return out;
}

void hadukenn_arena_reset(HadukennArena *arena) {
    if (arena) {
        arena->offset_bytes = 0u;
    }
}

void hadukenn_arena_free(HadukennArena *arena) {
    if (arena && arena->base) {
#if defined(_WIN32)
        _aligned_free(arena->base);
#else
        free(arena->base);
#endif
        memset(arena, 0, sizeof(*arena));
    }
}

/* =========================================================================
 * 3. Multi-ISA Register-Blocked Primitives (AVX-512 / AVX2+FMA / NEON)
 * ========================================================================= */

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

#if defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
    return _mm_cvtsi128_si32(lo);
}
#endif

static inline float dot_f32_fast(const float *a, const float *b, int d) {
    int i = 0;
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= d; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }
    float sum = _mm512_reduce_add_ps(acc0);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= d; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= d; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float sum = hsum256_ps(acc0);
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= d; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    for (; i + 4 <= d; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(acc0);
#else
    float sum = 0.0f;
#endif
    for (; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline void axpy_f32_fast(float alpha, const float *x, float *y, int d) {
    int i = 0;
#if defined(__AVX512F__)
    __m512 va = _mm512_set1_ps(alpha);
    for (; i + 32 <= d; i += 32) {
        _mm512_storeu_ps(y + i,      _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i),      _mm512_loadu_ps(y + i)));
        _mm512_storeu_ps(y + i + 16, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(y + i + 16)));
    }
    for (; i + 16 <= d; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 16 <= d; i += 16) {
        _mm256_storeu_ps(y + i,     _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),     _mm256_loadu_ps(y + i)));
        _mm256_storeu_ps(y + i + 8, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
    }
    for (; i + 8 <= d; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 8 <= d; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     va, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), va, vld1q_f32(x + i + 4)));
    }
    for (; i + 4 <= d; i += 4) {
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    }
#endif
    for (; i < d; ++i) {
        y[i] += alpha * x[i];
    }
}

/* Quantized INT8 dot-product: sum_{i=0}^{d-1} (int32_t)a[i] * (int32_t)b[i] */
static inline int32_t dot_i8_fast(const int8_t *a, const int8_t *b, int d) {
    int i = 0;
    int32_t total = 0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* Shift signed a[-128..127] -> unsigned [0..255] via ^0x80 (+128), subtract 128 * sum(b) */
    const __m512i bias_xor = _mm512_set1_epi8((char)0x80);
    const __m512i ones_u8  = _mm512_set1_epi8(1);
    __m512i acc_prod = _mm512_setzero_si512();
    __m512i acc_bsum = _mm512_setzero_si512();
    for (; i + 64 <= d; i += 64) {
        __m512i va_s8 = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb_s8 = _mm512_loadu_si512((const void *)(b + i));
        __m512i va_u8 = _mm512_xor_si512(va_s8, bias_xor);
        acc_prod = _mm512_dpbusd_epi32(acc_prod, va_u8, vb_s8);
        acc_bsum = _mm512_dpbusd_epi32(acc_bsum, ones_u8, vb_s8);
    }
    int32_t prod_sum = _mm512_reduce_add_epi32(acc_prod);
    int32_t b_sum    = _mm512_reduce_add_epi32(acc_bsum);
    total += prod_sum - (128 * b_sum);
#elif defined(__AVXVNNI__)
    /* 256-bit AVX-VNNI vpdpbusd accumulates in epi32 (no int16 saturation).
     * Shift signed a[-128..127] -> unsigned [0..255] via ^0x80, then subtract 128 * sum(b). */
    const __m256i bias_xor = _mm256_set1_epi8((char)0x80);
    const __m256i ones_u8  = _mm256_set1_epi8(1);
    __m256i acc_prod = _mm256_setzero_si256();
    __m256i acc_bsum = _mm256_setzero_si256();
    for (; i + 32 <= d; i += 32) {
        __m256i va_s8 = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb_s8 = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i va_u8 = _mm256_xor_si256(va_s8, bias_xor);
        acc_prod = _mm256_dpbusd_avx_epi32(acc_prod, va_u8, vb_s8);
        acc_bsum = _mm256_dpbusd_avx_epi32(acc_bsum, ones_u8, vb_s8);
    }
    total += hsum256_epi32(acc_prod) - (128 * hsum256_epi32(acc_bsum));
#elif defined(__AVX2__)
    /* Portable AVX2 path: sign-extend INT8 to INT16 and madd into INT32.
     * The previous vpmaddubsw path saturated int16 after the unsigned-bias
     * transform (255*127 + 255*127 overflows INT16_MAX), silently corrupting
     * qgemm and the Zenbook INT8 NPU loop on AVX2 / AVX-VNNI hosts. */
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= d; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
    }
    total += hsum256_epi32(acc);
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    /* Zenbook Snapdragon X Oryon path: native signed SDOT (vdotq_s32, 16 INT8 MACs/inst) */
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (; i + 32 <= d; i += 32) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + i),      vld1q_s8(b + i));
        acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
    }
    acc0 = vaddq_s32(acc0, acc1);
    for (; i + 16 <= d; i += 16) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    }
    total += vaddvq_s32(acc0);
#elif defined(__aarch64__)
    int32x4_t acc = vdupq_n_s32(0);
    for (; i + 8 <= d; i += 8) {
        int16x8_t prod16 = vmull_s8(vld1_s8(a + i), vld1_s8(b + i));
        acc = vpadalq_s16(acc, prod16);
    }
    total += vaddvq_s32(acc);
#endif
    for (; i < d; ++i) {
        total += (int32_t)a[i] * (int32_t)b[i];
    }
    return total;
}

/* =========================================================================
 * 4. Exported Tri-Machine Compute Kernels
 * ========================================================================= */

void hadukenn_gemv_f32(
    const float *x,
    const float *W,
    float *out,
    int N,
    int M,
    int d,
    float scale
) {
    #pragma omp parallel for collapse(2) if((size_t)N * (size_t)M >= 512)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            out[(size_t)n * M + m] = scale * dot_f32_fast(
                x + (size_t)n * d,
                W + (size_t)m * d,
                d
            );
        }
    }
}

void hadukenn_qgemm_int8_f32(
    const int8_t *x_q,
    const float *x_scale,
    const int8_t *w_q,
    const float *w_scale,
    float *out,
    int N,
    int M,
    int d
) {
    #pragma omp parallel for schedule(static) if((size_t)N * (size_t)M >= 256)
    for (int n = 0; n < N; ++n) {
        const int8_t *xn = x_q + (size_t)n * d;
        const float sx = x_scale[n];
        float *out_n = out + (size_t)n * M;
        for (int m = 0; m < M; ++m) {
            int32_t acc = dot_i8_fast(xn, w_q + (size_t)m * d, d);
            out_n[m] = (float)acc * (sx * w_scale[m]);
        }
    }
}


enum { HADUKENN_CCE_BN = 4 };

/* scale * H[n0:n0+bn] @ E[v0:v0+tv].T  into C[bn, tv], reusing each E row across BN tokens. */
static void cce_logit_tile_bn(
    const float *H,
    const float *E,
    float *C,
    int n0,
    int bn,
    int v0,
    int tv,
    int d,
    float scale
) {
#if defined(__AVX2__) && defined(__FMA__)
    int ni = 0;
    for (; ni + 4 <= bn; ni += 4) {
        const float *h0 = H + (size_t)(n0 + ni) * d;
        const float *h1 = H + (size_t)(n0 + ni + 1) * d;
        const float *h2 = H + (size_t)(n0 + ni + 2) * d;
        const float *h3 = H + (size_t)(n0 + ni + 3) * d;
        for (int vi = 0; vi < tv; ++vi) {
            const float *e = E + (size_t)(v0 + vi) * d;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= d; i += 8) {
                __m256 ev = _mm256_loadu_ps(e + i);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(h0 + i), ev, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(h1 + i), ev, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(h2 + i), ev, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(h3 + i), ev, acc3);
            }
            float s0 = hsum256_ps(acc0);
            float s1 = hsum256_ps(acc1);
            float s2 = hsum256_ps(acc2);
            float s3 = hsum256_ps(acc3);
            for (; i < d; ++i) {
                const float ev = e[i];
                s0 += h0[i] * ev;
                s1 += h1[i] * ev;
                s2 += h2[i] * ev;
                s3 += h3[i] * ev;
            }
            C[(size_t)(ni + 0) * (size_t)tv + (size_t)vi] = scale * s0;
            C[(size_t)(ni + 1) * (size_t)tv + (size_t)vi] = scale * s1;
            C[(size_t)(ni + 2) * (size_t)tv + (size_t)vi] = scale * s2;
            C[(size_t)(ni + 3) * (size_t)tv + (size_t)vi] = scale * s3;
        }
    }
    for (; ni < bn; ++ni) {
        const float *h = H + (size_t)(n0 + ni) * d;
        for (int vi = 0; vi < tv; ++vi) {
            C[(size_t)ni * (size_t)tv + (size_t)vi] = scale * dot_f32_fast(
                h, E + (size_t)(v0 + vi) * d, d
            );
        }
    }
#else
    for (int ni = 0; ni < bn; ++ni) {
        const float *h = H + (size_t)(n0 + ni) * d;
        for (int vi = 0; vi < tv; ++vi) {
            C[(size_t)ni * (size_t)tv + (size_t)vi] = scale * dot_f32_fast(
                h, E + (size_t)(v0 + vi) * d, d
            );
        }
    }
#endif
}

void hadukenn_tiled_cce_mup_fwd_bwd_f32(
    const float *H,
    const float *E,
    const int32_t *targets,
    float *out_loss,
    float *dH,
    float *dE,
    int N,
    int d,
    int V,
    int tile_V,
    float logit_scale,
    int compute_grad,
    int num_threads
) {
    if (tile_V < 64) tile_V = 64;
    if (tile_V > 4096) tile_V = 4096;
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    if (compute_grad && dH && dE) {
        for (int n = 0; n < N; ++n) {
            const float *h = H + (size_t)n * d;
            const int32_t y = targets[n];
            float m_run = -1e30f;
            float s_run = 0.0f;
            const float z_y = logit_scale * dot_f32_fast(h, E + (size_t)y * d, d);
            float tile_buf[4096];
            for (int v0 = 0; v0 < V; v0 += tile_V) {
                int v1 = v0 + tile_V;
                if (v1 > V) v1 = V;
                float m_tile = -1e30f;
                for (int v = v0; v < v1; ++v) {
                    float z = logit_scale * dot_f32_fast(h, E + (size_t)v * d, d);
                    tile_buf[v - v0] = z;
                    if (z > m_tile) m_tile = z;
                }
                float m_new = (m_tile > m_run) ? m_tile : m_run;
                float s_tile = 0.0f;
                for (int v = v0; v < v1; ++v) {
                    s_tile += expf(tile_buf[v - v0] - m_new);
                }
                s_run = s_run * expf(m_run - m_new) + s_tile;
                m_run = m_new;
            }
            const float lse = m_run + logf(s_run);
            out_loss[n] = lse - z_y;
            float *dh = dH + (size_t)n * d;
            const float inv_N = logit_scale / (float)N;
            for (int v = 0; v < V; ++v) {
                float z = logit_scale * dot_f32_fast(h, E + (size_t)v * d, d);
                float coeff = expf(z - lse) * inv_N;
                if (v == y) coeff -= inv_N;
                axpy_f32_fast(coeff, E + (size_t)v * d, dh, d);
                axpy_f32_fast(coeff, h, dE + (size_t)v * d, d);
            }
        }
#if defined(__AVX512F__) || defined(__AVX2__)
        _mm256_zeroupper();
#endif
        return;
    }

    #pragma omp parallel if(N >= 64)
    {
        float *tile_buf = (float *)malloc((size_t)HADUKENN_CCE_BN * (size_t)tile_V * sizeof(float));
        #pragma omp for schedule(static)
        for (int n0 = 0; n0 < N; n0 += HADUKENN_CCE_BN) {
            int bn = N - n0;
            if (bn > HADUKENN_CCE_BN) bn = HADUKENN_CCE_BN;
            if (tile_buf == NULL) {
                for (int ni = 0; ni < bn; ++ni) {
                    const float *h = H + (size_t)(n0 + ni) * d;
                    const int32_t y = targets[n0 + ni];
                    float m_run = -1e30f;
                    float s_run = 0.0f;
                    const float z_y = logit_scale * dot_f32_fast(h, E + (size_t)y * d, d);
                    for (int v = 0; v < V; ++v) {
                        float z = logit_scale * dot_f32_fast(h, E + (size_t)v * d, d);
                        float m_new = (z > m_run) ? z : m_run;
                        s_run = s_run * expf(m_run - m_new) + expf(z - m_new);
                        m_run = m_new;
                    }
                    out_loss[n0 + ni] = (m_run + logf(s_run)) - z_y;
                }
                continue;
            }
            float m_run[HADUKENN_CCE_BN];
            float s_run[HADUKENN_CCE_BN];
            float z_y[HADUKENN_CCE_BN];
            for (int ni = 0; ni < bn; ++ni) {
                m_run[ni] = -1e30f;
                s_run[ni] = 0.0f;
                const float *h = H + (size_t)(n0 + ni) * d;
                z_y[ni] = logit_scale * dot_f32_fast(h, E + (size_t)targets[n0 + ni] * d, d);
            }
            for (int v0 = 0; v0 < V; v0 += tile_V) {
                int tv = V - v0;
                if (tv > tile_V) tv = tile_V;
                cce_logit_tile_bn(H, E, tile_buf, n0, bn, v0, tv, d, logit_scale);
                for (int ni = 0; ni < bn; ++ni) {
                    const float *row = tile_buf + (size_t)ni * (size_t)tv;
                    float m_tile = -1e30f;
                    for (int vi = 0; vi < tv; ++vi) {
                        if (row[vi] > m_tile) m_tile = row[vi];
                    }
                    float m_new = (m_tile > m_run[ni]) ? m_tile : m_run[ni];
                    float s_tile = 0.0f;
                    for (int vi = 0; vi < tv; ++vi) {
                        s_tile += expf(row[vi] - m_new);
                    }
                    s_run[ni] = s_run[ni] * expf(m_run[ni] - m_new) + s_tile;
                    m_run[ni] = m_new;
                }
            }
            for (int ni = 0; ni < bn; ++ni) {
                out_loss[n0 + ni] = (m_run[ni] + logf(s_run[ni])) - z_y[ni];
            }
        }
        free(tile_buf);
    }
#if defined(__AVX512F__) || defined(__AVX2__)
    _mm256_zeroupper();
#endif
}

void hadukenn_sheaf_kv_prefix_gate_f32(
    const float *query_stalk,
    const float *nbr_stalks,
    const float *nbr_values,
    float *out_gated_prefix,
    float *out_sheaf_energy,
    int N,
    int K_nbr,
    int d_stalk,
    int d_model,
    float beta
) {
    #pragma omp parallel for schedule(static) if(N >= 64)
    for (int n = 0; n < N; ++n) {
        const float *q = query_stalk + (size_t)n * d_stalk;
        float *out_v = out_gated_prefix + (size_t)n * d_model;
        memset(out_v, 0, (size_t)d_model * sizeof(float));

        float weight_sum = 0.0f;
        float energy_sum = 0.0f;
        float gates[256];
        const int k_cap = (K_nbr > 256) ? 256 : K_nbr;

        for (int k = 0; k < k_cap; ++k) {
            const float *nk = nbr_stalks + ((size_t)n * K_nbr + k) * d_stalk;
            /* Coboundary squared norm ||q - nk||_2^2 = ||q||^2 + ||nk||^2 - 2 <q, nk> */
            float qq = dot_f32_fast(q, q, d_stalk);
            float nn = dot_f32_fast(nk, nk, d_stalk);
            float qn = dot_f32_fast(q, nk, d_stalk);
            float e_k = qq + nn - 2.0f * qn;
            if (e_k < 0.0f) e_k = 0.0f;
            energy_sum += e_k;
            float g = expf(-beta * e_k);
            gates[k] = g;
            weight_sum += g;
        }
        const float inv_w = (weight_sum > 1e-12f) ? (1.0f / weight_sum) : 0.0f;
        for (int k = 0; k < k_cap; ++k) {
            const float *vk = nbr_values + ((size_t)n * K_nbr + k) * d_model;
            axpy_f32_fast(gates[k] * inv_w, vk, out_v, d_model);
        }
        out_sheaf_energy[n] = (k_cap > 0) ? (energy_sum / (float)k_cap) : 0.0f;
    }
}

int hadukenn_scan_causal_spans_u8(
    const uint8_t *bytes,
    int len,
    int max_span_len,
    uint32_t hash_mod_mask,
    uint32_t trigger_rem,
    int32_t *out_starts,
    int32_t *out_ends,
    uint64_t *out_hashes,
    int max_spans
) {
    /*
     * Strict causal prefix scanner: at byte position end_pos, only bytes[0..end_pos]
     * are ever inspected, guaranteeing suffix-invariance of span IDs and hashes.
     */
    int count = 0;
    if (max_span_len < 2) max_span_len = 2;
    for (int end_pos = 1; end_pos < len && count < max_spans; ++end_pos) {
        const int min_start = (end_pos - max_span_len + 1 > 0) ? (end_pos - max_span_len + 1) : 0;
        uint64_t h = 14695981039346656037ULL;
        /* Walk backward strictly inside the visible prefix [min_start .. end_pos] */
        for (int s = end_pos; s >= min_start; --s) {
            h ^= (uint64_t)bytes[s];
            h *= 1099511628211ULL;
            const int span_len = end_pos - s + 1;
            if (span_len >= 2 && ((uint32_t)h & hash_mod_mask) == trigger_rem) {
                out_starts[count] = s;
                out_ends[count]   = end_pos + 1;
                out_hashes[count] = h;
                count += 1;
                if (count >= max_spans) break;
            }
        }
    }
    return count;
}
