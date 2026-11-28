/*
 * HaDukeNN Tri-Machine Bare-Metal Hardware Abstraction Layer (C11 + OpenMP + Multi-ISA SIMD)
 *
 * Custom low-level implementations for:
 *   - WBeast   (Intel i9-9900K: 256-bit AVX2 + FMA3 + vpmaddubsw INT8, 64B cache-line blocking)
 *   - LapBeast (Core Ultra 9 275HX 8P+16E: AVX2 + FMA3 + AVX-VNNI vpdpbusd, asymmetric OpenMP)
 *   - Zenbook  (Snapdragon X Oryon ARMv8.7-A: 128-bit NEON + ASIMDDP vdotq_s32 + 13 GiB slab)
 *   - Server   (Xeon / Zen4+: 512-bit AVX-512F + AVX512-VNNI vpdpbusd, clean VZEROUPPER hygiene)
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#include "tri_machine_hal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#elif !defined(_WIN32)
#include <pthread.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif
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
#elif defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_DOTPROD)
    return "arm64_neon_i8mm_sdot";
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
    if (!arena || capacity_bytes == 0 || capacity_bytes > SIZE_MAX - 63u) return -1;
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
    if (size_bytes > SIZE_MAX - 63u) {
        arena->overflow_count += 1u;
        return NULL;
    }
    const size_t aligned = (size_bytes + 63u) & ~(size_t)63u;
    if (aligned < size_bytes || arena->offset_bytes > arena->capacity_bytes ||
        aligned > arena->capacity_bytes - arena->offset_bytes) {
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

#if defined(__AVX2__) && !defined(__AVX512F__)
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
 * 4. Multi-Core Thread Dispatch & Exported Tri-Machine Compute Kernels
 * ========================================================================= */

#if !defined(_OPENMP) && !defined(_WIN32)
#define HADUKENN_MAX_PTHREADS 16

typedef void (*hadukenn_par_fn)(int start, int end, void *ctx);

typedef struct {
    int start;
    int end;
    hadukenn_par_fn fn;
    void *ctx;
} HadukennParTask;

static void *hadukenn_par_trampoline(void *arg) {
    HadukennParTask *task = (HadukennParTask *)arg;
    task->fn(task->start, task->end, task->ctx);
    return NULL;
}

static int hadukenn_resolve_num_threads(int requested, int total_items, int min_items_per_thread) {
    if (total_items <= 0) return 1;
    int hw = 1;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu > 1) {
        hw = (ncpu > HADUKENN_MAX_PTHREADS) ? HADUKENN_MAX_PTHREADS : (int)ncpu;
    }
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        int aff_cnt = CPU_COUNT(&cpuset);
        if (aff_cnt >= 1 && aff_cnt < hw) {
            hw = aff_cnt;
        }
    }
#endif
    int nt = (requested > 0) ? requested : hw;
    if (nt < 1) nt = 1;
    if (nt > HADUKENN_MAX_PTHREADS) nt = HADUKENN_MAX_PTHREADS;
    if (min_items_per_thread > 0) {
        int max_useful = total_items / min_items_per_thread;
        if (max_useful < 1) max_useful = 1;
        if (nt > max_useful) nt = max_useful;
    }
    if (nt > total_items) nt = total_items;
    return nt;
}

static void hadukenn_parallel_for(
    int total_items,
    int requested_threads,
    int min_items_per_thread,
    hadukenn_par_fn fn,
    void *ctx
) {
    if (total_items <= 0) return;
    int nt = hadukenn_resolve_num_threads(requested_threads, total_items, min_items_per_thread);
    if (nt <= 1) {
        fn(0, total_items, ctx);
        return;
    }

    pthread_t threads[HADUKENN_MAX_PTHREADS];
    HadukennParTask tasks[HADUKENN_MAX_PTHREADS];
    int base = total_items / nt;
    int rem = total_items % nt;
    int cursor = 0;
    for (int t = 0; t < nt; ++t) {
        int chunk = base + (t < rem ? 1 : 0);
        tasks[t].start = cursor;
        tasks[t].end = cursor + chunk;
        tasks[t].fn = fn;
        tasks[t].ctx = ctx;
        cursor += chunk;
    }

    int spawned = 0;
    int caller_start = tasks[nt - 1].start;
    for (int t = 0; t < nt - 1; ++t) {
        if (pthread_create(&threads[t], NULL, hadukenn_par_trampoline, &tasks[t]) != 0) {
            caller_start = tasks[t].start;
            break;
        }
        spawned += 1;
    }

    fn(caller_start, total_items, ctx);

    for (int t = 0; t < spawned; ++t) {
        pthread_join(threads[t], NULL);
    }
}
#endif /* !defined(_OPENMP) && !defined(_WIN32) */

/* --- 4a. FP32 GEMV --- */

static inline void gemv_f32_range(
    const float *x,
    const float *W,
    float *out,
    int M,
    int d,
    float scale,
    int n_start,
    int n_end,
    int m_start,
    int m_end
) {
    for (int n = n_start; n < n_end; ++n) {
        const float *xn = x + (size_t)n * d;
        float *out_n = out + (size_t)n * M;
        for (int m = m_start; m < m_end; ++m) {
            out_n[m] = scale * dot_f32_fast(xn, W + (size_t)m * d, d);
        }
    }
}

#if !defined(_OPENMP) && !defined(_WIN32)
typedef struct {
    const float *x;
    const float *W;
    float *out;
    int N;
    int M;
    int d;
    float scale;
    int partition_rows;
} GemvParCtx;

static void gemv_par_worker(int start, int end, void *raw_ctx) {
    GemvParCtx *ctx = (GemvParCtx *)raw_ctx;
    if (ctx->partition_rows) {
        gemv_f32_range(ctx->x, ctx->W, ctx->out, ctx->M, ctx->d, ctx->scale, start, end, 0, ctx->M);
    } else {
        gemv_f32_range(ctx->x, ctx->W, ctx->out, ctx->M, ctx->d, ctx->scale, 0, ctx->N, start, end);
    }
}
#endif

void hadukenn_gemv_f32(
    const float *x,
    const float *W,
    float *out,
    int N,
    int M,
    int d,
    float scale
) {
    if (N <= 0 || M <= 0 || d < 0 || !x || !W || !out) return;
    if (d == 0) {
        memset(out, 0, (size_t)N * (size_t)M * sizeof(float));
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if((size_t)N * (size_t)M >= 512u && (size_t)N * (size_t)M * (size_t)d >= 32768u)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            out[(size_t)n * M + m] = scale * dot_f32_fast(
                x + (size_t)n * d,
                W + (size_t)m * d,
                d
            );
        }
    }
#elif !defined(_WIN32)
    if ((size_t)N * (size_t)M >= 512u && (size_t)N * (size_t)M * (size_t)d >= 32768u) {
        GemvParCtx ctx = {x, W, out, N, M, d, scale, (N >= 4) ? 1 : 0};
        if (ctx.partition_rows) {
            hadukenn_parallel_for(N, 0, 1, gemv_par_worker, &ctx);
        } else {
            hadukenn_parallel_for(M, 0, 16, gemv_par_worker, &ctx);
        }
    } else {
        gemv_f32_range(x, W, out, M, d, scale, 0, N, 0, M);
    }
#else
    gemv_f32_range(x, W, out, M, d, scale, 0, N, 0, M);
#endif
}

/* --- 4b. Symmetric INT8 Quantized GEMM (AVX512-VNNI / AVX2 / ARM64 i8mm+sdot) --- */

static inline void qgemm_int8_range(
    const int8_t *x_q,
    const float *x_scale,
    const int8_t *w_q,
    const float *w_scale,
    float *out,
    int n_start,
    int n_end,
    int m_start,
    int m_end,
    int M,
    int d
) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    int n = n_start;
    for (; n + 1 < n_end; n += 2) {
        const int8_t *x0 = x_q + (size_t)n * d;
        const int8_t *x1 = x_q + (size_t)(n + 1) * d;
        const float sx0 = x_scale[n];
        const float sx1 = x_scale[n + 1];
        float *out0 = out + (size_t)n * M;
        float *out1 = out + (size_t)(n + 1) * M;

        int m = m_start;
        for (; m + 1 < m_end; m += 2) {
            const int8_t *w0 = w_q + (size_t)m * d;
            const int8_t *w1 = w_q + (size_t)(m + 1) * d;
            int32x4_t acc = vdupq_n_s32(0);
            int k = 0;
            for (; k + 8 <= d; k += 8) {
                int8x16_t va = vcombine_s8(vld1_s8(x0 + k), vld1_s8(x1 + k));
                int8x16_t vb = vcombine_s8(vld1_s8(w0 + k), vld1_s8(w1 + k));
                acc = vmmlaq_s32(acc, va, vb);
            }
            int32_t a00 = vgetq_lane_s32(acc, 0);
            int32_t a01 = vgetq_lane_s32(acc, 1);
            int32_t a10 = vgetq_lane_s32(acc, 2);
            int32_t a11 = vgetq_lane_s32(acc, 3);
            if (k < d) {
                const int rem = d - k;
                a00 += dot_i8_fast(x0 + k, w0 + k, rem);
                a01 += dot_i8_fast(x0 + k, w1 + k, rem);
                a10 += dot_i8_fast(x1 + k, w0 + k, rem);
                a11 += dot_i8_fast(x1 + k, w1 + k, rem);
            }
            const float sw0 = w_scale[m];
            const float sw1 = w_scale[m + 1];
            out0[m]     = (float)a00 * (sx0 * sw0);
            out0[m + 1] = (float)a01 * (sx0 * sw1);
            out1[m]     = (float)a10 * (sx1 * sw0);
            out1[m + 1] = (float)a11 * (sx1 * sw1);
        }
        if (m < m_end) {
            const int8_t *wm = w_q + (size_t)m * d;
            const float swm = w_scale[m];
            out0[m] = (float)dot_i8_fast(x0, wm, d) * (sx0 * swm);
            out1[m] = (float)dot_i8_fast(x1, wm, d) * (sx1 * swm);
        }
    }
    if (n < n_end) {
        const int8_t *xn = x_q + (size_t)n * d;
        const float sx = x_scale[n];
        float *out_n = out + (size_t)n * M;
        for (int m = m_start; m < m_end; ++m) {
            out_n[m] = (float)dot_i8_fast(xn, w_q + (size_t)m * d, d) * (sx * w_scale[m]);
        }
    }
#else
    for (int n = n_start; n < n_end; ++n) {
        const int8_t *xn = x_q + (size_t)n * d;
        const float sx = x_scale[n];
        float *out_n = out + (size_t)n * M;
        for (int m = m_start; m < m_end; ++m) {
            int32_t acc = dot_i8_fast(xn, w_q + (size_t)m * d, d);
            out_n[m] = (float)acc * (sx * w_scale[m]);
        }
    }
#endif
}

#if !defined(_OPENMP) && !defined(_WIN32)
typedef struct {
    const int8_t *x_q;
    const float *x_scale;
    const int8_t *w_q;
    const float *w_scale;
    float *out;
    int N;
    int M;
    int d;
    int partition_row_pairs;
} QGemmParCtx;

static void qgemm_par_worker(int start, int end, void *raw_ctx) {
    QGemmParCtx *ctx = (QGemmParCtx *)raw_ctx;
    if (ctx->partition_row_pairs) {
        int n_start = start * 2;
        int n_end = end * 2;
        if (n_end > ctx->N) n_end = ctx->N;
        qgemm_int8_range(
            ctx->x_q, ctx->x_scale, ctx->w_q, ctx->w_scale, ctx->out,
            n_start, n_end, 0, ctx->M, ctx->M, ctx->d
        );
    } else {
        int m_start = start * 2;
        int m_end = end * 2;
        if (m_end > ctx->M) m_end = ctx->M;
        qgemm_int8_range(
            ctx->x_q, ctx->x_scale, ctx->w_q, ctx->w_scale, ctx->out,
            0, ctx->N, m_start, m_end, ctx->M, ctx->d
        );
    }
}
#endif

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
    if (N <= 0 || M <= 0 || d < 0 || !x_q || !x_scale || !w_q || !w_scale || !out) return;
    if (d == 0) {
        memset(out, 0, (size_t)N * (size_t)M * sizeof(float));
        return;
    }
#ifdef _OPENMP
    const int row_pairs = (N + 1) / 2;
    #pragma omp parallel for schedule(static) if((size_t)N * (size_t)M >= 256u && (size_t)N * (size_t)M * (size_t)d >= 32768u)
    for (int p = 0; p < row_pairs; ++p) {
        int n_start = p * 2;
        int n_end = n_start + 2;
        if (n_end > N) n_end = N;
        qgemm_int8_range(x_q, x_scale, w_q, w_scale, out, n_start, n_end, 0, M, M, d);
    }
#elif !defined(_WIN32)
    if ((size_t)N * (size_t)M >= 256u && (size_t)N * (size_t)M * (size_t)d >= 32768u) {
        QGemmParCtx ctx = {x_q, x_scale, w_q, w_scale, out, N, M, d, (N >= 8) ? 1 : 0};
        if (ctx.partition_row_pairs) {
            hadukenn_parallel_for((N + 1) / 2, 0, 1, qgemm_par_worker, &ctx);
        } else {
            hadukenn_parallel_for((M + 1) / 2, 0, 8, qgemm_par_worker, &ctx);
        }
    } else {
        qgemm_int8_range(x_q, x_scale, w_q, w_scale, out, 0, N, 0, M, M, d);
    }
#else
    qgemm_int8_range(x_q, x_scale, w_q, w_scale, out, 0, N, 0, M, M, d);
#endif
}

/* --- 4c. Tiled muP Cross-Entropy Forward/Backward --- */

#if !defined(__aarch64__) && !defined(_M_ARM64)
static inline void dot8x2_f32_fast(
    const float *h0, const float *h1, const float *h2, const float *h3,
    const float *h4, const float *h5, const float *h6, const float *h7,
    const float *e0, const float *e1,
    int d,
    float scale,
    float out_z0[8],
    float out_z1[8]
) {
    const float *hs[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int r = 0; r < 8; ++r) {
        out_z0[r] = scale * dot_f32_fast(hs[r], e0, d);
        out_z1[r] = scale * dot_f32_fast(hs[r], e1, d);
    }
}
#endif

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static int tiled_cce_fwd_8x2_range(
    const float * restrict H,
    const float * restrict E,
    const int32_t * restrict targets,
    float * restrict out_loss,
    int d,
    int V,
    int tile_V,
    float logit_scale,
    int n_start,
    int n_end
) {
    float stack_buf[8 * 512];
    float *heap_buf = NULL;
    float * restrict tile_buf = stack_buf;
    if (tile_V > 512) {
        heap_buf = (float *)malloc((size_t)8 * (size_t)tile_V * sizeof(float));
        if (!heap_buf) return n_start;
        tile_buf = heap_buf;
    }

    int n = n_start;
    for (; n + 7 < n_end; n += 8) {
        const float * restrict h0 = H + (size_t)n * d;
        const float * restrict h1 = H + (size_t)(n + 1) * d;
        const float * restrict h2 = H + (size_t)(n + 2) * d;
        const float * restrict h3 = H + (size_t)(n + 3) * d;
        const float * restrict h4 = H + (size_t)(n + 4) * d;
        const float * restrict h5 = H + (size_t)(n + 5) * d;
        const float * restrict h6 = H + (size_t)(n + 6) * d;
        const float * restrict h7 = H + (size_t)(n + 7) * d;
        const float z_y0 = logit_scale * dot_f32_fast(h0, E + (size_t)targets[n] * d, d);
        const float z_y1 = logit_scale * dot_f32_fast(h1, E + (size_t)targets[n + 1] * d, d);
        const float z_y2 = logit_scale * dot_f32_fast(h2, E + (size_t)targets[n + 2] * d, d);
        const float z_y3 = logit_scale * dot_f32_fast(h3, E + (size_t)targets[n + 3] * d, d);
        const float z_y4 = logit_scale * dot_f32_fast(h4, E + (size_t)targets[n + 4] * d, d);
        const float z_y5 = logit_scale * dot_f32_fast(h5, E + (size_t)targets[n + 5] * d, d);
        const float z_y6 = logit_scale * dot_f32_fast(h6, E + (size_t)targets[n + 6] * d, d);
        const float z_y7 = logit_scale * dot_f32_fast(h7, E + (size_t)targets[n + 7] * d, d);

        float m_run[8] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
        float s_run[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int v0 = 0; v0 < V; v0 += tile_V) {
            int v1 = v0 + tile_V;
            if (v1 > V) v1 = V;
            float m_tile[8] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
            int v = v0;
            for (; v + 1 < v1; v += 2) {
                const float * restrict e0 = E + (size_t)v * d;
                const float * restrict e1 = E + (size_t)(v + 1) * d;
                float z0[8], z1[8];
#if defined(__aarch64__) || defined(_M_ARM64)
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
                float32x4_t a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f);
                float32x4_t a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f);
                float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
                float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
                float32x4_t b4 = vdupq_n_f32(0.0f), b5 = vdupq_n_f32(0.0f);
                float32x4_t b6 = vdupq_n_f32(0.0f), b7 = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i + 8 <= d; i += 8) {
                    float32x4_t ve0 = vld1q_f32(e0 + i), ve1 = vld1q_f32(e1 + i);
                    float32x4_t vh0 = vld1q_f32(h0 + i), vh1 = vld1q_f32(h1 + i);
                    float32x4_t vh2 = vld1q_f32(h2 + i), vh3 = vld1q_f32(h3 + i);
                    float32x4_t vh4 = vld1q_f32(h4 + i), vh5 = vld1q_f32(h5 + i);
                    float32x4_t vh6 = vld1q_f32(h6 + i), vh7 = vld1q_f32(h7 + i);
                    a0 = vfmaq_f32(a0, vh0, ve0); a1 = vfmaq_f32(a1, vh1, ve0);
                    a2 = vfmaq_f32(a2, vh2, ve0); a3 = vfmaq_f32(a3, vh3, ve0);
                    a4 = vfmaq_f32(a4, vh4, ve0); a5 = vfmaq_f32(a5, vh5, ve0);
                    a6 = vfmaq_f32(a6, vh6, ve0); a7 = vfmaq_f32(a7, vh7, ve0);
                    b0 = vfmaq_f32(b0, vh0, ve1); b1 = vfmaq_f32(b1, vh1, ve1);
                    b2 = vfmaq_f32(b2, vh2, ve1); b3 = vfmaq_f32(b3, vh3, ve1);
                    b4 = vfmaq_f32(b4, vh4, ve1); b5 = vfmaq_f32(b5, vh5, ve1);
                    b6 = vfmaq_f32(b6, vh6, ve1); b7 = vfmaq_f32(b7, vh7, ve1);

                    ve0 = vld1q_f32(e0 + i + 4); ve1 = vld1q_f32(e1 + i + 4);
                    vh0 = vld1q_f32(h0 + i + 4); vh1 = vld1q_f32(h1 + i + 4);
                    vh2 = vld1q_f32(h2 + i + 4); vh3 = vld1q_f32(h3 + i + 4);
                    vh4 = vld1q_f32(h4 + i + 4); vh5 = vld1q_f32(h5 + i + 4);
                    vh6 = vld1q_f32(h6 + i + 4); vh7 = vld1q_f32(h7 + i + 4);
                    a0 = vfmaq_f32(a0, vh0, ve0); a1 = vfmaq_f32(a1, vh1, ve0);
                    a2 = vfmaq_f32(a2, vh2, ve0); a3 = vfmaq_f32(a3, vh3, ve0);
                    a4 = vfmaq_f32(a4, vh4, ve0); a5 = vfmaq_f32(a5, vh5, ve0);
                    a6 = vfmaq_f32(a6, vh6, ve0); a7 = vfmaq_f32(a7, vh7, ve0);
                    b0 = vfmaq_f32(b0, vh0, ve1); b1 = vfmaq_f32(b1, vh1, ve1);
                    b2 = vfmaq_f32(b2, vh2, ve1); b3 = vfmaq_f32(b3, vh3, ve1);
                    b4 = vfmaq_f32(b4, vh4, ve1); b5 = vfmaq_f32(b5, vh5, ve1);
                    b6 = vfmaq_f32(b6, vh6, ve1); b7 = vfmaq_f32(b7, vh7, ve1);
                }
                for (; i + 4 <= d; i += 4) {
                    float32x4_t ve0 = vld1q_f32(e0 + i), ve1 = vld1q_f32(e1 + i);
                    float32x4_t vh0 = vld1q_f32(h0 + i), vh1 = vld1q_f32(h1 + i);
                    float32x4_t vh2 = vld1q_f32(h2 + i), vh3 = vld1q_f32(h3 + i);
                    float32x4_t vh4 = vld1q_f32(h4 + i), vh5 = vld1q_f32(h5 + i);
                    float32x4_t vh6 = vld1q_f32(h6 + i), vh7 = vld1q_f32(h7 + i);
                    a0 = vfmaq_f32(a0, vh0, ve0); a1 = vfmaq_f32(a1, vh1, ve0);
                    a2 = vfmaq_f32(a2, vh2, ve0); a3 = vfmaq_f32(a3, vh3, ve0);
                    a4 = vfmaq_f32(a4, vh4, ve0); a5 = vfmaq_f32(a5, vh5, ve0);
                    a6 = vfmaq_f32(a6, vh6, ve0); a7 = vfmaq_f32(a7, vh7, ve0);
                    b0 = vfmaq_f32(b0, vh0, ve1); b1 = vfmaq_f32(b1, vh1, ve1);
                    b2 = vfmaq_f32(b2, vh2, ve1); b3 = vfmaq_f32(b3, vh3, ve1);
                    b4 = vfmaq_f32(b4, vh4, ve1); b5 = vfmaq_f32(b5, vh5, ve1);
                    b6 = vfmaq_f32(b6, vh6, ve1); b7 = vfmaq_f32(b7, vh7, ve1);
                }
                float s0[8] = {
                    vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3),
                    vaddvq_f32(a4), vaddvq_f32(a5), vaddvq_f32(a6), vaddvq_f32(a7)
                };
                float s1[8] = {
                    vaddvq_f32(b0), vaddvq_f32(b1), vaddvq_f32(b2), vaddvq_f32(b3),
                    vaddvq_f32(b4), vaddvq_f32(b5), vaddvq_f32(b6), vaddvq_f32(b7)
                };
                for (; i < d; ++i) {
                    float y0 = e0[i], y1 = e1[i];
                    float x0 = h0[i], x1 = h1[i], x2 = h2[i], x3 = h3[i];
                    float x4 = h4[i], x5 = h5[i], x6 = h6[i], x7 = h7[i];
                    s0[0] += x0 * y0; s0[1] += x1 * y0; s0[2] += x2 * y0; s0[3] += x3 * y0;
                    s0[4] += x4 * y0; s0[5] += x5 * y0; s0[6] += x6 * y0; s0[7] += x7 * y0;
                    s1[0] += x0 * y1; s1[1] += x1 * y1; s1[2] += x2 * y1; s1[3] += x3 * y1;
                    s1[4] += x4 * y1; s1[5] += x5 * y1; s1[6] += x6 * y1; s1[7] += x7 * y1;
                }
                for (int r = 0; r < 8; ++r) {
                    z0[r] = logit_scale * s0[r];
                    z1[r] = logit_scale * s1[r];
                }
#else
                dot8x2_f32_fast(h0, h1, h2, h3, h4, h5, h6, h7, e0, e1, d, logit_scale, z0, z1);
#endif
                const int idx = v - v0;
                for (int r = 0; r < 8; ++r) {
                    float *rb = tile_buf + (size_t)r * tile_V;
                    rb[idx] = z0[r];
                    rb[idx + 1] = z1[r];
                    if (z0[r] > m_tile[r]) m_tile[r] = z0[r];
                    if (z1[r] > m_tile[r]) m_tile[r] = z1[r];
                }
            }
            for (; v < v1; ++v) {
                const float *ev = E + (size_t)v * d;
                const int idx = v - v0;
                float zv[8] = {
                    logit_scale * dot_f32_fast(h0, ev, d),
                    logit_scale * dot_f32_fast(h1, ev, d),
                    logit_scale * dot_f32_fast(h2, ev, d),
                    logit_scale * dot_f32_fast(h3, ev, d),
                    logit_scale * dot_f32_fast(h4, ev, d),
                    logit_scale * dot_f32_fast(h5, ev, d),
                    logit_scale * dot_f32_fast(h6, ev, d),
                    logit_scale * dot_f32_fast(h7, ev, d)
                };
                for (int r = 0; r < 8; ++r) {
                    tile_buf[(size_t)r * tile_V + idx] = zv[r];
                    if (zv[r] > m_tile[r]) m_tile[r] = zv[r];
                }
            }
            const int len_v = v1 - v0;
            for (int r = 0; r < 8; ++r) {
                float m_new = (m_tile[r] > m_run[r]) ? m_tile[r] : m_run[r];
                float s_tile = 0.0f;
                const float *rb = tile_buf + (size_t)r * tile_V;
                for (int idx = 0; idx < len_v; ++idx) {
                    s_tile += expf(rb[idx] - m_new);
                }
                s_run[r] = s_run[r] * expf(m_run[r] - m_new) + s_tile;
                m_run[r] = m_new;
            }
        }
        out_loss[n]     = (m_run[0] + logf(s_run[0])) - z_y0;
        out_loss[n + 1] = (m_run[1] + logf(s_run[1])) - z_y1;
        out_loss[n + 2] = (m_run[2] + logf(s_run[2])) - z_y2;
        out_loss[n + 3] = (m_run[3] + logf(s_run[3])) - z_y3;
        out_loss[n + 4] = (m_run[4] + logf(s_run[4])) - z_y4;
        out_loss[n + 5] = (m_run[5] + logf(s_run[5])) - z_y5;
        out_loss[n + 6] = (m_run[6] + logf(s_run[6])) - z_y6;
        out_loss[n + 7] = (m_run[7] + logf(s_run[7])) - z_y7;
    }

    if (heap_buf) free(heap_buf);
    return n;
}

static void tiled_cce_range(
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
    int n_start,
    int n_end
) {
    float tile_buf[4096];
    int n = n_start;
    if (!compute_grad && n_end - n_start >= 8) {
        n = tiled_cce_fwd_8x2_range(H, E, targets, out_loss, d, V, tile_V, logit_scale, n_start, n_end);
    }

    for (; n < n_end; ++n) {
        const float *h = H + (size_t)n * d;
        const int32_t y = targets[n];
        float m_run = -1e30f;
        float s_run = 0.0f;
        const float z_y = logit_scale * dot_f32_fast(h, E + (size_t)y * d, d);

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

        if (compute_grad && dH && dE) {
            float *dh = dH + (size_t)n * d;
            const float inv_N = logit_scale / (float)N;
            for (int v0 = 0; v0 < V; v0 += tile_V) {
                int v1 = v0 + tile_V;
                if (v1 > V) v1 = V;
                for (int v = v0; v < v1; ++v) {
                    float z = logit_scale * dot_f32_fast(h, E + (size_t)v * d, d);
                    float coeff = expf(z - lse) * inv_N;
                    if (v == y) coeff -= inv_N;
                    axpy_f32_fast(coeff, E + (size_t)v * d, dh, d);
                    axpy_f32_fast(coeff, h, dE + (size_t)v * d, d);
                }
            }
        }
    }
}

#if !defined(_OPENMP) && !defined(_WIN32)
typedef struct {
    const float *H;
    const float *E;
    const int32_t *targets;
    float *out_loss;
    float *dH;
    float *dE;
    int N;
    int d;
    int V;
    int tile_V;
    float logit_scale;
    int compute_grad;
} TiledCceParCtx;

static void tiled_cce_par_worker(int start, int end, void *raw_ctx) {
    TiledCceParCtx *ctx = (TiledCceParCtx *)raw_ctx;
    tiled_cce_range(
        ctx->H, ctx->E, ctx->targets, ctx->out_loss, ctx->dH, ctx->dE,
        ctx->N, ctx->d, ctx->V, ctx->tile_V, ctx->logit_scale, ctx->compute_grad,
        start, end
    );
}
#endif

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
    if (N <= 0 || d <= 0 || V <= 0 || !H || !E || !targets || !out_loss) return;
    if (tile_V < 64) tile_V = 64;
    if (tile_V > 4096) tile_V = 4096;
#ifdef _OPENMP
    if (num_threads > 0 && !compute_grad) {
        omp_set_num_threads(num_threads);
    }
    #pragma omp parallel if(N >= 64 && !compute_grad && (size_t)N * (size_t)V * (size_t)d >= 65536u)
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        int base = N / nthr;
        int rem = N % nthr;
        int n0 = tid * base + (tid < rem ? tid : rem);
        int n1 = n0 + base + (tid < rem ? 1 : 0);
        tiled_cce_range(H, E, targets, out_loss, dH, dE, N, d, V, tile_V, logit_scale, compute_grad, n0, n1);
    }
#elif !defined(_WIN32)
    /* Keep compute_grad != 0 single-threaded to prevent concurrent dE[v, :] write races */
    if (!compute_grad && N >= 64 && (size_t)N * (size_t)V * (size_t)d >= 65536u) {
        TiledCceParCtx ctx = {H, E, targets, out_loss, dH, dE, N, d, V, tile_V, logit_scale, compute_grad};
        hadukenn_parallel_for(N, num_threads, 8, tiled_cce_par_worker, &ctx);
    } else {
        tiled_cce_range(H, E, targets, out_loss, dH, dE, N, d, V, tile_V, logit_scale, compute_grad, 0, N);
    }
#else
    (void)num_threads;
    tiled_cce_range(H, E, targets, out_loss, dH, dE, N, d, V, tile_V, logit_scale, compute_grad, 0, N);
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    /* Prevent dirty upper ZMM/YMM transition stalls on hybrid & Xeon cores */
    _mm256_zeroupper();
#endif
}

/* --- 4d. Sheaf Coboundary KV Prefix Gate --- */

static void sheaf_kv_prefix_gate_range(
    const float *query_stalk,
    const float *nbr_stalks,
    const float *nbr_values,
    float *out_gated_prefix,
    float *out_sheaf_energy,
    int K_nbr,
    int d_stalk,
    int d_model,
    float beta,
    int n_start,
    int n_end
) {
    if (n_start >= n_end) return;
    float energies_stack[256];
    float *heap_energies = NULL;
    float *energies = energies_stack;
    if (K_nbr > 256) {
        heap_energies = (float *)malloc((size_t)K_nbr * sizeof(float));
        if (!heap_energies) return;
        energies = heap_energies;
    }

    for (int n = n_start; n < n_end; ++n) {
        const float *q = query_stalk + (size_t)n * (size_t)d_stalk;
        float *out_v = out_gated_prefix + (size_t)n * (size_t)d_model;
        memset(out_v, 0, (size_t)d_model * sizeof(float));

        if (K_nbr <= 0) {
            out_sheaf_energy[n] = 0.0f;
            continue;
        }

        const float qq = dot_f32_fast(q, q, d_stalk);
        float energy_sum = 0.0f;
        float e_min = 0.0f;
        float e_max = 0.0f;

        for (int k = 0; k < K_nbr; ++k) {
            const float *nk = nbr_stalks + ((size_t)n * (size_t)K_nbr + (size_t)k) * (size_t)d_stalk;
            /* Coboundary squared norm ||q - nk||_2^2 = ||q||^2 + ||nk||^2 - 2 <q, nk> */
            float nn = dot_f32_fast(nk, nk, d_stalk);
            float qn = dot_f32_fast(q, nk, d_stalk);
            float e_k = qq + nn - 2.0f * qn;
            if (e_k < 0.0f) e_k = 0.0f;
            energies[k] = e_k;
            energy_sum += e_k;
            if (k == 0 || e_k < e_min) e_min = e_k;
            if (k == 0 || e_k > e_max) e_max = e_k;
        }

        const float e_ref = (beta >= 0.0f) ? e_min : e_max;
        float weight_sum = 0.0f;
        for (int k = 0; k < K_nbr; ++k) {
            float g = expf(-beta * (energies[k] - e_ref));
            energies[k] = g;
            weight_sum += g;
        }

        const float inv_w = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
        for (int k = 0; k < K_nbr; ++k) {
            const float *vk = nbr_values + ((size_t)n * (size_t)K_nbr + (size_t)k) * (size_t)d_model;
            axpy_f32_fast(energies[k] * inv_w, vk, out_v, d_model);
        }
        out_sheaf_energy[n] = energy_sum / (float)K_nbr;
    }

    free(heap_energies);
}

#if !defined(_OPENMP) && !defined(_WIN32)
typedef struct {
    const float *query_stalk;
    const float *nbr_stalks;
    const float *nbr_values;
    float *out_gated_prefix;
    float *out_sheaf_energy;
    int K_nbr;
    int d_stalk;
    int d_model;
    float beta;
} SheafKvParCtx;

static void sheaf_kv_par_worker(int start, int end, void *raw_ctx) {
    SheafKvParCtx *ctx = (SheafKvParCtx *)raw_ctx;
    sheaf_kv_prefix_gate_range(
        ctx->query_stalk, ctx->nbr_stalks, ctx->nbr_values,
        ctx->out_gated_prefix, ctx->out_sheaf_energy,
        ctx->K_nbr, ctx->d_stalk, ctx->d_model, ctx->beta,
        start, end
    );
}
#endif

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
    if (N <= 0 || K_nbr < 0 || d_stalk < 0 || d_model <= 0 ||
        !query_stalk || !out_gated_prefix || !out_sheaf_energy) {
        return;
    }
    if (K_nbr > 0 && (!nbr_stalks || !nbr_values)) return;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N >= 64 && (size_t)N * (size_t)K_nbr * (size_t)(d_stalk + d_model) >= 32768u)
    for (int n = 0; n < N; ++n) {
        sheaf_kv_prefix_gate_range(
            query_stalk, nbr_stalks, nbr_values, out_gated_prefix, out_sheaf_energy,
            K_nbr, d_stalk, d_model, beta, n, n + 1
        );
    }
#elif !defined(_WIN32)
    if (N >= 64 && (size_t)N * (size_t)K_nbr * (size_t)(d_stalk + d_model) >= 32768u) {
        SheafKvParCtx ctx = {
            query_stalk, nbr_stalks, nbr_values, out_gated_prefix, out_sheaf_energy,
            K_nbr, d_stalk, d_model, beta
        };
        hadukenn_parallel_for(N, 0, 4, sheaf_kv_par_worker, &ctx);
    } else {
        sheaf_kv_prefix_gate_range(
            query_stalk, nbr_stalks, nbr_values, out_gated_prefix, out_sheaf_energy,
            K_nbr, d_stalk, d_model, beta, 0, N
        );
    }
#else
    sheaf_kv_prefix_gate_range(
        query_stalk, nbr_stalks, nbr_values, out_gated_prefix, out_sheaf_energy,
        K_nbr, d_stalk, d_model, beta, 0, N
    );
#endif
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
