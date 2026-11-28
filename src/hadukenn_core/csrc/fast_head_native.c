/*
 * Clean-room C11 + POSIX pthreads + ARM64 Oryon NEON / SDOT / I8MM kernels
 * for HaDukeNN_LLM local Snapdragon X Plus deployment (Features F4-F9).
 *
 * Exports:
 *   - hadukenn_oryon_caps()
 *   - hadukenn_pool_threads()
 *   - hadukenn_dot_f32()
 *   - hadukenn_axpy_f32()
 *   - hadukenn_int8_gemm()
 *   - hadukenn_nf4_matvec_f32()
 *   - cluster_is_head_fwd_bwd_f32()
 *   - tiled_cce_fwd_bwd_f32()
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1UL << 1)
#endif
#ifndef HWCAP_FCMA
#define HWCAP_FCMA (1UL << 14)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1UL << 14)
#endif

#define HADUKENN_CAP_NEON    (1u << 0)
#define HADUKENN_CAP_DOTPROD (1u << 1)
#define HADUKENN_CAP_I8MM    (1u << 2)
#define HADUKENN_CAP_FCMA    (1u << 3)
#define HADUKENN_CAP_BF16    (1u << 4)
#define HADUKENN_CAP_POOL    (1u << 8)

#define HADUKENN_MAX_WORKERS 8

/* =========================================================================
 * Persistent 8-Core POSIX pthreads Pool (F5)
 * ========================================================================= */

typedef void (*hadukenn_task_fn)(void *ctx, int64_t start, int64_t end, int worker_id);

typedef struct {
    pthread_mutex_t call_mu;
    pthread_mutex_t work_mu;
    pthread_mutex_t claim_mu;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    pthread_t threads[HADUKENN_MAX_WORKERS];
    int num_workers;
    int initialized;
    int shutdown;
    uint64_t generation;
    int active_workers;
    hadukenn_task_fn fn;
    void *ctx;
    int64_t total_items;
    int64_t chunk_size;
    int64_t next_item;
} hadukenn_pool_t;

typedef struct {
    int worker_id;
} hadukenn_worker_arg_t;

static hadukenn_pool_t g_pool = {
    .call_mu = PTHREAD_MUTEX_INITIALIZER,
    .work_mu = PTHREAD_MUTEX_INITIALIZER,
    .claim_mu = PTHREAD_MUTEX_INITIALIZER,
    .work_cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
    .num_workers = 1,
    .initialized = 0,
    .shutdown = 0,
    .generation = 0,
    .active_workers = 0,
    .fn = NULL,
    .ctx = NULL,
    .total_items = 0,
    .chunk_size = 1,
    .next_item = 0,
};

static hadukenn_worker_arg_t g_worker_args[HADUKENN_MAX_WORKERS];
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void pool_run_chunks(int worker_id) {
    hadukenn_task_fn fn = g_pool.fn;
    void *ctx = g_pool.ctx;
    const int64_t total = g_pool.total_items;
    const int64_t chunk = g_pool.chunk_size;

    for (;;) {
        pthread_mutex_lock(&g_pool.claim_mu);
        const int64_t start = g_pool.next_item;
        if (start >= total) {
            pthread_mutex_unlock(&g_pool.claim_mu);
            break;
        }
        int64_t end = start + chunk;
        if (end > total) {
            end = total;
        }
        g_pool.next_item = end;
        pthread_mutex_unlock(&g_pool.claim_mu);

        fn(ctx, start, end, worker_id);
    }
}

static void *pool_worker_main(void *arg) {
    const int worker_id = ((const hadukenn_worker_arg_t *)arg)->worker_id;
    uint64_t seen_gen = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.work_mu);
        while (!g_pool.shutdown && g_pool.generation == seen_gen) {
            pthread_cond_wait(&g_pool.work_cv, &g_pool.work_mu);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.work_mu);
            return NULL;
        }
        seen_gen = g_pool.generation;
        pthread_mutex_unlock(&g_pool.work_mu);

        pool_run_chunks(worker_id);

        pthread_mutex_lock(&g_pool.work_mu);
        g_pool.active_workers -= 1;
        if (g_pool.active_workers == 0) {
            pthread_cond_signal(&g_pool.done_cv);
        }
        pthread_mutex_unlock(&g_pool.work_mu);
    }
}

static void pool_init_impl(void) {
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    int w = (nproc > 0) ? (int)nproc : 1;
    if (w > HADUKENN_MAX_WORKERS) {
        w = HADUKENN_MAX_WORKERS;
    }
    if (w < 1) {
        w = 1;
    }

    int spawned = 1;
    for (int i = 1; i < w; ++i) {
        g_worker_args[i].worker_id = i;
        if (pthread_create(&g_pool.threads[i], NULL, pool_worker_main, &g_worker_args[i]) != 0) {
            break;
        }
        spawned += 1;
    }
    g_pool.num_workers = spawned;
    g_pool.initialized = 1;
}

__attribute__((destructor)) static void pool_teardown(void) {
    pthread_mutex_lock(&g_pool.call_mu);
    pthread_mutex_lock(&g_pool.work_mu);
    if (!g_pool.initialized || g_pool.shutdown) {
        pthread_mutex_unlock(&g_pool.work_mu);
        pthread_mutex_unlock(&g_pool.call_mu);
        return;
    }
    g_pool.shutdown = 1;
    pthread_cond_broadcast(&g_pool.work_cv);
    pthread_mutex_unlock(&g_pool.work_mu);

    for (int i = 1; i < g_pool.num_workers; ++i) {
        pthread_join(g_pool.threads[i], NULL);
    }
    g_pool.initialized = 0;
    pthread_mutex_unlock(&g_pool.call_mu);
}

int hadukenn_pool_threads(void) {
    pthread_once(&g_pool_once, pool_init_impl);
    return g_pool.num_workers;
}

static void pool_dispatch(
    hadukenn_task_fn fn,
    void *ctx,
    int64_t total_items,
    int64_t min_chunk,
    int64_t parallel_threshold
) {
    if (total_items <= 0) {
        return;
    }
    const int w = hadukenn_pool_threads();
    if (w <= 1 || total_items < parallel_threshold || total_items <= min_chunk) {
        fn(ctx, 0, total_items, 0);
        return;
    }

    int64_t chunk = total_items / ((int64_t)w * 2);
    if (chunk < min_chunk) {
        chunk = min_chunk;
    }
    if (chunk < 1) {
        chunk = 1;
    }
    if (chunk >= total_items) {
        fn(ctx, 0, total_items, 0);
        return;
    }

    pthread_mutex_lock(&g_pool.call_mu);
    if (g_pool.shutdown || !g_pool.initialized) {
        pthread_mutex_unlock(&g_pool.call_mu);
        fn(ctx, 0, total_items, 0);
        return;
    }

    pthread_mutex_lock(&g_pool.work_mu);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.total_items = total_items;
    g_pool.chunk_size = chunk;
    g_pool.next_item = 0;
    g_pool.active_workers = w - 1;
    g_pool.generation += 1;
    pthread_cond_broadcast(&g_pool.work_cv);
    pthread_mutex_unlock(&g_pool.work_mu);

    pool_run_chunks(0);

    pthread_mutex_lock(&g_pool.work_mu);
    while (g_pool.active_workers > 0) {
        pthread_cond_wait(&g_pool.done_cv, &g_pool.work_mu);
    }
    pthread_mutex_unlock(&g_pool.work_mu);
    pthread_mutex_unlock(&g_pool.call_mu);
}

/* =========================================================================
 * Runtime HWCAP Capability Detection (F4 / F7)
 * ========================================================================= */

uint32_t hadukenn_oryon_caps(void) {
    uint32_t caps = 0u;
    if (hadukenn_pool_threads() >= 1) {
        caps |= HADUKENN_CAP_POOL;
    }
#if defined(__linux__) && defined(__aarch64__)
    const unsigned long hw1 = getauxval(AT_HWCAP);
    const unsigned long hw2 = getauxval(AT_HWCAP2);
#if defined(__ARM_NEON) || defined(__aarch64__)
    if ((hw1 & HWCAP_ASIMD) != 0UL) {
        caps |= HADUKENN_CAP_NEON;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if ((hw1 & HWCAP_ASIMDDP) != 0UL) {
        caps |= HADUKENN_CAP_DOTPROD;
    }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    if ((hw2 & HWCAP2_I8MM) != 0UL) {
        caps |= HADUKENN_CAP_I8MM;
    }
#endif
    if ((hw1 & HWCAP_FCMA) != 0UL) {
        caps |= HADUKENN_CAP_FCMA;
    }
    if ((hw2 & HWCAP2_BF16) != 0UL) {
        caps |= HADUKENN_CAP_BF16;
    }
#else
#if defined(__ARM_NEON) || defined(__aarch64__)
    caps |= HADUKENN_CAP_NEON;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    caps |= HADUKENN_CAP_DOTPROD;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    caps |= HADUKENN_CAP_I8MM;
#endif
#endif
    return caps;
}

/* =========================================================================
 * NEON FP32 Kernels: dot_f32 & axpy_f32 (F6)
 * ========================================================================= */

static inline float dot_f32_vec(const float *a, const float *b, int64_t d) {
    int64_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= d; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
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

static inline void axpy_f32_vec(float alpha, const float *x, float *y, int64_t d) {
    int64_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 16 <= d; i += 16) {
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), va, vld1q_f32(x + i + 4)));
        vst1q_f32(y + i + 8, vfmaq_f32(vld1q_f32(y + i + 8), va, vld1q_f32(x + i + 8)));
        vst1q_f32(y + i + 12, vfmaq_f32(vld1q_f32(y + i + 12), va, vld1q_f32(x + i + 12)));
    }
    for (; i + 4 <= d; i += 4) {
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    }
#endif
    for (; i < d; ++i) {
        y[i] += alpha * x[i];
    }
}

typedef struct {
    const float *a;
    const float *b;
    double partial[HADUKENN_MAX_WORKERS];
} dot_par_ctx_t;

static void dot_par_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    dot_par_ctx_t *ctx = (dot_par_ctx_t *)vctx;
    double acc = 0.0;
    const int64_t blk = 1024;
    for (int64_t p = start; p < end; p += blk) {
        int64_t len = end - p;
        if (len > blk) {
            len = blk;
        }
        acc += (double)dot_f32_vec(ctx->a + p, ctx->b + p, len);
    }
    ctx->partial[worker_id] += acc;
}

float hadukenn_dot_f32(const float *a, const float *b, int64_t n) {
    if (!a || !b || n <= 0) {
        return 0.0f;
    }
    if (n < 65536) {
        return dot_f32_vec(a, b, n);
    }
    dot_par_ctx_t ctx;
    ctx.a = a;
    ctx.b = b;
    for (int w = 0; w < HADUKENN_MAX_WORKERS; ++w) {
        ctx.partial[w] = 0.0;
    }
    pool_dispatch(dot_par_worker, &ctx, n, 8192, 65536);
    double total = 0.0;
    const int nw = hadukenn_pool_threads();
    for (int w = 0; w < nw; ++w) {
        total += ctx.partial[w];
    }
    return (float)total;
}

typedef struct {
    float alpha;
    const float *x;
    float *y;
} axpy_par_ctx_t;

static void axpy_par_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    (void)worker_id;
    axpy_par_ctx_t *ctx = (axpy_par_ctx_t *)vctx;
    axpy_f32_vec(ctx->alpha, ctx->x + start, ctx->y + start, end - start);
}

void hadukenn_axpy_f32(float alpha, const float *x, float *y, int64_t n) {
    if (!x || !y || n <= 0) {
        return;
    }
    if (n < 65536) {
        axpy_f32_vec(alpha, x, y, n);
        return;
    }
    axpy_par_ctx_t ctx = {.alpha = alpha, .x = x, .y = y};
    pool_dispatch(axpy_par_worker, &ctx, n, 8192, 65536);
}

/* =========================================================================
 * Race-Free Structured Output Head Kernels (F5, F6, F9)
 * ========================================================================= */

typedef struct {
    const float *H;
    const float *E_clustered;
    const float *C_root;
    const int32_t *target_c;
    const int32_t *target_m;
    const int32_t *neg_clusters;
    float *out_loss;
    float *dH;
    int N;
    int d;
    int NC;
    int S;
    int K_neg;
    int compute_grad;
    int32_t *all_active_c;
    float *all_coeffs;
} cluster_is_ctx_t;

static void cluster_is_token_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    (void)worker_id;
    cluster_is_ctx_t *ctx = (cluster_is_ctx_t *)vctx;
    const int N = ctx->N;
    const int d = ctx->d;
    const int NC = ctx->NC;
    const int S = ctx->S;
    const int K_neg = ctx->K_neg;
    const int active_clusters = 1 + K_neg;
    const int active_cols = active_clusters * S;

    float root_stack[4096];
    float active_stack[2048];
    int32_t act_c_stack[16];

    float *root_logits = (NC <= 4096) ? root_stack : (float *)malloc((size_t)NC * sizeof(float));
    float *active_logits = (active_cols <= 2048) ? active_stack : (float *)malloc((size_t)active_cols * sizeof(float));
    int32_t *active_c = (active_clusters <= 16) ? act_c_stack : (int32_t *)malloc((size_t)active_clusters * sizeof(int32_t));

    if (!root_logits || !active_logits || !active_c) {
        if (root_logits != root_stack) free(root_logits);
        if (active_logits != active_stack) free(active_logits);
        if (active_c != act_c_stack) free(active_c);
        return;
    }

    for (int64_t n = start; n < end; ++n) {
        const float *h = ctx->H + (size_t)n * (size_t)d;
        const int32_t tc = ctx->target_c[n];
        const int32_t tm = ctx->target_m[n];

        float max_r = -1e30f;
        for (int c = 0; c < NC; ++c) {
            float r = dot_f32_vec(h, ctx->C_root + (size_t)c * (size_t)d, d);
            root_logits[c] = r;
            if (r > max_r) {
                max_r = r;
            }
        }
        float sum_r = 0.0f;
        for (int c = 0; c < NC; ++c) {
            float p = expf(root_logits[c] - max_r);
            root_logits[c] = p;
            sum_r += p;
        }
        const float inv_sum_r = 1.0f / sum_r;

        active_c[0] = tc;
        for (int k = 0; k < K_neg; ++k) {
            active_c[1 + k] = ctx->neg_clusters[(size_t)n * (size_t)K_neg + (size_t)k];
        }

        float max_o = -1e30f;
        for (int blk = 0; blk < active_clusters; ++blk) {
            const int32_t c_idx = active_c[blk];
            const float *E_blk = ctx->E_clustered + ((size_t)c_idx * (size_t)S) * (size_t)d;
            float log_w = 0.0f;
            if (blk > 0) {
                const float q_tc = root_logits[tc] * inv_sum_r;
                const float q_c = root_logits[c_idx] * inv_sum_r;
                const float q_cond = q_c / fmaxf(1.0f - q_tc, 1e-7f);
                const float denom = fmaxf((float)K_neg * q_cond, 1e-7f);
                log_w = -logf(denom);
            }
            for (int m = 0; m < S; ++m) {
                float o = dot_f32_vec(h, E_blk + (size_t)m * (size_t)d, d) + log_w;
                active_logits[blk * S + m] = o;
                if (o > max_o) {
                    max_o = o;
                }
            }
        }

        float sum_o = 0.0f;
        for (int j = 0; j < active_cols; ++j) {
            float p = expf(active_logits[j] - max_o);
            active_logits[j] = p;
            sum_o += p;
        }
        const float target_logit = active_logits[tm];
        ctx->out_loss[n] = -logf(fmaxf(target_logit / sum_o, 1e-30f));

        if (ctx->compute_grad) {
            float *dh = ctx->dH + (size_t)n * (size_t)d;
            const float inv_sum_o = 1.0f / (sum_o * (float)N);
            const float inv_N = 1.0f / (float)N;
            int32_t *dst_c = ctx->all_active_c + (size_t)n * (size_t)active_clusters;
            float *dst_coeff = ctx->all_coeffs + (size_t)n * (size_t)active_cols;

            for (int blk = 0; blk < active_clusters; ++blk) {
                const int32_t c_idx = active_c[blk];
                const float *E_blk = ctx->E_clustered + ((size_t)c_idx * (size_t)S) * (size_t)d;
                dst_c[blk] = c_idx;
                for (int m = 0; m < S; ++m) {
                    float coeff = active_logits[blk * S + m] * inv_sum_o;
                    if (blk == 0 && m == tm) {
                        coeff -= inv_N;
                    }
                    dst_coeff[blk * S + m] = coeff;
                    axpy_f32_vec(coeff, E_blk + (size_t)m * (size_t)d, dh, d);
                }
            }
        }
    }

    if (root_logits != root_stack) free(root_logits);
    if (active_logits != active_stack) free(active_logits);
    if (active_c != act_c_stack) free(active_c);
}

typedef struct {
    const float *H;
    float *dE_clustered;
    const int32_t *all_active_c;
    const float *all_coeffs;
    int N;
    int d;
    int S;
    int active_clusters;
    int active_cols;
} cluster_is_de_ctx_t;

static void cluster_is_de_worker(void *vctx, int64_t c_start, int64_t c_end, int worker_id) {
    (void)worker_id;
    cluster_is_de_ctx_t *ctx = (cluster_is_de_ctx_t *)vctx;
    const int N = ctx->N;
    const int d = ctx->d;
    const int S = ctx->S;
    const int active_clusters = ctx->active_clusters;
    const int active_cols = ctx->active_cols;

    for (int n = 0; n < N; ++n) {
        const float *h = ctx->H + (size_t)n * (size_t)d;
        const int32_t *act_c = ctx->all_active_c + (size_t)n * (size_t)active_clusters;
        const float *coeffs = ctx->all_coeffs + (size_t)n * (size_t)active_cols;

        for (int blk = 0; blk < active_clusters; ++blk) {
            const int32_t c_idx = act_c[blk];
            if ((int64_t)c_idx >= c_start && (int64_t)c_idx < c_end) {
                float *dE_blk = ctx->dE_clustered + ((size_t)c_idx * (size_t)S) * (size_t)d;
                const float *blk_coeff = coeffs + (size_t)blk * (size_t)S;
                for (int m = 0; m < S; ++m) {
                    axpy_f32_vec(blk_coeff[m], h, dE_blk + (size_t)m * (size_t)d, d);
                }
            }
        }
    }
}

void cluster_is_head_fwd_bwd_f32(
    const float *H,
    const float *E_clustered,
    const float *C_root,
    const int32_t *target_c,
    const int32_t *target_m,
    const int32_t *neg_clusters,
    float *out_loss,
    float *dH,
    float *dE_clustered,
    int N,
    int d,
    int NC,
    int S,
    int K_neg,
    int compute_grad
) {
    if (N <= 0 || d <= 0 || NC <= 0 || S <= 0 || K_neg < 0) {
        return;
    }
    const int active_clusters = 1 + K_neg;
    const int active_cols = active_clusters * S;

    int32_t *all_active_c = NULL;
    float *all_coeffs = NULL;
    if (compute_grad) {
        all_active_c = (int32_t *)malloc((size_t)N * (size_t)active_clusters * sizeof(int32_t));
        all_coeffs = (float *)malloc((size_t)N * (size_t)active_cols * sizeof(float));
        if (!all_active_c || !all_coeffs) {
            free(all_active_c);
            free(all_coeffs);
            return;
        }
    }

    cluster_is_ctx_t ctx = {
        .H = H,
        .E_clustered = E_clustered,
        .C_root = C_root,
        .target_c = target_c,
        .target_m = target_m,
        .neg_clusters = neg_clusters,
        .out_loss = out_loss,
        .dH = dH,
        .N = N,
        .d = d,
        .NC = NC,
        .S = S,
        .K_neg = K_neg,
        .compute_grad = compute_grad,
        .all_active_c = all_active_c,
        .all_coeffs = all_coeffs,
    };
    pool_dispatch(cluster_is_token_worker, &ctx, (int64_t)N, 4, 16);

    if (compute_grad) {
        cluster_is_de_ctx_t de_ctx = {
            .H = H,
            .dE_clustered = dE_clustered,
            .all_active_c = all_active_c,
            .all_coeffs = all_coeffs,
            .N = N,
            .d = d,
            .S = S,
            .active_clusters = active_clusters,
            .active_cols = active_cols,
        };
        /* Partition disjoint clusters across workers -> zero write races, deterministic per cluster. */
        pool_dispatch(cluster_is_de_worker, &de_ctx, (int64_t)NC, 8, 32);
        free(all_active_c);
        free(all_coeffs);
    }
}

typedef struct {
    const float *H;
    const float *E;
    const int32_t *targets;
    float *out_loss;
    float *lse_buf;
    float *dH;
    float *dE;
    int N;
    int d;
    int V;
    int tile_V;
    int compute_grad;
} tiled_cce_ctx_t;

static void tiled_cce_fwd_dh_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    (void)worker_id;
    tiled_cce_ctx_t *ctx = (tiled_cce_ctx_t *)vctx;
    const int N = ctx->N;
    const int d = ctx->d;
    const int V = ctx->V;
    const int tile_V = ctx->tile_V;

    float tile_stack[1024];
    float *tile_buf = (tile_V <= 1024) ? tile_stack : (float *)malloc((size_t)tile_V * sizeof(float));
    if (!tile_buf) {
        return;
    }

    for (int64_t n = start; n < end; ++n) {
        const float *h = ctx->H + (size_t)n * (size_t)d;
        const int32_t y = ctx->targets[n];
        float m_run = -1e30f;
        float s_run = 0.0f;
        const float z_y = dot_f32_vec(h, ctx->E + (size_t)y * (size_t)d, d);

        for (int v0 = 0; v0 < V; v0 += tile_V) {
            int v1 = v0 + tile_V;
            if (v1 > V) {
                v1 = V;
            }
            float m_tile = -1e30f;
            for (int v = v0; v < v1; ++v) {
                float z = dot_f32_vec(h, ctx->E + (size_t)v * (size_t)d, d);
                tile_buf[v - v0] = z;
                if (z > m_tile) {
                    m_tile = z;
                }
            }
            const float m_new = (m_tile > m_run) ? m_tile : m_run;
            float s_tile = 0.0f;
            for (int v = v0; v < v1; ++v) {
                s_tile += expf(tile_buf[v - v0] - m_new);
            }
            s_run = s_run * expf(m_run - m_new) + s_tile;
            m_run = m_new;
        }
        const float lse = m_run + logf(s_run);
        ctx->out_loss[n] = lse - z_y;
        if (ctx->lse_buf) {
            ctx->lse_buf[n] = lse;
        }

        if (ctx->compute_grad) {
            float *dh = ctx->dH + (size_t)n * (size_t)d;
            const float inv_N = 1.0f / (float)N;
            for (int v = 0; v < V; ++v) {
                const float z = dot_f32_vec(h, ctx->E + (size_t)v * (size_t)d, d);
                float p = expf(z - lse) * inv_N;
                if (v == y) {
                    p -= inv_N;
                }
                axpy_f32_vec(p, ctx->E + (size_t)v * (size_t)d, dh, d);
            }
        }
    }

    if (tile_buf != tile_stack) {
        free(tile_buf);
    }
}

static void tiled_cce_de_worker(void *vctx, int64_t v_start, int64_t v_end, int worker_id) {
    (void)worker_id;
    tiled_cce_ctx_t *ctx = (tiled_cce_ctx_t *)vctx;
    const int N = ctx->N;
    const int d = ctx->d;
    const float inv_N = 1.0f / (float)N;

    for (int64_t v = v_start; v < v_end; ++v) {
        const float *e_v = ctx->E + (size_t)v * (size_t)d;
        float *de_v = ctx->dE + (size_t)v * (size_t)d;
        for (int n = 0; n < N; ++n) {
            const float *h = ctx->H + (size_t)n * (size_t)d;
            const float z = dot_f32_vec(h, e_v, d);
            float p = expf(z - ctx->lse_buf[n]) * inv_N;
            if (v == (int64_t)ctx->targets[n]) {
                p -= inv_N;
            }
            axpy_f32_vec(p, h, de_v, d);
        }
    }
}

void tiled_cce_fwd_bwd_f32(
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
    int compute_grad
) {
    if (N <= 0 || d <= 0 || V <= 0) {
        return;
    }
    if (tile_V < 1) {
        tile_V = 64;
    }

    float lse_stack[1024];
    float *lse_buf = NULL;
    if (compute_grad) {
        lse_buf = (N <= 1024) ? lse_stack : (float *)malloc((size_t)N * sizeof(float));
        if (!lse_buf) {
            return;
        }
    }

    tiled_cce_ctx_t ctx = {
        .H = H,
        .E = E,
        .targets = targets,
        .out_loss = out_loss,
        .lse_buf = lse_buf,
        .dH = dH,
        .dE = dE,
        .N = N,
        .d = d,
        .V = V,
        .tile_V = tile_V,
        .compute_grad = compute_grad,
    };

    const int64_t fwd_thresh = ((int64_t)N * (int64_t)V >= 8192) ? 8 : 64;
    pool_dispatch(tiled_cce_fwd_dh_worker, &ctx, (int64_t)N, 4, fwd_thresh);

    if (compute_grad) {
        const int64_t v_thresh = ((int64_t)N * (int64_t)V >= 8192) ? 32 : 256;
        pool_dispatch(tiled_cce_de_worker, &ctx, (int64_t)V, 16, v_thresh);
        if (lse_buf != lse_stack) {
            free(lse_buf);
        }
    }
}

/* =========================================================================
 * INT8 GEMM: i8mm (vmmlaq_s32) / sdot (vdotq_s32) / scalar (F7)
 * ========================================================================= */

typedef struct {
    const int8_t *A;
    const int8_t *B;
    const int8_t *B_T;
    int32_t *C;
    int M;
    int N;
    int K;
    int K_pad;
} int8_gemm_ctx_t;

static void int8_gemm_scalar_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    (void)worker_id;
    const int8_gemm_ctx_t *ctx = (const int8_gemm_ctx_t *)vctx;
    const int N = ctx->N;
    const int K = ctx->K;

    for (int64_t i = start; i < end; ++i) {
        const int8_t *a_row = ctx->A + (size_t)i * (size_t)K;
        int32_t *c_row = ctx->C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            uint32_t acc = 0u;
            for (int k = 0; k < K; ++k) {
                acc += (uint32_t)((int32_t)a_row[k] * (int32_t)ctx->B[(size_t)k * (size_t)N + (size_t)j]);
            }
            c_row[j] = (int32_t)acc;
        }
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
static void int8_gemm_sdot_matvec_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    (void)worker_id;
    const int8_gemm_ctx_t *ctx = (const int8_gemm_ctx_t *)vctx;
    const int K = ctx->K;
    const int8_t *B = ctx->B;

    for (int64_t i = start; i < end; ++i) {
        const int8_t *a_row = ctx->A + (size_t)i * (size_t)K;
        int32x4_t acc = vdupq_n_s32(0);
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            const int8x16_t va = vld1q_s8(a_row + k);
            const int8x16_t vb = vld1q_s8(B + k);
            acc = vdotq_s32(acc, va, vb);
        }
        int32_t sum = vaddvq_s32(acc);
        if (k < K) {
            int8_t ta[16];
            int8_t tb[16];
            vst1q_s8(ta, vdupq_n_s8(0));
            vst1q_s8(tb, vdupq_n_s8(0));
            const int rem = K - k;
            for (int t = 0; t < rem; ++t) {
                ta[t] = a_row[k + t];
                tb[t] = B[k + t];
            }
            sum += vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(ta), vld1q_s8(tb)));
        }
        ctx->C[i] = sum;
    }
}

static void int8_gemm_sdot_bt_worker(void *vctx, int64_t start, int64_t end, int worker_id) {
    (void)worker_id;
    const int8_gemm_ctx_t *ctx = (const int8_gemm_ctx_t *)vctx;
    const int N = ctx->N;
    const int K = ctx->K;
    const int K_pad = ctx->K_pad;
    const int k_full = K & ~15;
    const int rem = K - k_full;

    for (int64_t i = start; i < end; ++i) {
        const int8_t *a_row = ctx->A + (size_t)i * (size_t)K;
        int32_t *c_row = ctx->C + (size_t)i * (size_t)N;
        int8_t ta_tail[16];
        int8x16_t va_tail = vdupq_n_s8(0);
        if (rem > 0) {
            vst1q_s8(ta_tail, vdupq_n_s8(0));
            for (int t = 0; t < rem; ++t) {
                ta_tail[t] = a_row[k_full + t];
            }
            va_tail = vld1q_s8(ta_tail);
        }

        for (int j = 0; j < N; ++j) {
            const int8_t *b_col = ctx->B_T + (size_t)j * (size_t)K_pad;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 32 <= k_full; k += 32) {
                acc0 = vdotq_s32(acc0, vld1q_s8(a_row + k), vld1q_s8(b_col + k));
                acc1 = vdotq_s32(acc1, vld1q_s8(a_row + k + 16), vld1q_s8(b_col + k + 16));
            }
            acc0 = vaddq_s32(acc0, acc1);
            if (k < k_full) {
                acc0 = vdotq_s32(acc0, vld1q_s8(a_row + k), vld1q_s8(b_col + k));
            }
            if (rem > 0) {
                acc0 = vdotq_s32(acc0, va_tail, vld1q_s8(b_col + k_full));
            }
            c_row[j] = vaddvq_s32(acc0);
        }
    }
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
static inline int8x16_t load_tile_a_gemm_padded(
    const int8_t *A,
    int M,
    int K,
    int i,
    int k_off,
    int k_len
) {
    int8_t tile_a[16];
    vst1q_s8(tile_a, vdupq_n_s8(0));
    const int8_t *row0 = A + (size_t)i * (size_t)K + (size_t)k_off;
    for (int t = 0; t < k_len; ++t) {
        tile_a[t] = row0[t];
    }
    if (i + 1 < M) {
        const int8_t *row1 = A + (size_t)(i + 1) * (size_t)K + (size_t)k_off;
        for (int t = 0; t < k_len; ++t) {
            tile_a[8 + t] = row1[t];
        }
    }
    return vld1q_s8(tile_a);
}

static inline int8x16_t load_tile_b_matvec_padded(const int8_t *B, int k_off, int k_len) {
    int8_t tile_b[16];
    vst1q_s8(tile_b, vdupq_n_s8(0));
    for (int t = 0; t < k_len; ++t) {
        tile_b[t] = B[k_off + t];
    }
    return vld1q_s8(tile_b);
}

static void int8_gemm_i8mm_matvec_worker(void *vctx, int64_t pair_start, int64_t pair_end, int worker_id) {
    (void)worker_id;
    const int8_gemm_ctx_t *ctx = (const int8_gemm_ctx_t *)vctx;
    const int M = ctx->M;
    const int K = ctx->K;
    const int8_t *A = ctx->A;
    const int8_t *B = ctx->B;
    const int8x8_t zero8 = vdup_n_s8(0);

    for (int64_t p = pair_start; p < pair_end; ++p) {
        const int i = (int)(p * 2);
        const int have_row1 = (i + 1 < M);
        int32x4_t acc = vdupq_n_s32(0);
        int k = 0;
        if (have_row1) {
            const int8_t *r0 = A + (size_t)i * (size_t)K;
            const int8_t *r1 = A + (size_t)(i + 1) * (size_t)K;
            for (; k + 8 <= K; k += 8) {
                const int8x16_t va = vcombine_s8(vld1_s8(r0 + k), vld1_s8(r1 + k));
                const int8x16_t vb = vcombine_s8(vld1_s8(B + k), zero8);
                acc = vmmlaq_s32(acc, va, vb);
            }
        }
        for (; k < K; k += 8) {
            const int rem = (K - k) < 8 ? (K - k) : 8;
            const int8x16_t va = load_tile_a_gemm_padded(A, M, K, i, k, rem);
            const int8x16_t vb = load_tile_b_matvec_padded(B, k, rem);
            acc = vmmlaq_s32(acc, va, vb);
        }
        ctx->C[i] = vgetq_lane_s32(acc, 0);
        if (have_row1) {
            ctx->C[i + 1] = vgetq_lane_s32(acc, 2);
        }
    }
}

static void int8_gemm_i8mm_bt_worker(void *vctx, int64_t pair_start, int64_t pair_end, int worker_id) {
    (void)worker_id;
    const int8_gemm_ctx_t *ctx = (const int8_gemm_ctx_t *)vctx;
    const int M = ctx->M;
    const int N = ctx->N;
    const int K = ctx->K;
    const int K_pad = ctx->K_pad;
    const int8_t *A = ctx->A;
    const int8_t *B_T = ctx->B_T;
    const int k_full8 = K & ~7;
    const int rem = K - k_full8;

    for (int64_t p = pair_start; p < pair_end; ++p) {
        const int i = (int)(p * 2);
        const int have_row1 = (i + 1 < M);
        const int8_t *r0 = A + (size_t)i * (size_t)K;
        const int8_t *r1 = have_row1 ? (A + (size_t)(i + 1) * (size_t)K) : NULL;
        int32_t *c0 = ctx->C + (size_t)i * (size_t)N;
        int32_t *c1 = have_row1 ? (ctx->C + (size_t)(i + 1) * (size_t)N) : NULL;

        int8x16_t va_tail = vdupq_n_s8(0);
        if (rem > 0) {
            va_tail = load_tile_a_gemm_padded(A, M, K, i, k_full8, rem);
        }

        for (int j = 0; j < N; j += 2) {
            const int have_col1 = (j + 1 < N);
            const int8_t *b0 = B_T + (size_t)j * (size_t)K_pad;
            const int8_t *b1 = B_T + (size_t)(j + 1) * (size_t)K_pad;
            int32x4_t acc = vdupq_n_s32(0);

            int k = 0;
            if (have_row1) {
                for (; k + 8 <= k_full8; k += 8) {
                    const int8x16_t va = vcombine_s8(vld1_s8(r0 + k), vld1_s8(r1 + k));
                    const int8x16_t vb = vcombine_s8(vld1_s8(b0 + k), vld1_s8(b1 + k));
                    acc = vmmlaq_s32(acc, va, vb);
                }
            } else {
                for (; k + 8 <= k_full8; k += 8) {
                    const int8x16_t va = load_tile_a_gemm_padded(A, M, K, i, k, 8);
                    const int8x16_t vb = vcombine_s8(vld1_s8(b0 + k), vld1_s8(b1 + k));
                    acc = vmmlaq_s32(acc, va, vb);
                }
            }
            if (rem > 0) {
                const int8x16_t vb = vcombine_s8(vld1_s8(b0 + k_full8), vld1_s8(b1 + k_full8));
                acc = vmmlaq_s32(acc, va_tail, vb);
            }

            c0[j] = vgetq_lane_s32(acc, 0);
            if (have_col1) {
                c0[j + 1] = vgetq_lane_s32(acc, 1);
            }
            if (have_row1) {
                c1[j] = vgetq_lane_s32(acc, 2);
                if (have_col1) {
                    c1[j + 1] = vgetq_lane_s32(acc, 3);
                }
            }
        }
    }
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_DOTPROD)
static int8_t *alloc_padded_transpose_b(const int8_t *B, int K, int N, int *out_k_pad) {
    const int n_pad = (N + 1) & ~1;
    const int k_pad = ((K + 15) / 16) * 16;
    int8_t *B_T = (int8_t *)calloc((size_t)n_pad * (size_t)k_pad, sizeof(int8_t));
    if (!B_T) {
        return NULL;
    }
    for (int k = 0; k < K; ++k) {
        const int8_t *b_row = B + (size_t)k * (size_t)N;
        for (int j = 0; j < N; ++j) {
            B_T[(size_t)j * (size_t)k_pad + (size_t)k] = b_row[j];
        }
    }
    *out_k_pad = k_pad;
    return B_T;
}
#endif

int hadukenn_int8_gemm(
    const int8_t *A,
    const int8_t *B,
    int32_t *C,
    int M,
    int N,
    int K,
    int mode
) {
    if (M < 0 || N < 0 || K < 0) {
        return -1;
    }

    const uint32_t caps = hadukenn_oryon_caps();
    const int can_i8mm = (caps & HADUKENN_CAP_I8MM) != 0u;
    const int can_sdot = (caps & HADUKENN_CAP_DOTPROD) != 0u;

    int chosen = mode;
    if (mode == 0) {
        if (can_i8mm) {
            chosen = 1;
        } else if (can_sdot) {
            chosen = 2;
        } else {
            chosen = 3;
        }
    } else if (mode == 1) {
        if (!can_i8mm) {
            return -1;
        }
    } else if (mode == 2) {
        if (!can_sdot) {
            return -1;
        }
    } else if (mode != 3) {
        return -1;
    }
    (void)chosen;

    if (M == 0 || N == 0) {
        return 0;
    }
    if (K == 0) {
        if (!C) {
            return -1;
        }
        memset(C, 0, (size_t)M * (size_t)N * sizeof(int32_t));
        return 0;
    }
    if (!A || !B || !C) {
        return -1;
    }

    const int64_t work_ops = (int64_t)M * (int64_t)N * (int64_t)K;

    int8_gemm_ctx_t ctx = {
        .A = A,
        .B = B,
        .B_T = NULL,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .K_pad = 0,
    };

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (chosen == 1) {
        const int64_t n_pairs = ((int64_t)M + 1) / 2;
        const int64_t thresh = (work_ops >= 32768 && n_pairs >= 4) ? 4 : (n_pairs + 1);
        if (N == 1) {
            pool_dispatch(int8_gemm_i8mm_matvec_worker, &ctx, n_pairs, 4, thresh);
            return 0;
        }
        int k_pad = 0;
        int8_t *B_T = alloc_padded_transpose_b(B, K, N, &k_pad);
        if (B_T) {
            ctx.B_T = B_T;
            ctx.K_pad = k_pad;
            pool_dispatch(int8_gemm_i8mm_bt_worker, &ctx, n_pairs, 2, thresh);
            free(B_T);
            return 0;
        }
        chosen = 3;
    }
#endif

#if defined(__ARM_FEATURE_DOTPROD)
    if (chosen == 2) {
        const int64_t thresh = (work_ops >= 32768 && M >= 8) ? 8 : ((int64_t)M + 1);
        if (N == 1) {
            pool_dispatch(int8_gemm_sdot_matvec_worker, &ctx, (int64_t)M, 8, thresh);
            return 0;
        }
        int k_pad = 0;
        int8_t *B_T = alloc_padded_transpose_b(B, K, N, &k_pad);
        if (B_T) {
            ctx.B_T = B_T;
            ctx.K_pad = k_pad;
            pool_dispatch(int8_gemm_sdot_bt_worker, &ctx, (int64_t)M, 4, thresh);
            free(B_T);
            return 0;
        }
        chosen = 3;
    }
#endif

    const int64_t scalar_thresh = (work_ops >= 32768 && M >= 8) ? 8 : ((int64_t)M + 1);
    pool_dispatch(int8_gemm_scalar_worker, &ctx, (int64_t)M, 4, scalar_thresh);
    return 0;
}

/* =========================================================================
 * Fused NF4 Dequantize + MatVec Kernel (F8)
 * ========================================================================= */

typedef struct {
    const float *x_f32;
    const uint8_t *qweights_u8;
    const float *scales_f32;
    float *out_f32;
    float lut[16];
    int M;
    int K;
    int group_size;
    int packed_cols;
    int n_groups;
} nf4_matvec_ctx_t;

static void nf4_matvec_worker(void *vctx, int64_t m_start, int64_t m_end, int worker_id) {
    (void)worker_id;
    const nf4_matvec_ctx_t *ctx = (const nf4_matvec_ctx_t *)vctx;
    const float *x = ctx->x_f32;
    const int K = ctx->K;
    const int gs = ctx->group_size;
    const int packed_cols = ctx->packed_cols;
    const int n_groups = ctx->n_groups;
    const float *lut = ctx->lut;

#if defined(__ARM_NEON) || defined(__aarch64__)
    const uint8_t *lut_bytes = (const uint8_t *)lut;
    uint8x16x4_t lut_tbl;
    lut_tbl.val[0] = vld1q_u8(lut_bytes + 0);
    lut_tbl.val[1] = vld1q_u8(lut_bytes + 16);
    lut_tbl.val[2] = vld1q_u8(lut_bytes + 32);
    lut_tbl.val[3] = vld1q_u8(lut_bytes + 48);
    const uint8x8_t mask0f = vdup_n_u8(0x0F);
    const uint32x4_t mul04 = vdupq_n_u32(0x04040404u);
    const uint32x4_t off03 = vdupq_n_u32(0x03020100u);
    const uint8x16_t widen_mask = {
        0, 0xFF, 0xFF, 0xFF,
        1, 0xFF, 0xFF, 0xFF,
        2, 0xFF, 0xFF, 0xFF,
        3, 0xFF, 0xFF, 0xFF
    };
    const uint8x16_t widen_mask_hi = {
        4, 0xFF, 0xFF, 0xFF,
        5, 0xFF, 0xFF, 0xFF,
        6, 0xFF, 0xFF, 0xFF,
        7, 0xFF, 0xFF, 0xFF
    };
#endif

    for (int64_t m = m_start; m < m_end; ++m) {
        const uint8_t *qw_row = ctx->qweights_u8 + (size_t)m * (size_t)packed_cols;
        const float *sc_row = ctx->scales_f32 + (size_t)m * (size_t)n_groups;
        float row_acc = 0.0f;

        for (int g = 0; g < n_groups; ++g) {
            const int g_start = g * gs;
            int g_end = g_start + gs;
            if (g_end > K) {
                g_end = K;
            }
            int k = g_start;
            float g_acc = 0.0f;

            if ((k & 1) != 0 && k < g_end) {
                const uint8_t b = qw_row[k >> 1];
                g_acc += lut[b & 0x0Fu] * x[k];
                k += 1;
            }

#if defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);
            float32x4_t vacc2 = vdupq_n_f32(0.0f);
            float32x4_t vacc3 = vdupq_n_f32(0.0f);

            for (; k + 16 <= g_end; k += 16) {
                const uint8x8_t p8 = vld1_u8(qw_row + (k >> 1));
                const uint8x8_t hi = vshr_n_u8(p8, 4);
                const uint8x8_t lo = vand_u8(p8, mask0f);
                const uint8x8x2_t zz = vzip_u8(hi, lo);

                const uint8x16_t c_low8 = vcombine_u8(zz.val[0], zz.val[0]);
                const uint8x16_t c_hi8 = vcombine_u8(zz.val[1], zz.val[1]);

                const uint32x4_t idx0 = vaddq_u32(
                    vmulq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(c_low8, widen_mask)), mul04),
                    off03
                );
                const uint32x4_t idx1 = vaddq_u32(
                    vmulq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(c_low8, widen_mask_hi)), mul04),
                    off03
                );
                const uint32x4_t idx2 = vaddq_u32(
                    vmulq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(c_hi8, widen_mask)), mul04),
                    off03
                );
                const uint32x4_t idx3 = vaddq_u32(
                    vmulq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(c_hi8, widen_mask_hi)), mul04),
                    off03
                );

                const float32x4_t w0 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_tbl, vreinterpretq_u8_u32(idx0)));
                const float32x4_t w1 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_tbl, vreinterpretq_u8_u32(idx1)));
                const float32x4_t w2 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_tbl, vreinterpretq_u8_u32(idx2)));
                const float32x4_t w3 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_tbl, vreinterpretq_u8_u32(idx3)));

                vacc0 = vfmaq_f32(vacc0, w0, vld1q_f32(x + k));
                vacc1 = vfmaq_f32(vacc1, w1, vld1q_f32(x + k + 4));
                vacc2 = vfmaq_f32(vacc2, w2, vld1q_f32(x + k + 8));
                vacc3 = vfmaq_f32(vacc3, w3, vld1q_f32(x + k + 12));
            }
            vacc0 = vaddq_f32(vaddq_f32(vacc0, vacc1), vaddq_f32(vacc2, vacc3));
            g_acc += vaddvq_f32(vacc0);
#endif

            for (; k + 1 < g_end; k += 2) {
                const uint8_t b = qw_row[k >> 1];
                g_acc += lut[(b >> 4) & 0x0Fu] * x[k];
                g_acc += lut[b & 0x0Fu] * x[k + 1];
            }
            if (k < g_end) {
                const uint8_t b = qw_row[k >> 1];
                const uint8_t code = ((k & 1) != 0) ? (b & 0x0Fu) : ((b >> 4) & 0x0Fu);
                g_acc += lut[code] * x[k];
            }
            row_acc += sc_row[g] * g_acc;
        }
        ctx->out_f32[m] = row_acc;
    }
}

void hadukenn_nf4_matvec_f32(
    const float *x_f32,
    const uint8_t *qweights_u8,
    const float *scales_f32,
    const float *lut_f32,
    float *out_f32,
    int M,
    int K,
    int group_size
) {
    if (M <= 0 || K < 0 || group_size <= 0 || !out_f32) {
        return;
    }
    if (K == 0) {
        memset(out_f32, 0, (size_t)M * sizeof(float));
        return;
    }
    if (!x_f32 || !qweights_u8 || !scales_f32 || !lut_f32) {
        return;
    }
    nf4_matvec_ctx_t ctx;
    ctx.x_f32 = x_f32;
    ctx.qweights_u8 = qweights_u8;
    ctx.scales_f32 = scales_f32;
    ctx.out_f32 = out_f32;
    memcpy(ctx.lut, lut_f32, 16 * sizeof(float));
    ctx.M = M;
    ctx.K = K;
    ctx.group_size = group_size;
    ctx.packed_cols = (K + 1) / 2;
    ctx.n_groups = (K + group_size - 1) / group_size;

    const int64_t work_ops = (int64_t)M * (int64_t)K;
    const int64_t thresh = (work_ops >= 16384 && M >= 4) ? 4 : ((int64_t)M + 1);
    pool_dispatch(nf4_matvec_worker, &ctx, (int64_t)M, 4, thresh);
}
