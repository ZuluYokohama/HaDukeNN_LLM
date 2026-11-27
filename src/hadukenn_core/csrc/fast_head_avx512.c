/*
 * Clean-room C11 + OpenMP Multi-ISA (AVX-512 / AVX2+FMA3 / ARM64 NEON)
 * cluster-blocked & L2-tiled Cut-Cross-Entropy kernel for HaDukeNN.
 *
 * Upgraded in v0.9.0 so WBeast (i9-9900K AVX2+FMA3), LapBeast (275HX AVX2+FMA3),
 * Zenbook (Snapdragon X Oryon ARM64 NEON), and Xeon/Zen4+ (AVX-512F) all hit
 * register-unrolled vector FMA pipes rather than dropping to scalar C.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

static inline float dot_f32(const float *a, const float *b, int d) {
    int i = 0;
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    for (; i + 32 <= d; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }
    float sum = _mm512_reduce_add_ps(acc0);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= d; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum = _mm_cvtss_f32(sums);
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= d; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
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

static inline void axpy_f32(float alpha, const float *x, float *y, int d) {
    int i = 0;
#if defined(__AVX512F__)
    __m512 va = _mm512_set1_ps(alpha);
    for (; i + 16 <= d; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8 <= d; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= d; i += 4) {
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    }
#endif
    for (; i < d; ++i) {
        y[i] += alpha * x[i];
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
    const int active_cols = (1 + K_neg) * S;
    #pragma omp parallel if(N >= 128 && !compute_grad)
    {
        float root_logits[4096];
        float active_logits[2048];
        int32_t active_c[16];

        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float *h = H + (size_t)n * d;
            const int32_t tc = target_c[n];
            const int32_t tm = target_m[n];

            float max_r = -1e30f;
            for (int c = 0; c < NC; ++c) {
                float r = dot_f32(h, C_root + (size_t)c * d, d);
                root_logits[c] = r;
                if (r > max_r) max_r = r;
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
                active_c[1 + k] = neg_clusters[(size_t)n * K_neg + k];
            }

            float max_o = -1e30f;
            for (int blk = 0; blk < 1 + K_neg; ++blk) {
                const int32_t c_idx = active_c[blk];
                const float *E_blk = E_clustered + ((size_t)c_idx * S) * d;
                float log_w = 0.0f;
                if (blk > 0) {
                    float q_tc = root_logits[tc] * inv_sum_r;
                    float q_c = root_logits[c_idx] * inv_sum_r;
                    float q_cond = q_c / fmaxf(1.0f - q_tc, 1e-7f);
                    float denom = fmaxf((float)K_neg * q_cond, 1e-7f);
                    log_w = -logf(denom);
                }
                for (int m = 0; m < S; ++m) {
                    float o = dot_f32(h, E_blk + (size_t)m * d, d) + log_w;
                    active_logits[blk * S + m] = o;
                    if (o > max_o) max_o = o;
                }
            }

            float sum_o = 0.0f;
            for (int j = 0; j < active_cols; ++j) {
                float p = expf(active_logits[j] - max_o);
                active_logits[j] = p;
                sum_o += p;
            }
            const float target_logit = active_logits[tm];
            const float nll = -logf(fmaxf(target_logit / sum_o, 1e-30f));
            out_loss[n] = nll;

            if (compute_grad) {
                float *dh = dH + (size_t)n * d;
                const float inv_sum_o = 1.0f / (sum_o * (float)N);
                const float inv_N = 1.0f / (float)N;
                for (int blk = 0; blk < 1 + K_neg; ++blk) {
                    const int32_t c_idx = active_c[blk];
                    const float *E_blk = E_clustered + ((size_t)c_idx * S) * d;
                    float *dE_blk = dE_clustered + ((size_t)c_idx * S) * d;
                    for (int m = 0; m < S; ++m) {
                        float coeff = active_logits[blk * S + m] * inv_sum_o;
                        if (blk == 0 && m == tm) {
                            coeff -= inv_N;
                        }
                        axpy_f32(coeff, E_blk + (size_t)m * d, dh, d);
                        axpy_f32(coeff, h, dE_blk + (size_t)m * d, d);
                    }
                }
            }
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
    #pragma omp parallel if(N >= 128 && !compute_grad)
    {
        float tile_buf[1024];
        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float *h = H + (size_t)n * d;
            const int32_t y = targets[n];
            float m_run = -1e30f;
            float s_run = 0.0f;
            float z_y = dot_f32(h, E + (size_t)y * d, d);

            for (int v0 = 0; v0 < V; v0 += tile_V) {
                int v1 = v0 + tile_V;
                if (v1 > V) v1 = V;
                float m_tile = -1e30f;
                for (int v = v0; v < v1; ++v) {
                    float z = dot_f32(h, E + (size_t)v * d, d);
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
            float lse = m_run + logf(s_run);
            out_loss[n] = lse - z_y;

            if (compute_grad) {
                float *dh = dH + (size_t)n * d;
                const float inv_N = 1.0f / (float)N;
                for (int v0 = 0; v0 < V; v0 += tile_V) {
                    int v1 = v0 + tile_V;
                    if (v1 > V) v1 = V;
                    for (int v = v0; v < v1; ++v) {
                        float z = dot_f32(h, E + (size_t)v * d, d);
                        float p = expf(z - lse) * inv_N;
                        if (v == y) p -= inv_N;
                        axpy_f32(p, E + (size_t)v * d, dh, d);
                        axpy_f32(p, h, dE + (size_t)v * d, d);
                    }
                }
            }
        }
    }
}
