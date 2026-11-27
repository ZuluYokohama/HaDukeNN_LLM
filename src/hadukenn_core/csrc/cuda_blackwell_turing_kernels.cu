/*
 * HaDukeNN Multi-Arch CUDA Kernels (CUDA 12.8 / 13.1)
 *
 * Targets:
 *   - sm_120 (Blackwell): WBeast GPU 0 (RTX 5070 12GB, 48 MiB L2) & LapBeast GPU 0 (RTX 5060 8GB, 32 MiB L2)
 *     -> Warp-reduced vectorized float4 / BF16 L2-persistent muP Cut-Cross-Entropy (zero [N, V] VRAM logits)
 *   - sm_75  (Turing)   : WBeast GPU 1 (GTX 1660 Ti 6GB side-lane)
 *     -> Asynchronous CSR-masked Hyperedge-Scoped E2 kNN-LM Retrieval & Log-Sum-Exp Kernel
 *
 * Compile on WBeast (dual-GPU sm_120 + sm_75):
 *   nvcc -O3 --use_fast_math -std=c++17 \
 *        -gencode arch=compute_120,code=sm_120 \
 *        -gencode arch=compute_75,code=sm_75 \
 *        -Xcompiler -fPIC -shared cuda_blackwell_turing_kernels.cu -o hadukenn_cuda_kernels.dll
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float vec4_dot_f32(const float *h, const float *e, int d) {
    float acc = 0.0f;
    const int d4 = d >> 2;
    const float4 *h4 = reinterpret_cast<const float4 *>(h);
    const float4 *e4 = reinterpret_cast<const float4 *>(e);
    for (int i = threadIdx.x; i < d4; i += 32) {
        float4 vh = __ldg(&h4[i]);
        float4 ve = __ldg(&e4[i]);
        acc = fmaf(vh.x, ve.x, acc);
        acc = fmaf(vh.y, ve.y, acc);
        acc = fmaf(vh.z, ve.z, acc);
        acc = fmaf(vh.w, ve.w, acc);
    }
    for (int i = (d4 << 2) + threadIdx.x; i < d; i += 32) {
        acc = fmaf(__ldg(&h[i]), __ldg(&e[i]), acc);
    }
    return warp_reduce_sum(acc);
}

}  // namespace

extern "C" {

/*
 * Kernel 1 (sm_120 Blackwell WBeast RTX 5070 / LapBeast RTX 5060):
 * Warp-per-token L2-tiled Cut-Cross-Entropy forward NLL + optional dH backward.
 * Keeps peak auxiliary VRAM at O(N) floats instead of N * V * 4 bytes.
 */
__global__ void hadukenn_cuda_mup_cce_fwd_bwd_f32(
    const float *__restrict__ H,
    const float *__restrict__ E,
    const int32_t *__restrict__ targets,
    float *__restrict__ out_loss,
    float *__restrict__ dH,
    int N,
    int d,
    int V,
    float logit_scale,
    int compute_dH
) {
    /* Grid: 1 warp (blockDim.x == 32, blockDim.y == warps_per_block) per token n */
    const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (warp_id >= N) return;

    const float *h = H + static_cast<size_t>(warp_id) * d;
    const int32_t y = targets[warp_id];
    const float z_y = logit_scale * vec4_dot_f32(h, E + static_cast<size_t>(y) * d, d);

    float m_run = -CUDART_INF_F;
    float s_run = 0.0f;
    for (int v = 0; v < V; ++v) {
        float z = logit_scale * vec4_dot_f32(h, E + static_cast<size_t>(v) * d, d);
        float m_new = fmaxf(m_run, z);
        s_run = s_run * __expf(m_run - m_new) + __expf(z - m_new);
        m_run = m_new;
    }
    const float lse = m_run + __logf(s_run);
    if (threadIdx.x == 0) {
        out_loss[warp_id] = lse - z_y;
    }

    if (compute_dH && dH != nullptr) {
        float *dh = dH + static_cast<size_t>(warp_id) * d;
        const float inv_N = logit_scale / static_cast<float>(N);
        for (int col = threadIdx.x; col < d; col += 32) {
            dh[col] = 0.0f;
        }
        for (int v = 0; v < V; ++v) {
            float z = logit_scale * vec4_dot_f32(h, E + static_cast<size_t>(v) * d, d);
            float coeff = __expf(z - lse) * inv_N;
            if (v == y) coeff -= inv_N;
            const float *ev = E + static_cast<size_t>(v) * d;
            for (int col = threadIdx.x; col < d; col += 32) {
                dh[col] = fmaf(coeff, __ldg(&ev[col]), dh[col]);
            }
        }
    }
}

/*
 * Kernel 2 (sm_75 Turing WBeast GPU 1 GTX 1660 Ti Side-Lane):
 * Hyperedge-Scoped E2 kNN-LM Retrieval over CSR candidate lists.
 * Evaluates negative squared L2 distance only against datastore entries sharing
 * a causal wt103-v3 hyperedge with query q, accumulating log-probability mass
 * on the ground-truth target token.
 */
__global__ void hadukenn_cuda_scoped_knnlm_eval_f32(
    const float *__restrict__ queries,         /* [Q, d] */
    const int32_t *__restrict__ query_targets, /* [Q] */
    const int32_t *__restrict__ csr_indptr,    /* [Q + 1] */
    const int32_t *__restrict__ csr_indices,   /* [nnz] datastore slot IDs */
    const float *__restrict__ ds_keys,         /* [M_ds, d] */
    const int32_t *__restrict__ ds_tokens,     /* [M_ds] */
    float *__restrict__ out_knn_logp,          /* [Q] log P_knn(y | q) */
    int Q,
    int d,
    float inv_temp
) {
    const int q = blockIdx.x * blockDim.y + threadIdx.y;
    if (q >= Q) return;

    const int start = csr_indptr[q];
    const int end   = csr_indptr[q + 1];
    const int32_t target_tok = query_targets[q];
    const float *qv = queries + static_cast<size_t>(q) * d;

    if (start >= end) {
        if (threadIdx.x == 0) out_knn_logp[q] = -30.0f;
        return;
    }

    float max_all = -CUDART_INF_F;
    float sum_all = 0.0f;
    float max_tgt = -CUDART_INF_F;
    float sum_tgt = 0.0f;

    for (int idx = start; idx < end; ++idx) {
        const int ds_id = csr_indices[idx];
        const float *kv = ds_keys + static_cast<size_t>(ds_id) * d;
        float dist2 = 0.0f;
        for (int col = threadIdx.x; col < d; col += 32) {
            float diff = qv[col] - __ldg(&kv[col]);
            dist2 = fmaf(diff, diff, dist2);
        }
        dist2 = warp_reduce_sum(dist2);
        const float score = -dist2 * inv_temp;

        const float m_new = fmaxf(max_all, score);
        sum_all = sum_all * __expf(max_all - m_new) + __expf(score - m_new);
        max_all = m_new;

        if (ds_tokens[ds_id] == target_tok) {
            const float mt_new = fmaxf(max_tgt, score);
            sum_tgt = sum_tgt * __expf(max_tgt - mt_new) + __expf(score - mt_new);
            max_tgt = mt_new;
        }
    }

    if (threadIdx.x == 0) {
        const float lse_all = max_all + __logf(fmaxf(sum_all, 1e-30f));
        if (sum_tgt <= 0.0f) {
            out_knn_logp[q] = -30.0f;
        } else {
            const float lse_tgt = max_tgt + __logf(sum_tgt);
            out_knn_logp[q] = lse_tgt - lse_all;
        }
    }
}

}  // extern "C"
