/*
 * HaDukeNN OpenCL FP16 Co-Processor Kernels for Zenbook (Qualcomm Adreno X1-45 iGPU)
 *
 * Designed for unified LPDDR5x-8448 zero-copy buffers (CL_MEM_USE_HOST_PTR):
 * While Hexagon V73 HTP executes the persistent looped INT8 linear projections at ~14.5 W,
 * Adreno X1-45 executes FP16 RMSNorm + RoPE + Sheaf-Coboundary Softmax reductions in place
 * without PCIe host-device copies.
 */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hadukenn_adreno_rmsnorm_rope_fp16(
    __global const half *restrict x_in,     /* [N, d] */
    __global const half *restrict weight,   /* [d] */
    __global const half *restrict cos_tbl,  /* [N, d/2] */
    __global const half *restrict sin_tbl,  /* [N, d/2] */
    __global half *restrict x_out,          /* [N, d] */
    const int d,
    const float eps,
    __local float *scratch
) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    __global const half *xr = x_in + (size_t)row * d;
    __global half *yr = x_out + (size_t)row * d;

    float sq_sum = 0.0f;
    for (int i = lid; i < d; i += lsz) {
        float v = vload_half(i, xr);
        sq_sum = fma(v, v, sq_sum);
    }
    scratch[lid] = sq_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_rms = rsqrt(scratch[0] / (float)d + eps);

    const int half_d = d >> 1;
    __global const half *cr = cos_tbl + (size_t)row * half_d;
    __global const half *sr = sin_tbl + (size_t)row * half_d;

    for (int j = lid; j < half_d; j += lsz) {
        float x0 = vload_half(2 * j,     xr) * inv_rms * vload_half(2 * j,     weight);
        float x1 = vload_half(2 * j + 1, xr) * inv_rms * vload_half(2 * j + 1, weight);
        float c  = vload_half(j, cr);
        float s  = vload_half(j, sr);
        vstore_half_rte(x0 * c - x1 * s, 2 * j,     yr);
        vstore_half_rte(x0 * s + x1 * c, 2 * j + 1, yr);
    }
}
