/* Host launchers for the sm_89 / Ada build of cuda_blackwell_turing_kernels.cu. */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "../src/hadukenn_core/csrc/cuda_blackwell_turing_kernels.cu"

extern "C" int hadukenn_cuda_mup_cce_run(
    const float *H_h,
    const float *E_h,
    const int32_t *targets_h,
    float *loss_h,
    float *dH_h,
    int N,
    int d,
    int V,
    float logit_scale,
    int compute_dH,
    float *elapsed_ms
) {
    if (N <= 0 || d <= 0 || V <= 0) return 1;
    float *H_d = nullptr, *E_d = nullptr, *loss_d = nullptr, *dH_d = nullptr;
    int32_t *y_d = nullptr;
    cudaError_t err;
    auto fail = [&](int rc) {
        cudaFree(H_d); cudaFree(E_d); cudaFree(y_d); cudaFree(loss_d); cudaFree(dH_d);
        return rc;
    };

    if (cudaMalloc(&H_d, sizeof(float) * (size_t)N * d) != cudaSuccess) return fail(2);
    if (cudaMalloc(&E_d, sizeof(float) * (size_t)V * d) != cudaSuccess) return fail(3);
    if (cudaMalloc(&y_d, sizeof(int32_t) * (size_t)N) != cudaSuccess) return fail(4);
    if (cudaMalloc(&loss_d, sizeof(float) * (size_t)N) != cudaSuccess) return fail(5);
    if (compute_dH) {
        if (cudaMalloc(&dH_d, sizeof(float) * (size_t)N * d) != cudaSuccess) return fail(6);
        cudaMemset(dH_d, 0, sizeof(float) * (size_t)N * d);
    }
    if (cudaMemcpy(H_d, H_h, sizeof(float) * (size_t)N * d, cudaMemcpyHostToDevice) != cudaSuccess) return fail(7);
    if (cudaMemcpy(E_d, E_h, sizeof(float) * (size_t)V * d, cudaMemcpyHostToDevice) != cudaSuccess) return fail(8);
    if (cudaMemcpy(y_d, targets_h, sizeof(int32_t) * (size_t)N, cudaMemcpyHostToDevice) != cudaSuccess) return fail(9);

    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    dim3 grid((N + warps_per_block - 1) / warps_per_block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    hadukenn_cuda_mup_cce_fwd_bwd_f32<<<grid, block>>>(
        H_d, E_d, y_d, loss_d, dH_d, N, d, V, logit_scale, compute_dH
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return fail(10);
    }
    cudaEventRecord(stop);
    if (cudaEventSynchronize(stop) != cudaSuccess) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return fail(11);
    }
    if (elapsed_ms) cudaEventElapsedTime(elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (cudaMemcpy(loss_h, loss_d, sizeof(float) * (size_t)N, cudaMemcpyDeviceToHost) != cudaSuccess) return fail(12);
    if (compute_dH && dH_h) {
        if (cudaMemcpy(dH_h, dH_d, sizeof(float) * (size_t)N * d, cudaMemcpyDeviceToHost) != cudaSuccess) return fail(13);
    }
    return fail(0);
}
