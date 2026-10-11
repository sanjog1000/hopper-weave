#include<iostream>
#include<vector>
#include<cmath>
#include<cstddef>

#include "hopperkv/cuda_buffer.cuh"
#include "hopperkv/cuda_check.cuh"

// C[M x N] = A[M x K] * B[K x N]
__global__ void gemm_naive(float* A , float* B ,float* C , int M ,int N ,int K){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int k = 0 ; k < K ; k+){
            const float a = A[static_cast<std::size_t>(row)*K +k];
            const float b = B[static_cast<std::size_t>(k)*N + col];
            sum += a*b;
        }
        C[static_cast<std::size_t>(row)*N + col] = sum;
    }
}
int main(){
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 64;

    const std::size_t size_A = static_cast<std::size_t>(M) * K;
    const std::size_t size_B = static_cast<std::size_t>(K) * N;
    const std::size_t size_C = static_cast<std::size_t>(M) * N;

    std::vector<float> host_A(size_A);
    std::vector<float> host_B(size_B);
    std::vector<float> host_C(size_C);
    std::vector<float> reference_C(size_C);

    for (std::size_t i = 0; i < size_A; ++i) {
        host_A[i] = static_cast<float>(i % 7) * 0.1f;
    }

    for (std::size_t i = 0; i < size_B; ++i) {
        host_B[i] = static_cast<float>(i % 5) * 0.2f;
    }
    for (int row =0; row <M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k =0; k < K; ++k) {
                sum += host_A[static_cast<std::size_t>(row) * K + k] * host_B[static_cast<std::size_t>(k) * N + col];
            }
            reference_C[static_cast<std::size_t>(row) * N + col] = sum;
        }
    }
    hopperkv::CudaBuffer<float>device_A(size_A);
    hopperkv::CudaBuffer<float>device_B(size_B);
    hopperkv::CudaBuffer<float> device_C(size_C);

    device_A.copy_from_host(host_A.data(), size_A);
    device_B.copy_from_host(host_B.data(), size_B);

    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x,(M + block.y - 1) / block.y);

    gemm_naive<<<grid, block>>>(device_A.data(),device_B.data(),device_C.data(),M,N,K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_C.copy_to_host(host_C.data(), size_C);

    constexpr float TOLERANCE = 1e-3f;

    for (std::size_t i =0; i< size_C;++i) {
        if (std::fabs(host_C[i] - reference_C[i]) > TOLERANCE) {
            std::cerr<< "GEMM verification failed at index "<< i<< ": GPU = " << host_C[i]
                << ", CPU = " << reference_C[i]<< '\n';
            return 1;
        }
    }
    std::cout << "Naive GEMM verification passed.\n\n";
    std::cout << "First 2x2 output:\n";

    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            std::cout<< host_C[
                    static_cast<std::size_t>(row) * N + col]<< '\t';
        }
            }

    return 0;
}