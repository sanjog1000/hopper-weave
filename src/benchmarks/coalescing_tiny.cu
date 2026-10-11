#include<iostream>
#include<vector>
#include<cstddef>
#include <cuda_runtime.h>

#include "hopperkv/cuda_buffer.cuh"

constexpr std::size_t N= 1024;
constexpr std::size_t STRIDE = 32;

// Kernel A: COALESCED ACCESS
__global__ void coalesced_kernel(const float* input , float* output , std::size_t n){
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i];
    }
}
// Kernel B: STRIDED ACCESS
__global__ void strided_kernel(const float* input , float* output , std::size_t n){
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i*STRIDE];
    }
}
int main(){
    // Enough elements for the largest strided index.
    constexpr std::size_t INPUT_COUNT = (N - 1) * STRIDE + 1;

    std::vector<float> host_input(INPUT_COUNT);
    for (std::size_t i = 0; i < INPUT_COUNT; ++i) {
        host_input[i] = static_cast<float>(i);
    }

    hopperkv::CudaBuffer<float> device_input(INPUT_COUNT);
    hopperkv::CudaBuffer<float> device_output(N);

    device_input.copy_from_host(host_input.data(), INPUT_COUNT);

    constexpr int THREADS = 256;
    const int BLOCKS = static_cast<int>((N + THREADS - 1) / THREADS);

    // A: adjacent threads access adjacent input elements.
    coalesced_kernel<<<BLOCKS, THREADS>>>(device_input.data(),device_output.data(),N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(N);

    device_output.copy_to_host(result.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        if (result[i] != host_input[i]) {
            std::cerr << "Coalesced kernel failed.\n";
            return 1;
        }
    }

    std::cout << "Coalesced kernel: passed\n";

    // B: adjacent threads access elements 32 apart.
    strided_kernel<<<BLOCKS, THREADS>>>(device_input.data(),device_output.data(),N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_output.copy_to_host(result.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        if (result[i] != host_input[i * STRIDE]) {
            std::cerr << "Strided kernel failed.\n";
            return 1;
        }
    }

    std::cout << "Strided kernel: passed\n\n";
    std::cout << "First 8 thread accesses:\n";

    for (std::size_t i = 0; i < 8; ++i) {
        std::cout<< "Thread " << i<< ": input[" << i << "]"<< " versus input[" << i * STRIDE << "]\n";
    }

    return 0;
}