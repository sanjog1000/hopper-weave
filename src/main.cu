#include<iostream>
#include <vector>
#include<string>
#include<cuda_runtime.h>
#include<hopperkv/cuda_check.cuh>
#include<hopperkv/cuda_buffer.cuh>
#include<hopperkv/tensor.hpp>

__global__ void add_one(float* data, std::size_t n)
{
    std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i < n) {
        data[i] += 1.0f;
    }
}


int main()
{
    // A contiguous 2D tensor with 2 rows and 4 columns.
    hopperkv::Tensor<float> x({2, 4});

    std::vector<float> host(x.numel());

    // Initialize CPU memory with 0, 1, 2, ..., 7.
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<float>(i);
    }

    // CPU -> GPU
    x.copy_from_host(host.data(), host.size());

    // Launch a GPU kernel over the tensor's linear storage.
    constexpr int THREADS = 256;

    const int blocks = static_cast<int>(
        (x.numel() + THREADS - 1) / THREADS
    );

    add_one<<<blocks, THREADS>>>(x.data(), x.numel());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU -> CPU
    x.copy_to_host(host.data(), host.size());

    // Display tensor metadata.
    std::cout << "Shape: [";

    for (std::size_t i = 0; i < x.shape().size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }

        std::cout << x.shape()[i];
    }

    std::cout << "]\nStrides: [";

    for (std::size_t i = 0; i < x.strides().size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }

        std::cout << x.strides()[i];
    }

    std::cout << "]\nValues: ";

    for (float value : host) {
        std::cout << value << ' ';
    }

    std::cout << '\n';

    // Validate the result.
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (host[i] != static_cast<float>(i + 1)) {
            std::cerr << "Tensor verification failed.\n";
            return 1;
        }
    }

    std::cout << "Tensor verification passed.\n";

    return 0;
}