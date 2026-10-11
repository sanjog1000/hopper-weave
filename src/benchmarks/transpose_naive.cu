#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <vector>

#include "hopperkv/cuda_check.cuh"
#include "hopperkv/cuda_buffer.cuh"

__global__ void transpose_naive(const float* input,float* output,int rows,int cols){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // Read A[row][col].
        const std::size_t input_offset =static_cast<std::size_t>(row) * cols + col;

        // Write the same value to B[col][row].
        const std::size_t output_offset =static_cast<std::size_t>(col) * rows + row;

        output[output_offset] = input[input_offset];
    }
}

constexpr int TILE_DIM = 32;
__global__ void transpose_tiled(const float* input,float* output,int rows,int cols){
    // shared memory title for 32 * 33
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;

    if(row <rows && col <cols){
        tile[threadIdx.y][threadIdx.x] = input[static_cast<std::size_t>(row)*cols + col];
    }

    __syncthreads();

    // Exchange the roles of the tile coordinates.
    const int output_row = blockIdx.x * TILE_DIM + threadIdx.y;
    const int output_col = blockIdx.y * TILE_DIM + threadIdx.x;

    if(output_row < cols && output_col < rows){
        output[static_cast<std::size_t>(output_row)*rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}
int main(){
    constexpr int ROWS = 64;
    constexpr int COLS = 64;

    const std::size_t count = static_cast<std::size_t>(ROWS) * COLS;

    std::vector<float> input(count);
    std::vector<float> output(count, -1.0f);

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            input[static_cast<std::size_t>(r) * COLS + c] = static_cast<float>(r * COLS + c);
        }
    }

    hopperkv::CudaBuffer<float> d_input(count);
    hopperkv::CudaBuffer<float> d_output(count);

    d_input.copy_from_host(input.data(), count);

    const dim3 block(32, 8);
    const dim3 grid(
        (COLS + block.x - 1) / block.x,
        (ROWS + block.y - 1) / block.y
    );

    transpose_naive<<<grid, block>>>(d_input.data(),d_output.data(),ROWS,COLS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    d_output.copy_to_host(output.data(), count);

    // Verify every transposed element.
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            const float expected = input[static_cast<std::size_t>(r) * COLS + c];
            const float actual = output[static_cast<std::size_t>(c) * ROWS + r];

            if (actual != expected) {
                std::cerr << "Transpose verification failed.\n";
                return 1;
            }
        }
    }

    std::cout << "Transpose verification passed.\n\n";
    const dim3 tiled_block(TILE_DIM, TILE_DIM);

    const dim3 tiled_grid((COLS + TILE_DIM - 1) / TILE_DIM,(ROWS + TILE_DIM - 1) / TILE_DIM);

    transpose_tiled<<<tiled_grid, tiled_block>>>(d_input.data(),d_output.data(),ROWS,COLS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    d_output.copy_to_host(output.data(), count);

    if (!validate_transpose(
            input, output, ROWS, COLS, "Tiled transpose")) {
        return 1;
    }

    std::cout << "Original matrix (top-left 4x4):\n";
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            std::cout << input[r * COLS + c] << '\t';
        }
        std::cout << '\n';
    }

    std::cout << "\nTransposed matrix (top-left 4x4):\n";
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            std::cout << output[r * ROWS + c] << '\t';
        }
        std::cout << '\n';
    }
    return 0;
}