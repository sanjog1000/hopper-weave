#pragma once
#include <cuda_runtime.h>
#include "hopperkv/cuda_check.cuh"

#include<cstddef>
#include <stdexcept>
#include <utility>

namespace hopperkv{

    template <typename T>
    class CudaBuffer
    {
    private:
        T* data = nullptr;
        std::size_t size = 0;
    public:
        CudaBuffer() = default;

        explicit CudaBuffer(std::size_t count){
            allocate(count);
        }

        ~CudaBuffer()
        {
            reset();
        }
        // both copy constructor and copy assignment is blocked
        CudaBuffer(const CudaBuffer&) = delete;
        CudaBuffer& operator=(const CudaBuffer&) =delete;

        // moving is allowed
        CudaBuffer(CudaBuffer&& other)noexcept : data(other.data) , size(other.size){
            other.data = nullptr;
            other.size = 0;
        }
        CudaBuffer& operator=(CudaBuffer&& other)noexcept{
            if(this != &other){
                reset();

                data = other.data;
                size = other.size;
                other.data = nullptr;
                other.size = 0;
            }
            return *this;
        }

        // allocation
        void allocate(std::size_t count){
            reset();

            if(count == 0) return;
            T* ptr = nullptr;

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count*sizeof(T)));

            data = ptr;
            size = count;
        }

        // freeing
        void reset()
        {
            if(data != nullptr){
                cudaFree(data);

                data = nullptr;
                size = 0;
            }
        }

        T* data()noexcept{
            return data;
        }

        const T* data()const noexcept{
            return data;
        }

        std::size_t size()const noexcept{
            return size;
        }

        std::size_t bytes()const noexcept{
            return size * sizeof(T);
        }

        void copy_from_host(T* host_ptr , std::size_t count){
            if(count > size){
                throw std::runtime_error("copy_from_host: source does not fit in buffer");
            }

            CUDA_CHECK(cudaMemcpy(data , host_ptr , count*sizeof(T) , cudaMemcpyHostToDevice));
        }
        void copy_from_host(T* host_ptr , std::size_t count){
            if(count > size){
                throw std::runtime_error("copy_to_host: destination does not fit in buffer");
            }

            CUDA_CHECK(cudaMemcpy(host_ptr ,data ,  count*sizeof(T) , cudaMemcpyDeviceToHost));
        }

        bool empty()const noexcept{
            return size == 0;
        }

        explicit operator bool()const noexcept{
            return data != nullptr;
        }
    };
}
