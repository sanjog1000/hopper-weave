#pragma once
#include<utility>
#include<limits>
#include<vector>
#include<initializer_list>
#include<cstddef>
#include<stdexcept>

#include"hopperkv/cuda_buffer.cuh"

namespace hopperkv{

    template <typename T>
    class Tensor{
    private:
        std::vector<std::size_t> shape_;
        std::size_t numel_;
        std::vector<std::size_t> strides_;
        CudaBuffer<T> buffer_;

        // Calculate the total element count safely
        static std::size_t checked_numel(const std::vector<std::size_t>& shape){
            if(shape.empty()){
                throw std::invalid_argument("Tensor must have at least one dimension");
            }

            std::size_t count = 1;

            for(std::size_t dim : shape){
                if(dim == 0){
                    throw std::invalid_argument("Tensor dimensions must be positive");
                }
                if (count > std::numeric_limits<std::size_t>::max() / dim){
                    throw std::overflow_error("Tensor element count overflows size_t");
                }
                count *= dim;
            }

            if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)){
                throw std::overflow_error("Tensor byte size overflows size_t");
            }

            return count;
        }

        // Calculate row-major contiguous strides
        static std::vector<std::size_t> make_contiguous_strides(const std::vector<std::size_t> &shape){
            std::vector<std::size_t> result(shape.size());
            std::size_t running_stride =1;

            for(std::size_t i = shape.size() ; i-- >0;){
                result[i] = running_stride;
                running_stride *= shape[i];
            }
            return result;
        }

        public:
            explicit Tensor(std::vector< std::size_t>shape)
                : shape_(std::move(shape)),
                numel_(checked_numel(shape_)),
                strides_(make_contiguous_strides(shape_)),
                buffer_(numel_) {}

            Tensor(std::initializer_list<std::size_t>shape) : Tensor(std::vector<std::size_t>(shape)) {}
            
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;
            Tensor(Tensor&&) noexcept = default;
            Tensor& operator=(Tensor&&) noexcept = default;
            
            T* data()noexcept{
                return buffer_.data();
            }
            const T* data()const noexcept{
                return buffer_.data();
            }

            const std::vector<std::size_t>& shape() const noexcept{
                return shape_;
            }
            const std::vector<std::size_t>& strides() const noexcept{
                return strides_;
            }

            std::size_t rank()const noexcept{
                return shape_.size();
            }
            std::size_t numel() const noexcept{
                return numel_;
            }
            std::size_t bytes() const noexcept{
                return buffer_.bytes();
            }

            void copy_from_host(const T* source , std::size_t count){
                if(count != numel_){
                    throw std::invalid_argument("Tensor::copy_from_host: element count must equal tensor.numel()");
                }
                if(source == nullptr){
                    throw std::invalid_argument("Tensor::copy_from_host: source is null");
                }
                buffer_.copy_from_host(source,count);
            }
            void copy_to_host(const T* destination , std::size_t count)const{
                if(count != numel_){
                    throw std::invalid_argument("Tensor::copy_to_host: element count must equal tensor.numel()");
                }
                if(destination == nullptr){
                    throw std::invalid_argument("Tensor::copy_to_host: destination is null");
                }
                buffer_.copy_to_host(destination,count);
            }
    };
}