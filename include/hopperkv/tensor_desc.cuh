#pragma once
#include<cstddef>
#include<cstdint>
#include<cassert>

namespace hopperkv{
    inline constexpr std::size_t kMaxTensorRank = 8;

    template<typename T , std::size_t Max_Rank = kMaxTensorRank>
    struct TensorDesc{
        static_assert(Max_Rank > 0 , "MaxRank must be positive");
        T* data = nullptr;
    };
}