//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include "../math_dis.h"
namespace cobraml::core {

    template<typename NumType>
    NumType dot_product(const void * vector1, const void * vector2, size_t const len) {

        auto vec1 = static_cast<const NumType *>(vector1);
        auto vec2 = static_cast<const NumType *>(vector2);

        NumType sum{0};
        for (size_t i = 0; i < len; ++i) {
            sum += static_cast<NumType>(vec1[i] * vec2[i]);
        }

        return sum;
    }

    class StandardMath final : public Math {
        void batched_dot_product(const void *matrix, const void *vector, void * dest, size_t rows, size_t columns, Dtype dtype) override;
    };
};

#endif //STANDARD_MATH_H
