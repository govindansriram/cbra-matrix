//
// Created by sriram on 12/15/24.
//

#ifndef COSINE_SIMILARITY_H
#define COSINE_SIMILARITY_H

#include "../math_dispatcher.h"
#include "oneapi/tbb/parallel_for.h"

namespace cobraml::core {
#ifdef NAIVE
        // use naive solution

        template<typename T>
        void batched_dot_product(void * matrix, void * vector, void * dest, size_t const rows, size_t const columns) override{
            auto p_mat = static_cast<T *>(matrix);
            auto p_vec = static_cast<T *>(vector);
            auto p_dest = static_cast<T *>(dest);

            for (size_t i = 0; i < rows; ++i) {
                T total = 0;
                for (size_t j = 0; j < columns; ++j) {
                    total += *p_mat[(i * columns) + j] * *p_vec[j];
                }

                *p_dest = total;
                ++p_dest;
            }
        }

#else
    // use optimized naive solution

    template<typename T>
    void batched_dot_product(void *matrix, void *vector, void *dest, size_t const rows, size_t const columns) override {
        auto p_mat = static_cast<T *>(matrix);
        auto p_vec = static_cast<T *>(vector);
        auto p_dest = static_cast<T *>(dest);

        size_t start{0};

        tbb::parallel_for(
            start, rows,
            [=](size_t const row) {
                T total = 0;
                for (size_t j = 0; j < columns; ++j) {
                    total += *p_mat[row * columns + j] * *p_vec[j];
                }

                *(p_dest + row) = total;
            });
    }
};

#endif

#endif //COSINE_SIMILARITY_H
