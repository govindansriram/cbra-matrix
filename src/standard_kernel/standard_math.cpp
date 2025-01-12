//
// Created by Sriram Govindan on 12/23/24.
//

#include "standard_math.h"
#include "enums.h"


namespace cobraml::core {
#define LINE_SIZE 11264 // must be multiple of 64
#define CONSECUTIVE_ROWS 1 //

    bool has_remainder(size_t const n1, size_t const n2) {
        if (n1 % n2 == 0) {
            return false;
        }

        return true;
    }

    void StandardMath::batched_dot_product(
        const void *matrix,
        const void *vector,
        void *dest,
        const size_t rows,
        const size_t columns,
        const Dtype dtype) {

        switch (dtype) {
            case FLOAT64: {
                const auto casted_dest = static_cast<double *>(dest);
                const auto casted_mat = static_cast<const double *>(matrix);
                const auto casted_vec = static_cast<const double *>(vector);
                benchmarked_gemv<double>(casted_mat, casted_vec, casted_dest, rows, columns);
                return;
            }
            case FLOAT32: {
                const auto casted_dest = static_cast<float *>(dest);
                const auto casted_mat = static_cast<const float *>(matrix);
                const auto casted_vec = static_cast<const float *>(vector);
                benchmarked_gemv<float>(casted_mat, casted_vec, casted_dest, rows, columns);
                return;
            }
            case INT8: {
                const auto casted_dest = static_cast<int8_t *>(dest);
                const auto casted_mat = static_cast<const int8_t *>(matrix);
                const auto casted_vec = static_cast<const int8_t *>(vector);
                benchmarked_gemv<int8_t>(casted_mat, casted_vec, casted_dest, rows, columns);
                return;
            }
            case INT16: {
                const auto casted_dest = static_cast<int16_t *>(dest);
                const auto casted_mat = static_cast<const int16_t *>(matrix);
                const auto casted_vec = static_cast<const int16_t *>(vector);
                benchmarked_gemv<int16_t>(casted_mat, casted_vec, casted_dest, rows, columns);
                return;
            }
            case INT32: {
                const auto casted_dest = static_cast<int32_t *>(dest);
                const auto casted_mat = static_cast<const int32_t *>(matrix);
                const auto casted_vec = static_cast<const int32_t *>(vector);
                benchmarked_gemv<int32_t>(casted_mat, casted_vec, casted_dest, rows, columns);
                return;
            }
            case INT64: {
                const auto casted_dest = static_cast<int64_t *>(dest);
                const auto casted_mat = static_cast<const int64_t *>(matrix);
                const auto casted_vec = static_cast<const int64_t *>(vector);
                benchmarked_gemv<int64_t>(casted_mat, casted_vec, casted_dest, rows, columns);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate gemmv on invalid type");
            }
        }
    }
}
