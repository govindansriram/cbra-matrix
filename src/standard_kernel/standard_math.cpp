//
// Created by Sriram Govindan on 12/23/24.
//

#include "standard_math.h"
#include "enums.h"
#include "oneapi/tbb/parallel_for.h"

namespace cobraml::core {
    void type_insensitive_dot_product(
        const void *vec1,
        const void *vec2,
        void *dest,
        Dtype const dtype,
        size_t const len) {
        switch (dtype) {
            case FLOAT64: {
                const auto casted_dest = static_cast<double *>(dest);
                *casted_dest = dot_product<double>(vec1, vec2, len);
                return;
            }
            case FLOAT32: {
                auto casted_dest = static_cast<float *>(dest);
                *casted_dest = dot_product<float>(vec1, vec2, len);
                return;
            }
            case INT8: {
                const auto casted_dest = static_cast<int8_t *>(dest);
                *casted_dest = dot_product<int8_t>(vec1, vec2, len);
                return;
            }
            case INT16: {
                const auto casted_dest = static_cast<int16_t *>(dest);
                *casted_dest = dot_product<int16_t>(vec1, vec2, len);
                return;
            }
            case INT32: {
                const auto casted_dest = static_cast<int32_t *>(dest);
                *casted_dest = dot_product<int32_t>(vec1, vec2, len);
                return;
            }
            case INT64: {
                const auto casted_dest = static_cast<int64_t *>(dest);
                *casted_dest = dot_product<int64_t>(vec1, vec2, len);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate the dot product from an invalid type");
            }
        }
    }

#ifdef BENCHMARK
    void batched_dot_product_naive(
        const void * matrix,
        const void * vector,
        void * dest,
        size_t const rows,
        size_t const columns,
        Dtype const dtype){

        const size_t jump{dtype_to_bytes(dtype)};
        auto char_dest{static_cast<char *>(dest)};

        for (size_t i = 0; i < rows; ++i) {
            const auto vec1 = static_cast<const char *>(matrix) + (jump * i * columns);
            type_insensitive_dot_product(vec1, vector, char_dest, dtype, columns);
            char_dest += jump;
        }
    }

    void batched_dot_product_parallel(
        const void *matrix,
        const void *vector,
        void *dest,
        const size_t rows,
        const size_t columns,
        const Dtype dtype) {

        const size_t jump{dtype_to_bytes(dtype)};
        const size_t row_jump{columns * jump};
        const auto c_dest = static_cast<char *>(dest);
        const auto c_mat = static_cast<const char *>(matrix);
        const auto c_vec = static_cast<const char *>(vector);
        constexpr size_t start{0};

        tbb::parallel_for(start, rows, [&](size_t const row) {
            char *p_dest = c_dest + (jump * row);
            const char *vector1 = c_mat + (row * row_jump);
            type_insensitive_dot_product(vector1, c_vec, p_dest, dtype, columns);
        });
    }

    void StandardMath::batched_dot_product(
        const void *matrix,
        const void *vector,
        void *dest,
        const size_t rows,
        const size_t columns,
        const Dtype dtype) {

        switch (func_pos) {
            case 0: {
                batched_dot_product_naive(matrix, vector, dest, rows, columns, dtype);
                return;
            }
            case 1: {
                batched_dot_product_parallel(matrix, vector, dest, rows, columns, dtype);
                return;
            }
            default:
                throw std::runtime_error("invalid function_pos set");
        }
    }

#else
    void StandardMath::batched_dot_product(
        const void *matrix,
        const void *vector,
        void *dest,
        size_t rows,
        size_t columns,
        Dtype dtype) {
        size_t jump{dtype_to_bytes(dtype)};
        size_t row_jump{columns * jump};
        auto c_dest = static_cast<char *>(dest);
        auto c_mat = static_cast<const char *>(matrix);
        auto c_vec = static_cast<const char *>(vector);
        size_t start{0};

        tbb::parallel_for(start, rows, [&](size_t const row) {
            // std::cout << "Processing on thread: " << tbb::this_task_arena::current_thread_index() << "\n";
            char *p_dest = c_dest + (jump * row);
            const char *vector1 = c_mat + (row * row_jump);
            type_insensitive_dot_product(vector1, c_vec, p_dest, dtype, columns);
        });
    }
#endif
}
