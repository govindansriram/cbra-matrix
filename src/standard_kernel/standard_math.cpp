//
// Created by Sriram Govindan on 12/23/24.
//

#include "standard_math.h"

#include <iostream>

#include "enums.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/enumerable_thread_specific.h"

namespace cobraml::core {
#define LINE_SIZE 40000 // must be multiple of 64
#define CONSECUTIVE_ROWS 1

    bool has_remainder(size_t const n1, size_t const n2) {
        if (n1 % n2 == 0) {
            return false;
        }

        return true;
    }

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
                const auto casted_dest = static_cast<float *>(dest);
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

    void process_batched_segment(
        const char *vector,
        const char *batch,
        char *dest,
        size_t const rows,
        size_t const vec_length,
        size_t const batch_length,
        size_t const dest_length,
        const Dtype dtype) {
        constexpr size_t start{0};
        bool const remainder = (rows % CONSECUTIVE_ROWS) > 0;
        size_t const blocks = rows / CONSECUTIVE_ROWS + static_cast<unsigned char>(remainder);

        // std::cout << vec_length << std::endl;
        // std::cout << batch_length << std::endl;
        // std::cout << dest_length << std::endl;

        tbb::parallel_for(start, blocks, [&](size_t const block) {
            size_t start_row = block * CONSECUTIVE_ROWS;
            size_t const end_row = block == blocks - 1 ? rows : start_row + CONSECUTIVE_ROWS;

            char *p_dest = &dest[start_row * dest_length];
            const char *p_batch = &batch[start_row * batch_length];

            // TODO: make thread private vector

            for (; start_row < end_row; ++start_row) {
                type_insensitive_dot_product(vector, p_batch, p_dest, dtype, vec_length);
                p_dest = &p_dest[dest_length];
                p_batch = &p_batch[batch_length];
            }
        });
    }

    void type_insensitive_sum(
        const void *vec,
        void *dest,
        Dtype const dtype,
        size_t const len) {
        switch (dtype) {
            case FLOAT64: {
                const auto casted_dest = static_cast<double *>(dest);
                *casted_dest = sum<double>(vec, len);
                return;
            }
            case FLOAT32: {
                auto casted_dest = static_cast<float *>(dest);
                *casted_dest = sum<float>(vec, len);
                return;
            }
            case INT8: {
                const auto casted_dest = static_cast<int8_t *>(dest);
                *casted_dest = sum<int8_t>(vec, len);
                return;
            }
            case INT16: {
                const auto casted_dest = static_cast<int16_t *>(dest);
                *casted_dest = sum<int16_t>(vec, len);
                return;
            }
            case INT32: {
                const auto casted_dest = static_cast<int32_t *>(dest);
                *casted_dest = sum<int32_t>(vec, len);
                return;
            }
            case INT64: {
                const auto casted_dest = static_cast<int64_t *>(dest);
                *casted_dest = sum<int64_t>(vec, len);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate the dot product from an invalid type");
            }
        }
    }

#ifdef BENCHMARK
    void batched_dot_product_naive(
        const void *matrix,
        const void *vector,
        void *dest,
        size_t const rows,
        size_t const columns,
        Dtype const dtype) {
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

    /**
     * l1 cache size = 38.4kb
     * l2 cache size = 2.5mb
     */

    /////////////////


    void batched_dot_product_blocked(
        const void *matrix,
        const void *vector,
        void *dest,
        const size_t rows,
        const size_t columns,
        const Dtype dtype) {
        size_t const jump{dtype_to_bytes(dtype)};
        size_t line_size;

        if (LINE_SIZE > columns * jump) {
            line_size = columns * jump;
        } else {
            line_size = LINE_SIZE;
        }

        size_t lines{columns * jump / line_size};

        if (has_remainder(columns * jump, line_size)) {
            ++lines;
        }

        // std::cout << lines << std::endl;

        tbb::cache_aligned_allocator<char> allocator;

        const auto c_mat = static_cast<const char *>(matrix);
        const auto c_vec = static_cast<const char *>(vector);
        auto const c_dest = static_cast<char *>(dest);

        char *results = allocator.allocate(rows * jump * lines);

        tbb::parallel_for(static_cast<size_t>(0), lines, [&](const size_t start) {
            size_t const start_line = start * line_size;
            const char *p_mat = &c_mat[start_line];
            const char *p_vec = &c_vec[start_line];
            char *p_dest = &results[start * jump];

            size_t const vec_len = start == lines - 1 ? (columns * jump) - start_line : line_size;

            // std::cout << "vec len: " << vec_len << std::endl;

            process_batched_segment(
                p_vec,
                p_mat,
                p_dest,
                rows,
                vec_len / jump,
                columns * jump,
                lines * jump,
                dtype);
        });

        // for (auto& t : thread_vec) {
        //     t.join();
        // }

        // int * test = reinterpret_cast<int *>(results);
        //
        // for (size_t i = 0; i < rows * lines; ++ i) {
        //     std::cout << test[i] << std::endl;
        // }

        for (size_t row = 0; row < rows; ++row) {
            type_insensitive_sum(&results[row * jump * lines], &c_dest[row * jump], dtype, lines);
        }

        allocator.deallocate(results, rows * jump * lines);
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
            case 2: {
                batched_dot_product_blocked(matrix, vector, dest, rows, columns, dtype);
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
        const size_t rows,
        const size_t columns,
        const Dtype dtype) {
        size_t const jump{dtype_to_bytes(dtype)};
        size_t line_size;

        if (LINE_SIZE > columns * jump) {
            line_size = columns * jump;
        } else {
            line_size = LINE_SIZE;
        }

        size_t lines{columns * jump / line_size};

        if (has_remainder(columns * jump, line_size)) {
            ++lines;
        }

        std::cout << lines << std::endl;

        tbb::cache_aligned_allocator<char> allocator;

        const auto c_mat = static_cast<const char *>(matrix);
        const auto c_vec = static_cast<const char *>(vector);
        auto const c_dest = static_cast<char *>(dest);

        char *results = allocator.allocate(rows * jump * lines);

        tbb::parallel_for(static_cast<size_t>(0), lines, [&](const size_t start) {
            size_t const start_line = start * line_size;
            const char *p_mat = &c_mat[start_line];
            const char *p_vec = &c_vec[start_line];
            char *p_dest = &results[start * jump];

            size_t const vec_len = start == lines - 1 ? (columns * jump) - start_line : line_size;

            std::cout << "vec len: " << vec_len << std::endl;

            process_batched_segment(
                p_vec,
                p_mat,
                p_dest,
                rows,
                vec_len / jump,
                columns * jump,
                lines * jump,
                dtype);
        });

        // for (auto& t : thread_vec) {
        //     t.join();
        // }

        // int * test = reinterpret_cast<int *>(results);
        //
        // for (size_t i = 0; i < rows * lines; ++ i) {
        //     std::cout << test[i] << std::endl;
        // }

        for (size_t row = 0; row < rows; ++row) {
            type_insensitive_sum(&results[row * jump * lines], &c_dest[row * jump], dtype, lines);
        }

        allocator.deallocate(results, rows * jump * lines);
    }

    // void StandardMath::batched_dot_product(
    //     const void *matrix,
    //     const void *vector,
    //     void *dest,
    //     size_t rows,
    //     size_t columns,
    //     Dtype dtype) {
    //     size_t jump{dtype_to_bytes(dtype)};
    //     size_t row_jump{columns * jump};
    //     auto c_dest = static_cast<char *>(dest);
    //     auto c_mat = static_cast<const char *>(matrix);
    //     auto c_vec = static_cast<const char *>(vector);
    //     size_t start{0};
    //
    //     tbb::parallel_for(start, rows, [&](size_t const row) {
    //         // std::cout << "Processing on thread: " << tbb::this_task_arena::current_thread_index() << "\n";
    //         char *p_dest = c_dest + (jump * row);
    //         const char *vector1 = c_mat + (row * row_jump);
    //         type_insensitive_dot_product(vector1, c_vec, p_dest, dtype, columns);
    //     });
    // }
#endif
}
