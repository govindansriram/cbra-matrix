//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include <iostream>
#include "../math_dis.h"

namespace cobraml::core {
    template<typename NumType>
    void gemv_naive(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        for (size_t start = 0; start < rows; ++start) {
            for (size_t i = 0; i < columns; ++i) {
                dest[start] = vector[i] * matrix[start * columns + i];
            }
        }
    }

    template<typename NumType>
    void gemv_parallel(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        size_t start;

#pragma omp parallel for default(none) shared(matrix, vector, dest, rows, columns) private(start)
        for (start = 0; start < rows; ++start) {
            for (size_t i = 0; i < columns; ++i) {
                dest[start] += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }
        }
    }

    template<typename NumType>
    void gemv_parallel_block(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {

        constexpr size_t block_rows = 8; // best 15 // 8
        constexpr size_t block_columns = 8192 / sizeof(NumType);

        size_t blocks_per_row = columns / block_columns;
        blocks_per_row += columns % block_columns > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t blocks_per_column = rows / block_rows;
        blocks_per_column += rows % block_rows > 0 ? 1 : 0; // add one more block if there is a remainder

        auto *dest_partials = new NumType[rows * blocks_per_row]();

        for (size_t i = 0; i < blocks_per_row; ++i) {

            const NumType * vector_segment = &vector[i * block_columns];
            size_t vector_len = block_columns;

            if (i == blocks_per_row - 1) {
                vector_len = columns - (block_columns * i);
            }

            size_t j;

#pragma omp parallel for default(none) shared(dest_partials, block_rows, blocks_per_column, blocks_per_row, vector_segment, vector_len, matrix, dest, columns, rows, i) private(j)
            for (j = 0; j < blocks_per_column; ++j) {
                size_t row_start = j * block_rows;
                size_t row_end = row_start + block_rows;

                if (j == blocks_per_column - 1) {
                    row_end = row_start + rows - (block_rows * j);
                }

                for (;row_start < row_end; ++row_start) {
                    NumType partial = 0;

                    for (size_t k = 0; k < vector_len; ++k) {
                        partial += static_cast<NumType>(vector_segment[k] * matrix[row_start * columns + block_columns * i + k]);
                    }

                    dest_partials[row_start * blocks_per_row + i] += partial;
                }
            }
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < blocks_per_row; ++j) {
                dest[i] += dest_partials[i * blocks_per_row + j];
            }
        }

        delete[] dest_partials;
    }

#ifdef BENCHMARK

    template<typename NumType>
    void benchmarked_gemv(
        const NumType *mat,
        const NumType *vec,
        NumType *dest,
        size_t const rows,
        size_t const columns) {
        switch (func_pos) {
            case 0: {
                gemv_naive(mat, vec, dest, rows, columns);
                return;
            }
            case 1: {
                gemv_parallel(mat, vec, dest, rows, columns);
                return;
            }
            case 2: {
                gemv_parallel_block(mat, vec, dest, rows, columns);
                return;
            }
            default: {
                throw std::runtime_error("invalid gemv type provided");
            }
        }
    }

#else
    template<typename NumType>
    void benchmarked_gemv(
        const NumType *mat,
        const NumType *vec,
        NumType *dest,
        size_t const rows,
        size_t const columns) {

        gemv_parallel_block(mat, vec, dest, rows, columns);
    }
#endif

    class StandardMath final : public Math {
        void batched_dot_product(const void *matrix, const void *vector, void *dest, size_t rows, size_t columns,
                                 Dtype dtype) override;
    };
};

#endif //STANDARD_MATH_H
