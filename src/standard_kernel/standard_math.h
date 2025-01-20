//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include <iostream>
#include "../math_dis.h"

namespace cobraml::core {
    void set_num_threads();

    template<typename NumType>
    void gemv_naive(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        for (size_t start{0}; start < rows; ++start) {
            NumType partial = 0;
            for (size_t i = 0; i < columns; ++i) {
                partial = static_cast<NumType>(partial + vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

    template<typename NumType>
    void gemv_parallel(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns) private(start) schedule(dynamic)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;

            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

    template<typename NumType>
    void gemv_parallel_simd(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns) private(start) schedule(dynamic)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;

#pragma omp simd reduction(+:partial)
            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

#define ROW_COUNT 2


    template<typename NumType>
    void gemv_parallel_simd_2(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns) private(start) schedule(dynamic)
        for (start = 0; start < rows; start += ROW_COUNT) {
            NumType partial;

            size_t const end_row = start + ROW_COUNT;
            size_t start_row = start;

            if (end_row > rows) {
                for (; start_row < rows; ++start_row) {
                    partial = 0;
#pragma omp simd reduction(+:partial) aligned(vector: 32) aligned(matrix: 32)
                    for (size_t i = 0; i < columns; ++i) {
                        partial += static_cast<NumType>(vector[i] * matrix[start_row * columns + i]);
                    }

                    dest[start_row] = static_cast<NumType>(dest[start_row] * beta + partial * alpha);
                }
            }else {
                partial = 0;
                NumType partial_2 = 0;
#pragma omp simd reduction(+:partial) reduction(+:partial_2) aligned(vector: 32) aligned(matrix: 32)
                for (size_t i = 0; i < columns; ++i) {
                    partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
                    partial_2 += static_cast<NumType>(vector[i] * matrix[(start + 1) * columns + i]);
                }

                dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
                dest[start + 1] = static_cast<NumType>(dest[start + 1] * beta + partial_2 * alpha);
            }
        }
    }

#define ROWS 8
#define COLUMNS 8192 // bytes

    template<typename NumType>
    void gemv_parallel_block(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();

        constexpr size_t block_rows{ROWS};
        constexpr size_t block_columns{COLUMNS / sizeof(NumType)};

        size_t blocks_per_row{columns / block_columns};
        blocks_per_row += columns % block_columns > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t blocks_per_column{rows / block_rows};
        blocks_per_column += rows % block_rows > 0 ? 1 : 0; // add one more block if there is a remainder

        auto *dest_partials = new NumType[rows * blocks_per_row]();

        for (size_t i = 0; i < blocks_per_row; ++i) {
            const NumType *vector_segment{&vector[i * block_columns]};
            size_t vector_len{block_columns};

            if (i == blocks_per_row - 1) {
                vector_len = columns - (block_columns * i);
            }

            size_t j;

#pragma omp parallel for default(none) shared(dest_partials, block_rows, blocks_per_column, blocks_per_row, vector_segment, vector_len, matrix, dest, columns, rows, i) private(j) schedule(dynamic)
            for (j = 0; j < blocks_per_column; ++j) {
                size_t row_start{j * block_rows};
                size_t row_end{row_start + block_rows};

                if (j == blocks_per_column - 1) {
                    row_end = row_start + rows - (block_rows * j);
                }

                for (; row_start < row_end; ++row_start) {
                    NumType partial{0};

                    for (size_t k = 0; k < vector_len; ++k) {
                        partial += static_cast<NumType>(
                            vector_segment[k] * matrix[row_start * columns + block_columns * i + k]);
                    }

                    dest_partials[row_start * blocks_per_row + i] += partial;
                }
            }
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < blocks_per_row; ++j) {
                dest[i] = static_cast<NumType>(dest_partials[i * blocks_per_row + j] * alpha + beta * dest[i]);
            }
        }

        delete[] dest_partials;
    }

    template<typename NumType>
    void gemv_parallel_block_simd(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();

        constexpr size_t block_rows{ROWS};
        constexpr size_t block_columns{COLUMNS / sizeof(NumType)};

        size_t blocks_per_row{columns / block_columns};
        blocks_per_row += columns % block_columns > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t blocks_per_column{rows / block_rows};
        blocks_per_column += rows % block_rows > 0 ? 1 : 0; // add one more block if there is a remainder

        auto *dest_partials = new NumType[rows * blocks_per_row]();

        for (size_t i = 0; i < blocks_per_row; ++i) {
            const NumType *vector_segment{&vector[i * block_columns]};
            size_t vector_len{block_columns};

            if (i == blocks_per_row - 1) {
                vector_len = columns - (block_columns * i);
            }

            size_t j;

#pragma omp parallel for default(none) shared(dest_partials, block_rows, blocks_per_column, blocks_per_row, vector_segment, vector_len, matrix, dest, columns, rows, i) private(j) schedule(dynamic)
            for (j = 0; j < blocks_per_column; ++j) {
                size_t row_start{j * block_rows};
                size_t row_end{row_start + block_rows};

                if (j == blocks_per_column - 1) {
                    row_end = row_start + rows - (block_rows * j);
                }

                for (; row_start < row_end; ++row_start) {
                    NumType partial{0};

#pragma omp simd reduction(+:partial)
                    for (size_t k = 0; k < vector_len; ++k) {
                        partial += static_cast<NumType>(
                            vector_segment[k] * matrix[row_start * columns + block_columns * i + k]);
                    }

                    dest_partials[row_start * blocks_per_row + i] += partial;
                }
            }
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < blocks_per_row; ++j) {
                dest[i] = static_cast<NumType>(dest_partials[i * blocks_per_row + j] * alpha + beta * dest[i]);
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
        const NumType alpha,
        const NumType beta,
        size_t const rows,
        size_t const columns) {
        switch (func_pos) {
            case 0: {
                gemv_naive(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 1: {
                gemv_parallel(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 2: {
                gemv_parallel_simd(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 3: {
                gemv_parallel_block(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 4: {
                gemv_parallel_block_simd(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 5: {
                gemv_parallel_simd_2(mat, vec, dest, alpha, beta, rows, columns);
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
        const NumType alpha,
        const NumType beta,
        size_t const rows,
        size_t const columns) {
        gemv_parallel_simd_2(mat, vec, dest, alpha, beta, rows, columns);
    }
#endif

    class StandardMath final : public Math {
        void gemv(
            const void *matrix,
            const void *vector,
            void *dest,
            const void *alpha,
            const void *beta,
            size_t rows,
            size_t columns,
            Dtype dtype) override;
    };
}

#endif //STANDARD_MATH_H
