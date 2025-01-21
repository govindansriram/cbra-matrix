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
