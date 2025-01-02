//
// Created by Sriram Govindan on 12/23/24.
//

#include "standard_math.h"
#include "enums.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include "oneapi/tbb/blocked_range2d.h"


namespace cobraml::core {
#define LINE_SIZE 11264 // must be multiple of 64
#define CONSECUTIVE_ROWS 1 //

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
                // std::cout << *casted_dest << std::endl;
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
        size_t block_len;

        if (LINE_SIZE > columns * jump) {
            block_len = columns * jump;
        } else {
            block_len = LINE_SIZE;
        }

        size_t block_rows{rows / CONSECUTIVE_ROWS};
        size_t block_columns{columns * jump / block_len};

        if (has_remainder(columns * jump, block_len))
            ++block_columns;

        if (has_remainder(rows, CONSECUTIVE_ROWS))
            ++block_rows;

        tbb::cache_aligned_allocator<char> allocator;

        const auto c_mat = static_cast<const char *>(matrix);
        const auto c_vec = static_cast<const char *>(vector);
        auto const c_dest = static_cast<char *>(dest);

        char *results = allocator.allocate(rows * jump * block_columns);
        constexpr size_t zero = 0;

        parallel_for(tbb::blocked_range2d<size_t>(
                zero,
                rows,
                zero,
                block_columns), [&](const tbb::blocked_range2d<size_t> &r) {

                    for (size_t col_start = r.cols().begin(); col_start < r.cols().end(); ++col_start) {

                        const char * p_vec = &c_vec[col_start * block_len];
                        size_t vec_len;

                        if (col_start == block_columns - 1) {
                            vec_len = columns * jump - (col_start * block_len);
                        }
                        else {
                            vec_len = block_len;
                        }

                        for (size_t row_start = r.rows().begin(); row_start < r.rows().end(); ++row_start) {
                            const char * p_mat = &c_mat[row_start * columns * jump + (col_start * block_len)];
                            char * p_dest = &results[row_start * jump * block_columns  + (jump * col_start)];
                            type_insensitive_dot_product(p_vec, p_mat, p_dest, dtype, vec_len / jump);
                        }
                    }
        });

        // std::cout << "columns: " << block_columns << std::endl;
        // std::cout << "block_rows: " << block_rows << std::endl;

        // parallel_for(tbb::blocked_range2d<size_t>(
        //                  zero,
        //                  block_rows,
        //                  zero,
        //                  block_columns), [&](const tbb::blocked_range2d<size_t> &r) {
        //                  for (size_t col_start = r.cols().begin(); col_start < r.cols().end(); ++col_start) {
        //                      const char *p_vec = &c_vec[col_start * block_len];
        //                      size_t vec_len;
        //
        //                      if (col_start == block_columns - 1) {
        //                          vec_len = columns * jump - (col_start * block_len);
        //                      } else {
        //                          vec_len = block_len;
        //                      }
        //
        //                      for (size_t block_start = r.rows().begin(); block_start < r.rows().end(); ++block_start) {
        //                          size_t row_end;
        //
        //                          size_t row_start = block_start * CONSECUTIVE_ROWS;
        //
        //                          if (block_start == block_rows - 1) {
        //                              row_end = row_start + (rows - (block_start * CONSECUTIVE_ROWS));
        //                              // std::cout << "rows: " << rows << " row_start: " << row_start * CONSECUTIVE_ROWS << " row_end: " << row_end << std::endl;
        //                          } else {
        //                              row_end = row_start + CONSECUTIVE_ROWS;
        //                          }
        //
        //                          // std::cout <<  " row_start: " << row_start << " row_end: " << row_end << std::endl;
        //
        //                          for (; row_start < row_end; ++row_start) {
        //                              // std::cout <<  " row_start: " << row_start << std::endl;
        //                              const char *p_mat = &c_mat[row_start * columns * jump + (col_start * block_len)];
        //                              char *p_dest = &results[row_start * jump * block_columns + (jump * col_start)];
        //                              type_insensitive_dot_product(p_vec, p_mat, p_dest, dtype, vec_len / jump);
        //                          }
        //                      }
        //                  }
        //              });


        for (size_t row = 0; row < rows; ++row) {
            type_insensitive_sum(&results[row * jump * block_columns], &c_dest[row * jump], dtype, block_columns);
        }

        allocator.deallocate(results, rows * jump * block_columns);
    }

#ifdef BENCHMARK
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
        batched_dot_product_blocked(matrix, vector, dest, rows, columns, dtype);
    }
#endif
}
