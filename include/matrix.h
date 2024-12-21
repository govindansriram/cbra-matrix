//
// Created by sriram on 12/15/24.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <memory>
#include <vector>
#include "enums.h"

namespace cobraml::core {
    class Matrix {
        struct MatrixImpl;
        std::unique_ptr<MatrixImpl> impl;

        /**
         * get the raw pointer backing an array, do not free this pointer
         *
         * @return a raw void pointer to the buffer data
         */
        [[nodiscard]] void *get_raw_buffer() const;

        /**
         * private constructor that allows initialization of a matrix
         * from an existing buffer of data. This buffer must be on the cpu.
         *
         * @param rows the # of rows in the matrix
         * @param columns the # of columns in the matrix
         * @param device the device of the matrix being constructed
         * @param dtype the dtype of the matrix being constructed
         * @param p_data the data that will be copied
         */
        Matrix(size_t rows, size_t columns, Device device, Dtype dtype, const void *p_data);

    public:
        /**
         * constructor that creates a zero matrix of shape (rows, columns)
         * @param rows the # of rows in the matrix
         * @param columns the # of columns in the matrix
         * @param device the device of the matrix being constructed
         * @param dtype the dtype of the matrix being constructed
         */
        Matrix(size_t rows, size_t columns, Device device, Dtype dtype);

        Matrix();
        Matrix(Matrix &other);
        Matrix& operator=(const Matrix& other);

        /**
         * @return the dtype of the matrix
         */
        [[nodiscard]] Dtype get_dtype() const;

        /**
         * provides access to the underlying matrix buffer in row major format
         * @tparam T the type that the ptr should be cast too, it must match the Dtype
         * @return the raw ptr buffer
         */
        template<typename T>
        T *get_buffer();

        /**
         * prints the contents of the matrix in tabular format
         * @param hide_middle hide the center elements of an array
         */
        void print(bool hide_middle = true) const;

        ~Matrix();

        /**
         * Start of the friend API
         */

        /**
         * computes the batche dot product between a singular vector shape (1, N) and matrix of shape (M, N)
         * @return
         */
        friend size_t batched_dot_product(Matrix &matrix, Matrix &vector);

        template<typename T>
        friend Matrix from_vector(const std::vector<std::vector<T>> &mat, Device device);
    };

    template<typename T>
    Matrix from_vector(const std::vector<std::vector<T>> &mat, Device const device) {
        constexpr Dtype dtype{get_dtype_from_type<T>::type};
        is_invalid(dtype);

        const size_t rows{mat.size()};
        const size_t columns{mat[0].size()};

        for (const std::vector<T> &row: mat) {
            if (row.size() != columns) {
                throw std::runtime_error("matrix is not rectangular");
            }
        }

        // change this up

        std::cout << "byte: " << *static_cast<const int *>(mat.data()) << '\n';

        return Matrix(rows, columns, device, dtype, mat.data());
    }

    template<typename T>
    T *Matrix::get_buffer() {
        const Dtype current{get_dtype()};
        if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
            throw std::runtime_error(
                "provided buffer type does not match matrix type: " + dtype_to_string(current));
        }

        return static_cast<T *>(get_raw_buffer());
    }

}

#endif //MATRIX_H
