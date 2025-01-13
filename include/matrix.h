//
// Created by sriram on 12/15/24.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <memory>
#include <vector>
#include <sys/stat.h>

#include "enums.h"
#include "iostream"

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
         * replace a segment of the matrix buffer with a different buffer
         * @param source
         * @param offset
         * @param bytes
         */
        void replace_segment(const void * source, size_t offset, size_t bytes) const;

    public:
        struct Shape {
            size_t rows;
            size_t columns;
        };

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
        * @return the Device of the matrix
        */
        [[nodiscard]] Device get_device() const;

        /**
        * @return the shape of the matrix
        */
        [[nodiscard]] Shape get_shape() const;

        /**
         * prints the contents of the matrix in tabular format
         * @param hide_middle hide the center elements of an array
         */
        void print(bool hide_middle = true) const;

        ~Matrix();

        // Start of the Friend API

        /**
         * Generalized Matrix Vector Multiplication.
         * Performs y=αAx+βy
         *
         * @param matrix A
         * @param vector x
         * @param result y
         * @param alpha α
         * @param beta β
         */
        friend void gemv(const Matrix &matrix, const Matrix &vector, Matrix &result, const void * alpha, const void * beta);

        /**
         * provides access to the underlying matrix buffer in row major format
         * @tparam T the type that the ptr should be cast too, it must match the Dtype
         * @return the raw ptr buffer
         */
        template<typename T>
        friend const T *get_buffer(const Matrix &matrix);

        template<typename T>
        friend Matrix from_vector(const std::vector<std::vector<T>> &mat, Device device);
    };

    template<typename T>
    Matrix from_vector(const std::vector<std::vector<T>> &mat, Device const device) {
        constexpr Dtype dtype{get_dtype_from_type<T>::type};
        is_invalid(dtype);

        const size_t rows{mat.size()};
        const size_t columns{mat[0].size()};
        constexpr unsigned char data_size{dtype_to_bytes(dtype)};

        for (const std::vector<T> &row: mat) {
            if (row.size() != columns) {
                throw std::runtime_error("matrix is not rectangular");
            }
        }

        Matrix ret(rows, columns, device, dtype);

        size_t count{0};
        const size_t copy_amount{data_size * columns};

        for (const std::vector<T> &row: mat) {
            ret.replace_segment(row.data(), count * columns * data_size, copy_amount);
            ++count;
        }

        return ret;
    }

    template<typename T>
    const T *get_buffer(const Matrix &matrix) {
        const Dtype current{matrix.get_dtype()};
        if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
            throw std::runtime_error(
                "provided buffer type does not match matrix type: " + dtype_to_string(current));
        }

        return static_cast<T *>(matrix.get_raw_buffer());
    }
}

#endif //MATRIX_H
