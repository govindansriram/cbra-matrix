//
// Created by sriram on 12/15/24.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <memory>
#include "enums.h"

namespace cobraml::core {

    class Matrix {
        struct MatrixImpl;
        std::unique_ptr<MatrixImpl> impl;
        [[nodiscard]] void * get_raw_buffer() const;

    public:
        Matrix(size_t rows, size_t columns, Device device, Dtype dtype);

        template<typename T>
        T * get_buffer() {
            return static_cast<T *>(get_raw_buffer());
        }

        void print(bool hide_middle = true) const;

        ~Matrix();

        friend size_t batched_dot_product(Matrix &, Matrix &);
    };
}

#endif //MATRIX_H
