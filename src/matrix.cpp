//
// Created by sriram on 12/19/24.
//


#include "matrix.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include "allocator.h"
#include "enums.h"

namespace cobraml::core {

    struct Matrix::MatrixImpl {
        size_t rows;
        size_t columns;
        Device device;
        Dtype dtype;
        std::shared_ptr<Buffer> buffer;


        MatrixImpl(size_t const rows, size_t const columns, Device const device, Dtype const type):
        rows(rows),
        columns(columns),
        device(device),
        dtype(type),
        buffer(std::make_shared<Buffer>(rows * columns * dtype_to_bytes(type), device)){}

        [[nodiscard]] void * get_raw_buffer() const {
            return buffer->get_p_buffer();
        }
    };

    Matrix::Matrix(size_t rows,size_t columns, Device device, Dtype dtype):
    impl(std::make_unique<MatrixImpl>(rows, columns, device, dtype)) {}

    Matrix::~Matrix() = default;

    void * Matrix::get_raw_buffer() const {
        return impl->get_raw_buffer();
    }

    void print_num(void * buffer, Dtype const dtype) {
        switch (dtype) {
            case INT8: {
                const auto num = *static_cast<char *>(buffer);
                std::cout << static_cast<int>(num);
                return;
            }
            case INT16: {
                std::cout << *static_cast<int16_t *>(buffer);
                return;
            }
            case INT32: {
                std::cout << *static_cast<int32_t *>(buffer);
                return;
            }
            case INT64:{
                std::cout << *static_cast<int64_t *>(buffer);
                return;
            }
            case FLOAT32:{
                std::cout << *static_cast<float *>(buffer);
                return;
            }
            case FLOAT64:{
                std::cout << *static_cast<double *>(buffer);
            }
        }
    }

    size_t batched_dot_product(Matrix &matrix, Matrix &vector) {
        return matrix.impl->columns;
    }

    void Matrix::print(bool const hide_middle) const {
        char const shift = dtype_to_bytes(impl->dtype);

        std::cout << "[\n";
        for (int x = 0; x < 2; ++x) {
            size_t start = 0;
            size_t end = impl->rows;

            if (impl->rows > 20 && hide_middle) {
                start = x * impl->rows + (x == 1 ? -1 : 1) * 3;
                end = start + 3;
            }

            for (; start < end; ++start) {
                std::cout << "   [";

                for (int y = 0; y < 2; ++y) {
                    size_t start_inner = 0;
                    size_t end_inner = impl->columns;

                    if (impl->columns > 20 && hide_middle) {
                        start_inner = y * impl->rows + (y == 1 ? -1 : 1) * 3;
                        end_inner = start_inner + 3;
                    }

                    for (; start_inner < end_inner; ++start_inner) {
                        std::cout << std::fixed << std::setprecision(5);

                        auto buff = static_cast<char *>(impl->get_raw_buffer());
                        buff += (start * impl->columns + start_inner) * shift;
                        print_num(buff, impl->dtype);

                        if (start_inner != end_inner - 1 || (y == 0 && hide_middle)) {
                            std::cout << ", ";
                        }
                    }

                    if (impl->columns <= 20 || !hide_middle)
                        break;

                    if (y == 0) {
                        std::cout << "..., ";
                    }
                }

                std::cout << "]";
                std::cout << "\n";
            }

            if (impl->rows <= 20 || !hide_middle)
                break;

            if (x == 0) {
                std::cout << "   ..." << "\n";
            }
        }

        std::cout << "]\n";
    }
}
