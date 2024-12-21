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


        MatrixImpl(size_t const rows, size_t const columns, Device const device, Dtype const type): rows(rows),
            columns(columns),
            device(device),
            dtype(type),
            buffer(std::make_shared<Buffer>(rows * columns * dtype_to_bytes(type), device)) {
        }

        MatrixImpl(
            size_t const rows,
            size_t const columns,
            Device const device,
            Dtype const type,
            const void * p_data): rows(rows),
            columns(columns),
            device(device),
            dtype(type),
            buffer(std::make_shared<Buffer>(rows * columns * dtype_to_bytes(type), device, p_data)) {
        }

        [[nodiscard]] void *get_raw_buffer() const {
            return buffer->get_p_buffer();
        }
    };

    Matrix::Matrix(size_t rows, size_t columns, Device device, Dtype dtype): impl(
        std::make_unique<MatrixImpl>(rows, columns, device, dtype)) {
        is_invalid(dtype);
    }

    Matrix::Matrix(
        size_t const rows,
        size_t const columns,
        Device const device,
        Dtype const dtype,
        const void *p_data): impl(std::make_unique<MatrixImpl>(rows, columns, device, dtype, p_data)) {
        is_invalid(dtype);
    }


    Dtype Matrix::get_dtype() const {
        return this->impl->dtype;
    }


    Matrix::~Matrix() = default;

    void *Matrix::get_raw_buffer() const {
        return impl->get_raw_buffer();
    }

    Matrix::Matrix(): impl(nullptr) {}

    void print_num(void *buffer, Dtype const dtype) {
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
            case INT64: {
                std::cout << *static_cast<int64_t *>(buffer);
                return;
            }
            case FLOAT32: {
                std::cout << *static_cast<float *>(buffer);
                return;
            }
            case FLOAT64: {
                std::cout << *static_cast<double *>(buffer);
                return;
            }
            case INVALID: {
                is_invalid(dtype);
            }
        }
    }

    Matrix::Matrix(Matrix &other): impl(std::move(other.impl)) {}

    Matrix& Matrix::operator=(const Matrix &other) {
        if (impl == nullptr) {
            impl = std::make_unique<MatrixImpl>(
                other.impl->rows,
                other.impl->columns,
                other.impl->device,
                other.impl->dtype,
                other.impl->buffer->get_p_buffer());
        }

        if (this != &other) {
            impl->columns = other.impl->columns;
            impl->rows = other.impl->rows;
            impl->dtype = other.impl->dtype;
            impl->device = other.impl->device;
            impl->buffer = other.impl->buffer;
        }

        return *this;
    }

    // size_t batched_dot_product(Matrix &matrix, Matrix &vector) {
    //     return matrix.impl->columns;
    // }

    void Matrix::print(bool const hide_middle) const {
        unsigned char const shift = dtype_to_bytes(impl->dtype);

        auto dec3 = [this](size_t const x, size_t &start) {
            if (x == 1)
                start = impl->rows - 3;
        };

        std::cout << "[\n";
        for (size_t x = 0; x < 2; ++x) {
            size_t start = 0;
            size_t end = impl->rows;

            if (impl->rows > 20 && hide_middle) {
                dec3(x, start);
                end = start + 3;
            }

            for (; start < end; ++start) {
                std::cout << "   [";

                for (size_t y = 0; y < 2; ++y) {
                    size_t start_inner = 0;
                    size_t end_inner = impl->columns;

                    if (impl->columns > 20 && hide_middle) {
                        dec3(y, start_inner);
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
