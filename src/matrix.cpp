//
// Created by sriram on 12/19/24.
//


#include "matrix.h"
#include <iomanip>
#include <iostream>
#include "enums.h"

namespace cobraml::core {

    Matrix::Matrix(size_t const rows, size_t const columns, Device const device, Dtype const dtype):
        Array(rows * columns, device, dtype),
        rows(rows),
        columns(columns){
        is_invalid(dtype);
    }

    Matrix::Shape Matrix::get_shape() const {
        Shape sh{};
        sh.columns = columns;
        sh.rows = rows;
        return sh;
    }

    Matrix::Matrix(Array const &other): Array(other), rows(0), columns(0) {}
    Matrix::Matrix(Matrix const &other): Array(other), rows(other.rows), columns(other.columns) {}


    Matrix::~Matrix() = default;

    bool Matrix::Shape::operator==(const Shape &other) const {
        return this->rows == other.rows && this->columns == other.columns;
    }

    bool Matrix::is_vector() const {
        return rows == 1;
    }

    bool Matrix::is_scalar() const {
        return this->is_vector() && columns == 1;
    }

    Matrix Matrix::operator[](size_t const index) const{

        Matrix ret = *this;
        ret.rows = 1;

        if (this->is_vector()) {
            if (index >= this->columns) {
                throw std::out_of_range("index is out of range");
            }

            ret.columns = 1;
            ret.increment_offset(index);

            ret.set_length(1);

            return ret;
        }

        if (index >= this->get_shape().rows) {
            throw std::out_of_range("index is out of range");
        }

        ret.columns = this->get_shape().columns;

        ret.increment_offset(this->columns * index);
        ret.set_length(columns);

        return ret;
    }

    Matrix::Matrix(): rows(0), columns(0) {}

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

    Matrix& Matrix::operator=(const Matrix &other) {
        if (this == &other) {
            return *this;
        }

        // if (!(this->get_shape() == other.get_shape())) {
        //     throw std::runtime_error("matrices are not the same shape");
        // }

        Array::operator=(other);
        this->rows = other.rows;
        this->columns = other.columns;

        return *this;
    }

    void print_details(Device const device, Dtype const dtype, size_t const rows, size_t const columns) {
        std::cout << "############## Details ##############\n";
        std::cout << "Shape: " << "(" << rows << ", " << columns << ")" << '\n';
        std::cout << "Device: " << device_to_string(device) << '\n';
        std::cout << "Dtype: " << dtype_to_string(dtype) << '\n';
        std::cout << "#####################################\n";
    }

    // void Matrix::print(bool const hide_middle) const {
    //     print_details(impl->device, impl->dtype, rows, columns);
    //     unsigned char const shift = dtype_to_bytes(impl->dtype);
    //     auto dec3 = [this](size_t const x, size_t &start) {
    //         if (x == 1)
    //             start = rows - 3;
    //     };
    //
    //     std::cout << "[\n";
    //     for (size_t x = 0; x < 2; ++x) {
    //         size_t start = 0;
    //         size_t end = rows;
    //
    //         if (rows > 20 && hide_middle) {
    //             dec3(x, start);
    //             end = start + 3;
    //         }
    //
    //         for (; start < end; ++start) {
    //             std::cout << "   [";
    //
    //             for (size_t y = 0; y < 2; ++y) {
    //                 size_t start_inner = 0;
    //                 size_t end_inner = columns;
    //
    //                 bool inner_hiding{false};
    //
    //                 if (columns > 20 && hide_middle) {
    //                     dec3(y, start_inner);
    //                     end_inner = start_inner + 3;
    //                     inner_hiding = true;
    //                 }
    //
    //                 for (; start_inner < end_inner; ++start_inner) {
    //                     std::cout << std::fixed << std::setprecision(5);
    //
    //                     auto buff = static_cast<char *>(impl->get_raw_buffer());
    //                     buff += (start * columns + start_inner) * shift;
    //                     print_num(buff, impl->dtype);
    //
    //                     if (start_inner != end_inner - 1 || (y == 0 && inner_hiding)) {
    //                         std::cout << ", ";
    //                     }
    //                 }
    //
    //                 if (columns <= 20 || !hide_middle)
    //                     break;
    //
    //                 if (y == 0) {
    //                     std::cout << "..., ";
    //                 }
    //             }
    //
    //             std::cout << "]";
    //             std::cout << "\n";
    //         }
    //
    //         if (rows <= 20 || !hide_middle)
    //             break;
    //
    //         if (x == 0) {
    //             std::cout << "   ..." << "\n";
    //         }
    //     }
    //
    //     std::cout << "]\n\n";
    // }
}
