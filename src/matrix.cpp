//
// Created by sriram on 12/19/24.
//


#include "matrix.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include "allocator.h"
#include "enums.h"
#include "math_dis.h"

namespace cobraml::core {

    // static size_t sum_vec(std::vector<size_t> const &shape) {
    //
    //     if (shape.empty())
    //         return 0;
    //
    //     size_t sum{1};
    //     for (auto const &num: shape)
    //         sum *= num;
    //
    //     return sum;
    // }

    ///////////////////////// Start of Array ////////////////////////////////

    struct Array::ArrayImpl {
        size_t offset = 0;
        Device device = CPU;
        Dtype dtype = INVALID;
        std::shared_ptr<Buffer> buffer = nullptr;
        Math * m_dispatcher = nullptr;

        ArrayImpl(Device const device, Dtype const dtype, size_t const total_items):
            device(device),
            dtype(dtype),
            buffer(std::make_shared<Buffer>(total_items * dtype_to_bytes(dtype), device)),
            m_dispatcher(get_math_kernels(device)){}

        [[nodiscard]] void *get_raw_buffer() const {
            return static_cast<char *>(buffer->get_p_buffer()) + offset;
        }

        ArrayImpl() = default;
        ArrayImpl(const ArrayImpl&) = default;
        ArrayImpl& operator=(const ArrayImpl&) = default;
    };

    Array::Array(size_t total_bytes, Device device, Dtype dtype):
    impl(std::make_unique<ArrayImpl>(device, dtype, total_bytes)) {
        // std::cout << total_bytes << std::endl;
    }

    Dtype Array::get_dtype() const {
        return this->impl->dtype;
    }

    Device Array::get_device() const {
        return this->impl->device;
    }

    void *Array::get_raw_buffer() const {
        return static_cast<char *>(impl->buffer->get_p_buffer()) + impl->offset;
    }

    Array::~Array() = default;

    Array::Array():impl(std::make_unique<ArrayImpl>()){}

    ///////////////////////// Start of Matrix ////////////////////////////////

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

        Matrix ret;
        ret.rows = 1;
        ret.impl->device = this->get_device();
        ret.impl->dtype = this->get_dtype();
        ret.impl->buffer = this->impl->buffer;
        ret.impl->m_dispatcher = this->impl->m_dispatcher;

        if (this->is_vector()) {
            if (index >= this->get_shape().columns) {
                throw std::out_of_range("index is out of range");
            }

            ret.columns = 1;
            ret.impl->offset = this->impl->offset + dtype_to_bytes(this->get_dtype()) * index;

            return ret;
        }

        if (index >= this->get_shape().rows) {
            throw std::out_of_range("index is out of range");
        }

        ret.columns = this->get_shape().columns;

        ret.impl->offset =
            this->impl->offset + dtype_to_bytes(this->get_dtype()) * this->get_shape().columns * index;

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

    Matrix::Matrix(Matrix const &other): Array(), rows(0), columns(0) {
        columns = other.columns;
        rows = other.rows;
        impl->dtype = other.impl->dtype;
        impl->device = other.impl->device;
        impl->buffer = other.impl->buffer;
        impl->m_dispatcher = other.impl->m_dispatcher;
    }

    Matrix& Matrix::operator=(const Matrix &other) {
        if (this == &other) {
            return *this;
        }

        if (!(this->get_shape() == other.get_shape())) {
            throw std::runtime_error("matrices are not the same shape");
        }

        if (this->get_dtype() != other.get_dtype()) {
            throw std::runtime_error("cannot copy matrix dtype does not match");
        }

        this->impl->buffer->overwrite(
            other.get_raw_buffer(),
            other.get_shape().rows * other.get_shape().columns * dtype_to_bytes(this->get_dtype()),
            this->impl->offset);

        return *this;
    }

    void Matrix::replace_segment(const void *source, size_t const offset, size_t const bytes) const {
        this->impl->buffer->overwrite(source, bytes, offset);
    }

    void gemv(const Matrix &matrix, const Matrix &vector, Matrix &result, const void * alpha, const void * beta) {
        if (!vector.is_vector()) {
            throw std::runtime_error("vector is a matrix");
        }

        if (!result.is_vector()) {
            throw std::runtime_error("result is a matrix");
        }

        if (matrix.columns != vector.columns) {
            throw std::runtime_error("vector and matrix have different columns lengths");
        }

        if (matrix.rows != result.columns) {
            throw std::runtime_error("result must be size 1, rows(matrix)");
        }

        if (matrix.impl->device != vector.impl->device || matrix.impl->device != result.impl->device) {
            throw std::runtime_error("vector, matrix and result are not on the same device");
        }

        if (matrix.impl->dtype != vector.impl->dtype || matrix.impl->dtype != result.impl->dtype) {
            throw std::runtime_error("vector, matrix and result share different dtypes");
        }

        result.impl->m_dispatcher->gemv(
            matrix.get_raw_buffer(),
            vector.get_raw_buffer(),
            result.get_raw_buffer(),
            alpha,
            beta,
            matrix.rows,
            matrix.columns,
            matrix.impl->dtype);
    }

    void print_details(Device const device, Dtype const dtype, size_t const rows, size_t const columns) {
        std::cout << "############## Details ##############\n";
        std::cout << "Shape: " << "(" << rows << ", " << columns << ")" << '\n';
        std::cout << "Device: " << device_to_string(device) << '\n';
        std::cout << "Dtype: " << dtype_to_string(dtype) << '\n';
        std::cout << "#####################################\n";
    }

    void Matrix::print(bool const hide_middle) const {
        print_details(impl->device, impl->dtype, rows, columns);
        unsigned char const shift = dtype_to_bytes(impl->dtype);
        auto dec3 = [this](size_t const x, size_t &start) {
            if (x == 1)
                start = rows - 3;
        };

        std::cout << "[\n";
        for (size_t x = 0; x < 2; ++x) {
            size_t start = 0;
            size_t end = rows;

            if (rows > 20 && hide_middle) {
                dec3(x, start);
                end = start + 3;
            }

            for (; start < end; ++start) {
                std::cout << "   [";

                for (size_t y = 0; y < 2; ++y) {
                    size_t start_inner = 0;
                    size_t end_inner = columns;

                    bool inner_hiding{false};

                    if (columns > 20 && hide_middle) {
                        dec3(y, start_inner);
                        end_inner = start_inner + 3;
                        inner_hiding = true;
                    }

                    for (; start_inner < end_inner; ++start_inner) {
                        std::cout << std::fixed << std::setprecision(5);

                        auto buff = static_cast<char *>(impl->get_raw_buffer());
                        buff += (start * columns + start_inner) * shift;
                        print_num(buff, impl->dtype);

                        if (start_inner != end_inner - 1 || (y == 0 && inner_hiding)) {
                            std::cout << ", ";
                        }
                    }

                    if (columns <= 20 || !hide_middle)
                        break;

                    if (y == 0) {
                        std::cout << "..., ";
                    }
                }

                std::cout << "]";
                std::cout << "\n";
            }

            if (rows <= 20 || !hide_middle)
                break;

            if (x == 0) {
                std::cout << "   ..." << "\n";
            }
        }

        std::cout << "]\n\n";
    }

    // struct Tensor::TensorImpl {
    //     size_t offset = 0;
    //     std::vector<size_t> shape;
    //     Device device = CPU;
    //     Dtype dtype = INVALID;
    //     std::shared_ptr<Buffer> buffer = nullptr;
    //     Math * m_dispatcher = nullptr;
    //
    //     TensorImpl() = default;
    //     TensorImpl(const TensorImpl&) = default;
    //     TensorImpl& operator=(const TensorImpl&) = default;
    //
    //     TensorImpl(std::vector<size_t> shape, Device const device, Dtype const dtype):
    //         shape(std::move(shape)), device(device), dtype(dtype) {
    //         size_t const sum = sum_vec(shape);
    //         buffer = std::make_shared<Buffer>(sum * dtype_to_bytes(dtype), device);
    //         m_dispatcher = get_math_kernels(device);
    //     }
    // };
    //
    // Tensor::Tensor(std::vector<size_t> shape, Device device, Dtype dtype):
    //     impl(std::make_unique<TensorImpl>(std::move(shape), device, dtype)){
    // }
    //
    // void Tensor::test_nothing() {
    //     auto const m = Matrix();
    //     // Matrix const m(1, 1, this->impl->device, this->impl->dtype);
    //     std::cout << (m.impl->buffer == nullptr) << std::endl;
    // }
    //
    // Tensor::~Tensor() = default;
}
