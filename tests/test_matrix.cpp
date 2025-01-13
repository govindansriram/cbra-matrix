//
// Created by sriram on 12/15/24.
//


#include <omp.h>
#include <random>
#include <gtest/gtest.h>
#include "matrix.h"
#include "enums.h"

std::vector<std::vector<double> > create_vector(size_t const rows, size_t const columns) {
    std::vector ret(rows, std::vector(columns, 0.0));

    std::uniform_int_distribution<> unif{0, 10};
    const size_t seed = static_cast<size_t>(std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine gen{seed};

    for (auto &vector: ret) {
        for (auto &num: vector) {
            num = static_cast<double>(unif(gen));
        }
    }

    return ret;
}

bool check_dot_product(
    std::vector<std::vector<double> > const &vec,
    std::vector<std::vector<double> > const &mat,
    const double * result) {

    for (size_t i = 0; i < mat.size(); ++i) {
        double base = 0;

        for (size_t j = 0; j < vec[0].size(); ++j) {
            base += mat[i][j] * vec[0][j];
        }

        if (base != result[i]) {
            return false;
        }
    }

    return true;
}

bool arr_eq(const int *arr_one, const int *arr_two, size_t const len) {
    for (size_t i = 0; i < len; ++i) {
        if (arr_one[i] != arr_two[i])
            return false;
    }

    return true;
}

TEST(MatrixTestFunc, test_invalid_constructor) {
    ASSERT_THROW(
        cobraml::core::Matrix mat(10, 20, cobraml::core::CPU, cobraml::core::INVALID),
        std::runtime_error);
}

class MatrixTest : public testing::Test {
protected:
    cobraml::core::Matrix test_mat;
    cobraml::core::Matrix mat1{};
    cobraml::core::Matrix vec1{};
    cobraml::core::Matrix res1{};
    cobraml::core::Matrix mat2{};
    cobraml::core::Matrix vec2{};
    cobraml::core::Matrix res2{};

    MatrixTest(): test_mat(23, 1, cobraml::core::CPU, cobraml::core::INT8) {
        auto const mat{
            std::vector<std::vector<int> >{
                {0, 1, 2, 3, 4},
                {5, 6, 7, 8, 9},
                {10, 11, 12, 13, 14},
                {15, 16, 17, 18, 19},
            }
        };

        auto const vec{
            std::vector<std::vector<int> >{
                {1, 2, 3, 4, 5},
            }
        };

        auto const res{
            std::vector<std::vector<int> >{
                    {2, 1, 2, 1},
                }
        };

        std::vector<double> const temp_vec{
            {10.5, 2.5, 4.5, 10, 5.5},
        };

        std::vector _vec2{1, std::vector(100000, 0.0)};
        std::vector _res2{1, std::vector(1000, 0.0)};

        size_t start = 0;
        for (double &data: _vec2[0]) {
            if (start != temp_vec.size()) {
                data = temp_vec[start];
                ++start;
            }else {
                start = 0;
                data = temp_vec[start];
                ++start;
            }
        }

        std::vector const _mat2{1000, _vec2[0]};

        mat1 = cobraml::core::from_vector<int>(mat, cobraml::core::CPU);
        vec1 = cobraml::core::from_vector<int>(vec, cobraml::core::CPU);
        res1 = cobraml::core::from_vector<int>(res, cobraml::core::CPU);

        mat2 = cobraml::core::from_vector<double>(_mat2, cobraml::core::CPU);
        vec2 = cobraml::core::from_vector<double>(_vec2, cobraml::core::CPU);
        res2 = cobraml::core::from_vector<double>(_res2, cobraml::core::CPU);
    }
};

TEST_F(MatrixTest, test_meta_data) {
    ASSERT_EQ(test_mat.get_dtype(), cobraml::core::INT8);
    const auto [rows, columns] = test_mat.get_shape();
    ASSERT_EQ(columns, 1);
    ASSERT_EQ(rows, 23);
    ASSERT_EQ(test_mat.get_device(), cobraml::core::CPU);
}

TEST_F(MatrixTest, gemv_alpha_beta) {
    omp_set_num_threads(20); // Use the thread count from benchmark range

    constexpr int alpha = 2;
    constexpr int beta = -1;

    constexpr int expected[]{
        78, 229, 378, 529
    };

    gemv(mat1, vec1, res1, &alpha, &beta);
    const int *res1_buff = cobraml::core::get_buffer<int>(res1);
    ASSERT_EQ(arr_eq(res1_buff, expected, sizeof(expected) / sizeof(int)), true);
}

// TEST_F(MatrixTest, dot_product) {
//     omp_set_num_threads(20); // Use the thread count from benchmark range
//     cobraml::core::Matrix const res1 = gemv(mat1, vec1);
//
//     const int *res1_buff = cobraml::core::get_buffer<int>(res1);
//     constexpr int expected[]{
//         40, 115, 190, 265
//     };
//
//     ASSERT_EQ(arr_eq(res1_buff, expected, sizeof(expected) / sizeof(int)), true);
//
//     auto const matt = create_vector(1000, 100000);
//     auto const vect = create_vector(1, 100000);
//
//     auto const matt1 = cobraml::core::from_vector(matt, cobraml::core::CPU);
//     auto const vect2 = cobraml::core::from_vector(vect, cobraml::core::CPU);
//
//     cobraml::core::Matrix const res2 = gemv(matt1, vect2);
//     const auto *res2_buff = cobraml::core::get_buffer<double>(res2);
//
//     ASSERT_EQ(check_dot_product(vect, matt, res2_buff), true);
//
// }
