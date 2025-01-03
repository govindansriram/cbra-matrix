//
// Created by sriram on 12/15/24.
//


#include <random>
#include <gtest/gtest.h>
#include "matrix.h"
#include "enums.h"

std::vector<std::vector<double> > create_vector(size_t const rows, size_t const columns) {
    std::vector ret(rows, std::vector(columns, 0.0));

    std::uniform_int_distribution<> unif{0, 10};
    size_t seed = static_cast<size_t>(std::chrono::system_clock::now().time_since_epoch().count());
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

        // std::cout << "result " << result[i] << std::endl;
        // std::cout << "base " << base << std::endl;

        if (base != result[i]) {
            // std::cout << result[i] << std::endl;
            // std::cout << base << std::endl;
            // std::cout << i << std::endl;
            return false;
        }
    }

    return true;
}

class MatrixTest : public testing::Test {
protected:
    cobraml::core::Matrix m1{};
    cobraml::core::Matrix m2;
    cobraml::core::Matrix m3{};
    cobraml::core::Matrix m4{};
    cobraml::core::Matrix m5{};

    MatrixTest(): m2(23, 1, cobraml::core::CPU, cobraml::core::INT8) {
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

        std::vector<double> const temp_vec{
            {10.5, 2.5, 4.5, 10, 5.5},
        };

        std::vector vec2{1, std::vector(100000, 0.0)};

        size_t start = 0;
        for (double &data: vec2[0]) {
            if (start != temp_vec.size()) {
                data = temp_vec[start];
                ++start;
            }else {
                start = 0;
                data = temp_vec[start];
                ++start;
            }
        }

        // std::cout << vec2.size()<< std::endl;

        // for (double const &data: vec2[0]) {
        //     std::cout << data<< std::endl;
        // }

        std::vector const mat2{1000, vec2[0]};

        m1 = cobraml::core::from_vector<int>(mat, cobraml::core::CPU);
        m3 = cobraml::core::from_vector<int>(vec, cobraml::core::CPU);
        m4 = cobraml::core::from_vector<double>(mat2, cobraml::core::CPU);
        m5 = cobraml::core::from_vector<double>(vec2, cobraml::core::CPU);
    }
};

TEST_F(MatrixTest, test_meta_data) {
    ASSERT_EQ(m2.get_dtype(), cobraml::core::INT8);
    const auto [rows, columns] = m2.get_shape();
    ASSERT_EQ(columns, 1);
    ASSERT_EQ(rows, 23);
    ASSERT_EQ(m2.get_device(), cobraml::core::CPU);
}

bool arr_eq(const int *arr_one, const int *arr_two, size_t const len) {
    for (size_t i = 0; i < len; ++i) {
        if (arr_one[i] != arr_two[i])
            return false;
    }

    return true;
}

TEST_F(MatrixTest, dot_product) {
    cobraml::core::Matrix const res1 = batched_dot_product(m1, m3);

    const int *res1_buff = cobraml::core::get_buffer<int>(res1);
    constexpr int expected[]{
        40, 115, 190, 265
    };

    ASSERT_EQ(arr_eq(res1_buff, expected, 4), true);

    auto const matt = create_vector(1000, 100000);
    auto const vect = create_vector(1, 100000);

    auto const matt1 = cobraml::core::from_vector(matt, cobraml::core::CPU);
    auto const vect2 = cobraml::core::from_vector(vect, cobraml::core::CPU);

    cobraml::core::Matrix const res2 = batched_dot_product(matt1, vect2);
    const auto *res2_buff = cobraml::core::get_buffer<double>(res2);

    ASSERT_EQ(check_dot_product(vect, matt, res2_buff), true);

}

TEST(MatrixTestFunc, test_invalid_constructor) {
    ASSERT_THROW(
        cobraml::core::Matrix mat(10, 20, cobraml::core::CPU, cobraml::core::INVALID),
        std::runtime_error);
}
