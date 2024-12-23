//
// Created by sriram on 12/15/24.
//


#include <gtest/gtest.h>
#include "matrix.h"
#include "enums.h"

class MatrixTest : public testing::Test {
protected:

    cobraml::core::Matrix m1{};
    cobraml::core::Matrix m2;

    MatrixTest(): m2(23,1, cobraml::core::CPU, cobraml::core::INT8){
        auto const mat = std::vector<std::vector<int>>{
            {0, 1, 2, 3, 4},
            {5, 6, 7, 8, 9},
            {10, 11, 12, 13, 14},
            {15, 16, 17, 18, 19},
        };

        m1 = cobraml::core::from_vector<int>(mat, cobraml::core::CPU);
    }

    // ~QueueTest() override = default;
};

TEST_F(MatrixTest, test_meta_data) {
    ASSERT_EQ(m2.get_dtype(), cobraml::core::INT8);
    const auto [rows, columns] = m2.get_shape();
    ASSERT_EQ(columns, 1);
    ASSERT_EQ(rows, 23);
    ASSERT_EQ(m2.get_device(), cobraml::core::CPU);
}

TEST(MatrixTestFunc, test_invalid_constructor) {
    ASSERT_THROW(
        cobraml::core::Matrix mat(10, 20, cobraml::core::CPU, cobraml::core::INVALID),
        std::runtime_error);
}