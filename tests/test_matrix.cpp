//
// Created by sriram on 12/15/24.
//


#include <gtest/gtest.h>
#include "matrix.h"
#include "enums.h"

TEST(matrix_tests, print_matrices) {
    auto mat = cobraml::core::Matrix(21, 21, cobraml::core::CPU, cobraml::core::INT16);

    ASSERT_EQ(batched_dot_product(mat, mat), 21);
    // mat.print();
}