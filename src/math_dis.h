//
// Created by sriram on 12/19/24.
//

#ifndef MATHDISPATCHER_H
#define MATHDISPATCHER_H
#include "enums.h"
#include <memory>

namespace cobraml::core {

    class Math {
    public:
        virtual ~Math() = default;
        virtual void batched_dot_product(
            const void * matrix, const void * vector, void * dest, size_t rows, size_t columns, Dtype dtype) = 0;
    };

    extern std::array<std::unique_ptr<Math>, 3> global_math_kernels;

    Math * get_math_kernels(Device device);
}

#endif //MATHDISPATCHER_H
