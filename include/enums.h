//
// Created by sriram on 12/19/24.
//

#ifndef ENUMS_H
#define ENUMS_H
#include <cstdint>
#include <stdexcept>

namespace cobraml::core {
    enum Device {
        CPU,    // standard naive implementations
        GPU,    // GPU implementations
        CPU_X   // Accelerated CPU implementation
    };

    enum Dtype {
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
        INVALID
    };

    bool operator<(Dtype lhs, Dtype rhs);

    constexpr unsigned char dtype_to_bytes(Dtype const type) {
        switch (type) {
            case INT8: return 1;
            case INT16: return 2;
            case INT32: return 4;
            case INT64: return 8;
            case FLOAT32: return 4;
            case FLOAT64: return 8;
            case INVALID: return 0;
        }

        return 0;
    }

    std::string dtype_to_string(Dtype dtype);

    inline void is_invalid(Dtype const dtype) {
        if (dtype == INVALID)
            throw std::runtime_error("invalid dtype provided");
    }

    template<typename T>
    struct get_dtype_from_type {
        static constexpr Dtype type = INVALID;
    };

    template<>
    struct get_dtype_from_type<int8_t> {
        static constexpr Dtype type = INT8;
    };

    template<>
    struct get_dtype_from_type<int16_t> {
        static constexpr Dtype type = INT16;
    };

    template<>
    struct get_dtype_from_type<int32_t> {
        static constexpr Dtype type = INT32;
    };

    template<>
    struct get_dtype_from_type<int64_t> {
        static constexpr Dtype type = INT64;
    };

    template<>
    struct get_dtype_from_type<float> {
        static constexpr Dtype type = FLOAT32;
    };

    template<>
    struct get_dtype_from_type<double> {
        static constexpr Dtype type = FLOAT64;
    };
}

#endif //ENUMS_H
