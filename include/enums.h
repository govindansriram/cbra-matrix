//
// Created by sriram on 12/19/24.
//

#ifndef ENUMS_H
#define ENUMS_H

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
        FLOAT64
    };

    bool operator<(Dtype lhs, Dtype rhs);

    constexpr char dtype_to_bytes(Dtype const type) {
        switch (type) {
            case INT8: return 1;
            case INT16: return 2;
            case INT32: return 4;
            case INT64: return 8;
            case FLOAT32: return 4;
            case FLOAT64: return 8;
        }

        return -1;
    }
}

#endif //ENUMS_H
