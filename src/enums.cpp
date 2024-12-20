//
// Created by sriram on 12/19/24.
//

#include "enums.h"

namespace cobraml::core {
    bool is_float(Dtype const type) {
        return type == FLOAT32 || type == FLOAT64;
    }

    bool operator<(Dtype const lhs, Dtype const rhs) {
        bool const l_if = is_float(lhs);
        bool const r_if = is_float(rhs);

        if (l_if == r_if) {
            return dtype_to_bytes(lhs) < dtype_to_bytes(rhs);
        }

        if (l_if) {
            return false;
        }

        if (r_if) {
            return true;
        }

        return false;
    }
}