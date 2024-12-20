//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_ALLOCATOR_H
#define STANDARD_ALLOCATOR_H

#include "../allocator.h"

namespace cobraml::core {
    class StandardAllocator final : public Allocator {
        void *malloc(std::size_t bytes) override;
        void *calloc(std::size_t bytes) override;
        void mem_copy(void *dest, void *source, std::size_t bytes, bool overlap) override;
        void free(void *ptr) override;
    };
}

#endif //STANDARD_ALLOCATOR_H