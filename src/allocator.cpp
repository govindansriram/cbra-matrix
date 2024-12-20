//
// Created by sriram on 12/15/24.
//

#include "allocator.h"
#include "standard_allocator.h"

namespace cobraml::core {

    std::array<std::unique_ptr<Allocator>, 3> global_allocators{
        std::make_unique<StandardAllocator>(),
    };

    Allocator * get_allocator(Device const device) {
        return global_allocators[device].get();
    }

    Buffer::Buffer(size_t const bytes, Device const device)
        : p_allocator(get_allocator(device)), device(device) {
        p_buffer = p_allocator->calloc(bytes);
    }

    Buffer::~Buffer() {
        p_allocator->free(p_buffer);
    }

    void *Buffer::get_p_buffer() const {
        return p_buffer;
    }

}
