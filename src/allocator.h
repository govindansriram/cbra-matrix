//
// Created by sriram on 11/24/24.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cstddef>
#include <memory>
#include "enums.h"

namespace cobraml::core {

    class Allocator {
    public:
        virtual ~Allocator() = default;
        virtual void *malloc(std::size_t bytes) = 0;
        virtual void *calloc(std::size_t bytes) = 0;
        virtual void mem_copy(void *dest, void *source, std::size_t bytes, bool overlap) = 0;
        virtual void free(void *ptr) = 0;
    };

    extern std::array<std::unique_ptr<Allocator>, 3> global_allocators;

    Allocator * get_allocator(Device device);

    class Buffer {
        void * p_buffer;
        Allocator * p_allocator;
        Device device;

    public:
        Buffer() = delete;
        explicit Buffer(size_t bytes, Device device);
        ~Buffer();
        [[nodiscard]] void * get_p_buffer() const;
    };
}

#endif //ALLOCATOR_H
