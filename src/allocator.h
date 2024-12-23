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
        virtual void mem_copy(void *dest, const void *source, std::size_t bytes) = 0;
        virtual void free(void *ptr) = 0;
    };

    extern std::array<std::unique_ptr<Allocator>, 3> global_allocators;

    Allocator * get_allocator(Device device);

    class Buffer {
        void * p_buffer = nullptr;
        Allocator * p_allocator;
        Device device;

    public:
        Buffer() = delete;
        explicit Buffer(size_t bytes, Device device);
        ~Buffer();
        [[nodiscard]] void * get_p_buffer() const;
        Buffer(Buffer&) = delete;
        Buffer& operator=(Buffer&) = delete;

        /**
         * overwrite a segment of the buffer with new data
         * @param source the new data
         * @param byte_count how many bytes to replace
         * @param offset the starting position to start overwriting in the original buffer
         */
        void overwrite(const void * source, size_t byte_count, size_t offset = 0) const;
    };
}

#endif //ALLOCATOR_H
