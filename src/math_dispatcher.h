//
// Created by sriram on 12/19/24.
//

#ifndef MATHDISPATCHER_H
#define MATHDISPATCHER_H

namespace cobraml::core {

    class MathDispatcher {
    public:
        virtual ~MathDispatcher() = default;
        virtual void batched_dot_product(void * matrix, void * vector, void * dest, size_t  rows, size_t columns) = 0;
    };
}

#endif //MATHDISPATCHER_H
