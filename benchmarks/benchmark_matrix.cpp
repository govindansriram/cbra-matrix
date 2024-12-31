//
// Created by sriram on 12/19/24.
//

#include <benchmark/benchmark.h>
#include <random>
#include "matrix.h"

namespace {
    class CPUFixture : public benchmark::Fixture {
    public:
        double lower_bound{0};
        double upper_bound{10000};
        std::uniform_real_distribution<> unif{lower_bound, upper_bound};
        std::default_random_engine gen{108};

        std::vector<std::vector<double> > create_vector(size_t const rows, size_t const columns) {
            std::vector ret(rows, std::vector(columns, 0.0));

            for (auto &vector: ret) {
                for (auto &num: vector) {
                    num = unif(gen);
                }
            }

            return ret;
        }
    };

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProduct)(benchmark::State &st) {
        size_t const size = st.range(0);
        size_t const pos = st.range(1);

        cobraml::core::func_pos = pos;
        cobraml::core::Matrix const mat = from_vector(
            create_vector(size, size), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, size), cobraml::core::CPU);

        for (auto _: st) {
            batched_dot_product(mat, vec);
        }

        st.counters["rows"] = size;
        st.counters["columns"] = size;
        st.counters["type"] = pos;
    }

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProduct)
    // ->Args({100, 0})
    // ->Args({500, 0})
    // ->Args({1000, 0})
    // ->Args({3000, 0})
    ->Args({5000, 0})
    // ->Args({100, 1})
    // ->Args({500, 1})
    // ->Args({1000, 1})
    // ->Args({3000, 1})
    ->Args({5000, 1})
    ->Args({100, 2})
    ->Args({500, 2})
    ->Args({1000, 2})
    ->Args({3000, 2})
    ->Args({5000, 2})
    ->Threads(1);
}

BENCHMARK_MAIN();
