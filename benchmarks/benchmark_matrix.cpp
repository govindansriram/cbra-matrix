//
// Created by sriram on 12/19/24.
//

#include <omp.h>
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
        omp_set_num_threads(20); // Use the thread count from benchmark range

        size_t const rows = st.range(0);
        size_t const col = st.range(1);
        size_t const pos = st.range(2);

        cobraml::core::func_pos = pos;
        cobraml::core::Matrix const mat = from_vector(
            create_vector(rows, col), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, col), cobraml::core::CPU);

        for (auto _: st) {
            batched_dot_product(mat, vec);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = col;
        st.counters["type"] = pos;
    }

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProduct)
    // ->Args({100, 100, 0})
    // ->Args({500, 500, 0})
    // ->Args({1000, 1000, 0})
    // ->Args({3000, 3000, 0})
    // ->Args({5000, 5000, 0})
    // ->Args({1000, 1000000, 0})
    ->Args({100, 100, 1})
    ->Args({500, 500, 1})
    ->Args({1000, 1000, 1})
    ->Args({3000, 3000, 1})
    ->Args({5000, 5000, 1})
    ->Args({100, 100, 2})
    ->Args({500, 500, 2})
    ->Args({1000, 1000, 2})
    ->Args({3000, 3000, 2})
    ->Args({5000, 5000, 2})
    // ->Args({1000, 1000000, 1})
    ->Args({100, 100, 3})
    ->Args({500, 500, 3})
    ->Args({1000, 1000, 3})
    ->Args({3000, 3000, 3})
    ->Args({5000, 5000, 3})
    ->Threads(1);
}

BENCHMARK_MAIN();
