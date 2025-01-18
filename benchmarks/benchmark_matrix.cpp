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
        size_t const rows{static_cast<size_t>(st.range(0))};
        size_t const col{static_cast<size_t>(st.range(1))};
        size_t const pos{static_cast<size_t>(st.range(2))};

        cobraml::core::func_pos = pos;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(rows, col), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, col), cobraml::core::CPU);

        cobraml::core::Matrix res(1, rows, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, &alpha1, &alpha1);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = col;
        st.counters["type"] = pos;
    }

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProduct)
    ->Args({100, 100, 0})
    ->Args({500, 500, 0})
    ->Args({1000, 1000, 0})
    ->Args({1250, 1250, 0})
    ->Args({1500, 1500, 0})
    ->Args({1750, 1750, 0})
    ->Args({2000, 2000, 0})
    ->Args({2500, 2500, 0})
    ->Args({3000, 3000, 0})
    ->Args({5000, 5000, 0})
    // ->Args({1000, 1000000, 0})
    ->Args({100, 100, 1})
    ->Args({500, 500, 1})
    ->Args({1000, 1000, 1})
    ->Args({1250, 1250, 1})
    ->Args({1500, 1500, 1})
    ->Args({1750, 1750, 1})
    ->Args({2000, 2000, 1})
    ->Args({2500, 2500, 1})
    ->Args({3000, 3000, 1})
    ->Args({5000, 5000, 1})
    // ->Args({1000, 1000000, 1})
    ->Args({100, 100, 2})
    ->Args({500, 500, 2})
    ->Args({1000, 1000, 2})
    ->Args({1250, 1250, 2})
    ->Args({1500, 1500, 2})
    ->Args({1750, 1750, 2})
    ->Args({2000, 2000, 2})
    ->Args({2500, 2500, 2})
    ->Args({3000, 3000, 2})
    ->Args({5000, 5000, 2})
    // ->Args({1000, 1000000, 2})
    ->Args({100, 100, 3})
    ->Args({500, 500, 3})
    ->Args({1000, 1000, 3})
    ->Args({1250, 1250, 3})
    ->Args({1500, 1500, 3})
    ->Args({1750, 1750, 3})
    ->Args({2000, 2000, 3})
    ->Args({2500, 2500, 3})
    ->Args({3000, 3000, 3})
    ->Args({5000, 5000, 3})
    // ->Args({1000, 1000000, 3})
    ->Args({100, 100, 4})
    ->Args({500, 500, 4})
    ->Args({1000, 1000, 4})
    ->Args({1250, 1250, 4})
    ->Args({1500, 1500, 4})
    ->Args({1750, 1750, 4})
    ->Args({2000, 2000, 4})
    ->Args({2500, 2500, 4})
    ->Args({3000, 3000, 4})
    ->Args({5000, 5000, 4})
    // ->Args({1000, 1000000, 4})
    ->Threads(1);
}

BENCHMARK_MAIN();
