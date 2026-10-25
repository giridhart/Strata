// perf/sycl/perf_q8_quant_multi.cpp - Batched Multi-Activation Q8 Quantization Benchmark on Intel Xe2
//
// Evaluates performance improvements on Intel Xe2 execution units comparing:
//   1. Baseline Separate: N independent single-item kernel launches (q8_quant_single).
//   2. Batched Multi: Single 3D kernel dispatch (q8_quant_multi) aggregating all N activation tensors.
//   3. Batched with Deduplication: Demonstrates elimination of redundant activation quantizations
//      when multiple projection heads/experts share common inputs (e.g., dual gate/up projections).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/q8_quant_multi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct QuantBatchFixture {
    int N;
    int rows;
    int K;

    sycl::queue q;
    std::vector<float*> d_x;
    std::vector<int8_t*> d_q;
    std::vector<float*> d_d;

    strata::kernels::QuantItems batch_all;
    strata::kernels::QuantItems batch_dedup;

    QuantBatchFixture(int num_items, int num_rows, int k_dim)
        : N(num_items), rows(num_rows), K(k_dim),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {

        d_x.resize(N);
        d_q.resize(N);
        d_d.resize(N);

        const size_t ne = static_cast<size_t>(rows) * K;
        const size_t ns = static_cast<size_t>(rows) * (K / 32);

        std::mt19937 rng(1337 + N + rows + K);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        std::vector<float> h_x(ne);

        for (int i = 0; i < N; ++i) {
            d_x[i] = sycl::malloc_device<float>(ne, q);
            d_q[i] = sycl::malloc_device<int8_t>(ne, q);
            d_d[i] = sycl::malloc_device<float>(ns, q);

            for (auto& v : h_x) v = dist(rng);
            q.memcpy(d_x[i], h_x.data(), ne * sizeof(float));

            strata::kernels::quant_add(batch_all, d_x[i], K, rows, K, d_q[i], d_d[i], /*nodedup=*/true);
        }

        // Setup batch_dedup where every pair (2*i, 2*i+1) shares the same input pointer
        for (int i = 0; i < N; ++i) {
            float* src_x = d_x[(i / 2) * 2]; // 50% duplicate inputs
            strata::kernels::quant_add(batch_dedup, src_x, K, rows, K, d_q[i], d_d[i], /*nodedup=*/false);
        }

        q.wait();
    }

    ~QuantBatchFixture() {
        for (int i = 0; i < N; ++i) {
            sycl::free(d_x[i], q);
            sycl::free(d_q[i], q);
            sycl::free(d_d[i], q);
        }
    }
};

static void BM_Q8Quant_Separate(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    const int K = static_cast<int>(state.range(1));
    const int rows = 2;

    QuantBatchFixture f(N, rows, K);

    // Warm-up
    for (int it = 0; it < 5; ++it) {
        for (int i = 0; i < f.N; ++i) {
            strata::kernels::q8_quant_single(f.d_x[i], f.K, f.rows, f.K, f.d_q[i], f.d_d[i], f.q);
        }
        f.q.wait();
    }

    for (auto _ : state) {
        for (int i = 0; i < f.N; ++i) {
            strata::kernels::q8_quant_single(f.d_x[i], f.K, f.rows, f.K, f.d_q[i], f.d_d[i], f.q);
        }
        f.q.wait();
    }

    const double bytes_read = static_cast<double>(N) * rows * K * sizeof(float);
    const double bytes_written = static_cast<double>(N) * rows * (K * sizeof(int8_t) + (K / 32) * sizeof(float));
    state.SetBytesProcessed(static_cast<int64_t>(bytes_read + bytes_written) * static_cast<int64_t>(state.iterations()));
    state.SetItemsProcessed(static_cast<int64_t>(N) * static_cast<int64_t>(state.iterations()));
}

static void BM_Q8Quant_Batched(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    const int K = static_cast<int>(state.range(1));
    const int rows = 2;

    QuantBatchFixture f(N, rows, K);

    // Warm-up
    for (int it = 0; it < 5; ++it) {
        strata::kernels::q8_quant_multi(f.batch_all, f.q);
        f.q.wait();
    }

    for (auto _ : state) {
        strata::kernels::q8_quant_multi(f.batch_all, f.q);
        f.q.wait();
    }

    const double bytes_read = static_cast<double>(N) * rows * K * sizeof(float);
    const double bytes_written = static_cast<double>(N) * rows * (K * sizeof(int8_t) + (K / 32) * sizeof(float));
    state.SetBytesProcessed(static_cast<int64_t>(bytes_read + bytes_written) * static_cast<int64_t>(state.iterations()));
    state.SetItemsProcessed(static_cast<int64_t>(N) * static_cast<int64_t>(state.iterations()));
}

static void BM_Q8Quant_BatchedDedup(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    const int K = static_cast<int>(state.range(1));
    const int rows = 2;

    QuantBatchFixture f(N, rows, K);

    // Warm-up
    for (int it = 0; it < 5; ++it) {
        strata::kernels::q8_quant_multi(f.batch_dedup, f.q);
        f.q.wait();
    }

    for (auto _ : state) {
        strata::kernels::q8_quant_multi(f.batch_dedup, f.q);
        f.q.wait();
    }

    const double bytes_read = static_cast<double>(N) * rows * K * sizeof(float);
    const double bytes_written = static_cast<double>(N) * rows * (K * sizeof(int8_t) + (K / 32) * sizeof(float));
    state.SetBytesProcessed(static_cast<int64_t>(bytes_read + bytes_written) * static_cast<int64_t>(state.iterations()));
    state.SetItemsProcessed(static_cast<int64_t>(N) * static_cast<int64_t>(state.iterations()));
}

void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int n : {4, 8, 16}) {
        for (int k : {2048, 2560, 4096}) {
            b->Args({n, k});
        }
    }
}

BENCHMARK(BM_Q8Quant_Separate)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Q8Quant_Batched)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Q8Quant_BatchedDedup)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond);

} // namespace

BENCHMARK_MAIN();
