// perf/sycl/perf_prefill_gemm.cpp - Intel Xe2 Native Prefill Tiled GEMM Scaling Benchmark
//
// Benchmarks prompt prefilling throughput (TFLOPS) using native SYCL tiled GEMM across chunk sizes:
// M in [128, 256, 512, 1024, 2048, 4096]
//
// Target Projections from Qwen3.8-Flash-Next:
// - Gate/Up Projection: [K=2560, N=10240]
// - Down Projection: [K=10240, N=2560]
// - Attention/PLE Projection: [K=2560, N=2560]
//
// Evaluates computational throughput against Intel Arc B580 peak (117 TFLOPS BF16).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl::ext::oneapi::experimental::matrix;

namespace {

// Tiled XMX Prefill GEMM Kernel (Tile16: 16x16x16)
template <size_t TM = 16, size_t TN = 16, size_t TK = 16>
void prefill_tiled_gemm_kernel(sycl::nd_item<2> item, const sycl::half* __restrict__ a,
                               const sycl::half* __restrict__ b, float* __restrict__ c, size_t M, size_t N,
                               size_t K) {
    auto sg = item.get_sub_group();
    const size_t tile_m = item.get_group(0) * TM;
    const size_t tile_n = item.get_group(1) * TN;

    if (tile_m >= M || tile_n >= N) return;

    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
    joint_matrix_fill(sg, sub_c, 0.0f);

#pragma unroll 4
    for (size_t k = 0; k < K; k += TK) {
        joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
        joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;

        auto ptr_a = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            a + tile_m * K + k);
        auto ptr_b = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            b + k * N + tile_n);

        joint_matrix_load(sg, sub_a, ptr_a, K);
        joint_matrix_load(sg, sub_b, ptr_b, N);
        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
    }

    auto ptr_c = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        c + tile_m * N + tile_n);
    joint_matrix_store(sg, sub_c, ptr_c, N, layout::row_major);
}

}  // namespace

// ============================================================================
// Benchmark: BM_Xe2_Prefill_GEMM
// ============================================================================

static void BM_Xe2_Prefill_GEMM(benchmark::State& state) {
    const size_t M = state.range(0);
    const size_t K = state.range(1);
    const size_t N = state.range(2);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    auto* d_a = sycl::malloc_device<sycl::half>(M * K, q);
    auto* d_b = sycl::malloc_device<sycl::half>(K * N, q);
    auto* d_c = sycl::malloc_device<float>(M * N, q);

    q.fill(d_a, sycl::half(0.1f), M * K);
    q.fill(d_b, sycl::half(0.05f), K * N);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                           prefill_tiled_gemm_kernel<16, 16, 16>(item, d_a, d_b, d_c, M, N, K);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev = q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                                 [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                     prefill_tiled_gemm_kernel<16, 16, 16>(item, d_a, d_b, d_c, M, N, K);
                                 });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double ops = 2.0 * (double)M * (double)K * (double)N;
    state.counters["TFLOPS"]         = benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)M, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Chunk_M"]        = benchmark::Counter((double)M);

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
}

// Sweep Chunk Sizes across Gate/Up, Down, and Attention Projections
BENCHMARK(BM_Xe2_Prefill_GEMM)->Args({128, 2560, 10240})->Args({512, 2560, 10240})->Args({2048, 2560, 10240})->UseManualTime();
BENCHMARK(BM_Xe2_Prefill_GEMM)->Args({128, 10240, 2560})->Args({512, 10240, 2560})->Args({2048, 10240, 2560})->UseManualTime();
BENCHMARK(BM_Xe2_Prefill_GEMM)->Args({128, 2560, 2560})->Args({512, 2560, 2560})->Args({2048, 2560, 2560})->UseManualTime();

BENCHMARK_MAIN();
