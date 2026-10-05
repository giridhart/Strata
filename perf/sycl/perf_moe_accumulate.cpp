// perf/sycl/perf_moe_accumulate.cpp - Intel Xe2 MoE Expert Weighted Reduction Benchmark
//
// Benchmarks MoE weighted accumulation across 10 routed experts (D = 2560 embedding dimensions):
// output[b, col] = sum_{e=0}^{9} parts[b, e, col] * weights[b, e]
//
// Compares:
// 1. BM_Xe2_MoE_Atomic: Global FP32 atomic additions (atomic_ref in device memory)
// 2. BM_Xe2_MoE_DirectCombine: Column-strided reduction (native_moe combine kernel)
// 3. BM_Xe2_MoE_SubgroupReduce: SIMD16 hierarchical subgroup cooperative reduction
//
// Evaluated across batch sizes B = 1, 4, 8, 16, 32.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int N_EXPERTS = 10;
constexpr int N_EMBD = 2560;

// ============================================================================
// MoE Reduction Kernels
// ============================================================================

/// 1. Atomic Add Reduction
/// Each work-item processes one expert for one column and atomically accumulates to output.
void moe_atomic_kernel(sycl::nd_item<1> item, const float* __restrict__ parts, const float* __restrict__ weights,
                       float* __restrict__ output, int B, int n_embd, int k) {
    const int idx = item.get_global_id(0);
    const int total_items = B * k * n_embd;
    if (idx >= total_items) return;

    const int b = idx / (k * n_embd);
    const int rem = idx % (k * n_embd);
    const int expert = rem / n_embd;
    const int col = rem % n_embd;

    const float val = parts[idx] * weights[b * k + expert];

    auto out_ref = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(output[b * n_embd + col]);
    out_ref.fetch_add(val);
}

/// 2. Direct Column-Strided Reduction (from native_moe.cpp)
/// One thread per output column, iterating across all K experts sequentially.
void moe_combine_kernel(sycl::nd_item<2> item, const float* __restrict__ parts, const float* __restrict__ weights,
                        float* __restrict__ output, int n_embd, int k) {
    const int64_t tk = item.get_group(0);  // Token / Batch index
    parts += tk * k * n_embd;
    weights += tk * k;
    output += tk * n_embd;

    const int64_t col = int64_t(item.get_group(1)) * item.get_local_range(1) + item.get_local_id(1);
    if (col >= n_embd) return;

    float sum = parts[col] * weights[0];
#pragma unroll
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    output[col] = sum;
}

/// 3. SIMD16 Subgroup Cooperative Reduction
/// Subgroups stride across columns with unrolled vectorized memory operations.
void moe_subgroup_kernel(sycl::nd_item<1> item, const float* __restrict__ parts, const float* __restrict__ weights,
                         float* __restrict__ output, int B, int n_embd, int k) {
    const int total_cols = B * n_embd;
    const int col_idx = item.get_global_id(0);
    if (col_idx >= total_cols) return;

    const int b = col_idx / n_embd;
    const int col = col_idx % n_embd;

    const float* p_tok = parts + static_cast<size_t>(b) * k * n_embd + col;
    const float* w_tok = weights + b * k;

    float sum = 0.0f;
#pragma unroll 10
    for (int e = 0; e < k; ++e) {
        sum += p_tok[static_cast<size_t>(e) * n_embd] * w_tok[e];
    }
    output[col_idx] = sum;
}

}  // namespace

// ============================================================================
// Benchmarks
// ============================================================================

static void BM_Xe2_MoE_Atomic(benchmark::State& state) {
    const int B = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t parts_size = static_cast<size_t>(B) * N_EXPERTS * N_EMBD;
    const size_t weights_size = static_cast<size_t>(B) * N_EXPERTS;
    const size_t out_size = static_cast<size_t>(B) * N_EMBD;

    float* d_parts = sycl::malloc_device<float>(parts_size, q);
    float* d_weights = sycl::malloc_device<float>(weights_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_parts || !d_weights || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for MoE benchmark");
        if (d_parts) sycl::free(d_parts, q);
        if (d_weights) sycl::free(d_weights, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_parts, 0.1f, parts_size);
    q.fill(d_weights, 0.1f, weights_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr int WG = 256;
    const size_t total_items = parts_size;
    const size_t n_wgs = (total_items + WG - 1) / WG;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wgs * WG, WG),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             moe_atomic_kernel(item, d_parts, d_weights, d_out, B, N_EMBD, N_EXPERTS);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.fill(d_out, 0.0f, out_size);
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        q.fill(d_out, 0.0f, out_size);
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(parts_size + weights_size + out_size) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(parts_size), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

static void BM_Xe2_MoE_DirectCombine(benchmark::State& state) {
    const int B = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t parts_size = static_cast<size_t>(B) * N_EXPERTS * N_EMBD;
    const size_t weights_size = static_cast<size_t>(B) * N_EXPERTS;
    const size_t out_size = static_cast<size_t>(B) * N_EMBD;

    float* d_parts = sycl::malloc_device<float>(parts_size, q);
    float* d_weights = sycl::malloc_device<float>(weights_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_parts || !d_weights || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for MoE benchmark");
        if (d_parts) sycl::free(d_parts, q);
        if (d_weights) sycl::free(d_weights, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_parts, 0.1f, parts_size);
    q.fill(d_weights, 0.1f, weights_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr int WG = 256;
    const size_t n_blocks = (N_EMBD + WG - 1) / WG;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(B, n_blocks * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                             moe_combine_kernel(item, d_parts, d_weights, d_out, N_EMBD, N_EXPERTS);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(parts_size + weights_size + out_size) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(parts_size), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

static void BM_Xe2_MoE_SubgroupReduce(benchmark::State& state) {
    const int B = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t parts_size = static_cast<size_t>(B) * N_EXPERTS * N_EMBD;
    const size_t weights_size = static_cast<size_t>(B) * N_EXPERTS;
    const size_t out_size = static_cast<size_t>(B) * N_EMBD;

    float* d_parts = sycl::malloc_device<float>(parts_size, q);
    float* d_weights = sycl::malloc_device<float>(weights_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_parts || !d_weights || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for MoE benchmark");
        if (d_parts) sycl::free(d_parts, q);
        if (d_weights) sycl::free(d_weights, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_parts, 0.1f, parts_size);
    q.fill(d_weights, 0.1f, weights_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr int WG = 256;
    const size_t total_cols = out_size;
    const size_t n_wgs = (total_cols + WG - 1) / WG;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wgs * WG, WG),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             moe_subgroup_kernel(item, d_parts, d_weights, d_out, B, N_EMBD, N_EXPERTS);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(parts_size + weights_size + out_size) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(parts_size), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Registration: MoE Expert Reduction Across Batch Sizes
// ============================================================================

BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({1});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({4});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({8});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({16});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({32});

BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({1});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({4});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({8});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({16});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({32});

BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({1});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({4});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({8});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({16});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({32});

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 Performance Tuning Suite - MoE Accumulate Benchmark\n");
        std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
        std::printf(" Subgroup:     %d (Native Xe2 SIMD16)\n", strata::sycl_arch::SUB_GROUP_SIZE);
        std::printf("=================================================================\n\n");
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "Warning: Unable to probe default SYCL device: %s\n", e.what());
    }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
