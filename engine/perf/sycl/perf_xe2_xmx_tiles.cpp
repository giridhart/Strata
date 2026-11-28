// perf/sycl/perf_xe2_xmx_tiles.cpp - Intel Xe2 XMX Systolic Tile Microbenchmark Suite
//
// Benchmarks Intel Xe2 hardware XMX systolic matrix acceleration across tile shapes:
// - Single Systolic Tile: 8x16x16 (TM=8, TN=16, TK=16)
// - 2D Systolic Tile: 16x16x16 (TM=16, TN=16, TK=16)
// - Tiled 2D Blocked GEMM across large Qwen tensor projections (attn_qkv [2560, 10240], etc.)
//
// Uses Google Benchmark with benchmark::UseManualTime() and SYCL event profiling.

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

// ============================================================================
// XMX Systolic Matrix Kernels
// ============================================================================

/// Single Systolic Tile Kernel: 8x16x16 (TM=8, TN=16, TK=16)
template <size_t TM = 8, size_t TN = 16, size_t TK = 16>
void xmx_gemm_tile8_kernel(sycl::nd_item<2> item, const sycl::half* __restrict__ in, const sycl::half* __restrict__ w,
                           float* __restrict__ out, size_t M, size_t N, size_t K) {
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
            in + tile_m * K + k);
        auto ptr_b = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            w + k * N + tile_n);

        joint_matrix_load(sg, sub_a, ptr_a, K);
        joint_matrix_load(sg, sub_b, ptr_b, N);
        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
    }

    auto ptr_c = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        out + tile_m * N + tile_n);
    joint_matrix_store(sg, sub_c, ptr_c, N, layout::row_major);
}

/// 2D Systolic Tile Kernel: 16x16x16 (TM=16, TN=16, TK=16)
template <size_t TM = 16, size_t TN = 16, size_t TK = 16>
void xmx_gemm_tile16_kernel(sycl::nd_item<2> item, const sycl::half* __restrict__ in, const sycl::half* __restrict__ w,
                            float* __restrict__ out, size_t M, size_t N, size_t K) {
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
            in + tile_m * K + k);
        auto ptr_b = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            w + k * N + tile_n);

        joint_matrix_load(sg, sub_a, ptr_a, K);
        joint_matrix_load(sg, sub_b, ptr_b, N);
        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
    }

    auto ptr_c = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        out + tile_m * N + tile_n);
    joint_matrix_store(sg, sub_c, ptr_c, N, layout::row_major);
}

}  // namespace

// ============================================================================
// Benchmark: BM_Xe2_XMX_Tile8 (8x16x16 Systolic Matrix Tile)
// ============================================================================

static void BM_Xe2_XMX_Tile8(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, q);
    sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, q);
    float* d_C = sycl::malloc_device<float>(M * N, q);

    if (!d_A || !d_B || !d_C) {
        state.SkipWithError("Failed to allocate USM memory for XMX benchmark");
        if (d_A) sycl::free(d_A, q);
        if (d_B) sycl::free(d_B, q);
        if (d_C) sycl::free(d_C, q);
        return;
    }

    q.fill(d_A, sycl::half{0.01f}, M * K);
    q.fill(d_B, sycl::half{0.01f}, K * N);
    q.fill(d_C, 0.0f, M * N);
    q.wait_and_throw();

    constexpr size_t TM = 8;
    constexpr size_t TN = 16;
    const size_t grid_m = (M + TM - 1) / TM;
    const size_t grid_n = (N + TN - 1) / TN;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n * 16), sycl::range<2>(1, 16)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                             xmx_gemm_tile8_kernel<8, 16, 16>(item, d_A, d_B, d_C, M, N, K);
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

    const double bytes =
        static_cast<double>(M * K + K * N) * sizeof(sycl::half) + static_cast<double>(M * N) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(M * K * N), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(M), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);
}

// ============================================================================
// Benchmark: BM_Xe2_XMX_Tile16 (16x16x16 Systolic Matrix Tile)
// ============================================================================

static void BM_Xe2_XMX_Tile16(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, q);
    sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, q);
    float* d_C = sycl::malloc_device<float>(M * N, q);

    if (!d_A || !d_B || !d_C) {
        state.SkipWithError("Failed to allocate USM memory for XMX benchmark");
        if (d_A) sycl::free(d_A, q);
        if (d_B) sycl::free(d_B, q);
        if (d_C) sycl::free(d_C, q);
        return;
    }

    q.fill(d_A, sycl::half{0.01f}, M * K);
    q.fill(d_B, sycl::half{0.01f}, K * N);
    q.fill(d_C, 0.0f, M * N);
    q.wait_and_throw();

    constexpr size_t TM = 16;
    constexpr size_t TN = 16;
    const size_t grid_m = (M + TM - 1) / TM;
    const size_t grid_n = (N + TN - 1) / TN;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n * 16), sycl::range<2>(1, 16)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                             xmx_gemm_tile16_kernel<16, 16, 16>(item, d_A, d_B, d_C, M, N, K);
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

    const double bytes =
        static_cast<double>(M * K + K * N) * sizeof(sycl::half) + static_cast<double>(M * N) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(M * K * N), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(M), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);
}

// ============================================================================
// Registration: XMX Systolic Tile Benchmark Sweeps
// ============================================================================

// 1. Heavy Attention Projection: attn_qkv [K=2560, N=10240] across token batch sizes
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({8, 2560, 10240});
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({16, 2560, 10240});
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({32, 2560, 10240});

BENCHMARK(BM_Xe2_XMX_Tile16)->UseManualTime()->Args({16, 2560, 10240});
BENCHMARK(BM_Xe2_XMX_Tile16)->UseManualTime()->Args({32, 2560, 10240});

// 2. Square Projection: [K=2560, N=2560]
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({8, 2560, 2560});
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({16, 2560, 2560});
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({32, 2560, 2560});

BENCHMARK(BM_Xe2_XMX_Tile16)->UseManualTime()->Args({16, 2560, 2560});
BENCHMARK(BM_Xe2_XMX_Tile16)->UseManualTime()->Args({32, 2560, 2560});

// 3. HC Down Projection: [K=10240, N=320]
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({16, 10240, 320});
BENCHMARK(BM_Xe2_XMX_Tile8)->UseManualTime()->Args({32, 10240, 320});

BENCHMARK(BM_Xe2_XMX_Tile16)->UseManualTime()->Args({16, 10240, 320});
BENCHMARK(BM_Xe2_XMX_Tile16)->UseManualTime()->Args({32, 10240, 320});

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 Performance Tuning Suite - XMX Systolic Tile Benchmark\n");
        std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());

        bool has_xmx = false;
        try {
            auto combos = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
            if (!combos.empty()) {
                has_xmx = true;
                std::printf(" XMX Status:   Hardware XMX systolic matrix engines DETECTED (%zu combos)\n",
                            combos.size());
            }
        } catch (...) {
            std::printf(" XMX Status:   Matrix combination query unsupported\n");
        }

        if (!has_xmx) {
            std::fprintf(stderr, "Error: XMX joint_matrix extension is not available on this device.\n");
            return 1;
        }
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
