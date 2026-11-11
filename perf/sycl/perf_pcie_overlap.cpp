// perf/sycl/perf_pcie_overlap.cpp - Intel Xe2 PCIe Copy vs XMX Compute Concurrency Benchmark
//
// Benchmarks Strata's asymmetric memory tiering on Intel Xe2:
// Measuring whether asynchronous USM expert streaming over PCIe 4.0 degrades
// concurrent XMX systolic matrix compute on the GPU.
//
// Compares:
// 1. BM_Xe2_Compute_Isolated: Pure XMX GEMM [M=32, K=2560, N=10240] throughput (TFLOPS) with 0 PCIe traffic
// 2. BM_Xe2_PCIe_Isolated: Host-to-Device USM DMA copy throughput (GB/s) for expert chunks (2MB, 8MB, 16MB, 32MB)
// 3. BM_Xe2_PCIe_Compute_Concurrent: Concurrently executing XMX compute on Queue 1 and PCIe DMA on Queue 2
//
// Uses Google Benchmark with benchmark::UseManualTime() and SYCL event profiling.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>

#include "strata/core/sycl_arch.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl::ext::oneapi::experimental::matrix;

namespace {

constexpr size_t DEFAULT_M = 32;
constexpr size_t DEFAULT_K = 2560;
constexpr size_t DEFAULT_N = 10240;

// XMX 16x16x16 Systolic Tile Kernel
template <size_t TM = 16, size_t TN = 16, size_t TK = 16>
void xmx_gemm_kernel(sycl::nd_item<2> item, const sycl::half* __restrict__ in, const sycl::half* __restrict__ w,
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
// Benchmark: BM_Xe2_Compute_Isolated (Pure XMX TFLOPS with 0 PCIe traffic)
// ============================================================================

static void BM_Xe2_Compute_Isolated(benchmark::State& state) {
    const size_t M = DEFAULT_M;
    const size_t K = DEFAULT_K;
    const size_t N = DEFAULT_N;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};

    auto* d_in  = sycl::malloc_device<sycl::half>(M * K, q);
    auto* d_w   = sycl::malloc_device<sycl::half>(K * N, q);
    auto* d_out = sycl::malloc_device<float>(M * N, q);

    q.fill(d_in, sycl::half(0.1f), M * K);
    q.fill(d_w, sycl::half(0.05f), K * N);
    q.wait();

    // Warmup
    for (int w = 0; w < 5; ++w) {
        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                           xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev = q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                                 [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                     xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
                                 });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double ops = 2.0 * (double)M * (double)K * (double)N;
    state.counters["TFLOPS"]         = benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)M, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_in, q);
    sycl::free(d_w, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Benchmark: BM_Xe2_PCIe_Isolated (Pure Host-to-Device DMA Transfer)
// ============================================================================

static void BM_Xe2_PCIe_Isolated(benchmark::State& state) {
    const size_t copy_bytes = state.range(0);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};

    auto* h_src = sycl::malloc_host<uint8_t>(copy_bytes, q);
    auto* d_dst = sycl::malloc_device<uint8_t>(copy_bytes, q);

    std::memset(h_src, 0x42, copy_bytes);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.memcpy(d_dst, h_src, copy_bytes).wait();
    }

    for (auto _ : state) {
        auto ev = q.memcpy(d_dst, h_src, copy_bytes);
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    state.counters["PCIe_GBps"] = benchmark::Counter((double)copy_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Chunk_MB"]  = benchmark::Counter(copy_bytes / (1024.0 * 1024.0));

    sycl::free(h_src, q);
    sycl::free(d_dst, q);
}

// ============================================================================
// Benchmark: BM_Xe2_PCIe_Compute_Concurrent (Dual-Queue Concurrent Pipeline)
// ============================================================================

static void BM_Xe2_PCIe_Compute_Concurrent(benchmark::State& state) {
    const size_t copy_bytes = state.range(0);
    const size_t M = DEFAULT_M;
    const size_t K = DEFAULT_K;
    const size_t N = DEFAULT_N;

    sycl::device dev{sycl::default_selector_v};
    sycl::context ctx{dev};

    sycl::queue q_compute{ctx, dev, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};
    sycl::queue q_copy{ctx, dev, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};

    auto* d_in  = sycl::malloc_device<sycl::half>(M * K, q_compute);
    auto* d_w   = sycl::malloc_device<sycl::half>(K * N, q_compute);
    auto* d_out = sycl::malloc_device<float>(M * N, q_compute);

    auto* h_src = sycl::malloc_host<uint8_t>(copy_bytes, q_copy);
    auto* d_dst = sycl::malloc_device<uint8_t>(copy_bytes, q_copy);

    q_compute.fill(d_in, sycl::half(0.1f), M * K);
    q_compute.fill(d_w, sycl::half(0.05f), K * N);
    std::memset(h_src, 0x42, copy_bytes);
    q_compute.wait();
    q_copy.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q_copy.memcpy(d_dst, h_src, copy_bytes);
        q_compute.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                               [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                   xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
                               });
        q_copy.wait();
        q_compute.wait();
    }

    for (auto _ : state) {
        // Launch both concurrently
        auto ev_copy = q_copy.memcpy(d_dst, h_src, copy_bytes);
        auto ev_comp = q_compute.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
            });

        ev_copy.wait();
        ev_comp.wait();

        const auto comp_t0 = ev_comp.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto comp_t1 = ev_comp.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((comp_t1 - comp_t0) * 1e-9);
    }

    const double ops = 2.0 * (double)M * (double)K * (double)N;
    state.counters["TFLOPS"]         = benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["PCIe_Bytes"]     = benchmark::Counter((double)copy_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Chunk_MB"]       = benchmark::Counter(copy_bytes / (1024.0 * 1024.0));

    sycl::free(d_in, q_compute);
    sycl::free(d_w, q_compute);
    sycl::free(d_out, q_compute);
    sycl::free(h_src, q_copy);
    sycl::free(d_dst, q_copy);
}

// ============================================================================
// Benchmark: BM_Xe2_Prefill_Speculative_Route (Router GEMV + Top-8 + D2H)
// ============================================================================
static void BM_Xe2_Prefill_Speculative_Route(benchmark::State& state) {
    const size_t tokens = state.range(0);
    const size_t dim = 2560;
    const size_t experts = 64;
    const size_t top_k = 8;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};

    auto* d_act = sycl::malloc_device<sycl::half>(tokens * dim, q);
    auto* d_rw  = sycl::malloc_device<sycl::half>(experts * dim, q);
    auto* d_logits = sycl::malloc_device<float>(tokens * experts, q);
    auto* d_ids = sycl::malloc_device<int32_t>(tokens * top_k, q);
    auto* h_pred = sycl::malloc_host<int32_t>(tokens * top_k, q);

    q.fill(d_act, sycl::half(0.01f), tokens * dim);
    q.fill(d_rw, sycl::half(0.02f), experts * dim);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::range<1>(tokens * experts), [=](sycl::id<1> idx) {
            size_t t = idx[0] / experts;
            size_t e = idx[0] % experts;
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                sum += (float)d_act[t * dim + d] * (float)d_rw[e * dim + d];
            }
            d_logits[t * experts + e] = sum;
        });
        q.parallel_for(sycl::range<1>(tokens), [=](sycl::id<1> idx) {
            size_t t = idx[0];
            for (size_t k = 0; k < top_k; ++k) {
                d_ids[t * top_k + k] = (int32_t)(k);
            }
        });
        q.memcpy(h_pred, d_ids, tokens * top_k * sizeof(int32_t));
        q.wait();
    }

    for (auto _ : state) {
        auto t_start = std::chrono::steady_clock::now();
        q.parallel_for(sycl::range<1>(tokens * experts), [=](sycl::id<1> idx) {
            size_t t = idx[0] / experts;
            size_t e = idx[0] % experts;
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                sum += (float)d_act[t * dim + d] * (float)d_rw[e * dim + d];
            }
            d_logits[t * experts + e] = sum;
        });
        q.parallel_for(sycl::range<1>(tokens), [=](sycl::id<1> idx) {
            size_t t = idx[0];
            for (size_t k = 0; k < top_k; ++k) {
                d_ids[t * top_k + k] = (int32_t)(k);
            }
        });
        q.memcpy(h_pred, d_ids, tokens * top_k * sizeof(int32_t));
        q.wait();
        auto t_end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t_end - t_start).count());
    }

    state.counters["Tokens"] = benchmark::Counter((double)tokens);

    sycl::free(d_act, q);
    sycl::free(d_rw, q);
    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(h_pred, q);
}

// ============================================================================
// Benchmark: BM_Xe2_Prefill_Sequential_PCIe_Exposed (Non-Overlapped Baseline)
// ============================================================================
static void BM_Xe2_Prefill_Sequential_PCIe_Exposed(benchmark::State& state) {
    const size_t copy_bytes = state.range(0);
    const size_t M = DEFAULT_M;
    const size_t K = DEFAULT_K;
    const size_t N = DEFAULT_N;

    sycl::device dev{sycl::default_selector_v};
    sycl::context ctx{dev};
    sycl::queue q{ctx, dev, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};

    auto* d_in  = sycl::malloc_device<sycl::half>(M * K, q);
    auto* d_w   = sycl::malloc_device<sycl::half>(K * N, q);
    auto* d_out = sycl::malloc_device<float>(M * N, q);
    auto* h_src = sycl::malloc_host<uint8_t>(copy_bytes, q);
    auto* d_dst = sycl::malloc_device<uint8_t>(copy_bytes, q);

    q.fill(d_in, sycl::half(0.1f), M * K);
    q.fill(d_w, sycl::half(0.05f), K * N);
    std::memset(h_src, 0x42, copy_bytes);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                           xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
                       });
        q.memcpy(d_dst, h_src, copy_bytes);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        // Sequential execution: Attention compute, THEN PCIe copy
        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                           xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
                       });
        q.memcpy(d_dst, h_src, copy_bytes);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Chunk_MB"] = benchmark::Counter(copy_bytes / (1024.0 * 1024.0));
    state.counters["PCIe_Bytes"] = benchmark::Counter((double)copy_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_in, q);
    sycl::free(d_w, q);
    sycl::free(d_out, q);
    sycl::free(h_src, q);
    sycl::free(d_dst, q);
}

// ============================================================================
// Benchmark: BM_Xe2_Prefill_Speculative_PCIe_Overlapped (STRATA_PF_PREDICT)
// ============================================================================
static void BM_Xe2_Prefill_Speculative_PCIe_Overlapped(benchmark::State& state) {
    const size_t copy_bytes = state.range(0);
    const size_t M = DEFAULT_M;
    const size_t K = DEFAULT_K;
    const size_t N = DEFAULT_N;

    sycl::device dev{sycl::default_selector_v};
    sycl::context ctx{dev};
    sycl::queue q_compute{ctx, dev, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};
    sycl::queue q_copy{ctx, dev, {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()}};

    auto* d_in  = sycl::malloc_device<sycl::half>(M * K, q_compute);
    auto* d_w   = sycl::malloc_device<sycl::half>(K * N, q_compute);
    auto* d_out = sycl::malloc_device<float>(M * N, q_compute);
    auto* h_src = sycl::malloc_host<uint8_t>(copy_bytes, q_copy);
    auto* d_dst = sycl::malloc_device<uint8_t>(copy_bytes, q_copy);

    q_compute.fill(d_in, sycl::half(0.1f), M * K);
    q_compute.fill(d_w, sycl::half(0.05f), K * N);
    std::memset(h_src, 0x42, copy_bytes);
    q_compute.wait();
    q_copy.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q_copy.memcpy(d_dst, h_src, copy_bytes);
        q_compute.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
                               [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                   xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
                               });
        q_copy.wait();
        q_compute.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        // Concurrent execution: Attention compute on q_compute WHILE PCIe copy streams on q_copy
        auto ev_copy = q_copy.memcpy(d_dst, h_src, copy_bytes);
        auto ev_comp = q_compute.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(M / 16, (N / 16) * 16), sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                xmx_gemm_kernel<16, 16, 16>(item, d_in, d_w, d_out, M, N, K);
            });

        ev_copy.wait();
        ev_comp.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Chunk_MB"] = benchmark::Counter(copy_bytes / (1024.0 * 1024.0));
    state.counters["PCIe_Bytes"] = benchmark::Counter((double)copy_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_in, q_compute);
    sycl::free(d_w, q_compute);
    sycl::free(d_out, q_compute);
    sycl::free(h_src, q_copy);
    sycl::free(d_dst, q_copy);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_Compute_Isolated)->UseManualTime();
BENCHMARK(BM_Xe2_PCIe_Isolated)->Arg(2 * 1024 * 1024)->Arg(8 * 1024 * 1024)->Arg(16 * 1024 * 1024)->Arg(32 * 1024 * 1024)->UseManualTime();
BENCHMARK(BM_Xe2_PCIe_Compute_Concurrent)->Arg(2 * 1024 * 1024)->Arg(8 * 1024 * 1024)->Arg(16 * 1024 * 1024)->Arg(32 * 1024 * 1024)->UseManualTime();
BENCHMARK(BM_Xe2_Prefill_Speculative_Route)->Arg(64)->Arg(256)->Arg(1024)->UseManualTime();
BENCHMARK(BM_Xe2_Prefill_Sequential_PCIe_Exposed)->Arg(8 * 1024 * 1024)->Arg(16 * 1024 * 1024)->Arg(32 * 1024 * 1024)->UseManualTime();
BENCHMARK(BM_Xe2_Prefill_Speculative_PCIe_Overlapped)->Arg(8 * 1024 * 1024)->Arg(16 * 1024 * 1024)->Arg(32 * 1024 * 1024)->UseManualTime();

BENCHMARK_MAIN();
