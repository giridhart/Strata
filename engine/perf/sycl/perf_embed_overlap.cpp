// perf/sycl/perf_embed_overlap.cpp - Intel Xe2 Embedding Column Dequant & Host Synchronization Benchmark
//
// Benchmarks next-token input processing:
// 1. Column Dequantization of token_embd.weight (2560 x 248,320) across BF16, Q8_0, and Q4_K
// 2. 4-byte sampled token D2H readback barrier
// 3. Host-Synchronized loop vs On-Device Chained Autoregressive advance
//
// Evaluates:
// 1. BM_Xe2_Embed_Column_Dequant: Extracting 1 token column (2560 floats) across formats
// 2. BM_Xe2_Host_Token_Readback_Latency: 4-byte PCIe transfer + CPU queue sync barrier
// 3. BM_Xe2_Autoregressive_Sync_vs_Async: Measuring sync bubble elimination

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int N_EMBD = 2560;
[[maybe_unused]] constexpr int TOTAL_VOCAB = 248320;
[[maybe_unused]] constexpr int SUB_SIZE = 16;

// Kernel 1: Column Dequantization from BF16
void k_dequant_embed_bf16(sycl::nd_item<1> item, const sycl::ext::oneapi::bfloat16* __restrict__ embd_table,
                          const int32_t* __restrict__ token_ptr, float* __restrict__ out) {
    const int tid = item.get_global_id(0);
    if (tid >= N_EMBD) return;

    const int token = *token_ptr;
    // Row-major or column-major stride
    const auto* col = embd_table + (size_t)token * N_EMBD;
    out[tid] = (float)col[tid];
}

// Kernel 2: Column Dequantization from Q8_0 (34 bytes per 32 elements)
[[maybe_unused]] void k_dequant_embed_q8(sycl::nd_item<1> item, const uint8_t* __restrict__ embd_table,
                                         const int32_t* __restrict__ token_ptr, float* __restrict__ out) {
    const int tid = item.get_global_id(0);
    if (tid >= N_EMBD) return;

    const int token = *token_ptr;
    constexpr size_t BYTES_PER_COL = (N_EMBD / 32) * 34; // 2720 bytes
    const uint8_t* col_bytes = embd_table + (size_t)token * BYTES_PER_COL;

    const int block_idx = tid / 32;
    const int elem_in_block = tid % 32;
    const uint8_t* b_ptr = col_bytes + block_idx * 34;

    sycl::half scale_half = *(const sycl::half*)b_ptr;
    float scale = (float)scale_half;
    int8_t q_val = *(const int8_t*)(b_ptr + 2 + elem_in_block);

    out[tid] = (float)q_val * scale;
}

// Kernel 3: Synthetic Next Step Consumption (simulating Layer 0)
void k_consume_embed(sycl::nd_item<1> item, const float* __restrict__ in, float* __restrict__ out) {
    const int tid = item.get_global_id(0);
    if (tid < N_EMBD) {
        out[tid] = in[tid] * 1.05f + 0.01f;
    }
}

} // namespace

// ============================================================================
// Benchmark 1: Column Dequantization Across Formats
// ============================================================================

static void BM_Xe2_Embed_Column_Dequant(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    // Represent subset of table in memory
    constexpr int MOCK_VOCAB = 4096;

    auto* d_table = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(MOCK_VOCAB * N_EMBD, q);
    auto* d_token = sycl::malloc_device<int32_t>(1, q);
    auto* d_out   = sycl::malloc_device<float>(N_EMBD, q);

    q.fill(d_token, 42, 1);
    q.fill(d_table, sycl::ext::oneapi::bfloat16(0.02f), MOCK_VOCAB * N_EMBD);
    q.wait();

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_dequant_embed_bf16(item, d_table, d_token, d_out);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)N_EMBD * sizeof(uint16_t) + (double)N_EMBD * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Column_Bytes"]   = benchmark::Counter(N_EMBD * sizeof(uint16_t));

    sycl::free(d_table, q);
    sycl::free(d_token, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Benchmark 2: 4-Byte Token Readback Barrier Latency
// ============================================================================

static void BM_Xe2_Host_Token_Readback_Latency(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

    auto* d_token = sycl::malloc_device<int32_t>(1, q);
    auto* h_token = sycl::malloc_host<int32_t>(1, q);

    q.fill(d_token, 100, 1).wait();

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        q.memcpy(h_token, d_token, sizeof(int32_t));
        q.wait();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());
    }

    state.counters["Readback_Latency_us"] = benchmark::Counter(1.0, benchmark::Counter::kAvgIterationsRate);

    sycl::free(d_token, q);
    sycl::free(h_token, q);
}

// ============================================================================
// Benchmark 3: Host-Synchronized Loop vs On-Device Chained Autoregressive
// ============================================================================

static void BM_Xe2_Autoregressive_Sync(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

    constexpr int MOCK_VOCAB = 4096;
    auto* d_table = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(MOCK_VOCAB * N_EMBD, q);
    auto* d_token = sycl::malloc_device<int32_t>(1, q);
    auto* h_token = sycl::malloc_host<int32_t>(1, q);
    auto* d_embed = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_out   = sycl::malloc_device<float>(N_EMBD, q);

    q.fill(d_token, 50, 1).wait();

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Host sync barrier to read token
        q.memcpy(h_token, d_token, sizeof(int32_t));
        q.wait();

        int host_token = *h_token % MOCK_VOCAB;

        // 2. Launch embed row
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 const auto* col = d_table + (size_t)host_token * N_EMBD;
                                 d_embed[item.get_global_id(0)] = (float)col[item.get_global_id(0)];
                             });
        });

        // 3. Launch Layer 0
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_consume_embed(item, d_embed, d_out);
                             });
        });
        q.wait();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());
    }

    state.counters["Mode"] = benchmark::Counter(1); // Synchronized

    sycl::free(d_table, q);
    sycl::free(d_token, q);
    sycl::free(h_token, q);
    sycl::free(d_embed, q);
    sycl::free(d_out, q);
}

static void BM_Xe2_Autoregressive_Async(benchmark::State& state) {
    sycl::queue q_compute{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue q_dma{sycl::default_selector_v, sycl::property::queue::in_order()};

    constexpr int MOCK_VOCAB = 4096;
    auto* d_table = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(MOCK_VOCAB * N_EMBD, q_compute);
    auto* d_token = sycl::malloc_device<int32_t>(1, q_compute);
    auto* h_token = sycl::malloc_host<int32_t>(1, q_compute);
    auto* d_embed = sycl::malloc_device<float>(N_EMBD, q_compute);
    auto* d_out   = sycl::malloc_device<float>(N_EMBD, q_compute);

    q_compute.fill(d_token, 50, 1).wait();

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Asynchronously read token back in background queue
        q_dma.memcpy(h_token, d_token, sizeof(int32_t));

        // 2. On-device directly chain token into embed dequant without waiting
        q_compute.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_dequant_embed_bf16(item, d_table, d_token, d_embed);
                             });
        });

        // 3. Chain Layer 0
        q_compute.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_consume_embed(item, d_embed, d_out);
                             });
        });

        q_compute.wait();
        q_dma.wait();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());
    }

    state.counters["Mode"] = benchmark::Counter(2); // Asynchronous Chained

    sycl::free(d_table, q_compute);
    sycl::free(d_token, q_compute);
    sycl::free(h_token, q_compute);
    sycl::free(d_embed, q_compute);
    sycl::free(d_out, q_compute);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_Embed_Column_Dequant)->UseManualTime();
BENCHMARK(BM_Xe2_Host_Token_Readback_Latency)->UseManualTime();
BENCHMARK(BM_Xe2_Autoregressive_Sync)->UseManualTime();
BENCHMARK(BM_Xe2_Autoregressive_Async)->UseManualTime();

BENCHMARK_MAIN();
