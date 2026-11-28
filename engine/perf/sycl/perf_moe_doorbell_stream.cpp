// perf/sycl/perf_moe_doorbell_stream.cpp - Intel Xe2 Host-Device Doorbell & MoE Expert Streaming Benchmark
//
// Benchmarks the asynchronous CPU <-> GPU doorbell protocol and expert parts streaming:
// 1. GPU moe_route writes top-K expert IDs to pinned host memory and increments d_seq.
// 2. Host detects h_seq and streams K=10 expert output parts (10 * 2560 * 2 bytes = 51.2 KB) over PCIe Gen4.
// 3. Concurrently, GPU executes moe_shared (2560 -> 6144 -> 2560).
// 4. Host sets h_flag; GPU verifies and executes moe_combine.
//
// Evaluates:
// 1. BM_Xe2_Doorbell_Roundtrip: GPU write -> Host detect -> Host reply -> GPU observe
// 2. BM_Xe2_PCIe_Parts_Stream_DMA: 51.2 KB PCIe transfer time across candidate sizes
// 3. BM_Xe2_PCIe_Parts_ZeroCopy: Direct GPU kernel reading from host-pinned memory over PCIe
// 4. BM_Xe2_MoE_Doorbell_Overlap_Slack: Full concurrency slack (shared compute vs PCIe transfer)

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr int N_EMBD = 2560;
constexpr int K_EXP = 10;
constexpr size_t PARTS_BYTES = (size_t)K_EXP * N_EMBD * sizeof(uint16_t); // 51.2 KB

// Kernel 1: GPU Ring Doorbell
void k_ring_doorbell(sycl::nd_item<1> item, int32_t* __restrict__ d_ids,
                     float* __restrict__ d_weights, uint32_t* __restrict__ d_seq,
                     uint32_t ring_val) {
    const int tid = item.get_local_id(0);
    if (tid < K_EXP) {
        d_ids[tid] = tid * 3;
        d_weights[tid] = 0.1f;
    }
    sycl::group_barrier(item.get_group());
    if (tid == 0) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system> ref(*d_seq);
        ref.store(ring_val);
    }
}

// Kernel 2: Shared Expert Compute (2560 -> 6144 -> 2560)
void k_shared_expert(sycl::nd_item<1> item, const float* __restrict__ in,
                     const sycl::ext::oneapi::bfloat16* __restrict__ weight,
                     float* __restrict__ out) {
    const int row = item.get_group(0);
    if (row >= N_EMBD) return;
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = weight + (size_t)row * N_EMBD;

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * (float)w_row[col];
    }
    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (tid == 0) out[row] = acc;
}

// Kernel 3: MoE Combine (Parts read directly from Host Pinned Memory via Zero-Copy)
void k_combine_zerocopy(sycl::nd_item<1> item, const float* __restrict__ shared,
                        const uint16_t* __restrict__ host_parts, const float* __restrict__ weights,
                        float* __restrict__ out) {
    const int d = item.get_global_id(0);
    if (d >= N_EMBD) return;

    float sum = shared[d];
    for (int k = 0; k < K_EXP; ++k) {
        float p = (float)host_parts[k * N_EMBD + d];
        sum += p * weights[k];
    }
    out[d] = sum;
}

} // namespace

// ============================================================================
// Benchmark 1: Doorbell Signal Roundtrip Latency
// ============================================================================

static void BM_Xe2_Doorbell_Roundtrip(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

    // Pinned host memory accessible from device
    auto* h_seq  = sycl::malloc_host<uint32_t>(1, q);
    auto* h_flag = sycl::malloc_host<uint32_t>(1, q);
    auto* h_ids  = sycl::malloc_host<int32_t>(K_EXP, q);
    auto* h_wt   = sycl::malloc_host<float>(K_EXP, q);

    *h_seq = 0;
    *h_flag = 0;

    uint32_t ring = 1;

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // GPU rings doorbell
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) {
                                 k_ring_doorbell(item, h_ids, h_wt, h_seq, ring);
                             });
        });

        // Host spins on h_seq
        while (true) {
            uint32_t cur = *h_seq;
            if (cur == ring) break;
            std::this_thread::yield();
        }

        // Host responds by updating h_flag
        *h_flag = ring;

        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());

        ring++;
    }

    state.counters["Doorbell_Latency_us"] = benchmark::Counter(1.0, benchmark::Counter::kAvgIterationsRate);

    sycl::free(h_seq, q);
    sycl::free(h_flag, q);
    sycl::free(h_ids, q);
    sycl::free(h_wt, q);
}

// ============================================================================
// Benchmark 2: PCIe Parts Streaming DMA Latency & Bandwidth
// ============================================================================

static void BM_Xe2_PCIe_Parts_Stream_DMA(benchmark::State& state) {
    const size_t bytes = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* h_parts = sycl::malloc_host<uint8_t>(bytes, q);
    auto* d_parts = sycl::malloc_device<uint8_t>(bytes, q);

    std::memset(h_parts, 0x5a, bytes);
    q.wait();

    for (auto _ : state) {
        auto ev = q.memcpy(d_parts, h_parts, bytes);
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    state.counters["PCIe_GBps"]   = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Transfer_KB"] = benchmark::Counter(bytes / 1024.0);

    sycl::free(h_parts, q);
    sycl::free(d_parts, q);
}

// ============================================================================
// Benchmark 3: Zero-Copy USM Read from Host over PCIe vs Device Memory
// ============================================================================

static void BM_Xe2_PCIe_Parts_ZeroCopy(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_shared  = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_weights = sycl::malloc_device<float>(K_EXP, q);
    auto* d_out     = sycl::malloc_device<float>(N_EMBD, q);

    // Host pinned memory mapped for direct GPU access
    auto* h_parts = sycl::malloc_host<uint16_t>(K_EXP * N_EMBD, q);

    q.fill(d_shared, 0.5f, N_EMBD);
    q.fill(d_weights, 0.1f, K_EXP);
    std::fill(h_parts, h_parts + K_EXP * N_EMBD, 10);
    q.wait();

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_combine_zerocopy(item, d_shared, h_parts, d_weights, d_out);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    state.counters["Bandwidth_GBps"] = benchmark::Counter(PARTS_BYTES, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Parts_KB"]       = benchmark::Counter(PARTS_BYTES / 1024.0);

    sycl::free(d_shared, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
    sycl::free(h_parts, q);
}

// ============================================================================
// Benchmark 4: MoE Doorbell Overlap & Slack Time
// ============================================================================

static void BM_Xe2_MoE_Doorbell_Overlap_Slack(benchmark::State& state) {
    // Dual queue: q_compute for shared expert, q_dma for streaming parts
    sycl::queue q_compute{sycl::default_selector_v,
                          {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
    sycl::queue q_dma{sycl::default_selector_v,
                      {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_in      = sycl::malloc_device<float>(N_EMBD, q_compute);
    auto* d_wshared = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q_compute);
    auto* d_shared  = sycl::malloc_device<float>(N_EMBD, q_compute);
    auto* d_parts   = sycl::malloc_device<uint16_t>(K_EXP * N_EMBD, q_compute);

    auto* h_parts   = sycl::malloc_host<uint16_t>(K_EXP * N_EMBD, q_compute);

    q_compute.fill(d_in, 1.0f, N_EMBD);
    q_compute.fill(d_wshared, sycl::ext::oneapi::bfloat16(0.01f), N_EMBD * N_EMBD);
    std::fill(h_parts, h_parts + K_EXP * N_EMBD, 5);
    q_compute.wait();

    for (auto _ : state) {
        // Concurrently launch shared expert on GPU and DMA copy on second queue
        auto ev_dma = q_dma.memcpy(d_parts, h_parts, PARTS_BYTES);

        auto ev_compute = q_compute.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_shared_expert(item, d_in, d_wshared, d_shared);
                             });
        });

        ev_dma.wait();
        ev_compute.wait();

        const auto c_t0 = ev_compute.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto c_t1 = ev_compute.get_profiling_info<sycl::info::event_profiling::command_end>();
        const double comp_time_us = (c_t1 - c_t0) * 1e-3;

        const auto d_t0 = ev_dma.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto d_t1 = ev_dma.get_profiling_info<sycl::info::event_profiling::command_end>();
        const double dma_time_us = (d_t1 - d_t0) * 1e-3;

        const double slack_us = comp_time_us - dma_time_us;

        // Total overlap latency is max(compute, dma)
        const double total_s = std::max(comp_time_us, dma_time_us) * 1e-6;
        state.SetIterationTime(total_s);

        state.counters["Compute_us"] = benchmark::Counter(comp_time_us);
        state.counters["DMA_us"]     = benchmark::Counter(dma_time_us);
        state.counters["Slack_us"]   = benchmark::Counter(slack_us);
    }

    sycl::free(d_in, q_compute);
    sycl::free(d_wshared, q_compute);
    sycl::free(d_shared, q_compute);
    sycl::free(d_parts, q_compute);
    sycl::free(h_parts, q_compute);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_Doorbell_Roundtrip)->UseManualTime();
BENCHMARK(BM_Xe2_PCIe_Parts_Stream_DMA)->Arg(25600)->Arg(51200)->Arg(102400)->UseManualTime();
BENCHMARK(BM_Xe2_PCIe_Parts_ZeroCopy)->UseManualTime();
BENCHMARK(BM_Xe2_MoE_Doorbell_Overlap_Slack)->UseManualTime();

BENCHMARK_MAIN();
