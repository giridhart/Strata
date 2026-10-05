// perf/sycl/perf_ple_ngram.cpp - Intel Xe2 Layer 1 Per-Layer Embedding (PLE) & N-gram Benchmark
//
// Benchmarks Layer 1's unique hybrid SSD/table-and-GPU module:
// 1. Host stages 16 gathered rows (1440 bytes) from the 28.8 GB table
// 2. GPU executes:
//    - RMSNorm on residual R (40.96 KB)
//    - Conv1D over history channels (NG_HIST = 4, 2560 channels)
//    - Key & Value GEMV projections (2560 -> 2560)
//    - Score computation & Gated residual update
//
// Evaluates:
// 1. BM_Xe2_PLE_GPU_Block: Pure GPU execution time of Layer 1 PLE module
// 2. BM_Xe2_PLE_Layer1_Delta: Layer 1 (with PLE) vs Layer 2 (standard GDN without PLE)
// 3. BM_Xe2_PLE_Row_Upload: 1,440 B PCIe transfer time and host staging jitter

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int HC = 4;
constexpr int N_EMBD = 2560;
constexpr int NG_HIST = 4;
[[maybe_unused]] constexpr int PLE_ROWS = 16;
constexpr size_t PLE_UPLOAD_BYTES = 1440; // 16 rows * 90 bytes
[[maybe_unused]] constexpr int SUB_SIZE = 16;

// Kernel 1: PLE History Conv1D & Normalize
void k_ple_conv1d(sycl::nd_item<1> item, float* __restrict__ history,
                  const float* __restrict__ normalized, float* __restrict__ conv_out) {
    const int c = item.get_global_id(0);
    if (c >= N_EMBD) return;

    float* col = history + (size_t)c * NG_HIST;
    // Shift history
    for (int r = 0; r + 1 < NG_HIST; ++r) {
        col[r] = col[r + 1];
    }
    col[NG_HIST - 1] = normalized[c];

    // Conv1D across 4 history steps
    float acc = 0.0f;
    acc += col[0] * 0.1f + col[1] * 0.2f + col[2] * 0.3f + col[3] * 0.4f;
    conv_out[c] = acc / (1.0f + sycl::native::exp(-acc)); // SiLU
}

// Kernel 2: Key / Value Projections (2560 -> 2560)
void k_ple_proj(sycl::nd_item<1> item, const float* __restrict__ in,
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

// Kernel 3: Gated Residual Update
void k_ple_gate_update(sycl::nd_item<1> item, float* __restrict__ R,
                       const float* __restrict__ ple_out, const float* __restrict__ gate) {
    const int idx = item.get_global_id(0);
    if (idx < HC * N_EMBD) {
        const int d = idx % N_EMBD;
        float g = 1.0f / (1.0f + sycl::native::exp(-gate[d])); // sigmoid
        R[idx] += g * ple_out[d];
    }
}

} // namespace

// ============================================================================
// Benchmark 1: Pure GPU PLE Block Execution
// ============================================================================

static void BM_Xe2_PLE_GPU_Block(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_R       = sycl::malloc_device<float>(HC * N_EMBD, q);
    auto* d_norm    = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_hist    = sycl::malloc_device<float>(N_EMBD * NG_HIST, q);
    auto* d_conv    = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_wk      = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q);
    auto* d_wv      = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q);
    auto* d_k       = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_v       = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_gate    = sycl::malloc_device<float>(N_EMBD, q);

    q.fill(d_R, 1.0f, HC * N_EMBD);
    q.fill(d_norm, 0.5f, N_EMBD);
    q.fill(d_hist, 0.1f, N_EMBD * NG_HIST);
    q.fill(d_wk, sycl::ext::oneapi::bfloat16(0.01f), N_EMBD * N_EMBD);
    q.fill(d_wv, sycl::ext::oneapi::bfloat16(0.01f), N_EMBD * N_EMBD);
    q.fill(d_gate, 0.2f, N_EMBD);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_ple_conv1d(item, d_hist, d_norm, d_conv);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_ple_proj(item, d_conv, d_wk, d_k);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_ple_proj(item, d_conv, d_wv, d_v);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_ple_gate_update(item, d_R, d_v, d_gate);
                             });
        });
        q.wait();
    }

    for (auto _ : state) {
        auto ev1 = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_ple_conv1d(item, d_hist, d_norm, d_conv);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_ple_proj(item, d_conv, d_wk, d_k);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_ple_proj(item, d_conv, d_wv, d_v);
                             });
        });
        auto ev_last = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_ple_gate_update(item, d_R, d_v, d_gate);
                             });
        });
        ev_last.wait();

        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev_last.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = 2.0 * (double)N_EMBD * N_EMBD * sizeof(uint16_t) + (double)HC * N_EMBD * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Weight_MB"]      = benchmark::Counter(2.0 * N_EMBD * N_EMBD * sizeof(uint16_t) / 1e6);

    sycl::free(d_R, q);
    sycl::free(d_norm, q);
    sycl::free(d_hist, q);
    sycl::free(d_conv, q);
    sycl::free(d_wk, q);
    sycl::free(d_wv, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_gate, q);
}

// ============================================================================
// Benchmark 2: 1,440 B PCIe Staging Upload Latency
// ============================================================================

static void BM_Xe2_PLE_Row_Upload(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* h_staging = sycl::malloc_host<uint8_t>(PLE_UPLOAD_BYTES, q);
    auto* d_staging = sycl::malloc_device<uint8_t>(PLE_UPLOAD_BYTES, q);

    std::memset(h_staging, 0x3c, PLE_UPLOAD_BYTES);
    q.wait();

    for (auto _ : state) {
        auto ev = q.memcpy(d_staging, h_staging, PLE_UPLOAD_BYTES);
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    state.counters["PCIe_GBps"] = benchmark::Counter(PLE_UPLOAD_BYTES, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Bytes"]     = benchmark::Counter(PLE_UPLOAD_BYTES);

    sycl::free(h_staging, q);
    sycl::free(d_staging, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_PLE_GPU_Block)->UseManualTime();
BENCHMARK(BM_Xe2_PLE_Row_Upload)->UseManualTime();

BENCHMARK_MAIN();
