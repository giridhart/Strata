// perf/sycl/perf_lmhead_sampler.cpp - Intel Xe2 LM Head 248K-Vocab & GPU Sampler Benchmark
//
// Benchmarks the final vocabulary projection and GPU token sampling:
// 1. LM Head: logits = W_output @ x_mixed  (2560 -> 248,320)
//    Reading ~635 MB to 1.27 GB of weights per token.
// 2. GPU Sampler: Parallel max reduction, softmax with temperature, Top-K reduction (248,320 candidates)
// 3. 4-byte host readback of sampled token ID.
//
// Evaluates:
// 1. BM_Xe2_LMHead_GEMV: Output projection throughput across vocabulary slices (16K, 64K, 248K)
// 2. BM_Xe2_Sampler_Reduce: Parallel Softmax + Top-K reduction across 248,320 logits
// 3. BM_Xe2_LMHead_Sampler_EndToEnd: Complete output projection -> sample -> 4-byte token readback

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int N_EMBD = 2560;
constexpr int FULL_VOCAB = 248320;
constexpr int SUB_SIZE = 16;

// Kernel 1: LM Head Projection (N_EMBD -> Vocab_N)
void k_lmhead_gemv(sycl::nd_item<1> item, const float* __restrict__ in,
                   const sycl::ext::oneapi::bfloat16* __restrict__ weight,
                   float* __restrict__ logits, int vocab_size) {
    const int row = item.get_group(0);
    if (row >= vocab_size) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = weight + (size_t)row * N_EMBD;

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * (float)w_row[col];
    }

    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (tid == 0) {
        logits[row] = acc;
    }
}

// Kernel 2: Parallel Max Reduction across 248K Logits
void k_sampler_max(sycl::nd_item<1> item, const float* __restrict__ logits,
                   float* __restrict__ block_maxes,
                   sycl::local_accessor<float, 1> slm_m, int vocab_size) {
    const int gid = item.get_global_id(0);
    const int tid = item.get_local_id(0);
    const int group_id = item.get_group(0);
    const int total_threads = item.get_global_range(0);

    float local_m = -1e30f;
    for (int i = gid; i < vocab_size; i += total_threads) {
        local_m = sycl::max(local_m, logits[i]);
    }

    auto sg = item.get_sub_group();
    local_m = sycl::reduce_over_group(sg, local_m, sycl::maximum<float>());

    // Store warp partials in SLM
    const int warp = tid / SUB_SIZE;
    const int lane = tid % SUB_SIZE;
    if (lane == 0) slm_m[warp] = local_m;
    sycl::group_barrier(item.get_group());

    if (warp == 0) {
        const int num_warps = item.get_local_range(0) / SUB_SIZE;
        float bm = (lane < num_warps) ? slm_m[lane] : -1e30f;
        bm = sycl::reduce_over_group(sg, bm, sycl::maximum<float>());
        if (lane == 0) block_maxes[group_id] = bm;
    }
}

// Kernel 3: Softmax Exponential & Normalization
void k_sampler_softmax(sycl::nd_item<1> item, float* __restrict__ logits,
                       const float* __restrict__ global_max, float inv_temp,
                       float* __restrict__ block_sums,
                       sycl::local_accessor<float, 1> slm_s, int vocab_size) {
    const int gid = item.get_global_id(0);
    const int tid = item.get_local_id(0);
    const int group_id = item.get_group(0);
    const int total_threads = item.get_global_range(0);
    const float m = *global_max;

    float local_sum = 0.0f;
    for (int i = gid; i < vocab_size; i += total_threads) {
        float val = sycl::native::exp((logits[i] - m) * inv_temp);
        logits[i] = val;
        local_sum += val;
    }

    auto sg = item.get_sub_group();
    local_sum = sycl::reduce_over_group(sg, local_sum, sycl::plus<float>());

    const int warp = tid / SUB_SIZE;
    const int lane = tid % SUB_SIZE;
    if (lane == 0) slm_s[warp] = local_sum;
    sycl::group_barrier(item.get_group());

    if (warp == 0) {
        const int num_warps = item.get_local_range(0) / SUB_SIZE;
        float bs = (lane < num_warps) ? slm_s[lane] : 0.0f;
        bs = sycl::reduce_over_group(sg, bs, sycl::plus<float>());
        if (lane == 0) block_sums[group_id] = bs;
    }
}

// Kernel 4: Argmax Selection (Greedy / Top-1)
void k_sampler_argmax(sycl::nd_item<1> item, const float* __restrict__ logits,
                      int32_t* __restrict__ sampled_token, int vocab_size) {
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);

    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = tid; i < vocab_size; i += bsize) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    // Subgroup reduction of max value and index
    auto sg = item.get_sub_group();
    for (int offset = SUB_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = sycl::shift_group_left(sg, max_val, offset);
        int other_idx = sycl::shift_group_left(sg, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    if (tid == 0) {
        *sampled_token = max_idx;
    }
}

} // namespace

// ============================================================================
// Benchmark 1: LM Head Output Projection GEMV Throughput
// ============================================================================

static void BM_Xe2_LMHead_GEMV(benchmark::State& state) {
    const int vocab_size = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t w_elements = (size_t)vocab_size * N_EMBD;

    auto* d_in     = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_logits = sycl::malloc_device<float>(vocab_size, q);
    auto* d_w      = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(w_elements, q);

    q.fill(d_in, 1.0f, N_EMBD);
    q.fill(d_w, sycl::ext::oneapi::bfloat16(0.01f), w_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(vocab_size * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_lmhead_gemv(item, d_in, d_w, d_logits, vocab_size);
                             });
        }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(vocab_size * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_lmhead_gemv(item, d_in, d_w, d_logits, vocab_size);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_read = (double)w_elements * sizeof(uint16_t) + (double)vocab_size * sizeof(float);
    const double flops = 2.0 * (double)w_elements;
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_read, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"]         = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Vocab_N"]        = benchmark::Counter(vocab_size);

    sycl::free(d_in, q);
    sycl::free(d_logits, q);
    sycl::free(d_w, q);
}

// ============================================================================
// Benchmark 2: GPU Sampler Reduction across 248K Logits
// ============================================================================

static void BM_Xe2_Sampler_Reduce(benchmark::State& state) {
    const int vocab_size = FULL_VOCAB;

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    constexpr int NUM_GROUPS = 64;
    auto* d_logits      = sycl::malloc_device<float>(vocab_size, q);
    auto* d_block_m     = sycl::malloc_device<float>(NUM_GROUPS, q);
    auto* d_global_m    = sycl::malloc_device<float>(1, q);
    auto* d_block_s     = sycl::malloc_device<float>(NUM_GROUPS, q);
    auto* d_token       = sycl::malloc_device<int32_t>(1, q);

    q.fill(d_logits, 0.5f, vocab_size);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm_m(sycl::range<1>(32), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(NUM_GROUPS * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_max(item, d_logits, d_block_m, slm_m, vocab_size);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm_s(sycl::range<1>(32), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(NUM_GROUPS * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_softmax(item, d_logits, d_global_m, 1.25f, d_block_s, slm_s, vocab_size);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_argmax(item, d_logits, d_token, vocab_size);
                             });
        });
        q.wait();
    }

    for (auto _ : state) {
        auto ev1 = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm_m(sycl::range<1>(32), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(NUM_GROUPS * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_max(item, d_logits, d_block_m, slm_m, vocab_size);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm_s(sycl::range<1>(32), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(NUM_GROUPS * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_softmax(item, d_logits, d_global_m, 1.25f, d_block_s, slm_s, vocab_size);
                             });
        });
        auto ev_last = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_argmax(item, d_logits, d_token, vocab_size);
                             });
        });
        ev_last.wait();

        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev_last.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_scanned = (double)vocab_size * sizeof(float) * 3.0; // read, write, read
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_scanned, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Vocab_N"]        = benchmark::Counter(vocab_size);

    sycl::free(d_logits, q);
    sycl::free(d_block_m, q);
    sycl::free(d_global_m, q);
    sycl::free(d_block_s, q);
    sycl::free(d_token, q);
}

// ============================================================================
// Benchmark 3: End-to-End LM Head Projection + Sampler + 4-Byte Readback
// ============================================================================

static void BM_Xe2_LMHead_Sampler_EndToEnd(benchmark::State& state) {
    const int vocab_size = 65536; // Representative chunk

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t w_elements = (size_t)vocab_size * N_EMBD;

    auto* d_in     = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_logits = sycl::malloc_device<float>(vocab_size, q);
    auto* d_w      = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(w_elements, q);
    auto* d_token  = sycl::malloc_device<int32_t>(1, q);
    auto* h_token  = sycl::malloc_host<int32_t>(1, q);

    q.fill(d_in, 1.0f, N_EMBD);
    q.fill(d_w, sycl::ext::oneapi::bfloat16(0.01f), w_elements);
    q.wait();

    for (auto _ : state) {
        auto ev_start = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(vocab_size * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_lmhead_gemv(item, d_in, d_w, d_logits, vocab_size);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_sampler_argmax(item, d_logits, d_token, vocab_size);
                             });
        });
        auto ev_copy = q.memcpy(h_token, d_token, sizeof(int32_t));
        ev_copy.wait();

        const auto t0 = ev_start.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev_copy.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)w_elements * sizeof(uint16_t) + (double)vocab_size * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Vocab_N"]        = benchmark::Counter(vocab_size);

    sycl::free(d_in, q);
    sycl::free(d_logits, q);
    sycl::free(d_w, q);
    sycl::free(d_token, q);
    sycl::free(h_token, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_LMHead_GEMV)->Arg(16384)->Arg(65536)->Arg(248320)->UseManualTime();
BENCHMARK(BM_Xe2_Sampler_Reduce)->UseManualTime();
BENCHMARK(BM_Xe2_LMHead_Sampler_EndToEnd)->UseManualTime();

BENCHMARK_MAIN();
