// perf/sycl/perf_xe2_gemv.cpp - Intel Xe2 GEMV Microbenchmark Suite using Google Benchmark
//
// Parameterized performance tuning suite for Intel Arc Battlemage (B580 / Xe2-HPG)
// and Lunar Lake (Xe2-LPG) GPUs running Qwen3.8-Flash-Next tensor projections.
//
// Key Advantages of this Google Benchmark Suite:
// 1. Hardware Profiling Timers: Uses benchmark::UseManualTime() with SYCL event profiling
//    (command_start to command_end) to bypass host CPU dispatch latency and queue jitter.
// 2. Steady-State Warmup: Mitigates dynamic GPU clock ramp (e.g. 800 MHz idle to 2850 MHz boost).
// 3. Automated Sweeps: Sweeps all 8 decode projections, expert layers, chunked prefill (B=24, 32),
//    subgroup sizes (SIMD16 vs SIMD32), and work-group configurations in a single run.
// 4. Native Hardware Counters: Reports Bandwidth (GB/s), TFLOPS, and tokens/sec directly.
// 5. Commit-to-Commit Comparison: JSON output (--benchmark_out=baseline.json) compatible with
//    tools/compare.py for regression tracking.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// ============================================================================
// Xe2 Native SYCL Kernels
// ============================================================================

/// Pure SYCL Xe2 subgroup-per-row GEMV kernel.
/// Optimized for native SIMD16 execution on Intel Xe2 Vector Engines (XVE).
/// One subgroup computes one output token/channel (b, o).
template <int SG_SIZE, int WG_SIZE>
void xe2_subgroup_gemv_kernel(sycl::nd_item<1> item, const sycl::ext::oneapi::bfloat16* __restrict__ in,
                              const sycl::ext::oneapi::bfloat16* __restrict__ w,
                              sycl::ext::oneapi::bfloat16* __restrict__ out, int K, int N, int total_tasks) {
    auto sg = item.get_sub_group();
    constexpr int sgs_per_wg = WG_SIZE / SG_SIZE;
    const int gid = item.get_group(0);
    const int lid = item.get_local_id(0);
    const int sg_in_wg = lid / SG_SIZE;
    const int task_id = gid * sgs_per_wg + sg_in_wg;

    if (task_id >= total_tasks) return;

    const int b = task_id / N;
    const int o = task_id % N;
    const int lane = sg.get_local_linear_id();

    const sycl::ext::oneapi::bfloat16* x_b = in + b * K;
    const sycl::ext::oneapi::bfloat16* w_o = w + o * K;

    float acc = 0.0f;
#pragma unroll 4
    for (int i = lane; i < K; i += SG_SIZE) {
        acc += static_cast<float>(x_b[i]) * static_cast<float>(w_o[i]);
    }

    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) {
        out[task_id] = static_cast<sycl::ext::oneapi::bfloat16>(acc);
    }
}

/// Xe2 Split Reduction Kernel (Threads-Per-Row with Shared Local Memory).
/// Used for narrow output shapes (e.g. ssm_alpha [2560, 48], indexer.k_proj [2560, 128])
/// where output width is too small to saturate all Xe-cores.
template <int TPR>
void xe2_split_gemv_kernel(sycl::nd_item<1> item, const sycl::ext::oneapi::bfloat16* __restrict__ in,
                           const sycl::ext::oneapi::bfloat16* __restrict__ w,
                           sycl::ext::oneapi::bfloat16* __restrict__ out, int K, int N, int total_tasks,
                           sycl::local_accessor<float, 1> scratch) {
    const int task_id = item.get_group(0);
    if (task_id >= total_tasks) return;

    const int b = task_id / N;
    const int o = task_id % N;
    const int t = item.get_local_id(0);

    const sycl::ext::oneapi::bfloat16* x_b = in + b * K;
    const sycl::ext::oneapi::bfloat16* w_o = w + o * K;

    float acc = 0.0f;
#pragma unroll 4
    for (int i = t; i < K; i += TPR) {
        acc += static_cast<float>(x_b[i]) * static_cast<float>(w_o[i]);
    }

    scratch[t] = acc;
    item.barrier(sycl::access::fence_space::local_space);

    for (int off = TPR >> 1; off > 0; off >>= 1) {
        if (t < off) {
            scratch[t] += scratch[t + off];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    if (t == 0) {
        out[task_id] = static_cast<sycl::ext::oneapi::bfloat16>(scratch[0]);
    }
}

}  // namespace

// ============================================================================
// 1. Primary Benchmark: BM_Xe2_GEMV
// ============================================================================
// Benchmarks native Xe2 SIMD16 subgroup GEMV across all Qwen3.8-Flash-Next shapes.

static void BM_Xe2_GEMV(benchmark::State& state) {
    const int K = static_cast<int>(state.range(0));
    const int N = static_cast<int>(state.range(1));
    const int B = static_cast<int>(state.range(2));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    // Allocate USM device buffers
    auto* in = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * K, q);
    auto* w = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(K * N, q);
    auto* out = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * N, q);

    if (!in || !w || !out) {
        state.SkipWithError("Failed to allocate SYCL device memory");
        if (in) sycl::free(in, q);
        if (w) sycl::free(w, q);
        if (out) sycl::free(out, q);
        return;
    }

    // Initialize buffers to realistic non-zero values
    q.fill(in, sycl::ext::oneapi::bfloat16{0.01f}, B * K);
    q.fill(w, sycl::ext::oneapi::bfloat16{0.01f}, K * N);
    q.fill(out, sycl::ext::oneapi::bfloat16{0.0f}, B * N);
    q.wait_and_throw();

    constexpr int SG_SIZE = 16;   // Native Xe2 SIMD16
    constexpr int WG_SIZE = 256;  // 16 subgroups per work-group
    constexpr int sgs_per_wg = WG_SIZE / SG_SIZE;
    const int total_tasks = B * N;
    const int num_wgs = (total_tasks + sgs_per_wg - 1) / sgs_per_wg;

    // GPU Steady-State Warmup: ramp clocks from idle (800 MHz) to peak boost (2850 MHz)
    for (int warmup = 0; warmup < 10; ++warmup) {
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                  xe2_subgroup_gemv_kernel<SG_SIZE, WG_SIZE>(item, in, w, out, K, N, total_tasks);
                              });
         }).wait();
    }

    // Timed microbenchmark loop using hardware event profiling
    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 xe2_subgroup_gemv_kernel<SG_SIZE, WG_SIZE>(item, in, w, out, K, N, total_tasks);
                             });
        });
        ev.wait();

        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    // Hardware performance metrics & counters
    const double bytes = static_cast<double>(K * N + B * K + B * N) * sizeof(sycl::ext::oneapi::bfloat16);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] = benchmark::Counter((2.0 * B * K * N), benchmark::Counter::kIsIterationInvariantRate,
                                                  benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(in, q);
    sycl::free(w, q);
    sycl::free(out, q);
}

// ============================================================================
// 2. Subgroup & Work-Group Tuning Benchmark: BM_Xe2_GEMV_TuneSubgroup
// ============================================================================
// Sweeps combinations of subgroup sizes (SIMD16 vs SIMD32) and work-group sizes (64, 128, 256)
// to verify Xe2 SIMD16 architectural advantages and occupancy scaling.

static void BM_Xe2_GEMV_TuneSubgroup(benchmark::State& state) {
    const int K = static_cast<int>(state.range(0));
    const int N = static_cast<int>(state.range(1));
    const int B = static_cast<int>(state.range(2));
    const int SG_SIZE = static_cast<int>(state.range(3));
    const int WG_SIZE = static_cast<int>(state.range(4));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    auto* in = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * K, q);
    auto* w = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(K * N, q);
    auto* out = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * N, q);

    if (!in || !w || !out) {
        state.SkipWithError("Failed to allocate SYCL device memory");
        if (in) sycl::free(in, q);
        if (w) sycl::free(w, q);
        if (out) sycl::free(out, q);
        return;
    }

    q.fill(in, sycl::ext::oneapi::bfloat16{0.01f}, B * K);
    q.fill(w, sycl::ext::oneapi::bfloat16{0.01f}, K * N);
    q.fill(out, sycl::ext::oneapi::bfloat16{0.0f}, B * N);
    q.wait_and_throw();

    const int sgs_per_wg = WG_SIZE / SG_SIZE;
    const int total_tasks = B * N;
    const int num_wgs = (total_tasks + sgs_per_wg - 1) / sgs_per_wg;

    auto launch_kernel = [&](sycl::handler& cgh) {
        if (SG_SIZE == 16) {
            if (WG_SIZE == 64) {
                cgh.parallel_for(sycl::nd_range<1>(num_wgs * 64, 64),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     xe2_subgroup_gemv_kernel<16, 64>(item, in, w, out, K, N, total_tasks);
                                 });
            } else if (WG_SIZE == 128) {
                cgh.parallel_for(sycl::nd_range<1>(num_wgs * 128, 128),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     xe2_subgroup_gemv_kernel<16, 128>(item, in, w, out, K, N, total_tasks);
                                 });
            } else {
                cgh.parallel_for(sycl::nd_range<1>(num_wgs * 256, 256),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     xe2_subgroup_gemv_kernel<16, 256>(item, in, w, out, K, N, total_tasks);
                                 });
            }
        } else {
            if (WG_SIZE == 64) {
                cgh.parallel_for(sycl::nd_range<1>(num_wgs * 64, 64),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                                     xe2_subgroup_gemv_kernel<32, 64>(item, in, w, out, K, N, total_tasks);
                                 });
            } else if (WG_SIZE == 128) {
                cgh.parallel_for(sycl::nd_range<1>(num_wgs * 128, 128),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                                     xe2_subgroup_gemv_kernel<32, 128>(item, in, w, out, K, N, total_tasks);
                                 });
            } else {
                cgh.parallel_for(sycl::nd_range<1>(num_wgs * 256, 256),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                                     xe2_subgroup_gemv_kernel<32, 256>(item, in, w, out, K, N, total_tasks);
                                 });
            }
        }
    };

    // Warmup
    for (int warmup = 0; warmup < 5; ++warmup) {
        q.submit(launch_kernel).wait();
    }

    // Timed execution
    for (auto _ : state) {
        auto ev = q.submit(launch_kernel);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(K * N + B * K + B * N) * sizeof(sycl::ext::oneapi::bfloat16);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] = benchmark::Counter((2.0 * B * K * N), benchmark::Counter::kIsIterationInvariantRate,
                                                  benchmark::Counter::kIs1000);

    sycl::free(in, q);
    sycl::free(w, q);
    sycl::free(out, q);
}

// ============================================================================
// 3. Shared Memory Split Reduction Benchmark: BM_Xe2_GEMV_SplitReduction
// ============================================================================
// Tests SLM reduction across threads_per_row = 64, 128, 256 to measure whether
// multi-thread cooperative reductions saturate Xe-cores better on narrow shapes.

static void BM_Xe2_GEMV_SplitReduction(benchmark::State& state) {
    const int K = static_cast<int>(state.range(0));
    const int N = static_cast<int>(state.range(1));
    const int B = static_cast<int>(state.range(2));
    const int TPR = static_cast<int>(state.range(3));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    auto* in = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * K, q);
    auto* w = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(K * N, q);
    auto* out = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * N, q);

    if (!in || !w || !out) {
        state.SkipWithError("Failed to allocate SYCL device memory");
        if (in) sycl::free(in, q);
        if (w) sycl::free(w, q);
        if (out) sycl::free(out, q);
        return;
    }

    q.fill(in, sycl::ext::oneapi::bfloat16{0.01f}, B * K);
    q.fill(w, sycl::ext::oneapi::bfloat16{0.01f}, K * N);
    q.fill(out, sycl::ext::oneapi::bfloat16{0.0f}, B * N);
    q.wait_and_throw();

    const int total_tasks = B * N;

    auto launch_split = [&](sycl::handler& cgh) {
        if (TPR == 64) {
            sycl::local_accessor<float, 1> scratch(sycl::range<1>(64), cgh);
            cgh.parallel_for(sycl::nd_range<1>(total_tasks * 64, 64),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 xe2_split_gemv_kernel<64>(item, in, w, out, K, N, total_tasks, scratch);
                             });
        } else if (TPR == 128) {
            sycl::local_accessor<float, 1> scratch(sycl::range<1>(128), cgh);
            cgh.parallel_for(sycl::nd_range<1>(total_tasks * 128, 128),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 xe2_split_gemv_kernel<128>(item, in, w, out, K, N, total_tasks, scratch);
                             });
        } else {
            sycl::local_accessor<float, 1> scratch(sycl::range<1>(256), cgh);
            cgh.parallel_for(sycl::nd_range<1>(total_tasks * 256, 256),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 xe2_split_gemv_kernel<256>(item, in, w, out, K, N, total_tasks, scratch);
                             });
        }
    };

    // Warmup
    for (int warmup = 0; warmup < 5; ++warmup) {
        q.submit(launch_split).wait();
    }

    // Timed execution
    for (auto _ : state) {
        auto ev = q.submit(launch_split);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(K * N + B * K + B * N) * sizeof(sycl::ext::oneapi::bfloat16);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] = benchmark::Counter((2.0 * B * K * N), benchmark::Counter::kIsIterationInvariantRate,
                                                  benchmark::Counter::kIs1000);

    sycl::free(in, q);
    sycl::free(w, q);
    sycl::free(out, q);
}

// ============================================================================
// Registration: Qwen3.8-Flash-Next Decode & Prefill Projections
// ============================================================================

// Register all 8 Qwen3.8-Flash-Next decode projections (B=1)
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 48, 1});    // ssm_alpha / ssm_beta [2560, 48]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 128, 1});   // indexer.k_proj [2560, 128]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 512, 1});   // indexer.q_proj [2560, 512]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 640, 1});   // ffn_gate_shexp [2560, 640]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({640, 2560, 1});   // ffn_down_shexp [640, 2560]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({10240, 320, 1});  // hc_attn_down [10240, 320]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({320, 10240, 1});  // hc_attn_up [320, 10240]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 2560, 1});  // ple_value / q_proj / out_proj [2560, 2560]

// Register expert projections
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 10240, 1});  // gate / up expert [2560, 10240]
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({10240, 2560, 1});  // down expert [10240, 2560]

// Register prefill context chunking (B=24, B=32)
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({10240, 320, 24});  // prefill hc_attn_down chunking
BENCHMARK(BM_Xe2_GEMV)->UseManualTime()->Args({2560, 2560, 32});  // prefill attention/ffn chunking

// ============================================================================
// Registration: Microarchitectural Parameter Tuning Sweeps
// ============================================================================

// Tuning: Subgroup size (16 vs 32) and Workgroup size (64, 128, 256) on narrow and square shapes
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 128, 1, 16, 64});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 128, 1, 16, 128});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 128, 1, 16, 256});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 128, 1, 32, 64});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 128, 1, 32, 128});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 128, 1, 32, 256});

BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 2560, 1, 16, 64});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 2560, 1, 16, 128});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 2560, 1, 16, 256});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 2560, 1, 32, 64});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 2560, 1, 32, 128});
BENCHMARK(BM_Xe2_GEMV_TuneSubgroup)->UseManualTime()->Args({2560, 2560, 1, 32, 256});

// Tuning: SLM Split reduction threads_per_row (64, 128, 256) on narrow shapes
BENCHMARK(BM_Xe2_GEMV_SplitReduction)->UseManualTime()->Args({2560, 48, 1, 64});
BENCHMARK(BM_Xe2_GEMV_SplitReduction)->UseManualTime()->Args({2560, 48, 1, 128});
BENCHMARK(BM_Xe2_GEMV_SplitReduction)->UseManualTime()->Args({2560, 48, 1, 256});

BENCHMARK(BM_Xe2_GEMV_SplitReduction)->UseManualTime()->Args({10240, 320, 1, 64});
BENCHMARK(BM_Xe2_GEMV_SplitReduction)->UseManualTime()->Args({10240, 320, 1, 128});
BENCHMARK(BM_Xe2_GEMV_SplitReduction)->UseManualTime()->Args({10240, 320, 1, 256});

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 Performance Tuning Suite - GEMV Microbenchmark\n");
        std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
        std::printf(" Architecture: %s (Native Subgroup Size: %d)\n",
                    strata::sycl_arch::IS_XE2 ? "Intel Xe2 (Battlemage/Lunar Lake)" : "Generic",
                    strata::sycl_arch::SUB_GROUP_SIZE);
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
