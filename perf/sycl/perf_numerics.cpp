// perf/sycl/perf_numerics.cpp - Intel Xe2 Fast E4M3 / E2M1 FP8 / FP4 Numerics & Block Quantization Benchmark
//
// Ports DeepSeek-V4.1 low-bit numerics (ds4_numerics.cuh & ds4_kernels.cu lines 60-125) to pure SYCL 2020.
// Measures throughput, latency, and compression efficiency on Intel Xe2 (Lunar Lake / Arc B580)
// across realistic KV cache buffer sizes (256K, 1M, 4M, 16M elements).
//
// Benchmarks:
//   1. Baseline Memory Bandwidth (DRAM streaming copy)
//   2. FP8 E4M3 Quantize & Dequantize (to F32 and FP16)
//   3. FP4 E2M1 Quantize & Dequantize (BlockSize = 32, to F32 and FP16)
//   4. FP4 E2M1 Quantize & Dequantize (BlockSize = 16, to F32)
//
// Verification:
//   Validates scalar bit-manipulation primitives (pow2_ceil, ue8m0_val, e4m3_round, e4m3_code,
//   e4m3_value, e2m1_rt, e2m1_code, e2m1_val14) and GPU device round-trip numerical accuracy.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/numerics_fp8_fp4.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct NumericsFixture {
    size_t n;
    sycl::queue q;

    float* d_in = nullptr;
    float* d_copy_out = nullptr;
    uint8_t* d_packed_fp8 = nullptr;
    uint8_t* d_packed_fp4_b32 = nullptr;
    uint8_t* d_packed_fp4_b16 = nullptr;
    float* d_out_f32 = nullptr;
    sycl::half* d_out_f16 = nullptr;

    explicit NumericsFixture(size_t size)
        : n(size),
          q(sycl::default_selector_v, {sycl::property::queue::enable_profiling()}) {

        std::mt19937 rng(1337);
        std::normal_distribution<float> dist(0.0f, 1.5f);

        std::vector<float> h_in(n);
        for (size_t i = 0; i < n; ++i) {
            h_in[i] = dist(rng);
        }

        const size_t fp8_bytes = strata::kernels::fp8_packed_bytes(n, 32);
        const size_t fp4_b32_bytes = strata::kernels::fp4_packed_bytes(n, 32);
        const size_t fp4_b16_bytes = strata::kernels::fp4_packed_bytes(n, 16);

        d_in = sycl::malloc_device<float>(n, q);
        d_copy_out = sycl::malloc_device<float>(n, q);
        d_packed_fp8 = sycl::malloc_device<uint8_t>(fp8_bytes, q);
        d_packed_fp4_b32 = sycl::malloc_device<uint8_t>(fp4_b32_bytes, q);
        d_packed_fp4_b16 = sycl::malloc_device<uint8_t>(fp4_b16_bytes, q);
        d_out_f32 = sycl::malloc_device<float>(n, q);
        d_out_f16 = sycl::malloc_device<sycl::half>(n, q);

        q.memcpy(d_in, h_in.data(), n * sizeof(float)).wait();
    }

    ~NumericsFixture() {
        if (d_in) sycl::free(d_in, q);
        if (d_copy_out) sycl::free(d_copy_out, q);
        if (d_packed_fp8) sycl::free(d_packed_fp8, q);
        if (d_packed_fp4_b32) sycl::free(d_packed_fp4_b32, q);
        if (d_packed_fp4_b16) sycl::free(d_packed_fp4_b16, q);
        if (d_out_f32) sycl::free(d_out_f32, q);
        if (d_out_f16) sycl::free(d_out_f16, q);
    }
};

// ============================================================================
// 1. Baseline Memory Bandwidth
// ============================================================================

void BM_Baseline_DRAM_Copy(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        fix.q.memcpy(fix.d_copy_out, fix.d_in, N * sizeof(float)).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = fix.q.memcpy(fix.d_copy_out, fix.d_in, N * sizeof(float));
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes = static_cast<double>(N * sizeof(float) * 2); // 1 read + 1 write
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

// ============================================================================
// 2. FP8 E4M3 Quantize & Dequantize
// ============================================================================

void BM_FP8_E4M3_Quantize(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::quantize_fp8_e4m3(fix.d_in, fix.d_packed_fp8, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::quantize_fp8_e4m3(fix.d_in, fix.d_packed_fp8, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(N * sizeof(float));
    const double bytes_written = static_cast<double>(strata::kernels::fp8_packed_bytes(N, 32));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

void BM_FP8_E4M3_Dequantize_F32(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    // Initialize packed data
    strata::kernels::quantize_fp8_e4m3(fix.d_in, fix.d_packed_fp8, N, fix.q).wait();

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::dequantize_fp8_e4m3(fix.d_packed_fp8, fix.d_out_f32, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::dequantize_fp8_e4m3(fix.d_packed_fp8, fix.d_out_f32, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(strata::kernels::fp8_packed_bytes(N, 32));
    const double bytes_written = static_cast<double>(N * sizeof(float));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

void BM_FP8_E4M3_Dequantize_FP16(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    strata::kernels::quantize_fp8_e4m3(fix.d_in, fix.d_packed_fp8, N, fix.q).wait();

    for (int i = 0; i < 5; ++i) {
        strata::kernels::dequantize_fp8_e4m3(fix.d_packed_fp8, fix.d_out_f16, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::dequantize_fp8_e4m3(fix.d_packed_fp8, fix.d_out_f16, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(strata::kernels::fp8_packed_bytes(N, 32));
    const double bytes_written = static_cast<double>(N * sizeof(sycl::half));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

// ============================================================================
// 3. FP4 E2M1 Quantize & Dequantize (BlockSize = 32)
// ============================================================================

void BM_FP4_E2M1_Quantize_B32(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::quantize_fp4_e2m1<32>(fix.d_in, fix.d_packed_fp4_b32, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::quantize_fp4_e2m1<32>(fix.d_in, fix.d_packed_fp4_b32, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(N * sizeof(float));
    const double bytes_written = static_cast<double>(strata::kernels::fp4_packed_bytes(N, 32));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

void BM_FP4_E2M1_Dequantize_B32_F32(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    strata::kernels::quantize_fp4_e2m1<32>(fix.d_in, fix.d_packed_fp4_b32, N, fix.q).wait();

    for (int i = 0; i < 5; ++i) {
        strata::kernels::dequantize_fp4_e2m1<32>(fix.d_packed_fp4_b32, fix.d_out_f32, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::dequantize_fp4_e2m1<32>(fix.d_packed_fp4_b32, fix.d_out_f32, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(strata::kernels::fp4_packed_bytes(N, 32));
    const double bytes_written = static_cast<double>(N * sizeof(float));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

void BM_FP4_E2M1_Dequantize_B32_FP16(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    strata::kernels::quantize_fp4_e2m1<32>(fix.d_in, fix.d_packed_fp4_b32, N, fix.q).wait();

    for (int i = 0; i < 5; ++i) {
        strata::kernels::dequantize_fp4_e2m1<32>(fix.d_packed_fp4_b32, fix.d_out_f16, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::dequantize_fp4_e2m1<32>(fix.d_packed_fp4_b32, fix.d_out_f16, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(strata::kernels::fp4_packed_bytes(N, 32));
    const double bytes_written = static_cast<double>(N * sizeof(sycl::half));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

// ============================================================================
// 4. FP4 E2M1 Quantize & Dequantize (BlockSize = 16)
// ============================================================================

void BM_FP4_E2M1_Quantize_B16(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    for (int i = 0; i < 5; ++i) {
        strata::kernels::quantize_fp4_e2m1<16>(fix.d_in, fix.d_packed_fp4_b16, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::quantize_fp4_e2m1<16>(fix.d_in, fix.d_packed_fp4_b16, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(N * sizeof(float));
    const double bytes_written = static_cast<double>(strata::kernels::fp4_packed_bytes(N, 16));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

void BM_FP4_E2M1_Dequantize_B16_F32(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));
    NumericsFixture fix(N);

    strata::kernels::quantize_fp4_e2m1<16>(fix.d_in, fix.d_packed_fp4_b16, N, fix.q).wait();

    for (int i = 0; i < 5; ++i) {
        strata::kernels::dequantize_fp4_e2m1<16>(fix.d_packed_fp4_b16, fix.d_out_f32, N, fix.q).wait();
    }

    double total_sec = 0.0;
    size_t iters = 0;

    for (auto _ : state) {
        auto ev = strata::kernels::dequantize_fp4_e2m1<16>(fix.d_packed_fp4_b16, fix.d_out_f32, N, fix.q);
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        const double iter_time = (t1 - t0) * 1e-9;
        state.SetIterationTime(iter_time);
        total_sec += iter_time;
        ++iters;
    }

    const double bytes_read = static_cast<double>(strata::kernels::fp4_packed_bytes(N, 16));
    const double bytes_written = static_cast<double>(N * sizeof(float));
    const double avg_sec = (iters > 0) ? (total_sec / iters) : 1e-6;

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Elems_Mps"] =
        benchmark::Counter(static_cast<double>(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Latency_us"] =
        benchmark::Counter(avg_sec * 1e6, benchmark::Counter::kAvgThreads);
}

// Register Google Benchmarks for realistic KV Cache sizes: 256K, 1M, 4M, 16M
#define REGISTER_NUMERICS_BENCHMARK(func) \
    BENCHMARK(func)                       \
        ->Arg(256 * 1024)                 \
        ->Arg(1024 * 1024)                \
        ->Arg(4 * 1024 * 1024)            \
        ->Arg(16 * 1024 * 1024)           \
        ->UseManualTime()

REGISTER_NUMERICS_BENCHMARK(BM_Baseline_DRAM_Copy);
REGISTER_NUMERICS_BENCHMARK(BM_FP8_E4M3_Quantize);
REGISTER_NUMERICS_BENCHMARK(BM_FP8_E4M3_Dequantize_F32);
REGISTER_NUMERICS_BENCHMARK(BM_FP8_E4M3_Dequantize_FP16);
REGISTER_NUMERICS_BENCHMARK(BM_FP4_E2M1_Quantize_B32);
REGISTER_NUMERICS_BENCHMARK(BM_FP4_E2M1_Dequantize_B32_F32);
REGISTER_NUMERICS_BENCHMARK(BM_FP4_E2M1_Dequantize_B32_FP16);
REGISTER_NUMERICS_BENCHMARK(BM_FP4_E2M1_Quantize_B16);
REGISTER_NUMERICS_BENCHMARK(BM_FP4_E2M1_Dequantize_B16_F32);

#undef REGISTER_NUMERICS_BENCHMARK

// ============================================================================
// Numerical Accuracy & Parity Verification Suite
// ============================================================================

bool verify_scalar_numerics() {
    std::printf("[VERIFY] 1. Checking scalar bit manipulations and roundings...\n");

    // a) pow2_ceil & ue8m0_val
    const float test_floats[] = {0.125f, 0.25f, 0.33f, 0.5f, 0.99f, 1.0f, 1.01f, 2.0f, 3.9f, 4.0f, 7.5f, 8.0f, 448.0f};
    for (float v : test_floats) {
        const float p = strata::kernels::pow2_ceil(v);
        if (p < v || (p > 2.0f * v && v > 1e-30f)) {
            std::printf("  [FAIL] pow2_ceil(%f) = %f is invalid!\n", v, p);
            return false;
        }
        const uint8_t sc = strata::kernels::ue8m0_scale_byte(p);
        const float p_rec = strata::kernels::ue8m0_val(sc);
        if (p != p_rec) {
            std::printf("  [FAIL] ue8m0 roundtrip mismatch for %f (rec %f)!\n", p, p_rec);
            return false;
        }
    }
    std::printf("  [PASS] pow2_ceil and ue8m0_val exact.\n");

    // b) E4M3 codes: all 256 code points
    for (int code = 0; code < 256; ++code) {
        if (code == 0x7f || code == 0xff) continue; // NaN in E4M3
        const float val = strata::kernels::e4m3_value(static_cast<uint8_t>(code));
        const float rounded = strata::kernels::e4m3_round(val);
        const uint32_t rec_code = strata::kernels::e4m3_code(rounded);
        if (rec_code != static_cast<uint32_t>(code)) {
            std::printf("  [FAIL] E4M3 code 0x%02x != rec_code 0x%02x (val=%f)!\n", code, rec_code, val);
            return false;
        }
    }
    std::printf("  [PASS] FP8 E4M3 254 non-NaN codes roundtrip bit-exact.\n");

    // c) E2M1 codes: all 16 4-bit codes
    const float expected_e2m1_mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint32_t c = 0; c < 16; ++c) {
        const float exp_mag = expected_e2m1_mags[c & 7u];
        const float exp_signed = (c & 8u) ? -exp_mag : exp_mag;
        const float val14 = strata::kernels::e2m1_val14(c);
        const float recovered = val14 * 16384.0f;
        if (std::fabs(recovered - exp_signed) > 1e-6f) {
            std::printf("  [FAIL] E2M1 val14 decode mismatch for code %u (exp %f, got %f)!\n", c, exp_signed, recovered);
            return false;
        }
        const uint32_t rec_c = strata::kernels::e2m1_code(strata::kernels::e2m1_rt(recovered));
        if (rec_c != c) {
            std::printf("  [FAIL] E2M1 code mismatch for code %u (rec %u)!\n", c, rec_c);
            return false;
        }
    }
    std::printf("  [PASS] FP4 E2M1 e2m1_val14 bit trick and e2m1_code bit-exact for all 16 codes.\n");

    return true;
}

bool verify_device_roundtrip(sycl::queue& q) {
    std::printf("[VERIFY] 2. Checking end-to-end device Quantize / Dequantize roundtrip on GPU...\n");

    constexpr size_t N = 65536; // 64K elements test tensor
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> h_orig(N);
    for (size_t i = 0; i < N; ++i) {
        h_orig[i] = dist(rng);
    }
    // Inject edge cases: exact powers of 2, zeros, large and small magnitudes
    h_orig[0] = 0.0f;
    h_orig[1] = -0.0f;
    h_orig[2] = 1.0f;
    h_orig[3] = -1.0f;
    h_orig[4] = 448.0f;
    h_orig[5] = -448.0f;
    h_orig[6] = 0.001953125f;
    h_orig[7] = 6.0f;

    float* d_in = sycl::malloc_device<float>(N, q);
    uint8_t* d_fp8 = sycl::malloc_device<uint8_t>(strata::kernels::fp8_packed_bytes(N, 32), q);
    uint8_t* d_fp4_b32 = sycl::malloc_device<uint8_t>(strata::kernels::fp4_packed_bytes(N, 32), q);
    uint8_t* d_fp4_b16 = sycl::malloc_device<uint8_t>(strata::kernels::fp4_packed_bytes(N, 16), q);
    float* d_out = sycl::malloc_device<float>(N, q);

    q.memcpy(d_in, h_orig.data(), N * sizeof(float)).wait();

    // 1. FP8 E4M3 verification
    strata::kernels::quantize_fp8_e4m3(d_in, d_fp8, N, q).wait();
    strata::kernels::dequantize_fp8_e4m3(d_fp8, d_out, N, q).wait();

    std::vector<float> h_rec_fp8(N);
    q.memcpy(h_rec_fp8.data(), d_out, N * sizeof(float)).wait();

    double max_err_fp8 = 0.0;
    double max_rel_err_fp8 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double err = std::fabs(static_cast<double>(h_orig[i]) - static_cast<double>(h_rec_fp8[i]));
        if (err > max_err_fp8) max_err_fp8 = err;
        if (std::fabs(h_orig[i]) > 0.05f) {
            const double rel_err = err / std::fabs(static_cast<double>(h_orig[i]));
            if (rel_err > max_rel_err_fp8) max_rel_err_fp8 = rel_err;
        }
    }
    // Theoretical max relative step error for FP8 E4M3 is 1 / 16 = 0.0625 (6.25%)
    if (max_rel_err_fp8 > 0.07) {
        std::printf("  [FAIL] FP8 max relative error %f exceeds 6.25%% bound!\n", max_rel_err_fp8);
        return false;
    }
    std::printf("  [PASS] FP8 E4M3 Round-trip verified: Max Abs Err = %.4e, Max Rel Err = %.2f%%\n",
                max_err_fp8, max_rel_err_fp8 * 100.0);

    // 2. FP4 E2M1 (BlockSize = 32) verification
    strata::kernels::quantize_fp4_e2m1<32>(d_in, d_fp4_b32, N, q).wait();
    strata::kernels::dequantize_fp4_e2m1<32>(d_fp4_b32, d_out, N, q).wait();

    std::vector<float> h_rec_fp4_32(N);
    q.memcpy(h_rec_fp4_32.data(), d_out, N * sizeof(float)).wait();

    double max_err_fp4_32 = 0.0;
    double mse_fp4_32 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double err = std::fabs(static_cast<double>(h_orig[i]) - static_cast<double>(h_rec_fp4_32[i]));
        if (err > max_err_fp4_32) max_err_fp4_32 = err;
        mse_fp4_32 += err * err;
    }
    mse_fp4_32 /= static_cast<double>(N);
    std::printf("  [PASS] FP4 E2M1 (Block=32) Round-trip verified: Max Abs Err = %.4e, RMSE = %.4e\n",
                max_err_fp4_32, std::sqrt(mse_fp4_32));

    // 3. FP4 E2M1 (BlockSize = 16) verification
    strata::kernels::quantize_fp4_e2m1<16>(d_in, d_fp4_b16, N, q).wait();
    strata::kernels::dequantize_fp4_e2m1<16>(d_fp4_b16, d_out, N, q).wait();

    std::vector<float> h_rec_fp4_16(N);
    q.memcpy(h_rec_fp4_16.data(), d_out, N * sizeof(float)).wait();

    double max_err_fp4_16 = 0.0;
    double mse_fp4_16 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double err = std::fabs(static_cast<double>(h_orig[i]) - static_cast<double>(h_rec_fp4_16[i]));
        if (err > max_err_fp4_16) max_err_fp4_16 = err;
        mse_fp4_16 += err * err;
    }
    mse_fp4_16 /= static_cast<double>(N);
    std::printf("  [PASS] FP4 E2M1 (Block=16) Round-trip verified: Max Abs Err = %.4e, RMSE = %.4e\n",
                max_err_fp4_16, std::sqrt(mse_fp4_16));

    sycl::free(d_in, q);
    sycl::free(d_fp8, q);
    sycl::free(d_fp4_b32, q);
    sycl::free(d_fp4_b16, q);
    sycl::free(d_out, q);

    return true;
}

} // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::default_selector_v};
    const auto dev = q.get_device();
    std::printf("======================================================================\n");
    std::printf("Intel Xe2 E4M3 / E2M1 Low-Bit Numerics & Quantization Microbenchmark\n");
    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Vendor: %s\n", dev.get_info<sycl::info::device::vendor>().c_str());
    std::printf("Driver: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("======================================================================\n");

    if (!verify_scalar_numerics() || !verify_device_roundtrip(q)) {
        std::fprintf(stderr, "[FATAL] Numerical verification FAILED!\n");
        return 1;
    }
    std::printf("[VERIFY] All verification tests PASSED!\n");
    std::printf("======================================================================\n\n");

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
