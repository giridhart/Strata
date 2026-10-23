// perf/sycl/perf_quantize_act.cpp - Microbenchmark measuring fused quantize_act_images vs unfused sequences on Intel Xe2.
#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct ActQuantFixture {
    int64_t n;
    sycl::queue q;

    float* d_x;
    uint8_t* d_q8_0;
    uint8_t* d_q8_K;
    uint16_t* d_bf16;

    ActQuantFixture(int64_t size)
        : n(size),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

        std::vector<float> h_x((size_t)n);
        for (auto& v : h_x) v = dist(rng);

        const size_t nb_q80 = (size_t)(n / 32);
        const size_t nb_q8K = (size_t)(n / 256);

        d_x = sycl::malloc_device<float>((size_t)n, q);
        d_q8_0 = sycl::malloc_device<uint8_t>(nb_q80 * 34, q);
        d_q8_K = (n % 256 == 0) ? sycl::malloc_device<uint8_t>(nb_q8K * 292, q) : nullptr;
        d_bf16 = sycl::malloc_device<uint16_t>((size_t)n, q);

        q.memcpy(d_x, h_x.data(), (size_t)n * sizeof(float)).wait();
    }

    ~ActQuantFixture() {
        if (d_x) sycl::free(d_x, q);
        if (d_q8_0) sycl::free(d_q8_0, q);
        if (d_q8_K) sycl::free(d_q8_K, q);
        if (d_bf16) sycl::free(d_bf16, q);
    }
};

void BM_Unfused_ActImages_3Kernel(benchmark::State& state) {
    const int64_t N = state.range(0);
    ActQuantFixture fix(N);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
        strata::kernels::quantize_q8_K(fix.d_x, fix.d_q8_K, N, &fix.q);
        strata::kernels::f32_to_bf16_bulk(fix.d_x, fix.d_bf16, N, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
        strata::kernels::quantize_q8_K(fix.d_x, fix.d_q8_K, N, &fix.q);
        strata::kernels::f32_to_bf16_bulk(fix.d_x, fix.d_bf16, N, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double bytes_read = (double)N * sizeof(float) * 3.0; // 3 reads of x
    const double bytes_written = (double)(N / 32) * 34.0 + (double)(N / 256) * 292.0 + (double)N * sizeof(uint16_t);
    state.counters["Bytes/s"] = benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate);
}

void BM_Fused_ActImages_3Output(benchmark::State& state) {
    const int64_t N = state.range(0);
    ActQuantFixture fix(N);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, fix.d_q8_K, fix.d_bf16, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, fix.d_q8_K, fix.d_bf16, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double bytes_read = (double)N * sizeof(float); // 1 read of x
    const double bytes_written = (double)(N / 32) * 34.0 + (double)(N / 256) * 292.0 + (double)N * sizeof(uint16_t);
    state.counters["Bytes/s"] = benchmark::Counter(bytes_read + bytes_written, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_Unfused_ActImages_3Kernel)->UseManualTime()->Args({2560});
BENCHMARK(BM_Fused_ActImages_3Output)->UseManualTime()->Args({2560});

void run_direct_summary() {
    std::printf("========================================================================================\n");
    std::printf(" Intel Xe2 Microbenchmark: Fused Multi-Format Quantizer (quantize_act_images) vs Unfused\n");
    std::printf("========================================================================================\n");

    const int warmup = 20;
    const int iters = 200;

    // Test 1: Decode shape N=2560 with all 3 images (Q8_0 + Q8_K + BF16)
    {
        const int64_t N = 2560;
        ActQuantFixture fix(N);

        for (int i = 0; i < warmup; ++i) {
            strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
            strata::kernels::quantize_q8_K(fix.d_x, fix.d_q8_K, N, &fix.q);
            strata::kernels::f32_to_bf16_bulk(fix.d_x, fix.d_bf16, N, &fix.q);
            fix.q.wait();
        }

        auto t0_unfused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
            strata::kernels::quantize_q8_K(fix.d_x, fix.d_q8_K, N, &fix.q);
            strata::kernels::f32_to_bf16_bulk(fix.d_x, fix.d_bf16, N, &fix.q);
            fix.q.wait();
        }
        auto t1_unfused = std::chrono::high_resolution_clock::now();
        const double lat_unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

        for (int i = 0; i < warmup; ++i) {
            strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, fix.d_q8_K, fix.d_bf16, &fix.q);
            fix.q.wait();
        }

        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, fix.d_q8_K, fix.d_bf16, &fix.q);
            fix.q.wait();
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        const double lat_fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

        const double bytes_fused = (double)N * sizeof(float) + (double)(N / 32) * 34.0 + (double)(N / 256) * 292.0 + (double)N * sizeof(uint16_t);
        const double bw_gbs = (bytes_fused / (lat_fused_us * 1e-6)) * 1e-9;
        const double speedup = lat_unfused_us / lat_fused_us;
        const double reduction_pct = (lat_unfused_us - lat_fused_us) / lat_unfused_us * 100.0;

        std::printf("Configuration: Shape N=%lld, 3-Format (Q8_0 + Q8_K + BF16)\n", (long long)N);
        std::printf("  Unfused (3 launches: q8_0 + q8_K + bf16): %7.2f µs\n", lat_unfused_us);
        std::printf("  Fused   (1 launch: quantize_act_images):  %7.2f µs\n", lat_fused_us);
        std::printf("  Speedup:                                  %7.2fx (latency -%.1f%%)\n", speedup, reduction_pct);
        std::printf("  Effective Bandwidth:                      %7.2f GB/s\n\n", bw_gbs);
    }

    // Test 2: GDN/ssm_out shape N=2560 with 2 images (Q8_0 + Q8_K)
    {
        const int64_t N = 2560;
        ActQuantFixture fix(N);

        for (int i = 0; i < warmup; ++i) {
            strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
            strata::kernels::quantize_q8_K(fix.d_x, fix.d_q8_K, N, &fix.q);
            fix.q.wait();
        }

        auto t0_unfused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
            strata::kernels::quantize_q8_K(fix.d_x, fix.d_q8_K, N, &fix.q);
            fix.q.wait();
        }
        auto t1_unfused = std::chrono::high_resolution_clock::now();
        const double lat_unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

        for (int i = 0; i < warmup; ++i) {
            strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, fix.d_q8_K, nullptr, &fix.q);
            fix.q.wait();
        }

        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, fix.d_q8_K, nullptr, &fix.q);
            fix.q.wait();
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        const double lat_fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

        const double bytes_fused = (double)N * sizeof(float) + (double)(N / 32) * 34.0 + (double)(N / 256) * 292.0;
        const double bw_gbs = (bytes_fused / (lat_fused_us * 1e-6)) * 1e-9;
        const double speedup = lat_unfused_us / lat_fused_us;
        const double reduction_pct = (lat_unfused_us - lat_fused_us) / lat_unfused_us * 100.0;

        std::printf("Configuration: Shape N=%lld, 2-Format (Q8_0 + Q8_K)\n", (long long)N);
        std::printf("  Unfused (2 launches: q8_0 + q8_K):        %7.2f µs\n", lat_unfused_us);
        std::printf("  Fused   (1 launch: quantize_act_images):  %7.2f µs\n", lat_fused_us);
        std::printf("  Speedup:                                  %7.2fx (latency -%.1f%%)\n", speedup, reduction_pct);
        std::printf("  Effective Bandwidth:                      %7.2f GB/s\n\n", bw_gbs);
    }

    // Test 3: Shape N=640 with 2 images (Q8_0 + BF16)
    {
        const int64_t N = 640;
        ActQuantFixture fix(N);

        for (int i = 0; i < warmup; ++i) {
            strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
            strata::kernels::f32_to_bf16_bulk(fix.d_x, fix.d_bf16, N, &fix.q);
            fix.q.wait();
        }

        auto t0_unfused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::quantize_q8_0(fix.d_x, fix.d_q8_0, N, &fix.q);
            strata::kernels::f32_to_bf16_bulk(fix.d_x, fix.d_bf16, N, &fix.q);
            fix.q.wait();
        }
        auto t1_unfused = std::chrono::high_resolution_clock::now();
        const double lat_unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

        for (int i = 0; i < warmup; ++i) {
            strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, nullptr, fix.d_bf16, &fix.q);
            fix.q.wait();
        }

        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::quantize_act_images(fix.d_x, N, fix.d_q8_0, nullptr, fix.d_bf16, &fix.q);
            fix.q.wait();
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        const double lat_fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

        const double bytes_fused = (double)N * sizeof(float) + (double)(N / 32) * 34.0 + (double)N * sizeof(uint16_t);
        const double bw_gbs = (bytes_fused / (lat_fused_us * 1e-6)) * 1e-9;
        const double speedup = lat_unfused_us / lat_fused_us;
        const double reduction_pct = (lat_unfused_us - lat_fused_us) / lat_unfused_us * 100.0;

        std::printf("Configuration: Shape N=%lld, 2-Format (Q8_0 + BF16)\n", (long long)N);
        std::printf("  Unfused (2 launches: q8_0 + bf16):        %7.2f µs\n", lat_unfused_us);
        std::printf("  Fused   (1 launch: quantize_act_images):  %7.2f µs\n", lat_fused_us);
        std::printf("  Speedup:                                  %7.2fx (latency -%.1f%%)\n", speedup, reduction_pct);
        std::printf("  Effective Bandwidth:                      %7.2f GB/s\n\n", bw_gbs);
    }

    std::printf("========================================================================================\n");
}

}  // namespace

int main(int argc, char** argv) {
    bool run_summary = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark_filter" || std::string(argv[i]) == "--benchmark_list_tests") {
            run_summary = false;
        }
    }

    if (run_summary) {
        run_direct_summary();
    }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
