// perf/sycl/perf_bf16_gemv_pair.cpp - Microbenchmark measuring fused bf16_gemv_pair vs two standalone bf16_gemv calls on Intel Xe2.
#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;

struct Bf16PairFixture {
    int64_t n_in;
    int64_t n_out1;
    int64_t n_out2;
    sycl::queue q;

    uint16_t* d_x;
    uint16_t* d_w1;
    uint16_t* d_w2;
    float* d_y1;
    float* d_y2;

    Bf16PairFixture(int64_t in_dim, int64_t out1, int64_t out2)
        : n_in(in_dim), n_out1(out1), n_out2(out2),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        std::vector<uint16_t> h_x((size_t)n_in);
        for (auto& v : h_x) v = bf16_from_f32(dist(rng));

        std::vector<uint16_t> h_w1((size_t)(n_in * n_out1));
        for (auto& v : h_w1) v = bf16_from_f32(dist(rng) * 0.05f);

        std::vector<uint16_t> h_w2((size_t)(n_in * n_out2));
        for (auto& v : h_w2) v = bf16_from_f32(dist(rng) * 0.05f);

        d_x = sycl::malloc_device<uint16_t>((size_t)n_in, q);
        d_w1 = n_out1 > 0 ? sycl::malloc_device<uint16_t>((size_t)(n_in * n_out1), q) : nullptr;
        d_w2 = n_out2 > 0 ? sycl::malloc_device<uint16_t>((size_t)(n_in * n_out2), q) : nullptr;
        d_y1 = n_out1 > 0 ? sycl::malloc_device<float>((size_t)n_out1, q) : nullptr;
        d_y2 = n_out2 > 0 ? sycl::malloc_device<float>((size_t)n_out2, q) : nullptr;

        q.memcpy(d_x, h_x.data(), (size_t)n_in * sizeof(uint16_t));
        if (n_out1 > 0) q.memcpy(d_w1, h_w1.data(), (size_t)(n_in * n_out1) * sizeof(uint16_t));
        if (n_out2 > 0) q.memcpy(d_w2, h_w2.data(), (size_t)(n_in * n_out2) * sizeof(uint16_t));
        q.wait();
    }

    ~Bf16PairFixture() {
        if (d_x) sycl::free(d_x, q);
        if (d_w1) sycl::free(d_w1, q);
        if (d_w2) sycl::free(d_w2, q);
        if (d_y1) sycl::free(d_y1, q);
        if (d_y2) sycl::free(d_y2, q);
    }
};

static void BM_Unfused_BF16_Pair(benchmark::State& state) {
    const int64_t K = state.range(0);
    const int64_t N1 = state.range(1);
    const int64_t N2 = state.range(2);
    Bf16PairFixture fix(K, N1, N2);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        if (N1 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w1, fix.d_y1, K, N1, &fix.q);
        if (N2 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w2, fix.d_y2, K, N2, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (N1 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w1, fix.d_y1, K, N1, &fix.q);
        if (N2 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w2, fix.d_y2, K, N2, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double flops = 2.0 * (double)K * (double)(N1 + N2);
    const double bytes = (double)K * 2.0 * 2.0 + (double)(N1 + N2) * (double)K * 2.0 + (double)(N1 + N2) * 4.0;
    state.counters["FLOP/s"] = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["Bytes/s"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_Fused_BF16_Pair(benchmark::State& state) {
    const int64_t K = state.range(0);
    const int64_t N1 = state.range(1);
    const int64_t N2 = state.range(2);
    Bf16PairFixture fix(K, N1, N2);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::bf16_gemv_pair(fix.d_x, fix.d_w1, fix.d_y1, N1, fix.d_w2, fix.d_y2, N2, K, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::bf16_gemv_pair(fix.d_x, fix.d_w1, fix.d_y1, N1, fix.d_w2, fix.d_y2, N2, K, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double flops = 2.0 * (double)K * (double)(N1 + N2);
    const double bytes = (double)K * 2.0 + (double)(N1 + N2) * (double)K * 2.0 + (double)(N1 + N2) * 4.0;
    state.counters["FLOP/s"] = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["Bytes/s"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_Unfused_BF16_Pair)->UseManualTime()->Args({2560, 48, 48});
BENCHMARK(BM_Fused_BF16_Pair)->UseManualTime()->Args({2560, 48, 48});
BENCHMARK(BM_Unfused_BF16_Pair)->UseManualTime()->Args({2560, 128, 512});
BENCHMARK(BM_Fused_BF16_Pair)->UseManualTime()->Args({2560, 128, 512});

void run_direct_summary() {
    std::printf("========================================================================================\n");
    std::printf(" Intel Xe2 Microbenchmark: Fused Paired GEMV (bf16_gemv_pair) vs Standalone GEMVs\n");
    std::printf("========================================================================================\n");

    struct BenchmarkShape {
        int64_t K, N1, N2;
        const char* label;
    };
    const std::vector<BenchmarkShape> shapes = {
        {2560, 48, 48, "GDN ssm_alpha + ssm_beta [2560, 48+48]"},
        {2560, 128, 512, "QSA indexer k_proj + q_proj [2560, 128+512]"},
        {2560, 64, 256, "Asymmetric decode [2560, 64+256]"}
    };
    const int warmup = 10;
    const int iters = 100;

    for (const auto& s : shapes) {
        Bf16PairFixture fix(s.K, s.N1, s.N2);

        // Warmup unfused
        for (int i = 0; i < warmup; ++i) {
            if (s.N1 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w1, fix.d_y1, s.K, s.N1, &fix.q);
            if (s.N2 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w2, fix.d_y2, s.K, s.N2, &fix.q);
            fix.q.wait();
        }

        // Time unfused
        auto t0_unfused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (s.N1 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w1, fix.d_y1, s.K, s.N1, &fix.q);
            if (s.N2 > 0) strata::kernels::bf16_gemv(fix.d_x, fix.d_w2, fix.d_y2, s.K, s.N2, &fix.q);
            fix.q.wait();
        }
        auto t1_unfused = std::chrono::high_resolution_clock::now();
        const double lat_unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

        // Warmup fused
        for (int i = 0; i < warmup; ++i) {
            strata::kernels::bf16_gemv_pair(fix.d_x, fix.d_w1, fix.d_y1, s.N1, fix.d_w2, fix.d_y2, s.N2, s.K, &fix.q);
            fix.q.wait();
        }

        // Time fused
        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::bf16_gemv_pair(fix.d_x, fix.d_w1, fix.d_y1, s.N1, fix.d_w2, fix.d_y2, s.N2, s.K, &fix.q);
            fix.q.wait();
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        const double lat_fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

        const double flops = 2.0 * (double)s.K * (double)(s.N1 + s.N2);
        const double gflops_fused = (flops / (lat_fused_us * 1e-6)) * 1e-9;
        const double bytes = (double)s.K * 2.0 + (double)(s.N1 + s.N2) * (double)s.K * 2.0 + (double)(s.N1 + s.N2) * 4.0;
        const double bw_fused_gbs = (bytes / (lat_fused_us * 1e-6)) * 1e-9;
        const double speedup = lat_unfused_us / lat_fused_us;
        const double reduction_pct = (lat_unfused_us - lat_fused_us) / lat_unfused_us * 100.0;

        std::printf("Shape: %s [K=%lld, N1=%lld, N2=%lld]\n", s.label, (long long)s.K, (long long)s.N1, (long long)s.N2);
        std::printf("  Unfused (2x standalone GEMV): %7.2f µs\n", lat_unfused_us);
        std::printf("  Fused   (bf16_gemv_pair):     %7.2f µs\n", lat_fused_us);
        std::printf("  Speedup:                      %7.2fx (latency -%.1f%%)\n", speedup, reduction_pct);
        std::printf("  Throughput:                   %7.2f GFLOPS, %7.2f GB/s\n\n", gflops_fused, bw_fused_gbs);
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
