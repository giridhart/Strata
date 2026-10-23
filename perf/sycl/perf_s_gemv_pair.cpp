// perf/sycl/perf_s_gemv_pair.cpp - Microbenchmark measuring fused s_gemv_pair_silu vs unfused pair on Intel Xe2.
#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using strata::kernels::Codebook;
using strata::kernels::SForm;

void gemv_unfused(const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                  const uint8_t* x80, const uint8_t* x8k, float* y, int64_t n_in, int64_t n_out,
                  sycl::queue& q) {
    if (f.code_bits == 2) {
        strata::kernels::s2_gemv_q8(x80, codes, scales, y, n_in, n_out, strata::sycl_arch::SUB_GROUP_SIZE, &q);
    } else if (f.act_kind == 1) {
        strata::kernels::s_gemv_q8k_split(x8k, codes, scales, off, y, n_in, n_out, f, &q);
    } else {
        strata::kernels::s_gemv_q8_0_split(x80, codes, scales, off, y, n_in, n_out, f, &q);
    }
}

void run_swiglu(const float* g, const float* u, float* out, int64_t n, int swilu_kind, sycl::queue& q) {
    const int threads = 128;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                   [=](sycl::nd_item<1> item) {
                       const int i = item.get_global_id(0);
                       if (i >= n) return;
                       if (swilu_kind == 0) {
                           const double x = (double)g[i];
                           out[i] = (float)(x / (1.0 + sycl::exp(-x))) * u[i];
                       } else {
                           out[i] = (g[i] / (1.0f + sycl::native::exp(-g[i]))) * u[i];
                       }
                   });
}

// Benchmark fixture for dual gate/up GEMV + SwiGLU
struct GemvPairFixture {
    int64_t n_in;
    int64_t n_out;
    SForm fg;
    SForm fu;
    sycl::queue q;

    uint8_t *d_cg, *d_cu, *d_x80, *d_x8k;
    float *d_sg, *d_su, *d_og, *d_ou;
    float *d_gate, *d_up, *d_out;

    GemvPairFixture(int64_t kin, int64_t nout, SForm form_g, SForm form_u)
        : n_in(kin), n_out(nout), fg(form_g), fu(form_u),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> act_dist(-2.0f, 2.0f);
        std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);

        std::vector<float> h_x((size_t)n_in);
        for (auto& v : h_x) v = act_dist(rng);

        const int per_byte_g = 8 / fg.code_bits;
        const int per_byte_u = 8 / fu.code_bits;
        const int64_t n_groups_g = n_in / fg.group_elems;
        const int64_t n_groups_u = n_in / fu.group_elems;

        std::vector<uint8_t> h_cg((size_t)(n_out * (n_in / per_byte_g)));
        std::vector<uint8_t> h_cu((size_t)(n_out * (n_in / per_byte_u)));
        for (auto& v : h_cg) v = (uint8_t)rng();
        for (auto& v : h_cu) v = (uint8_t)rng();

        std::vector<float> h_sg((size_t)(n_out * n_groups_g), 0.01f);
        std::vector<float> h_su((size_t)(n_out * n_groups_u), 0.01f);

        const size_t q80_bytes = (size_t)(n_in / 32) * 34;
        const size_t q8k_bytes = (size_t)(n_in / 256) * 292;
        float* d_xf = sycl::malloc_device<float>((size_t)n_in, q);
        d_x80 = sycl::malloc_device<uint8_t>(q80_bytes, q);
        d_x8k = (fg.act_kind == 1 || fu.act_kind == 1) ? sycl::malloc_device<uint8_t>(q8k_bytes, q) : nullptr;

        q.memcpy(d_xf, h_x.data(), (size_t)n_in * sizeof(float)).wait();
        strata::kernels::quantize_q8_0(d_xf, d_x80, n_in, &q);
        if (d_x8k) strata::kernels::quantize_q8_K(d_xf, d_x8k, n_in, &q);
        q.wait();
        sycl::free(d_xf, q);

        d_cg = sycl::malloc_device<uint8_t>(h_cg.size(), q);
        d_cu = sycl::malloc_device<uint8_t>(h_cu.size(), q);
        d_sg = sycl::malloc_device<float>(h_sg.size(), q);
        d_su = sycl::malloc_device<float>(h_su.size(), q);
        d_og = nullptr;
        d_ou = nullptr;

        q.memcpy(d_cg, h_cg.data(), h_cg.size()).wait();
        q.memcpy(d_cu, h_cu.data(), h_cu.size()).wait();
        q.memcpy(d_sg, h_sg.data(), h_sg.size() * sizeof(float)).wait();
        q.memcpy(d_su, h_su.data(), h_su.size() * sizeof(float)).wait();

        d_gate = sycl::malloc_device<float>((size_t)n_out, q);
        d_up = sycl::malloc_device<float>((size_t)n_out, q);
        d_out = sycl::malloc_device<float>((size_t)n_out, q);
    }

    ~GemvPairFixture() {
        sycl::free(d_x80, q);
        if (d_x8k) sycl::free(d_x8k, q);
        sycl::free(d_cg, q);
        sycl::free(d_cu, q);
        sycl::free(d_sg, q);
        sycl::free(d_su, q);
        sycl::free(d_gate, q);
        sycl::free(d_up, q);
        sycl::free(d_out, q);
    }
};

static void BM_Unfused_GateUp_SwiGLU(benchmark::State& state) {
    const int64_t K = state.range(0);
    const int64_t N = state.range(1);
    const SForm f4{4, -8, 64, Codebook::Affine, false, /*act_kind=*/0};
    GemvPairFixture fix(K, N, f4, f4);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        gemv_unfused(fix.fg, fix.d_cg, fix.d_sg, nullptr, fix.d_x80, fix.d_x8k, fix.d_gate, K, N, fix.q);
        gemv_unfused(fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_x80, fix.d_x8k, fix.d_up, K, N, fix.q);
        run_swiglu(fix.d_gate, fix.d_up, fix.d_out, N, 1, fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        gemv_unfused(fix.fg, fix.d_cg, fix.d_sg, nullptr, fix.d_x80, fix.d_x8k, fix.d_gate, K, N, fix.q);
        gemv_unfused(fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_x80, fix.d_x8k, fix.d_up, K, N, fix.q);
        run_swiglu(fix.d_gate, fix.d_up, fix.d_out, N, 1, fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double flops = 4.0 * (double)K * (double)N + 3.0 * (double)N;
    const double bytes = ((double)(K * N) / 2.0 + (double)(K * N / 64) * 4.0) * 2.0 + (double)(K / 32) * 34.0 * 2.0 + (double)N * 4.0 * 3.0;
    state.counters["FLOP/s"] = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["Bytes/s"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_Fused_GateUp_SwiGLU(benchmark::State& state) {
    const int64_t K = state.range(0);
    const int64_t N = state.range(1);
    const SForm f4{4, -8, 64, Codebook::Affine, false, /*act_kind=*/0};
    GemvPairFixture fix(K, N, f4, f4);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::s_gemv_pair_silu(fix.d_x80, fix.d_x8k, fix.fg, fix.d_cg, fix.d_sg, nullptr,
                                          fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_out, K, N,
                                          32, 1, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::s_gemv_pair_silu(fix.d_x80, fix.d_x8k, fix.fg, fix.d_cg, fix.d_sg, nullptr,
                                          fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_out, K, N,
                                          32, 1, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double flops = 4.0 * (double)K * (double)N + 3.0 * (double)N;
    const double bytes = ((double)(K * N) / 2.0 + (double)(K * N / 64) * 4.0) * 2.0 + (double)(K / 32) * 34.0 + (double)N * 4.0;
    state.counters["FLOP/s"] = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["Bytes/s"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_Unfused_GateUp_SwiGLU)->UseManualTime()->Args({2560, 640});
BENCHMARK(BM_Fused_GateUp_SwiGLU)->UseManualTime()->Args({2560, 640});
BENCHMARK(BM_Unfused_GateUp_SwiGLU)->UseManualTime()->Args({2560, 6144});
BENCHMARK(BM_Fused_GateUp_SwiGLU)->UseManualTime()->Args({2560, 6144});

void run_direct_summary() {
    std::printf("========================================================================================\n");
    std::printf(" Intel Xe2 Microbenchmark: Fused Dual GEMV + SwiGLU (s_gemv_pair_silu) vs Unfused\n");
    std::printf("========================================================================================\n");

    const SForm f4{4, -8, 64, Codebook::Affine, false, /*act_kind=*/0};
    const std::vector<std::pair<int64_t, int64_t>> shapes = {{2560, 640}, {2560, 6144}};
    const int warmup = 10;
    const int iters = 100;

    for (auto [K, N] : shapes) {
        GemvPairFixture fix(K, N, f4, f4);

        // Warmup unfused
        for (int i = 0; i < warmup; ++i) {
            gemv_unfused(fix.fg, fix.d_cg, fix.d_sg, nullptr, fix.d_x80, fix.d_x8k, fix.d_gate, K, N, fix.q);
            gemv_unfused(fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_x80, fix.d_x8k, fix.d_up, K, N, fix.q);
            run_swiglu(fix.d_gate, fix.d_up, fix.d_out, N, 1, fix.q);
            fix.q.wait();
        }

        // Time unfused
        auto t0_unfused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            gemv_unfused(fix.fg, fix.d_cg, fix.d_sg, nullptr, fix.d_x80, fix.d_x8k, fix.d_gate, K, N, fix.q);
            gemv_unfused(fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_x80, fix.d_x8k, fix.d_up, K, N, fix.q);
            run_swiglu(fix.d_gate, fix.d_up, fix.d_out, N, 1, fix.q);
            fix.q.wait();
        }
        auto t1_unfused = std::chrono::high_resolution_clock::now();
        const double lat_unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

        // Warmup fused
        for (int i = 0; i < warmup; ++i) {
            strata::kernels::s_gemv_pair_silu(fix.d_x80, fix.d_x8k, fix.fg, fix.d_cg, fix.d_sg, nullptr,
                                              fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_out, K, N,
                                              32, 1, &fix.q);
            fix.q.wait();
        }

        // Time fused
        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strata::kernels::s_gemv_pair_silu(fix.d_x80, fix.d_x8k, fix.fg, fix.d_cg, fix.d_sg, nullptr,
                                              fix.fu, fix.d_cu, fix.d_su, nullptr, fix.d_out, K, N,
                                              32, 1, &fix.q);
            fix.q.wait();
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        const double lat_fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

        const double flops = 4.0 * (double)K * (double)N + 3.0 * (double)N;
        const double gflops_fused = (flops / (lat_fused_us * 1e-6)) * 1e-9;
        const double bytes = ((double)(K * N) / 2.0 + (double)(K * N / 64) * 4.0) * 2.0 + (double)(K / 32) * 34.0 + (double)N * 4.0;
        const double bw_fused_gbs = (bytes / (lat_fused_us * 1e-6)) * 1e-9;
        const double speedup = lat_unfused_us / lat_fused_us;
        const double reduction_pct = (lat_unfused_us - lat_fused_us) / lat_unfused_us * 100.0;

        std::printf("Shape: [K=%lld, N=%lld]\n", (long long)K, (long long)N);
        std::printf("  Unfused (GEMV+GEMV+SwiGLU): %7.2f µs\n", lat_unfused_us);
        std::printf("  Fused   (s_gemv_pair_silu): %7.2f µs\n", lat_fused_us);
        std::printf("  Speedup:                    %7.2fx (latency -%.1f%%)\n", speedup, reduction_pct);
        std::printf("  Throughput:                 %7.2f GFLOPS, %7.2f GB/s\n\n", gflops_fused, bw_fused_gbs);
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
