// perf/sycl/perf_gemv_rows.cpp - Multi-Row Matrix-Vector Multiplication (gemv_f16_rows) Benchmark on Intel Xe2
//
// Ports the DeepSeek-V4.1 multi-row GEMV technique (src/dsv41/ds4_kernels.cu lines 370-435) to pure SYCL 2020.
// Compares:
//   1. Baseline GEMV: Standard single-row GEMV launched across T tokens (re-reading weights per token from VRAM).
//   2. Multi-Row GEMV (Tile-2): gemv_f16_rows with ROWS_T = 2 (direct port of DeepSeek-V4.1 kRowsT = 2).
//   3. Multi-Row GEMV (Tuned): gemv_f16_rows with ROWS_T = 4 (optimized for Intel Xe2 register capacity).
//
// Evaluates on Intel Xe2 across realistic Qwen projection shapes:
//   - K=2048, N=2048
//   - K=2560, N=2560
//   - K=2560, N=6144 (Qwen gate/up projection)
//   - K=6144, N=2560 (Qwen down projection)
//   for T in [1, 2, 4, 8] tokens.
//
// Verification:
//   Full device numerical equivalence check comparing baseline against multi-row kernels (max abs diff < 1e-4).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/gemv_f16_rows.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

struct GemvFixture {
    int K;
    int N;
    int T;
    float alpha = 1.0f;

    sycl::queue q;
    sycl::half* d_w = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    GemvFixture(int k_dim, int n_dim, int t_tok)
        : K(k_dim), N(n_dim), T(t_tok),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {

        const size_t w_elems = static_cast<size_t>(N) * static_cast<size_t>(K);
        const size_t x_elems = static_cast<size_t>(T) * static_cast<size_t>(K);
        const size_t y_elems = static_cast<size_t>(T) * static_cast<size_t>(N);

        std::mt19937 rng(1337 + K + N + T);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<sycl::half> h_w(w_elems);
        for (auto& v : h_w) v = sycl::half(dist(rng) * 0.1f);

        std::vector<float> h_x(x_elems);
        for (auto& v : h_x) v = dist(rng) * 0.1f;

        d_w = sycl::malloc_device<sycl::half>(w_elems, q);
        d_x = sycl::malloc_device<float>(x_elems, q);
        d_y = sycl::malloc_device<float>(y_elems, q);

        q.memcpy(d_w, h_w.data(), w_elems * sizeof(sycl::half));
        q.memcpy(d_x, h_x.data(), x_elems * sizeof(float));
        q.fill(d_y, 0.0f, y_elems);
        q.wait_and_throw();
    }

    ~GemvFixture() {
        if (d_w) sycl::free(d_w, q);
        if (d_x) sycl::free(d_x, q);
        if (d_y) sycl::free(d_y, q);
    }
};

// -----------------------------------------------------------------------------
// Baseline GEMV Benchmark
// -----------------------------------------------------------------------------
static void BM_GEMV_Baseline(benchmark::State& state) {
    const int K = static_cast<int>(state.range(0));
    const int N = static_cast<int>(state.range(1));
    const int T = static_cast<int>(state.range(2));

    GemvFixture fix(K, N, T);

    // Steady-state GPU warmup
    for (int w = 0; w < 10; ++w) {
        strata::kernels::gemv_f16(fix.d_w, fix.d_x, N, K, T, fix.alpha, fix.d_y, fix.q);
        fix.q.wait();
    }

    // Timed microbenchmark loop
    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::gemv_f16(fix.d_w, fix.d_x, N, K, T, fix.alpha, fix.d_y, fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = static_cast<double>(N) * K * sizeof(sycl::half) +
                                static_cast<double>(T) * K * sizeof(float) +
                                static_cast<double>(T) * N * sizeof(float);
    const double total_flops = 2.0 * static_cast<double>(T) * K * N;

    state.counters["BW_GBps"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] = benchmark::Counter(total_flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// -----------------------------------------------------------------------------
// Multi-Row GEMV Benchmark (Tuned: ROWS_T = 4)
// -----------------------------------------------------------------------------
static void BM_GEMV_MultiRow(benchmark::State& state) {
    const int K = static_cast<int>(state.range(0));
    const int N = static_cast<int>(state.range(1));
    const int T = static_cast<int>(state.range(2));

    GemvFixture fix(K, N, T);

    for (int w = 0; w < 10; ++w) {
        strata::kernels::gemv_f16_rows<4>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::gemv_f16_rows<4>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = static_cast<double>(N) * K * sizeof(sycl::half) +
                                static_cast<double>(T) * K * sizeof(float) +
                                static_cast<double>(T) * N * sizeof(float);
    const double total_flops = 2.0 * static_cast<double>(T) * K * N;

    state.counters["BW_GBps"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] = benchmark::Counter(total_flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// -----------------------------------------------------------------------------
// Multi-Row GEMV Benchmark (Tile-2: ROWS_T = 2)
// -----------------------------------------------------------------------------
static void BM_GEMV_MultiRow_Tile2(benchmark::State& state) {
    const int K = static_cast<int>(state.range(0));
    const int N = static_cast<int>(state.range(1));
    const int T = static_cast<int>(state.range(2));

    GemvFixture fix(K, N, T);

    for (int w = 0; w < 10; ++w) {
        strata::kernels::gemv_f16_rows<2>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::gemv_f16_rows<2>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = static_cast<double>(N) * K * sizeof(sycl::half) +
                                static_cast<double>(T) * K * sizeof(float) +
                                static_cast<double>(T) * N * sizeof(float);
    const double total_flops = 2.0 * static_cast<double>(T) * K * N;

    state.counters["BW_GBps"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] = benchmark::Counter(total_flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

void RegisterGemvShapes(benchmark::internal::Benchmark* b) {
    const std::vector<std::pair<int, int>> shapes = {
        {2048, 2048},
        {2560, 2560},
        {2560, 6144},
        {6144, 2560}
    };
    for (const auto& s : shapes) {
        for (int T : {1, 2, 4, 8}) {
            b->Args({s.first, s.second, T});
        }
    }
}

BENCHMARK(BM_GEMV_Baseline)->Apply(RegisterGemvShapes)->UseManualTime();
BENCHMARK(BM_GEMV_MultiRow_Tile2)->Apply(RegisterGemvShapes)->UseManualTime();
BENCHMARK(BM_GEMV_MultiRow)->Apply(RegisterGemvShapes)->UseManualTime();

// -----------------------------------------------------------------------------
// Verification Suite: Checks numerical equivalence across all shapes and tokens
// -----------------------------------------------------------------------------
bool verify_numerical_equivalence() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}};
    std::printf("\n====================================================================================================\n");
    std::printf(" Verifying Device Numerical Equivalence (Baseline GEMV vs Multi-Row GEMV)\n");
    std::printf(" Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("====================================================================================================\n");

    const std::vector<std::pair<int, int>> shapes = {
        {2048, 2048},
        {2560, 2560},
        {2560, 6144},
        {6144, 2560}
    };

    bool all_passed = true;
    for (const auto& [K, N] : shapes) {
        for (int T : {1, 2, 4, 8}) {
            GemvFixture fix(K, N, T);

            const size_t y_elems = static_cast<size_t>(T) * static_cast<size_t>(N);
            float* d_y_base = sycl::malloc_device<float>(y_elems, q);
            float* d_y_row2 = sycl::malloc_device<float>(y_elems, q);
            float* d_y_row4 = sycl::malloc_device<float>(y_elems, q);

            strata::kernels::gemv_f16(fix.d_w, fix.d_x, N, K, T, fix.alpha, d_y_base, q);
            strata::kernels::gemv_f16_rows<2>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, d_y_row2, N, q);
            strata::kernels::gemv_f16_rows<4>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, d_y_row4, N, q);
            q.wait_and_throw();

            std::vector<float> h_base(y_elems);
            std::vector<float> h_row2(y_elems);
            std::vector<float> h_row4(y_elems);

            q.memcpy(h_base.data(), d_y_base, y_elems * sizeof(float));
            q.memcpy(h_row2.data(), d_y_row2, y_elems * sizeof(float));
            q.memcpy(h_row4.data(), d_y_row4, y_elems * sizeof(float));
            q.wait_and_throw();

            float max_diff2 = 0.0f;
            float max_diff4 = 0.0f;
            for (size_t i = 0; i < y_elems; ++i) {
                max_diff2 = std::max(max_diff2, std::abs(h_base[i] - h_row2[i]));
                max_diff4 = std::max(max_diff4, std::abs(h_base[i] - h_row4[i]));
            }

            const bool ok2 = (max_diff2 < 1e-4f);
            const bool ok4 = (max_diff4 < 1e-4f);
            const bool pass = ok2 && ok4;
            if (!pass) all_passed = false;

            std::printf("  Shape [%4d, %4d], T=%d: diff(Tile-2)=%.2e, diff(Tile-4)=%.2e [%s]\n",
                        K, N, T, max_diff2, max_diff4, pass ? "PASSED" : "FAILED");

            sycl::free(d_y_base, q);
            sycl::free(d_y_row2, q);
            sycl::free(d_y_row4, q);
        }
    }
    std::printf("====================================================================================================\n\n");
    return all_passed;
}

// -----------------------------------------------------------------------------
// Direct Performance Summary Table
// -----------------------------------------------------------------------------
void run_direct_summary() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}};
    std::printf("========================================================================================================================\n");
    std::printf(" Intel Xe2 Multi-Row Matrix-Vector Multiplication (gemv_f16_rows) Performance Summary\n");
    std::printf(" Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("========================================================================================================================\n");
    std::printf("%-16s | %-2s | %-13s | %-13s | %-14s | %-9s | %-12s | %-10s\n",
                "Shape [K, N]", "T", "Baseline (µs)", "Tile-2 (µs)", "Multi-Row (µs)", "Speedup", "Eff BW(GB/s)", "GFLOPS");
    std::printf("-----------------+----+---------------+---------------+----------------+-----------+--------------+-----------\n");

    const std::vector<std::pair<int, int>> shapes = {
        {2048, 2048},
        {2560, 2560},
        {2560, 6144},
        {6144, 2560}
    };

    constexpr int iters = 100;
    for (const auto& [K, N] : shapes) {
        for (int T : {1, 2, 4, 8}) {
            GemvFixture fix(K, N, T);

            // Warmup
            for (int i = 0; i < 20; ++i) {
                strata::kernels::gemv_f16(fix.d_w, fix.d_x, N, K, T, fix.alpha, fix.d_y, q);
                strata::kernels::gemv_f16_rows<2>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, q);
                strata::kernels::gemv_f16_rows<4>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, q);
            }
            q.wait();

            // Baseline timing
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                strata::kernels::gemv_f16(fix.d_w, fix.d_x, N, K, T, fix.alpha, fix.d_y, q);
            }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            const double lat_base_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            // Tile-2 timing
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                strata::kernels::gemv_f16_rows<2>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, q);
            }
            q.wait();
            t1 = std::chrono::high_resolution_clock::now();
            const double lat_tile2_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            // Tuned Tile-4 timing
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                strata::kernels::gemv_f16_rows<4>(fix.d_w, fix.d_x, K, N, K, T, fix.alpha, fix.d_y, N, q);
            }
            q.wait();
            t1 = std::chrono::high_resolution_clock::now();
            const double lat_row4_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            const double speedup = lat_base_us / lat_row4_us;
            const double useful_bytes = static_cast<double>(N) * K * sizeof(sycl::half) +
                                        static_cast<double>(T) * K * sizeof(float) +
                                        static_cast<double>(T) * N * sizeof(float);
            const double total_flops = 2.0 * static_cast<double>(T) * K * N;
            const double eff_bw_gbs = (useful_bytes / (lat_row4_us * 1e-6)) * 1e-9;
            const double gflops = (total_flops / (lat_row4_us * 1e-6)) * 1e-9;

            char shape_buf[32];
            std::snprintf(shape_buf, sizeof(shape_buf), "%d x %d", K, N);
            std::printf("%-16s | %-2d | %10.2f µs | %10.2f µs | %11.2f µs | %8.2fx | %10.2f | %9.1f\n",
                        shape_buf, T, lat_base_us, lat_tile2_us, lat_row4_us, speedup, eff_bw_gbs, gflops);
        }
    }
    std::printf("========================================================================================================================\n\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (!verify_numerical_equivalence()) {
        std::fprintf(stderr, "[FATAL] Device verification failed!\n");
        return 1;
    }

    bool run_summary = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]).rfind("--benchmark_filter", 0) == 0 ||
            std::string(argv[i]) == "--benchmark_list_tests") {
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
