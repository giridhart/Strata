// perf/sycl/perf_trellis.cpp - Intel Xe2 Gyro Trellis Quantization (TQ) GEMV Microbenchmark.
#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/tq_gemv.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct TqGemvFixture {
    int type;
    int64_t n_out;
    int64_t n_in;
    sycl::queue q;

    uint8_t* d_w;
    float* d_x;
    float* d_y;

    size_t weight_bytes;

    TqGemvFixture(int ty, int64_t out_dim, int64_t in_dim)
        : type(ty), n_out(out_dim), n_in(in_dim),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}) {
        const int64_t n_blocks = n_in / 128;
        const size_t bsize = strata::kernels::tq_block_size(type);
        const size_t row_bytes = static_cast<size_t>(n_blocks * bsize);
        weight_bytes = static_cast<size_t>(n_out * row_bytes);

        std::mt19937 rng(42 + type * 101);
        std::vector<uint8_t> h_w(weight_bytes);
        for (size_t r = 0; r < static_cast<size_t>(n_out); ++r) {
            uint8_t* row = h_w.data() + r * row_bytes;
            for (size_t b = 0; b < static_cast<size_t>(n_blocks); ++b) {
                uint8_t* blk = row + b * bsize;
                float scale = 0.05f;
                sycl::half sh(scale);
                *reinterpret_cast<uint16_t*>(blk) = *reinterpret_cast<uint16_t*>(&sh);
                for (size_t k = 2; k < bsize; ++k) {
                    blk[k] = static_cast<uint8_t>(rng() & 0xFF);
                }
            }
        }

        std::vector<float> h_x(n_in, 0.5f);

        d_w = sycl::malloc_device<uint8_t>(weight_bytes, q);
        d_x = sycl::malloc_device<float>(n_in, q);
        d_y = sycl::malloc_device<float>(n_out, q);

        q.memcpy(d_w, h_w.data(), weight_bytes);
        q.memcpy(d_x, h_x.data(), n_in * sizeof(float));
        q.wait();
    }

    ~TqGemvFixture() {
        sycl::free(d_w, q);
        sycl::free(d_x, q);
        sycl::free(d_y, q);
    }
};

static void BM_Xe2_TQ_GEMV_SLM(benchmark::State& state) {
    const int type = static_cast<int>(state.range(0));
    const int64_t n_out = state.range(1);
    const int64_t n_in = state.range(2);

    TqGemvFixture fix(type, n_out, n_in);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::tq_gemv_ext(type, fix.d_w, fix.d_x, fix.d_y, n_out, n_in, /*stage_slm=*/true, &fix.q);
    }
    fix.q.wait();

    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();
        strata::kernels::tq_gemv_ext(type, fix.d_w, fix.d_x, fix.d_y, n_out, n_in, /*stage_slm=*/true, &fix.q);
        fix.q.wait();
        const auto t1 = std::chrono::steady_clock::now();

        const double dt = std::chrono::duration<double>(t1 - t0).count();
        state.SetIterationTime(dt);
    }

    const double bytes_transferred = static_cast<double>(fix.weight_bytes + n_in * sizeof(float) + n_out * sizeof(float));
    const double flops = 2.0 * static_cast<double>(n_out * n_in);

    state.counters["Effective_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"]         = benchmark::Counter(flops * 1e-9, benchmark::Counter::kAvgIterationsRate);
    state.counters["N_out"]          = benchmark::Counter(n_out);
    state.counters["N_in"]           = benchmark::Counter(n_in);
    state.counters["Weight_KB"]      = benchmark::Counter(fix.weight_bytes / 1024.0);
}

static void BM_Xe2_TQ_GEMV_Global(benchmark::State& state) {
    const int type = static_cast<int>(state.range(0));
    const int64_t n_out = state.range(1);
    const int64_t n_in = state.range(2);

    TqGemvFixture fix(type, n_out, n_in);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::tq_gemv_ext(type, fix.d_w, fix.d_x, fix.d_y, n_out, n_in, /*stage_slm=*/false, &fix.q);
    }
    fix.q.wait();

    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();
        strata::kernels::tq_gemv_ext(type, fix.d_w, fix.d_x, fix.d_y, n_out, n_in, /*stage_slm=*/false, &fix.q);
        fix.q.wait();
        const auto t1 = std::chrono::steady_clock::now();

        const double dt = std::chrono::duration<double>(t1 - t0).count();
        state.SetIterationTime(dt);
    }

    const double bytes_transferred = static_cast<double>(fix.weight_bytes + n_in * sizeof(float) + n_out * sizeof(float));
    const double flops = 2.0 * static_cast<double>(n_out * n_in);

    state.counters["Effective_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"]         = benchmark::Counter(flops * 1e-9, benchmark::Counter::kAvgIterationsRate);
    state.counters["N_out"]          = benchmark::Counter(n_out);
    state.counters["N_in"]           = benchmark::Counter(n_in);
    state.counters["Weight_KB"]      = benchmark::Counter(fix.weight_bytes / 1024.0);
}

// Register across shapes K=2560, N=640 (down) and N=10240 (gate/up) for types 144, 145, 146
// Args: {type, n_out, n_in}
BENCHMARK(BM_Xe2_TQ_GEMV_SLM)->Args({144, 640, 2560})->Args({144, 10240, 2560})
                             ->Args({145, 640, 2560})->Args({145, 10240, 2560})
                             ->Args({146, 640, 2560})->Args({146, 10240, 2560})
                             ->UseManualTime();

BENCHMARK(BM_Xe2_TQ_GEMV_Global)->Args({144, 640, 2560})->Args({144, 10240, 2560})
                                ->Args({145, 640, 2560})->Args({145, 10240, 2560})
                                ->Args({146, 640, 2560})->Args({146, 10240, 2560})
                                ->UseManualTime();

} // namespace

BENCHMARK_MAIN();
