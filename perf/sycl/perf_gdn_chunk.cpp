// perf/sycl/perf_gdn_chunk.cpp - Intel Xe2 Chunked GDN vs Sequential Recurrence Benchmark
//
// Evaluates FlashQLA chunked GDN prompt recurrence (STRATA_GDN_CHUNKED=1)
// against sequential token-by-token recurrence across sequence lengths:
// T in [128, 512, 1024, 2048, 4096, 8192] on Intel Arc B580 (Xe2).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gdn_chunk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int S = 128;
constexpr int HV = 48;
constexpr int C = 10240;
constexpr int ROW = HV * S;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct BenchmarkContext {
    int64_t T;
    sycl::queue q;
    float* d_h = nullptr;
    float* d_gate = nullptr;
    float* d_beta = nullptr;
    float* d_z = nullptr;
    float* d_gamma = nullptr;
    float* d_state = nullptr;
    float* d_y = nullptr;
    uint16_t* d_y16 = nullptr;

    BenchmarkContext(int64_t tokens)
        : T(tokens), q(sycl::default_selector_v, {sycl::property::queue::in_order()}) {
        d_h = sycl::malloc_device<float>((size_t)T * C, q);
        d_gate = sycl::malloc_device<float>((size_t)T * HV, q);
        d_beta = sycl::malloc_device<float>((size_t)T * HV, q);
        d_z = sycl::malloc_device<float>((size_t)T * ROW, q);
        d_gamma = sycl::malloc_device<float>((size_t)S, q);
        d_state = sycl::malloc_device<float>((size_t)S * ROW, q);
        d_y = sycl::malloc_device<float>((size_t)T * ROW, q);
        d_y16 = sycl::malloc_device<uint16_t>((size_t)T * ROW, q);

        q.memset(d_h, 0x3f, (size_t)T * C * sizeof(float));
        q.memset(d_gate, 0xbf, (size_t)T * HV * sizeof(float));  // negative
        q.memset(d_beta, 0x3e, (size_t)T * HV * sizeof(float));  // ~0.18
        q.memset(d_z, 0x3d, (size_t)T * ROW * sizeof(float));
        q.memset(d_gamma, 0x3f, (size_t)S * sizeof(float));      // ~1.0
        q.memset(d_state, 0, (size_t)S * ROW * sizeof(float));
        q.wait();
    }

    ~BenchmarkContext() {
        if (d_h) sycl::free(d_h, q);
        if (d_gate) sycl::free(d_gate, q);
        if (d_beta) sycl::free(d_beta, q);
        if (d_z) sycl::free(d_z, q);
        if (d_gamma) sycl::free(d_gamma, q);
        if (d_state) sycl::free(d_state, q);
        if (d_y) sycl::free(d_y, q);
        if (d_y16) sycl::free(d_y16, q);
    }
};

void run_benchmark(benchmark::State& state, bool chunked) {
    const int64_t T = state.range(0);
    BenchmarkContext ctx(T);
    set_env("STRATA_GDN_CHUNKED", chunked ? "1" : "0");

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::prefill::gdn_recurrence(ctx.d_state, ctx.d_h, ctx.d_gate, ctx.d_beta, ctx.d_z,
                                        ctx.d_gamma, 1e-6f, ctx.d_y, ctx.d_y16, ctx.T, &ctx.q, ROW);
        ctx.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::prefill::gdn_recurrence(ctx.d_state, ctx.d_h, ctx.d_gate, ctx.d_beta, ctx.d_z,
                                        ctx.d_gamma, 1e-6f, ctx.d_y, ctx.d_y16, ctx.T, &ctx.q, ROW);
        ctx.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        state.SetIterationTime(elapsed_sec);
    }

    state.counters["tok_per_sec"] = benchmark::Counter((double)T, benchmark::Counter::kIsIterationInvariantRate,
                                                       benchmark::Counter::kIs1000);
}

}  // namespace

static void BM_GDN_Sequential(benchmark::State& state) {
    run_benchmark(state, false);
}

static void BM_GDN_Chunked(benchmark::State& state) {
    run_benchmark(state, true);
}

BENCHMARK(BM_GDN_Sequential)->Arg(128)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->UseManualTime();
BENCHMARK(BM_GDN_Chunked)->Arg(128)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->UseManualTime();

BENCHMARK_MAIN();
