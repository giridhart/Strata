// perf/sycl/perf_mmq_direct.cpp - Intel Xe2 MMQ Direct Pointer-Table vs Gathered Benchmark
//
// Evaluates direct pointer-table expert MMQ streaming optimization (STRATA_MMQ_DIRECT=1)
// against standard contiguous-gathered MMQ on Intel Arc B580 (Battlemage Xe2).
//
// Measures latency (us), throughput (tokens/sec), and DRAM traffic savings by eliminating
// redundant weight gather copies for resident experts.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

constexpr int G = 8;  // 8 experts per group / token
constexpr size_t TAIL = 4096;

struct DevBuf {
    sycl::queue* q = nullptr;
    void* p = nullptr;
    size_t bytes = 0;

    DevBuf(size_t n, sycl::queue& queue) : q(&queue), bytes(n) {
        p = sycl::malloc_device(bytes > 0 ? bytes : 1, *q);
    }
    ~DevBuf() {
        if (p && q) sycl::free(p, *q);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    uint8_t* b() const { return (uint8_t*)p; }
};

// Benchmark helper for Gate/Up and Down products
void benchmark_mmq(benchmark::State& state, int type, int64_t rows, int64_t cols, bool is_direct) {
    const int64_t tokens = state.range(0);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t eb = mmq::matrix_bytes(type, rows, cols);
    const size_t over = mmq::overread_bytes(type, cols);
    const int64_t total_rows = tokens;
    const int64_t max_rows = tokens / G;

    // Activations
    DevBuf x(total_rows * cols * sizeof(float), q);
    DevBuf xq(mmq::q8_bytes(total_rows, cols), q);
    DevBuf ids((total_rows + 2048) * sizeof(int32_t), q);
    DevBuf bounds((G + 1) * sizeof(int32_t), q);
    DevBuf dst(total_rows * rows * sizeof(float), q);

    mmq::iota((int32_t*)ids.p, total_rows + 2048, &q);
    std::vector<int32_t> h_bounds(G + 1);
    for (int i = 0; i <= G; ++i) h_bounds[i] = (int32_t)(i * (tokens / G));
    q.memcpy(bounds.p, h_bounds.data(), (G + 1) * sizeof(int32_t)).wait();

    // Weight allocations
    const size_t stride = (eb + mmq::kSlotGuard + 255) / 256 * 256;
    DevBuf arena(G * stride + 4096, q);
    DevBuf grp(G * eb + TAIL, q);
    q.memset(arena.p, 0x55, arena.bytes);
    q.memset(grp.p, 0x55, grp.bytes).wait();

    mmq::WeightTab tab;
    mmq::GuardTab guards;
    for (int i = 0; i < G; ++i) {
        uint8_t* slot = arena.b() + i * stride;
        tab.p[i] = slot;
        guards.p[guards.n++] = slot + eb;
    }

    mmq::Context ctx;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        if (is_direct) {
            mmq::prep_group(guards, over, nullptr, nullptr, 0, &q);
            mmq::Product p;
            p.type = type; p.w_rows = rows; p.w_cols = cols;
            p.n = G; p.xq = xq.p; p.bounds = (const int32_t*)bounds.p;
            p.ids = (const int32_t*)ids.p; p.total_rows = total_rows;
            p.max_rows = max_rows; p.dst = (float*)dst.p; p.ld_dst = rows;
            p.tab = &tab;
            ctx.run(p, &q);
        } else {
            for (int i = 0; i < G; ++i) {
                q.memcpy(grp.b() + i * eb, arena.b() + i * stride, eb);
            }
            q.memset(grp.b() + G * eb, 0, TAIL);
            mmq::Product p;
            p.w = grp.p; p.expert_bytes = eb;
            p.type = type; p.w_rows = rows; p.w_cols = cols;
            p.n = G; p.xq = xq.p; p.bounds = (const int32_t*)bounds.p;
            p.ids = (const int32_t*)ids.p; p.total_rows = total_rows;
            p.max_rows = max_rows; p.dst = (float*)dst.p; p.ld_dst = rows;
            ctx.run(p, &q);
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (is_direct) {
            mmq::prep_group(guards, over, nullptr, nullptr, 0, &q);
            mmq::Product p;
            p.type = type; p.w_rows = rows; p.w_cols = cols;
            p.n = G; p.xq = xq.p; p.bounds = (const int32_t*)bounds.p;
            p.ids = (const int32_t*)ids.p; p.total_rows = total_rows;
            p.max_rows = max_rows; p.dst = (float*)dst.p; p.ld_dst = rows;
            p.tab = &tab;
            ctx.run(p, &q);
        } else {
            for (int i = 0; i < G; ++i) {
                q.memcpy(grp.b() + i * eb, arena.b() + i * stride, eb);
            }
            q.memset(grp.b() + G * eb, 0, TAIL);
            mmq::Product p;
            p.w = grp.p; p.expert_bytes = eb;
            p.type = type; p.w_rows = rows; p.w_cols = cols;
            p.n = G; p.xq = xq.p; p.bounds = (const int32_t*)bounds.p;
            p.ids = (const int32_t*)ids.p; p.total_rows = total_rows;
            p.max_rows = max_rows; p.dst = (float*)dst.p; p.ld_dst = rows;
            ctx.run(p, &q);
        }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        state.SetIterationTime(secs);
    }

    const double ops = 2.0 * (double)tokens * (double)rows * (double)cols;
    state.counters["TFLOPS"] = benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["tok_per_sec"] = benchmark::Counter((double)tokens, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["gather_bytes_saved"] = benchmark::Counter(is_direct ? (double)(G * eb) : 0.0);
}

}  // namespace

// Benchmarks across chunk sizes 512, 1024, 2048, 4096
static void BM_MMQ_GateUp_IQ3S_Gathered(benchmark::State& state) {
    benchmark_mmq(state, GGML_TYPE_IQ3_S, 1280, 2560, false);
}
static void BM_MMQ_GateUp_IQ3S_Direct(benchmark::State& state) {
    benchmark_mmq(state, GGML_TYPE_IQ3_S, 1280, 2560, true);
}

static void BM_MMQ_Down_IQ4NL_Gathered(benchmark::State& state) {
    benchmark_mmq(state, GGML_TYPE_IQ4_NL, 2560, 640, false);
}
static void BM_MMQ_Down_IQ4NL_Direct(benchmark::State& state) {
    benchmark_mmq(state, GGML_TYPE_IQ4_NL, 2560, 640, true);
}

static void BM_MMQ_Down_Q8_0_Gathered(benchmark::State& state) {
    benchmark_mmq(state, GGML_TYPE_Q8_0, 2560, 640, false);
}
static void BM_MMQ_Down_Q8_0_Direct(benchmark::State& state) {
    benchmark_mmq(state, GGML_TYPE_Q8_0, 2560, 640, true);
}

BENCHMARK(BM_MMQ_GateUp_IQ3S_Gathered)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();
BENCHMARK(BM_MMQ_GateUp_IQ3S_Direct)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();

BENCHMARK(BM_MMQ_Down_IQ4NL_Gathered)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();
BENCHMARK(BM_MMQ_Down_IQ4NL_Direct)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();

BENCHMARK(BM_MMQ_Down_Q8_0_Gathered)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();
BENCHMARK(BM_MMQ_Down_Q8_0_Direct)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();

BENCHMARK_MAIN();
