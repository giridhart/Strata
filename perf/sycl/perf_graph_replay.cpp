// perf/sycl/perf_graph_replay.cpp - Intel Xe2 SYCL Command Graph Replay Benchmark
//
// Benchmarks host CPU dispatch latency across a multi-layer transformer decode pipeline:
// Compares direct individual queue submissions against strata::core::CapturedGraph replay
// (using sycl::ext::oneapi::experimental::command_graph).
//
// Evaluates:
// 1. BM_Xe2_Direct_Submission: Submitting N kernel nodes sequentially via q.submit()
// 2. BM_Xe2_Graph_Replay: Replaying pre-recorded CapturedGraph with 1 queue execution call
//
// Parameters:
// - Layers: 12, 24, 48 layers (Qwen3.8-Flash-Next has 48 layers, ~6 kernels/layer = 288 nodes)

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/graph.hpp"
#include "strata/core/sycl_arch.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr size_t N_ELEMENTS = 2560;

}  // namespace

// ============================================================================
// Benchmark: BM_Xe2_Direct_Submission (Host Driver Queue Loop)
// ============================================================================

static void BM_Xe2_Direct_Submission(benchmark::State& state) {
    const int num_layers = state.range(0);
    const int nodes_per_layer = 6;
    const int total_nodes = num_layers * nodes_per_layer;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto* d_data = sycl::malloc_device<float>(N_ELEMENTS, q);
    q.fill(d_data, 1.0f, N_ELEMENTS).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 0.99f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] += 0.01f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 1.01f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] -= 0.005f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 0.98f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] += 0.02f; });
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int l = 0; l < num_layers; ++l) {
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 0.99f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] += 0.01f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 1.01f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] -= 0.005f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 0.98f; });
            q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] += 0.02f; });
        }
        q.wait();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());
    }

    state.counters["Total_Nodes"]   = benchmark::Counter(total_nodes);
    state.counters["Layers"]        = benchmark::Counter(num_layers);
    state.counters["us_per_node"]   = benchmark::Counter(total_nodes, benchmark::Counter::kAvgIterationsRate);

    sycl::free(d_data, q);
}

// ============================================================================
// Benchmark: BM_Xe2_Graph_Replay (CapturedGraph Replay Loop)
// ============================================================================

static void BM_Xe2_Graph_Replay(benchmark::State& state) {
    const int num_layers = state.range(0);
    const int nodes_per_layer = 6;
    const int total_nodes = num_layers * nodes_per_layer;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto* d_data = sycl::malloc_device<float>(N_ELEMENTS, q);
    q.fill(d_data, 1.0f, N_ELEMENTS).wait();

    strata::core::CapturedGraph g;
    std::string err;

    if (!g.begin(&q, err)) {
        state.SkipWithError(("CapturedGraph::begin failed: " + err).c_str());
        sycl::free(d_data, q);
        return;
    }

    for (int l = 0; l < num_layers; ++l) {
        q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 0.99f; });
        q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] += 0.01f; });
        q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 1.01f; });
        q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] -= 0.005f; });
        q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] *= 0.98f; });
        q.parallel_for(sycl::range<1>(N_ELEMENTS), [=](sycl::id<1> idx) { d_data[idx] += 0.02f; });
    }

    if (!g.end(&q, err)) {
        state.SkipWithError(("CapturedGraph::end failed: " + err).c_str());
        sycl::free(d_data, q);
        return;
    }

    // Warmup
    for (int w = 0; w < 3; ++w) {
        g.launch(&q, err);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (!g.launch(&q, err)) {
            state.SkipWithError(("CapturedGraph::launch failed: " + err).c_str());
            break;
        }
        q.wait();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());
    }

    state.counters["Total_Nodes"]   = benchmark::Counter(total_nodes);
    state.counters["Layers"]        = benchmark::Counter(num_layers);
    state.counters["Graph_Nodes"]   = benchmark::Counter(g.nodes());

    sycl::free(d_data, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_Direct_Submission)->Arg(12)->Arg(24)->Arg(48)->UseManualTime();
BENCHMARK(BM_Xe2_Graph_Replay)->Arg(12)->Arg(24)->Arg(48)->UseManualTime();

BENCHMARK_MAIN();
