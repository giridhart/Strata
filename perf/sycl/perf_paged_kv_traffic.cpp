// perf/sycl/perf_paged_kv_traffic.cpp - Intel Xe2 Paged KV Cache Memory Traffic & Indirection Penalty Benchmark
//
// Benchmarks memory subsystem performance for non-contiguous paged KV cache lookups vs contiguous memory:
// In QSA attention layers:
//   physical_page = page_table[logical_page]
//   offset = (physical_page * PAGE_SIZE + token_in_page) * HEAD_BYTES
//
// Evaluates:
// 1. BM_Xe2_KV_Contiguous: Linear sequential memory access across context lengths
// 2. BM_Xe2_KV_Paged: Non-contiguous paged gather with physical page table indirection
// 3. BM_Xe2_KV_Page_Size_Sweep: Page granularity sweep (16, 32, 64, 128 tokens) for Xe2 cache lines
// 4. BM_Xe2_KV_Quant_Paged: INT8 (1024 B/token) vs FP16 (2048 B/token) paged memory bandwidth

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int N_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int BYTES_PER_CELL_FP16 = N_KV_HEADS * HEAD_DIM * sizeof(uint16_t); // 2048 bytes
constexpr int BYTES_PER_CELL_INT8 = N_KV_HEADS * HEAD_DIM * sizeof(int8_t);   // 1024 bytes

// Kernel 1: Contiguous Linear KV Gather
void k_kv_gather_contiguous(sycl::nd_item<1> item, const uint16_t* __restrict__ kv_pool,
                            const float* __restrict__ query, float* __restrict__ scores,
                            int total_tokens) {
    const int t = item.get_group(0);
    if (t >= total_tokens) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* k_tok = kv_pool + (size_t)t * (N_KV_HEADS * HEAD_DIM);

    float dot = 0.0f;
    for (int i = tid; i < N_KV_HEADS * HEAD_DIM; i += bsize) {
        dot += (float)k_tok[i] * query[i];
    }
    auto sg = item.get_sub_group();
    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
    if (tid == 0) scores[t] = dot;
}

// Kernel 2: Paged KV Gather via Page Table Indirection
void k_kv_gather_paged(sycl::nd_item<1> item, const uint16_t* __restrict__ kv_pool,
                       const int32_t* __restrict__ page_table, const float* __restrict__ query,
                       float* __restrict__ scores, int total_tokens, int page_size) {
    const int t = item.get_group(0);
    if (t >= total_tokens) return;

    const int logical_page = t / page_size;
    const int token_in_page = t % page_size;
    const int physical_page = page_table[logical_page];

    const size_t offset = ((size_t)physical_page * page_size + token_in_page) * (N_KV_HEADS * HEAD_DIM);
    const auto* k_tok = kv_pool + offset;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);

    float dot = 0.0f;
    for (int i = tid; i < N_KV_HEADS * HEAD_DIM; i += bsize) {
        dot += (float)k_tok[i] * query[i];
    }
    auto sg = item.get_sub_group();
    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
    if (tid == 0) scores[t] = dot;
}

// Kernel 3: INT8 Paged KV Gather
void k_kv_gather_int8_paged(sycl::nd_item<1> item, const int8_t* __restrict__ kv_pool,
                            const int32_t* __restrict__ page_table, const float* __restrict__ query,
                            float* __restrict__ scores, int total_tokens, int page_size) {
    const int t = item.get_group(0);
    if (t >= total_tokens) return;

    const int logical_page = t / page_size;
    const int token_in_page = t % page_size;
    const int physical_page = page_table[logical_page];

    const size_t offset = ((size_t)physical_page * page_size + token_in_page) * (N_KV_HEADS * HEAD_DIM);
    const auto* k_tok = kv_pool + offset;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);

    float dot = 0.0f;
    for (int i = tid; i < N_KV_HEADS * HEAD_DIM; i += bsize) {
        dot += (float)k_tok[i] * query[i];
    }
    auto sg = item.get_sub_group();
    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
    if (tid == 0) scores[t] = dot;
}

} // namespace

// ============================================================================
// Benchmark 1: Contiguous Linear Memory Gather
// ============================================================================

static void BM_Xe2_KV_Contiguous(benchmark::State& state) {
    const int total_tokens = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t pool_elements = (size_t)total_tokens * N_KV_HEADS * HEAD_DIM;
    const size_t query_elements = N_KV_HEADS * HEAD_DIM;

    auto* d_pool   = sycl::malloc_device<uint16_t>(pool_elements, q);
    auto* d_query  = sycl::malloc_device<float>(query_elements, q);
    auto* d_scores = sycl::malloc_device<float>(total_tokens, q);

    q.fill(d_pool, (uint16_t)1, pool_elements);
    q.fill(d_query, 0.05f, query_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_kv_gather_contiguous(item, d_pool, d_query, d_scores, total_tokens);
                             });
        }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_kv_gather_contiguous(item, d_pool, d_query, d_scores, total_tokens);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)total_tokens * BYTES_PER_CELL_FP16;
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter(total_tokens);
    state.counters["VRAM_MB"]        = benchmark::Counter(bytes_transferred / 1e6);

    sycl::free(d_pool, q);
    sycl::free(d_query, q);
    sycl::free(d_scores, q);
}

// ============================================================================
// Benchmark 2: Non-Contiguous Paged Memory Gather
// ============================================================================

static void BM_Xe2_KV_Paged(benchmark::State& state) {
    const int total_tokens = state.range(0);
    const int page_size = 16;
    const int num_pages = (total_tokens + page_size - 1) / page_size;

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    // Allocate 1.5x physical pool to simulate scattered physical pages
    const int num_phys_pages = num_pages * 3 / 2;
    const size_t pool_elements = (size_t)num_phys_pages * page_size * N_KV_HEADS * HEAD_DIM;
    const size_t query_elements = N_KV_HEADS * HEAD_DIM;

    // Generate pseudo-random scattered physical page mapping
    std::vector<int32_t> h_table(num_pages);
    std::vector<int32_t> phys_order(num_phys_pages);
    for (int i = 0; i < num_phys_pages; ++i) phys_order[i] = i;
    std::mt19937 g(42);
    std::shuffle(phys_order.begin(), phys_order.end(), g);
    for (int i = 0; i < num_pages; ++i) h_table[i] = phys_order[i];

    auto* d_pool   = sycl::malloc_device<uint16_t>(pool_elements, q);
    auto* d_table  = sycl::malloc_device<int32_t>(num_pages, q);
    auto* d_query  = sycl::malloc_device<float>(query_elements, q);
    auto* d_scores = sycl::malloc_device<float>(total_tokens, q);

    q.fill(d_pool, (uint16_t)1, pool_elements);
    q.memcpy(d_table, h_table.data(), num_pages * sizeof(int32_t));
    q.fill(d_query, 0.05f, query_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_kv_gather_paged(item, d_pool, d_table, d_query, d_scores, total_tokens, page_size);
                             });
        }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_kv_gather_paged(item, d_pool, d_table, d_query, d_scores, total_tokens, page_size);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)total_tokens * BYTES_PER_CELL_FP16;
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter(total_tokens);
    state.counters["Page_Size"]      = benchmark::Counter(page_size);

    sycl::free(d_pool, q);
    sycl::free(d_table, q);
    sycl::free(d_query, q);
    sycl::free(d_scores, q);
}

// ============================================================================
// Benchmark 3: Page Size Granularity Sweep (16, 32, 64, 128)
// ============================================================================

static void BM_Xe2_KV_Page_Size_Sweep(benchmark::State& state) {
    const int page_size = state.range(0);
    const int total_tokens = 4096;
    const int num_pages = (total_tokens + page_size - 1) / page_size;

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const int num_phys_pages = num_pages * 3 / 2;
    const size_t pool_elements = (size_t)num_phys_pages * page_size * N_KV_HEADS * HEAD_DIM;
    const size_t query_elements = N_KV_HEADS * HEAD_DIM;

    std::vector<int32_t> h_table(num_pages);
    std::vector<int32_t> phys_order(num_phys_pages);
    for (int i = 0; i < num_phys_pages; ++i) phys_order[i] = i;
    std::mt19937 g(42);
    std::shuffle(phys_order.begin(), phys_order.end(), g);
    for (int i = 0; i < num_pages; ++i) h_table[i] = phys_order[i];

    auto* d_pool   = sycl::malloc_device<uint16_t>(pool_elements, q);
    auto* d_table  = sycl::malloc_device<int32_t>(num_pages, q);
    auto* d_query  = sycl::malloc_device<float>(query_elements, q);
    auto* d_scores = sycl::malloc_device<float>(total_tokens, q);

    q.fill(d_pool, (uint16_t)1, pool_elements);
    q.memcpy(d_table, h_table.data(), num_pages * sizeof(int32_t));
    q.fill(d_query, 0.05f, query_elements);
    q.wait();

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_kv_gather_paged(item, d_pool, d_table, d_query, d_scores, total_tokens, page_size);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)total_tokens * BYTES_PER_CELL_FP16;
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Page_Size"]      = benchmark::Counter(page_size);

    sycl::free(d_pool, q);
    sycl::free(d_table, q);
    sycl::free(d_query, q);
    sycl::free(d_scores, q);
}

// ============================================================================
// Benchmark 4: INT8 Paged KV Gather
// ============================================================================

static void BM_Xe2_KV_Quant_Paged(benchmark::State& state) {
    const int total_tokens = state.range(0);
    const int page_size = 16;
    const int num_pages = (total_tokens + page_size - 1) / page_size;

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const int num_phys_pages = num_pages * 3 / 2;
    const size_t pool_elements = (size_t)num_phys_pages * page_size * N_KV_HEADS * HEAD_DIM;
    const size_t query_elements = N_KV_HEADS * HEAD_DIM;

    std::vector<int32_t> h_table(num_pages);
    std::vector<int32_t> phys_order(num_phys_pages);
    for (int i = 0; i < num_phys_pages; ++i) phys_order[i] = i;
    std::mt19937 g(42);
    std::shuffle(phys_order.begin(), phys_order.end(), g);
    for (int i = 0; i < num_pages; ++i) h_table[i] = phys_order[i];

    auto* d_pool   = sycl::malloc_device<int8_t>(pool_elements, q);
    auto* d_table  = sycl::malloc_device<int32_t>(num_pages, q);
    auto* d_query  = sycl::malloc_device<float>(query_elements, q);
    auto* d_scores = sycl::malloc_device<float>(total_tokens, q);

    q.fill(d_pool, (int8_t)1, pool_elements);
    q.memcpy(d_table, h_table.data(), num_pages * sizeof(int32_t));
    q.fill(d_query, 0.05f, query_elements);
    q.wait();

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_kv_gather_int8_paged(item, d_pool, d_table, d_query, d_scores, total_tokens, page_size);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)total_tokens * BYTES_PER_CELL_INT8;
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter(total_tokens);

    sycl::free(d_pool, q);
    sycl::free(d_table, q);
    sycl::free(d_query, q);
    sycl::free(d_scores, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_KV_Contiguous)->Arg(1024)->Arg(4096)->Arg(16384)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Paged)->Arg(1024)->Arg(4096)->Arg(16384)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Page_Size_Sweep)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Quant_Paged)->Arg(1024)->Arg(4096)->Arg(16384)->UseManualTime();

BENCHMARK_MAIN();
