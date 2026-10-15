// perf/sycl/perf_qsa_attention.cpp - Intel Xe2 QSA FlashAttention & Sparse Top-K Gather Benchmark
//
// Benchmarks:
// 1. QSA FlashAttention Prompt Processing:
//    - Multi-head attention across context lengths (T = 64, 256, 1024, 4096).
//    - H = 24 query heads, H_kv = 2 heads (GQA factor 12), head_dim = 256.
//    - Online softmax and V accumulation using Xe2 SIMD16 subgroup execution.
// 2. Top-K Sparse Block Gather Bandwidth:
//    - Active block gather (K = 256 blocks, block_size = 4 cells) from sparse contexts (up to 32K cells).
//    - Measures memory subsystem efficiency and L2 cache resilience under non-contiguous block gather.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/qsa.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// ============================================================================
// Attention Kernels
// ============================================================================

/// QSA FlashAttention Subgroup Kernel for Xe2
/// Each work-group computes one query head h in [0, 24).
void qsa_flash_attn_kernel(sycl::nd_item<1> item, const float* __restrict__ q, const sycl::half* __restrict__ k,
                           const sycl::half* __restrict__ v, float* __restrict__ out, int width, float scale,
                           sycl::local_accessor<float, 1> s_dots) {
    const int h = item.get_group(0);        // 0 .. 23
    const int kv_head = h / 12;             // GQA: 2 KV heads for 24 query heads
    const int lane = item.get_local_id(0);  // 0 .. 255
    auto sg = item.get_sub_group();

    const float* q_h = q + h * 256;
    const float q_val = (lane < 256) ? (q_h[lane] * scale) : 0.0f;

    // Online softmax state
    float max_s = -1e30f;
    float sum_exp = 0.0f;
    float acc_v = 0.0f;

    for (int cell = 0; cell < width; ++cell) {
        const sycl::half* k_cell = k + (static_cast<size_t>(cell) * 2 + kv_head) * 256;
        const sycl::half* v_cell = v + (static_cast<size_t>(cell) * 2 + kv_head) * 256;

        float k_val = (lane < 256) ? static_cast<float>(k_cell[lane]) : 0.0f;
        float prod = q_val * k_val;

        // Reduce dot product within subgroup
        float dot_sg = sycl::reduce_over_group(sg, prod, sycl::plus<float>());
        // In local memory across subgroups (16 subgroups of 16 threads = 256 threads)
        s_dots[sg.get_group_linear_id()] = dot_sg;
        item.barrier(sycl::access::fence_space::local_space);

        float dot = 0.0f;
        for (int i = 0; i < 16; ++i) {
            dot += s_dots[i];
        }

        // Online softmax update
        float old_max = max_s;
        if (dot > max_s) max_s = dot;
        float exp_diff = sycl::native::exp(old_max - max_s);
        float exp_score = sycl::native::exp(dot - max_s);

        sum_exp = sum_exp * exp_diff + exp_score;
        acc_v = acc_v * exp_diff;

        float v_val = (lane < 256) ? static_cast<float>(v_cell[lane]) : 0.0f;
        acc_v += exp_score * v_val;
    }

    if (lane < 256) {
        out[h * 256 + lane] = acc_v / (sum_exp + 1e-6f);
    }
}

/// Sparse Top-K Block Gather Kernel
/// Gathers active KV blocks based on block index array.
void qsa_topk_gather_kernel(sycl::nd_item<1> item, const sycl::half* __restrict__ kv_pool,
                            const int32_t* __restrict__ active_bids, sycl::half* __restrict__ gathered_kv,
                            int num_active_blocks, int block_size, int head_dim, int n_kv_heads) {
    const int idx = item.get_global_id(0);
    const int total_elements = num_active_blocks * block_size * n_kv_heads * head_dim;
    if (idx >= total_elements) return;

    const int elements_per_block = block_size * n_kv_heads * head_dim;
    const int b = idx / elements_per_block;
    const int offset_in_block = idx % elements_per_block;

    const int src_bid = active_bids[b];
    gathered_kv[idx] = kv_pool[static_cast<size_t>(src_bid) * elements_per_block + offset_in_block];
}

}  // namespace

// ============================================================================
// Benchmark: BM_Xe2_QSA_PromptAttention
// ============================================================================

static void BM_Xe2_QSA_PromptAttention(benchmark::State& state) {
    const int width = static_cast<int>(state.range(0));
    constexpr int n_head = 24;
    constexpr int n_head_kv = 2;
    constexpr int head_dim = 256;
    constexpr float scale = 0.0625f;  // 1.0 / sqrt(256)

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    float* d_q = sycl::malloc_device<float>(n_head * head_dim, q);
    sycl::half* d_k = sycl::malloc_device<sycl::half>(static_cast<size_t>(width) * n_head_kv * head_dim, q);
    sycl::half* d_v = sycl::malloc_device<sycl::half>(static_cast<size_t>(width) * n_head_kv * head_dim, q);
    float* d_out = sycl::malloc_device<float>(n_head * head_dim, q);

    if (!d_q || !d_k || !d_v || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for QSA Attention benchmark");
        if (d_q) sycl::free(d_q, q);
        if (d_k) sycl::free(d_k, q);
        if (d_v) sycl::free(d_v, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_q, 0.05f, n_head * head_dim);
    q.fill(d_k, sycl::half{0.05f}, static_cast<size_t>(width) * n_head_kv * head_dim);
    q.fill(d_v, sycl::half{0.05f}, static_cast<size_t>(width) * n_head_kv * head_dim);
    q.fill(d_out, 0.0f, n_head * head_dim);
    q.wait_and_throw();

    auto launch = [&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_dots(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_head * 256, 256),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             qsa_flash_attn_kernel(item, d_q, d_k, d_v, d_out, width, scale, s_dots);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    // Flops: 2 * (QK dot) + 2 * (AV dot) = 4 * n_head * width * head_dim
    const double flops = 4.0 * n_head * static_cast<double>(width) * head_dim;
    // Bytes: Q (read once) + K (read once) + V (read once) + Out (write once)
    const double bytes = static_cast<double>(n_head * head_dim) * sizeof(float) +
                         static_cast<double>(width * n_head_kv * head_dim * 2) * sizeof(sycl::half) +
                         static_cast<double>(n_head * head_dim) * sizeof(float);

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(width), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Benchmark: BM_Xe2_QSA_TopK_Gather
// ============================================================================

static void BM_Xe2_QSA_TopK_Gather(benchmark::State& state) {
    const int total_cells = static_cast<int>(state.range(0));
    constexpr int block_size = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int num_active_blocks = 256;  // Top-256 blocks = 1024 cells

    const int total_blocks = total_cells / block_size;
    const size_t pool_size = static_cast<size_t>(total_blocks) * block_size * n_kv_heads * head_dim;
    const size_t gather_size = static_cast<size_t>(num_active_blocks) * block_size * n_kv_heads * head_dim;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    sycl::half* d_pool = sycl::malloc_device<sycl::half>(pool_size, q);
    int32_t* d_active_bids = sycl::malloc_device<int32_t>(num_active_blocks, q);
    sycl::half* d_gathered = sycl::malloc_device<sycl::half>(gather_size, q);

    if (!d_pool || !d_active_bids || !d_gathered) {
        state.SkipWithError("Failed to allocate USM memory for Top-K Gather benchmark");
        if (d_pool) sycl::free(d_pool, q);
        if (d_active_bids) sycl::free(d_active_bids, q);
        if (d_gathered) sycl::free(d_gathered, q);
        return;
    }

    // Generate pseudo-random strided block IDs to stress non-contiguous gather
    std::vector<int32_t> h_bids(num_active_blocks);
    const int stride = (total_blocks > num_active_blocks) ? (total_blocks / num_active_blocks) : 1;
    for (int i = 0; i < num_active_blocks; ++i) {
        h_bids[i] = (i * stride) % total_blocks;
    }

    q.fill(d_pool, sycl::half{0.1f}, pool_size);
    q.memcpy(d_active_bids, h_bids.data(), num_active_blocks * sizeof(int32_t));
    q.fill(d_gathered, sycl::half{0.0f}, gather_size);
    q.wait_and_throw();

    constexpr int WG_SIZE = 256;
    const size_t num_wgs = (gather_size + WG_SIZE - 1) / WG_SIZE;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             qsa_topk_gather_kernel(item, d_pool, d_active_bids, d_gathered, num_active_blocks,
                                                    block_size, head_dim, n_kv_heads);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    // Bytes transferred: read indices + read sparse KV blocks + write dense KV blocks
    const double bytes = static_cast<double>(num_active_blocks * sizeof(int32_t)) +
                         static_cast<double>(gather_size * 2) * sizeof(sycl::half);

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_pool, q);
    sycl::free(d_active_bids, q);
    sycl::free(d_gathered, q);
}

// ============================================================================
// Benchmark: BM_Xe2_QSA_LaneCell_Scoring
// ============================================================================

static void BM_Xe2_QSA_LaneCell_Scoring(benchmark::State& state) {
    const int n_cells = static_cast<int>(state.range(0));
    constexpr int D = 128;
    constexpr int HEADS = 4;
    constexpr int R = 4;
    const int max_blocks = n_cells / R + 1;
    const int max_cells = n_cells;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    float* d_pooled = sycl::malloc_device<float>(static_cast<size_t>(max_blocks) * D, q);
    float* d_query = sycl::malloc_device<float>(HEADS * D, q);
    float* d_bias = sycl::malloc_device<float>(max_blocks, q);
    float* d_scores = sycl::malloc_device<float>(max_cells, q);
    int32_t* d_step = sycl::malloc_device<int32_t>(strata::kernels::kStepCount, q);

    if (!d_pooled || !d_query || !d_bias || !d_scores || !d_step) {
        state.SkipWithError("Failed to allocate USM memory for QSA LaneCell Scoring benchmark");
        if (d_pooled) sycl::free(d_pooled, q);
        if (d_query) sycl::free(d_query, q);
        if (d_bias) sycl::free(d_bias, q);
        if (d_scores) sycl::free(d_scores, q);
        if (d_step) sycl::free(d_step, q);
        return;
    }

    int32_t h_step[strata::kernels::kStepCount] = {
        n_cells - 1,
        n_cells,
        n_cells / R,
        n_cells < 2051 ? n_cells : 2051
    };

    q.fill(d_pooled, 0.05f, static_cast<size_t>(max_blocks) * D);
    q.fill(d_query, 0.05f, HEADS * D);
    q.fill(d_bias, 0.01f, max_blocks);
    q.fill(d_scores, 0.0f, max_cells);
    q.memcpy(d_step, h_step, sizeof(h_step));
    q.wait_and_throw();

    strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::native_qsa_score(d_pooled, d_query, d_bias, shapes, d_step,
                                         max_blocks, max_cells, d_scores, &q);
        q.wait();
    }

    // Benchmark loop
    for (auto _ : state) {
        auto t_start = std::chrono::high_resolution_clock::now();
        strata::kernels::native_qsa_score(d_pooled, d_query, d_bias, shapes, d_step,
                                         max_blocks, max_cells, d_scores, &q);
        q.wait();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t_end - t_start;
        state.SetIterationTime(diff.count());
    }

    // Flops: for each block, HEADS * D * 2 flops (dot product) = max_blocks * HEADS * D * 2
    const double flops = static_cast<double>(max_blocks) * HEADS * D * 2.0;
    // Bytes transferred: pooled (read) + query (read) + bias (read) + scores (write)
    const double bytes = static_cast<double>(max_blocks * D) * sizeof(float) +
                         static_cast<double>(HEADS * D) * sizeof(float) +
                         static_cast<double>(max_blocks) * sizeof(float) +
                         static_cast<double>(max_cells) * sizeof(float);

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] =
        benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["cells_per_sec"] =
        benchmark::Counter(static_cast<double>(n_cells), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_pooled, q);
    sycl::free(d_query, q);
    sycl::free(d_bias, q);
    sycl::free(d_scores, q);
    sycl::free(d_step, q);
}

// ============================================================================
// Registration: Attention Context Sweeps & Sparse Top-K Gather
// ============================================================================

// 1. QSA FlashAttention across active context width (T)
BENCHMARK(BM_Xe2_QSA_PromptAttention)->UseManualTime()->Args({64});
BENCHMARK(BM_Xe2_QSA_PromptAttention)->UseManualTime()->Args({256});
BENCHMARK(BM_Xe2_QSA_PromptAttention)->UseManualTime()->Args({1024});
BENCHMARK(BM_Xe2_QSA_PromptAttention)->UseManualTime()->Args({4096});

// 2. Top-K Sparse Gather from total context capacity (cells)
BENCHMARK(BM_Xe2_QSA_TopK_Gather)->UseManualTime()->Args({1024});
BENCHMARK(BM_Xe2_QSA_TopK_Gather)->UseManualTime()->Args({4096});
BENCHMARK(BM_Xe2_QSA_TopK_Gather)->UseManualTime()->Args({16384});
BENCHMARK(BM_Xe2_QSA_TopK_Gather)->UseManualTime()->Args({32768});
BENCHMARK(BM_Xe2_QSA_TopK_Gather)->UseManualTime()->Args({65536});

// 3. Lane-Cell Scoring across sequence context lengths
BENCHMARK(BM_Xe2_QSA_LaneCell_Scoring)->UseManualTime()->Args({1024});
BENCHMARK(BM_Xe2_QSA_LaneCell_Scoring)->UseManualTime()->Args({4096});
BENCHMARK(BM_Xe2_QSA_LaneCell_Scoring)->UseManualTime()->Args({16384});
BENCHMARK(BM_Xe2_QSA_LaneCell_Scoring)->UseManualTime()->Args({65536});

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 Performance Tuning Suite - QSA Attention Benchmark\n");
        std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
        std::printf(" Subgroup:     %d (Native Xe2 SIMD16)\n", strata::sycl_arch::SUB_GROUP_SIZE);
        std::printf("=================================================================\n\n");
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "Warning: Unable to probe default SYCL device: %s\n", e.what());
    }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
