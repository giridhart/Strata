// perf/sycl/perf_dequant_dpas.cpp - Intel Xe2 Weight-Only Dequantization + XMX DPAS Benchmark
//
// Benchmarks the Critical LLM Decode Bottleneck on Intel Xe2:
// Streaming quantized weights from VRAM/System RAM, dequantizing to FP16/BF16, and feeding XMX DPAS engines.
//
// Compares two execution strategies:
// 1. Direct-to-Register (XVE -> XMX):
//    Dequantizes directly in Xe Vector Engine registers and feeds XMX systolic tiles without SLM traffic.
// 2. SLM-Buffered (VRAM -> SLM -> XMX):
//    Buffers dequantized FP16 tiles in Shared Local Memory (SLM) with group barriers before XMX load.
//
// Evaluated on key Qwen projection shapes:
// - Gate projection [K=2560, N=6144]
// - Expert projection [K=2560, N=10240]

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl::ext::oneapi::experimental::matrix;

namespace {

// ============================================================================
// Dequantization Kernels
// ============================================================================

/// Direct-to-Register Dequantization + XMX Systolic Tile (8x16x16)
/// Unpacks 4-bit weights in-register and feeds directly to joint_matrix_mad.
template <size_t TM = 8, size_t TN = 16, size_t TK = 16>
void dequant_register_dpas_kernel(sycl::nd_item<2> item, const sycl::half* __restrict__ act,
                                  const uint8_t* __restrict__ qweights,  // 4-bit packed weights
                                  const sycl::half* __restrict__ scales, float* __restrict__ out, size_t M, size_t N,
                                  size_t K, sycl::local_accessor<sycl::half, 1> dequant_tile) {
    auto sg = item.get_sub_group();
    const size_t tile_m = item.get_group(0) * TM;
    const size_t tile_n = item.get_group(1) * TN;

    if (tile_m >= M || tile_n >= N) return;

    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
    joint_matrix_fill(sg, sub_c, 0.0f);

    for (size_t k = 0; k < K; k += TK) {
        // Load activation tile
        joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
        auto ptr_a = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            act + tile_m * K + k);
        joint_matrix_load(sg, sub_a, ptr_a, K);

        // Dequantize 4-bit weights into local tile without global barrier
        const int lane = sg.get_local_linear_id();
        const float scale = static_cast<float>(scales[(k / 32) * N + tile_n + lane]);

#pragma unroll
        for (size_t row = 0; row < TK; ++row) {
            const size_t byte_idx = ((k + row) * N + tile_n + lane) / 2;
            const uint8_t byte_val = qweights[byte_idx];
            const int nibble = (lane % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
            dequant_tile[row * TN + lane] = static_cast<sycl::half>((static_cast<float>(nibble) - 8.0f) * scale);
        }

        joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;
        auto ptr_b = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
            &dequant_tile[0]);
        joint_matrix_load(sg, sub_b, ptr_b, TN);

        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
    }

    auto ptr_c = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        out + tile_m * N + tile_n);
    joint_matrix_store(sg, sub_c, ptr_c, N, layout::row_major);
}

/// SLM-Buffered Dequantization + XMX Systolic Tile
/// Buffers dequantized FP16 blocks in Shared Local Memory with barriers before XMX consumption.
template <size_t TM = 8, size_t TN = 16, size_t TK = 16>
void dequant_slm_dpas_kernel(sycl::nd_item<2> item, const sycl::half* __restrict__ act,
                             const uint8_t* __restrict__ qweights, const sycl::half* __restrict__ scales,
                             float* __restrict__ out, size_t M, size_t N, size_t K,
                             sycl::local_accessor<sycl::half, 1> slm_b) {
    auto sg = item.get_sub_group();
    const size_t tile_m = item.get_group(0) * TM;
    const size_t tile_n = item.get_group(1) * TN;

    if (tile_m >= M || tile_n >= N) return;

    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
    joint_matrix_fill(sg, sub_c, 0.0f);

    for (size_t k = 0; k < K; k += TK) {
        joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
        auto ptr_a = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            act + tile_m * K + k);
        joint_matrix_load(sg, sub_a, ptr_a, K);

        // Dequantize to SLM buffer
        const int lane = sg.get_local_linear_id();
        const float scale = static_cast<float>(scales[(k / 32) * N + tile_n + lane]);

#pragma unroll
        for (size_t row = 0; row < TK; ++row) {
            const size_t byte_idx = ((k + row) * N + tile_n + lane) / 2;
            const uint8_t byte_val = qweights[byte_idx];
            const int nibble = (lane % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
            slm_b[row * TN + lane] = static_cast<sycl::half>((static_cast<float>(nibble) - 8.0f) * scale);
        }
        item.barrier(sycl::access::fence_space::local_space);

        joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;
        auto ptr_b =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(&slm_b[0]);
        joint_matrix_load(sg, sub_b, ptr_b, TN);

        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
        item.barrier(sycl::access::fence_space::local_space);
    }

    auto ptr_c = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        out + tile_m * N + tile_n);
    joint_matrix_store(sg, sub_c, ptr_c, N, layout::row_major);
}

}  // namespace

// ============================================================================
// Benchmarks
// ============================================================================

static void BM_Xe2_Dequant_DirectRegister(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t act_size = M * K;
    const size_t qweight_bytes = (K * N) / 2;  // 4-bit packed
    const size_t scale_size = (K / 32) * N;
    const size_t out_size = M * N;

    sycl::half* d_act = sycl::malloc_device<sycl::half>(act_size, q);
    uint8_t* d_qweights = sycl::malloc_device<uint8_t>(qweight_bytes, q);
    sycl::half* d_scales = sycl::malloc_device<sycl::half>(scale_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_act || !d_qweights || !d_scales || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for Dequant DPAS benchmark");
        if (d_act) sycl::free(d_act, q);
        if (d_qweights) sycl::free(d_qweights, q);
        if (d_scales) sycl::free(d_scales, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_act, sycl::half{0.05f}, act_size);
    q.fill(d_qweights, static_cast<uint8_t>(0x55), qweight_bytes);
    q.fill(d_scales, sycl::half{0.01f}, scale_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr size_t TM = 8;
    constexpr size_t TN = 16;
    constexpr size_t TK = 16;
    const size_t grid_m = (M + TM - 1) / TM;
    const size_t grid_n = (N + TN - 1) / TN;

    auto launch = [&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> dequant_tile(sycl::range<1>(TK * TN), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n * 16), sycl::range<2>(1, 16)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                             dequant_register_dpas_kernel<8, 16, 16>(item, d_act, d_qweights, d_scales, d_out, M, N, K,
                                                                     dequant_tile);
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

    const double bytes = static_cast<double>(act_size) * sizeof(sycl::half) + static_cast<double>(qweight_bytes) +
                         static_cast<double>(scale_size) * sizeof(sycl::half) +
                         static_cast<double>(out_size) * sizeof(float);

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(M * K * N), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(M), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_act, q);
    sycl::free(d_qweights, q);
    sycl::free(d_scales, q);
    sycl::free(d_out, q);
}

static void BM_Xe2_Dequant_SLMBuffered(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t act_size = M * K;
    const size_t qweight_bytes = (K * N) / 2;
    const size_t scale_size = (K / 32) * N;
    const size_t out_size = M * N;

    sycl::half* d_act = sycl::malloc_device<sycl::half>(act_size, q);
    uint8_t* d_qweights = sycl::malloc_device<uint8_t>(qweight_bytes, q);
    sycl::half* d_scales = sycl::malloc_device<sycl::half>(scale_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_act || !d_qweights || !d_scales || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for Dequant DPAS benchmark");
        if (d_act) sycl::free(d_act, q);
        if (d_qweights) sycl::free(d_qweights, q);
        if (d_scales) sycl::free(d_scales, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_act, sycl::half{0.05f}, act_size);
    q.fill(d_qweights, static_cast<uint8_t>(0x55), qweight_bytes);
    q.fill(d_scales, sycl::half{0.01f}, scale_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr size_t TM = 8;
    constexpr size_t TN = 16;
    constexpr size_t TK = 16;
    const size_t grid_m = (M + TM - 1) / TM;
    const size_t grid_n = (N + TN - 1) / TN;

    auto launch = [&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> slm_b(sycl::range<1>(TK * TN), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n * 16), sycl::range<2>(1, 16)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                             dequant_slm_dpas_kernel<8, 16, 16>(item, d_act, d_qweights, d_scales, d_out, M, N, K,
                                                                slm_b);
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

    const double bytes = static_cast<double>(act_size) * sizeof(sycl::half) + static_cast<double>(qweight_bytes) +
                         static_cast<double>(scale_size) * sizeof(sycl::half) +
                         static_cast<double>(out_size) * sizeof(float);

    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(M * K * N), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(M), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_act, q);
    sycl::free(d_qweights, q);
    sycl::free(d_scales, q);
    sycl::free(d_out, q);
}

/// Standalone Q8_0 Dequantization Benchmark
/// Measures streaming throughput and bandwidth of dequant_q8_0
void BM_Xe2_Dequant_Q8_0(benchmark::State& state) {
    const size_t num_elements = state.range(0);
    const size_t num_blocks = num_elements / 32;
    const size_t block_bytes = num_blocks * 34;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

    std::vector<uint8_t> h_blocks(block_bytes, 0x12);
    uint8_t* d_blocks = sycl::malloc_device<uint8_t>(block_bytes, q);
    float* d_out = sycl::malloc_device<float>(num_elements, q);

    q.memcpy(d_blocks, h_blocks.data(), block_bytes).wait();

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::dequant_q8_0(d_blocks, d_out, num_elements, &q);
    }
    q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        strata::kernels::dequant_q8_0(d_blocks, d_out, num_elements, &q);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(block_bytes + num_elements * sizeof(float));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["elements_per_sec"] =
        benchmark::Counter(static_cast<double>(num_elements), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_blocks, q);
    sycl::free(d_out, q);
}

/// S2 GEMV with Q8_0 Activations (s2_gemv_q8)
static void BM_Xe2_S2_GEMV_Q8(benchmark::State& state) {
    const int64_t K = state.range(0);
    const int64_t N = state.range(1);
    const int tpr = 32;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order()}};

    const size_t act_bytes = (size_t)(K / 32) * 34;
    const size_t codes_bytes = (size_t)N * (K / 4);
    const size_t scales_elems = (size_t)N * (K / 64);

    uint8_t* d_act = sycl::malloc_device<uint8_t>(act_bytes, q);
    uint8_t* d_codes = sycl::malloc_device<uint8_t>(codes_bytes, q);
    float* d_scales = sycl::malloc_device<float>(scales_elems, q);
    float* d_y = sycl::malloc_device<float>(N, q);

    q.memset(d_act, 1, act_bytes).wait();
    q.memset(d_codes, 0x55, codes_bytes).wait();
    q.fill(d_scales, 0.01f, scales_elems).wait();

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::s2_gemv_q8(d_act, d_codes, d_scales, d_y, K, N, tpr, &q);
    }
    q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        strata::kernels::s2_gemv_q8(d_act, d_codes, d_scales, d_y, K, N, tpr, &q);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = (double)(codes_bytes + scales_elems * sizeof(float) + act_bytes + N * sizeof(float));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * K * N, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_act, q);
    sycl::free(d_codes, q);
    sycl::free(d_scales, q);
    sycl::free(d_y, q);
}

// ============================================================================
// Registration: Critical Decode Gate & Expert Projections
// ============================================================================

// Gate projection [K=2560, N=6144]
BENCHMARK(BM_Xe2_Dequant_DirectRegister)->UseManualTime()->Args({8, 2560, 6144});
BENCHMARK(BM_Xe2_Dequant_DirectRegister)->UseManualTime()->Args({16, 2560, 6144});
BENCHMARK(BM_Xe2_Dequant_DirectRegister)->UseManualTime()->Args({32, 2560, 6144});

BENCHMARK(BM_Xe2_Dequant_SLMBuffered)->UseManualTime()->Args({8, 2560, 6144});
BENCHMARK(BM_Xe2_Dequant_SLMBuffered)->UseManualTime()->Args({16, 2560, 6144});
BENCHMARK(BM_Xe2_Dequant_SLMBuffered)->UseManualTime()->Args({32, 2560, 6144});

// Expert projection [K=2560, N=10240]
BENCHMARK(BM_Xe2_Dequant_DirectRegister)->UseManualTime()->Args({16, 2560, 10240});
BENCHMARK(BM_Xe2_Dequant_DirectRegister)->UseManualTime()->Args({32, 2560, 10240});

BENCHMARK(BM_Xe2_Dequant_SLMBuffered)->UseManualTime()->Args({16, 2560, 10240});
BENCHMARK(BM_Xe2_Dequant_SLMBuffered)->UseManualTime()->Args({32, 2560, 10240});

// Standalone Q8_0 Dequantization
BENCHMARK(BM_Xe2_Dequant_Q8_0)->UseManualTime()->Arg(10240)->Arg(20480)->Arg(81920)->Arg(262144)->Arg(1048576);

// S2 GEMV with Q8_0 Activations
BENCHMARK(BM_Xe2_S2_GEMV_Q8)->UseManualTime()->Args({2560, 6144})->Args({2560, 10240});

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 Performance Tuning Suite - Dequant + DPAS Benchmark\n");
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
