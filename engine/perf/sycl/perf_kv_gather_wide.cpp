// perf/sycl/perf_kv_gather_wide.cpp - Microbenchmark measuring 8-wide vs 4-wide INT8 KV cache gather on Intel Xe2.
#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

struct KvGatherFixture {
    int T;
    int page_size;
    int n_ids;
    k::QsaShapes s;
    sycl::queue q;

    int8_t* d_kq;
    int8_t* d_vq;
    uint16_t* d_ks;
    uint16_t* d_vs;
    int32_t* d_table;
    int32_t* d_ids;
    int32_t* d_step;
    uint16_t* d_k_scratch;
    uint16_t* d_v_scratch;

    double total_bytes;

    KvGatherFixture(int context_len, int p_size = 64)
        : T(context_len), page_size(p_size),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {
        s = k::qsa_real_shapes();
        s.page_size = page_size;

        const int H = (int)s.n_head_kv;
        const int D = (int)s.head_dim;
        const int P = (int)s.page_size;
        const int G = D / k::KV_Q8_GROUP;
        const int pages = (T + P - 1) / P;
        const int cells = pages * P;

        n_ids = (int)k::qsa_selection_width(T, s);

        std::mt19937 rng(42 + T * 19);

        // Scattered pages
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = i;
        std::shuffle(table.begin(), table.end(), rng);

        // Random codes & scales
        std::vector<int8_t> h_kq((size_t)cells * H * D), h_vq((size_t)cells * H * D);
        std::vector<uint16_t> h_ks((size_t)cells * H * G), h_vs((size_t)cells * H * G);
        for (auto& x : h_kq) x = (int8_t)(rng() & 0xFF);
        for (auto& x : h_vq) x = (int8_t)(rng() & 0xFF);
        for (auto& x : h_ks) x = (uint16_t)(0x3000 | (rng() & 0x3FF));
        for (auto& x : h_vs) x = (uint16_t)(0x3000 | (rng() & 0x3FF));

        // Random cells gathered across context
        std::vector<int32_t> ids(n_ids);
        for (auto& x : ids) x = (int32_t)(rng() % T);

        d_kq = sycl::malloc_device<int8_t>(h_kq.size(), q);
        d_vq = sycl::malloc_device<int8_t>(h_vq.size(), q);
        d_ks = sycl::malloc_device<uint16_t>(h_ks.size(), q);
        d_vs = sycl::malloc_device<uint16_t>(h_vs.size(), q);
        d_table = sycl::malloc_device<int32_t>(table.size(), q);
        d_ids = sycl::malloc_device<int32_t>(ids.size(), q);
        d_step = sycl::malloc_device<int32_t>(k::kStepCount, q);

        const size_t out_elems = (size_t)n_ids * H * D;
        d_k_scratch = sycl::malloc_device<uint16_t>(out_elems, q);
        d_v_scratch = sycl::malloc_device<uint16_t>(out_elems, q);

        q.memcpy(d_kq, h_kq.data(), h_kq.size()).wait();
        q.memcpy(d_vq, h_vq.data(), h_vq.size()).wait();
        q.memcpy(d_ks, h_ks.data(), h_ks.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_vs, h_vs.data(), h_vs.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_table, table.data(), table.size() * sizeof(int32_t)).wait();
        q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t)).wait();

        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        q.memcpy(d_step, hstep, sizeof(hstep)).wait();

        // Effective memory traffic:
        // K + V INT8 codes: n_ids * H * D * 2 bytes
        // K + V FP16 scales: n_ids * H * G * 2 * 2 bytes
        // K + V FP16 scratch: n_ids * H * D * 2 * 2 bytes
        total_bytes = (double)n_ids * H * D * 2.0
                    + (double)n_ids * H * G * 4.0
                    + (double)n_ids * H * D * 4.0;
    }

    ~KvGatherFixture() {
        sycl::free(d_kq, q);
        sycl::free(d_vq, q);
        sycl::free(d_ks, q);
        sycl::free(d_vs, q);
        sycl::free(d_table, q);
        sycl::free(d_ids, q);
        sycl::free(d_step, q);
        sycl::free(d_k_scratch, q);
        sycl::free(d_v_scratch, q);
    }
};

static void BM_KV_Gather_Standard_4Wide(benchmark::State& state) {
    const int T = (int)state.range(0);
    KvGatherFixture fix(T);

    for (int i = 0; i < 5; ++i) {
        k::kv_gather_q8_standard_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                      fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        k::kv_gather_q8_standard_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                      fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Bytes/s"] = benchmark::Counter(fix.total_bytes, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["n_ids"] = benchmark::Counter((double)fix.n_ids);
}

static void BM_KV_Gather_Wide_8Wide(benchmark::State& state) {
    const int T = (int)state.range(0);
    KvGatherFixture fix(T);

    for (int i = 0; i < 5; ++i) {
        k::kv_gather_q8_wide_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                  fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        k::kv_gather_q8_wide_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                  fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Bytes/s"] = benchmark::Counter(fix.total_bytes, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["n_ids"] = benchmark::Counter((double)fix.n_ids);
}

BENCHMARK(BM_KV_Gather_Standard_4Wide)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(32768)->Arg(65536)->UseManualTime();
BENCHMARK(BM_KV_Gather_Wide_8Wide)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(32768)->Arg(65536)->UseManualTime();

void run_direct_summary() {
    std::printf("\n=================================================================================================\n");
    std::printf("   Intel Xe2 / Arc B580 INT8 KV Cache Gather Microbenchmark (8-Wide vs 4-Wide)\n");
    std::printf("=================================================================================================\n");
    std::printf("%-8s | %-6s | %-8s | %-12s | %-12s | %-8s | %-12s | %-10s\n",
                "Context", "Gather", "Traffic", "4-Wide (us)", "8-Wide (us)", "Speedup", "4-Wide GB/s", "8-Wide GB/s");
    std::printf("-------------------------------------------------------------------------------------------------\n");

    const std::vector<int> contexts = {1024, 4096, 16384, 32768, 65536};
    constexpr int warmup = 20;
    constexpr int reps = 100;

    for (int T : contexts) {
        KvGatherFixture fix(T);

        // Warmup 4-wide
        for (int i = 0; i < warmup; ++i) {
            k::kv_gather_q8_standard_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                          fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
            fix.q.wait();
        }

        // Measure 4-wide
        auto t0_4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            k::kv_gather_q8_standard_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                          fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
            fix.q.wait();
        }
        auto t1_4 = std::chrono::high_resolution_clock::now();
        const double us_4 = std::chrono::duration<double, std::micro>(t1_4 - t0_4).count() / reps;

        // Warmup 8-wide
        for (int i = 0; i < warmup; ++i) {
            k::kv_gather_q8_wide_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                      fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
            fix.q.wait();
        }

        // Measure 8-wide
        auto t0_8 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            k::kv_gather_q8_wide_step(fix.d_kq, fix.d_vq, fix.d_ks, fix.d_vs, fix.d_table, fix.d_ids,
                                      fix.d_step, fix.n_ids, fix.s, fix.d_k_scratch, fix.d_v_scratch, &fix.q);
            fix.q.wait();
        }
        auto t1_8 = std::chrono::high_resolution_clock::now();
        const double us_8 = std::chrono::duration<double, std::micro>(t1_8 - t0_8).count() / reps;

        const double mb = fix.total_bytes / 1e6;
        const double gbs_4 = (fix.total_bytes / (us_4 * 1e-6)) * 1e-9;
        const double gbs_8 = (fix.total_bytes / (us_8 * 1e-6)) * 1e-9;
        const double speedup = us_4 / us_8;
        const double red_pct = (us_4 - us_8) / us_4 * 100.0;

        std::printf("T=%-6d | %-6d | %5.2f MB | %10.2f us | %10.2f us | %6.2fx (%+4.1f%%) | %9.1f GB/s | %9.1f GB/s\n",
                    T, fix.n_ids, mb, us_4, us_8, speedup, red_pct, gbs_4, gbs_8);
    }
    std::printf("=================================================================================================\n\n");
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
