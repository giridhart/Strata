// perf/sycl/perf_hc_mix_split.cpp - Fused HC Mix + Split (hc_mix_split) Benchmark on Intel Xe2
//
// Evaluates performance improvements on Intel Xe2 execution units comparing:
//   1. Baseline Separate: hc_mix followed by hc_split in two independent kernel dispatches.
//   2. Fused hc_mix_split: Unified kernel launch with atomic completion handoff and in-subgroup split.
//
// Evaluates across token batch counts T in [1, 2, 4, 8] and hidden dimensions hcdim in [2560, 5120, 10240] (n_mix = 24, hc = 4).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/hc_mix_split.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct HcFixture {
    int T;
    int hcdim;
    int n_mix = 24;
    int hc = 4;
    int iters = 2;
    float eps = 1e-6f;
    float hc_eps = 1e-6f;

    sycl::queue q;
    float* d_h = nullptr;
    float* d_fn = nullptr;
    float* d_mixes = nullptr;
    float* d_scale = nullptr;
    float* d_base = nullptr;
    float* d_pre = nullptr;
    float* d_post = nullptr;
    float* d_comb = nullptr;
    uint32_t* d_cnt = nullptr;

    HcFixture(int tokens, int dim)
        : T(tokens), hcdim(dim),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {

        const size_t h_elems = static_cast<size_t>(T) * hcdim;
        const size_t fn_elems = static_cast<size_t>(n_mix) * hcdim;
        const size_t mixes_elems = static_cast<size_t>(T) * 32;
        const size_t pre_elems = static_cast<size_t>(T) * 8;
        const size_t post_elems = static_cast<size_t>(T) * 8;
        const size_t comb_elems = static_cast<size_t>(T) * 64;

        d_h = sycl::malloc_device<float>(h_elems, q);
        d_fn = sycl::malloc_device<float>(fn_elems, q);
        d_mixes = sycl::malloc_device<float>(mixes_elems, q);
        d_scale = sycl::malloc_device<float>(3, q);
        d_base = sycl::malloc_device<float>(32, q);
        d_pre = sycl::malloc_device<float>(pre_elems, q);
        d_post = sycl::malloc_device<float>(post_elems, q);
        d_comb = sycl::malloc_device<float>(comb_elems, q);
        d_cnt = sycl::malloc_device<uint32_t>(T, q);

        std::mt19937 rng(42 + tokens + dim);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> h_data(h_elems);
        for (auto& v : h_data) v = dist(rng) * 0.5f;
        std::vector<float> fn_data(fn_elems);
        for (auto& v : fn_data) v = dist(rng) * 0.1f;
        std::vector<float> scale_data = {1.2f, 0.8f, 1.5f};
        std::vector<float> base_data(32);
        for (auto& v : base_data) v = dist(rng) * 0.2f;

        q.memcpy(d_h, h_data.data(), h_elems * sizeof(float));
        q.memcpy(d_fn, fn_data.data(), fn_elems * sizeof(float));
        q.memcpy(d_scale, scale_data.data(), 3 * sizeof(float));
        q.memcpy(d_base, base_data.data(), 32 * sizeof(float));
        q.memset(d_cnt, 0, T * sizeof(uint32_t));
        q.wait();
    }

    ~HcFixture() {
        sycl::free(d_h, q);
        sycl::free(d_fn, q);
        sycl::free(d_mixes, q);
        sycl::free(d_scale, q);
        sycl::free(d_base, q);
        sycl::free(d_pre, q);
        sycl::free(d_post, q);
        sycl::free(d_comb, q);
        sycl::free(d_cnt, q);
    }
};

static void BM_HcMixSplit_Separate(benchmark::State& state) {
    const int T = static_cast<int>(state.range(0));
    const int hcdim = static_cast<int>(state.range(1));

    HcFixture f(T, hcdim);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        strata::kernels::hc_mix(f.d_h, f.d_fn, f.hcdim, f.n_mix, f.T, f.eps, f.d_mixes, f.q);
        strata::kernels::hc_split(f.d_mixes, f.d_scale, f.d_base, f.hc, f.iters, f.hc_eps, f.T,
                                  f.d_pre, f.d_post, f.d_comb, f.q);
        f.q.wait();
    }

    for (auto _ : state) {
        strata::kernels::hc_mix(f.d_h, f.d_fn, f.hcdim, f.n_mix, f.T, f.eps, f.d_mixes, f.q);
        strata::kernels::hc_split(f.d_mixes, f.d_scale, f.d_base, f.hc, f.iters, f.hc_eps, f.T,
                                  f.d_pre, f.d_post, f.d_comb, f.q);
        f.q.wait();
    }

    const double bytes_processed = (static_cast<double>(T) * hcdim * sizeof(float)) +
                                   (static_cast<double>(f.n_mix) * hcdim * sizeof(float));
    state.SetBytesProcessed(static_cast<int64_t>(bytes_processed) * static_cast<int64_t>(state.iterations()));
    state.SetItemsProcessed(static_cast<int64_t>(T) * static_cast<int64_t>(state.iterations()));
}

static void BM_HcMixSplit_Fused(benchmark::State& state) {
    const int T = static_cast<int>(state.range(0));
    const int hcdim = static_cast<int>(state.range(1));

    HcFixture f(T, hcdim);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        strata::kernels::hc_mix_split(f.d_h, f.d_fn, f.hcdim, f.n_mix, f.T, f.eps, f.d_mixes, f.d_cnt,
                                      f.d_scale, f.d_base, f.hc, f.iters, f.hc_eps, f.d_pre, f.d_post, f.d_comb, f.q);
        f.q.wait();
    }

    for (auto _ : state) {
        strata::kernels::hc_mix_split(f.d_h, f.d_fn, f.hcdim, f.n_mix, f.T, f.eps, f.d_mixes, f.d_cnt,
                                      f.d_scale, f.d_base, f.hc, f.iters, f.hc_eps, f.d_pre, f.d_post, f.d_comb, f.q);
        f.q.wait();
    }

    const double bytes_processed = (static_cast<double>(T) * hcdim * sizeof(float)) +
                                   (static_cast<double>(f.n_mix) * hcdim * sizeof(float));
    state.SetBytesProcessed(static_cast<int64_t>(bytes_processed) * static_cast<int64_t>(state.iterations()));
    state.SetItemsProcessed(static_cast<int64_t>(T) * static_cast<int64_t>(state.iterations()));
}

void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int t : {1, 2, 4, 8}) {
        for (int dim : {2560, 5120, 10240}) {
            b->Args({t, dim});
        }
    }
}

BENCHMARK(BM_HcMixSplit_Separate)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_HcMixSplit_Fused)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond);

} // namespace

BENCHMARK_MAIN();
