// tests/sycl/xe2_subgroup_gemv_test.cpp - Standalone test for Xe2 SIMD16 subgroup reductions across model dimensions.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

namespace {

void cpu_ref_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out) {
    for (int64_t o = 0; o < n_out; ++o) {
        double acc = 0.0;
        const uint16_t* row = w + o * n_in;
        for (int64_t i = 0; i < n_in; ++i) {
            acc += (double)f32_from_bf16(x[i]) * (double)f32_from_bf16(row[i]);
        }
        y[o] = (float)acc;
    }
}

} // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    printf("Running Xe2 Subgroup GEMV test on: %s (SUB_GROUP_SIZE=%d)\n",
           q.get_device().get_info<sycl::info::device::name>().c_str(),
           strata::sycl_arch::SUB_GROUP_SIZE);

    // Exact model projection shapes from Qwen3.8-Flash-Next layout:
    struct GemvShape {
        int64_t n_in;
        int64_t n_out;
        const char* name;
    };

    const GemvShape shapes[] = {
        {2560, 48,   "ssm_alpha / ssm_beta [2560, 48]"},
        {2560, 128,  "indexer.k_proj [2560, 128]"},
        {2560, 512,  "indexer.q_proj [2560, 512]"},
        {2560, 640,  "ffn_gate_shexp [2560, 640]"},
        {640,  2560, "ffn_down_shexp [640, 2560]"},
        {10240, 320, "hc_attn_down [10240, 320]"},
        {320, 10240, "hc_attn_up [320, 10240]"},
        {2560, 2560, "ple_value / ffn [2560, 2560]"}
    };

    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int total_bad = 0;

    for (const auto& s : shapes) {
        std::vector<uint16_t> h_x(s.n_in);
        std::vector<uint16_t> h_w(s.n_in * s.n_out);
        std::vector<float> h_ref(s.n_out);
        std::vector<float> h_out_warp(s.n_out, 0.0f);
        std::vector<float> h_out_split(s.n_out, 0.0f);

        for (int64_t i = 0; i < s.n_in; ++i) h_x[i] = bf16_from_f32(dist(rng) * 0.1f);
        for (int64_t i = 0; i < s.n_in * s.n_out; ++i) h_w[i] = bf16_from_f32(dist(rng) * 0.1f);

        cpu_ref_gemv(h_x.data(), h_w.data(), h_ref.data(), s.n_in, s.n_out);

        uint16_t* d_x = sycl::malloc_device<uint16_t>(s.n_in, q);
        uint16_t* d_w = sycl::malloc_device<uint16_t>(s.n_in * s.n_out, q);
        float* d_y = sycl::malloc_device<float>(s.n_out, q);

        q.memcpy(d_x, h_x.data(), s.n_in * sizeof(uint16_t)).wait();
        q.memcpy(d_w, h_w.data(), s.n_in * s.n_out * sizeof(uint16_t)).wait();

        // 1. Test subgroup-per-row path
        q.memset(d_y, 0, s.n_out * sizeof(float)).wait();
        strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, &q);
        q.wait_and_throw();
        q.memcpy(h_out_warp.data(), d_y, s.n_out * sizeof(float)).wait();

        // 2. Test split reduction path (threads_per_row = 256)
        q.memset(d_y, 0, s.n_out * sizeof(float)).wait();
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 256, &q);
        q.wait_and_throw();
        q.memcpy(h_out_split.data(), d_y, s.n_out * sizeof(float)).wait();

        double max_err_warp = 0.0, max_err_split = 0.0;
        int bad_warp = 0, bad_split = 0;

        for (int64_t i = 0; i < s.n_out; ++i) {
            double err_w = std::abs(h_out_warp[i] - h_ref[i]) / (std::abs(h_ref[i]) + 1e-4);
            double err_s = std::abs(h_out_split[i] - h_ref[i]) / (std::abs(h_ref[i]) + 1e-4);
            if (err_w > max_err_warp) max_err_warp = err_w;
            if (err_s > max_err_split) max_err_split = err_s;
            if (err_w > 5e-4) ++bad_warp;
            if (err_s > 5e-4) ++bad_split;
        }

        printf("  %-36s : warp err=%.2e (%d bad) | split256 err=%.2e (%d bad)\n",
               s.name, max_err_warp, bad_warp, max_err_split, bad_split);

        if (bad_warp > 0 || bad_split > 0) total_bad += (bad_warp + bad_split);

        sycl::free(d_x, q);
        sycl::free(d_w, q);
        sycl::free(d_y, q);
    }

    if (total_bad == 0) {
        printf("Xe2 Subgroup GEMV test OK (0 failures)\n");
        return 0;
    } else {
        printf("Xe2 Subgroup GEMV test FAILED with %d errors\n", total_bad);
        return 1;
    }
}
