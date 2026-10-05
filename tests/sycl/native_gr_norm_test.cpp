// tests/sycl/native_gr_norm_test.cpp - Standalone unit test for native GR RMSNorm on SYCL/Xe2.
#include <sycl/sycl.hpp>
#include "strata/kernels/native_gr_norm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void cpu_rmsnorm(const float* x, const float* gamma, float* y, int n_cols, int n_rows, float eps) {
    for (int r = 0; r < n_rows; ++r) {
        const float* row_in = x + (size_t)r * n_cols;
        const float* row_gamma = gamma + (size_t)r * n_cols;
        float* row_out = y + (size_t)r * n_cols;
        double sum_sq = 0.0;
        for (int c = 0; c < n_cols; ++c) {
            sum_sq += (double)row_in[c] * (double)row_in[c];
        }
        double mean = sum_sq / n_cols;
        float scale = 1.0f / (float)std::sqrt(mean + eps);
        for (int c = 0; c < n_cols; ++c) {
            row_out[c] = scale * row_in[c] * row_gamma[c];
        }
    }
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    printf("Running native_gr_norm SYCL test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Exact model dimensions for Qwen3.8-Flash-Next:
    // hc_attn_norm, hc_ffn_norm: 10240 cols
    // n_embd: 2560 cols
    // hc_lr: 320 cols
    struct Case {
        int n_cols;
        int n_rows;
        const char* name;
    };

    const Case cases[] = {{320, 1, "hc_lr [320, 1]"},       {320, 4, "hc_lr [320, 4]"},
                          {2560, 1, "n_embd [2560, 1]"},    {2560, 8, "n_embd [2560, 8]"},
                          {10240, 1, "hc_dim [10240, 1]"},  {10240, 4, "hc_dim [10240, 4]"},
                          {10240, 16, "hc_dim [10240, 16]"}};

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const float eps = 1e-6f;
    int total_failures = 0;

    for (const auto& c : cases) {
        const size_t total_elems = (size_t)c.n_cols * c.n_rows;
        std::vector<float> h_in(total_elems);
        std::vector<float> h_gamma(total_elems);
        std::vector<float> h_ref(total_elems);
        std::vector<float> h_out(total_elems, 0.0f);

        for (size_t i = 0; i < total_elems; ++i) {
            h_in[i] = dist(rng);
            h_gamma[i] = std::abs(dist(rng)) + 0.1f;
        }

        cpu_rmsnorm(h_in.data(), h_gamma.data(), h_ref.data(), c.n_cols, c.n_rows, eps);

        float* d_in = sycl::malloc_device<float>(total_elems, q);
        float* d_gamma = sycl::malloc_device<float>(total_elems, q);
        float* d_out = sycl::malloc_device<float>(total_elems, q);

        q.memcpy(d_in, h_in.data(), total_elems * sizeof(float)).wait();
        q.memcpy(d_gamma, h_gamma.data(), total_elems * sizeof(float)).wait();
        q.memset(d_out, 0, total_elems * sizeof(float)).wait();

        strata::kernels::native_gr_rms_norm_weighted(d_in, d_gamma, d_out, c.n_cols, c.n_rows, eps, &q);
        q.wait_and_throw();

        q.memcpy(h_out.data(), d_out, total_elems * sizeof(float)).wait();

        double max_err = 0.0;
        int bad = 0;
        for (size_t i = 0; i < total_elems; ++i) {
            double diff = std::abs(h_out[i] - h_ref[i]);
            double rel = diff / (std::abs(h_ref[i]) + 1e-6);
            if (rel > max_err) max_err = rel;
            if (rel > 1e-4) ++bad;
        }

        printf("  %-24s: max_rel = %.3e, bad = %d / %zu\n", c.name, max_err, bad, total_elems);

        if (bad > 0) total_failures += bad;

        sycl::free(d_in, q);
        sycl::free(d_gamma, q);
        sycl::free(d_out, q);
    }

    if (total_failures == 0) {
        printf("native_gr_norm SYCL test OK (0 failures)\n");
        return 0;
    } else {
        printf("native_gr_norm SYCL test FAILED with %d errors\n", total_failures);
        return 1;
    }
}
