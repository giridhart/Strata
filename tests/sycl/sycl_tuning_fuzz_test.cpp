// tests/sycl/sycl_tuning_fuzz_test.cpp - Boundary and stress testing for tuned SYCL kernel configurations.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

void ref_gemv(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
              int64_t n_in, int64_t n_out, std::vector<float>& y) {
    y.assign((size_t)n_out, 0.0f);
    for (int64_t o = 0; o < n_out; ++o) {
        double acc = 0.0;
        for (int64_t i = 0; i < n_in; ++i) {
            acc += (double)f32_from_bf16(x[(size_t)i]) * (double)f32_from_bf16(w[(size_t)(o * n_in + i)]);
        }
        y[(size_t)o] = (float)acc;
    }
}

void ref_rmsnorm(const float* x, const float* gamma, float* y, int n_cols, int n_rows, float eps) {
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

// ----------------------------------------------------------------------------
// 1. BF16 GEMV Split Reduction Boundary Fuzzing
// ----------------------------------------------------------------------------
void test_bf16_split_fuzz(sycl::queue& q) {
    printf("[1/3] Fuzzing BF16 GEMV split reduction across TPR & boundary shapes...\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    const int64_t in_dim = 256;
    // Test boundary output sizes: powers of 2, odd numbers, small, and non-multiples
    const int64_t out_dims[] = {1, 3, 7, 15, 16, 17, 31, 32, 48, 63, 64, 127, 128};
    const int tpr_candidates[] = {16, 32, 64, 128, 256};

    for (int64_t n_out : out_dims) {
        std::vector<uint16_t> h_x((size_t)in_dim);
        for (auto& v : h_x) v = bf16_from_f32(dist(rng));

        std::vector<uint16_t> h_w((size_t)(in_dim * n_out));
        for (auto& v : h_w) v = bf16_from_f32(dist(rng) * 0.1f);

        std::vector<float> ref_y;
        ref_gemv(h_x, h_w, in_dim, n_out, ref_y);

        auto* d_x = sycl::malloc_device<uint16_t>(h_x.size(), q);
        auto* d_w = sycl::malloc_device<uint16_t>(h_w.size(), q);
        auto* d_y = sycl::malloc_device<float>((size_t)n_out, q);

        q.copy(h_x.data(), d_x, h_x.size());
        q.copy(h_w.data(), d_w, h_w.size());
        q.wait_and_throw();

        for (int tpr : tpr_candidates) {
            q.fill(d_y, 0.0f, (size_t)n_out);
            q.wait_and_throw();

            strata::kernels::bf16_gemv_split(d_x, d_w, d_y, in_dim, n_out, tpr, &q);
            q.wait_and_throw();

            std::vector<float> gpu_y((size_t)n_out);
            q.copy(d_y, gpu_y.data(), gpu_y.size()).wait();

            double max_err = 0.0;
            for (size_t i = 0; i < (size_t)n_out; ++i) {
                double diff = std::fabs((double)gpu_y[i] - (double)ref_y[i]);
                if (diff > max_err) max_err = diff;
            }

            if (max_err > 0.05) { // BF16 accumulation tolerance
                printf("FAIL: n_out=%lld, tpr=%d, max_err=%f\n", (long long)n_out, tpr, max_err);
                g_failures++;
            }
        }

        sycl::free(d_x, q);
        sycl::free(d_w, q);
        sycl::free(d_y, q);
    }
    printf("      BF16 GEMV Split Boundary Fuzzing: %s\n", g_failures == 0 ? "PASSED" : "FAILED");
}

// ----------------------------------------------------------------------------
// 2. RMSNorm Dimension Boundary Fuzzing
// ----------------------------------------------------------------------------
void test_rmsnorm_fuzz(sycl::queue& q) {
    printf("[2/3] Fuzzing RMSNorm across arbitrary dimensions & row counts...\n");
    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const float eps = 1e-6f;

    const int cols[] = {16, 32, 48, 64, 128, 256, 320, 512, 1024, 2560};
    const int rows[] = {1, 2, 3, 5, 8};

    for (int n_cols : cols) {
        for (int n_rows : rows) {
            const size_t total_elems = (size_t)n_cols * n_rows;
            std::vector<float> h_in(total_elems);
            std::vector<float> h_gamma(total_elems);
            for (auto& v : h_in) v = dist(rng);
            for (auto& v : h_gamma) v = dist(rng) * 0.5f + 1.0f;

            std::vector<float> ref_out(total_elems);
            ref_rmsnorm(h_in.data(), h_gamma.data(), ref_out.data(), n_cols, n_rows, eps);

            auto* d_in = sycl::malloc_device<float>(total_elems, q);
            auto* d_gamma = sycl::malloc_device<float>(total_elems, q);
            auto* d_out = sycl::malloc_device<float>(total_elems, q);

            q.copy(h_in.data(), d_in, total_elems);
            q.copy(h_gamma.data(), d_gamma, total_elems);
            q.wait_and_throw();

            strata::kernels::native_gr_rms_norm_weighted(d_in, d_gamma, d_out, n_cols, n_rows, eps, &q);
            q.wait_and_throw();

            std::vector<float> gpu_out(total_elems);
            q.copy(d_out, gpu_out.data(), total_elems).wait();

            double max_err = 0.0;
            for (size_t i = 0; i < total_elems; ++i) {
                double diff = std::fabs((double)gpu_out[i] - (double)ref_out[i]);
                if (diff > max_err) max_err = diff;
            }

            if (max_err > 1e-4) {
                printf("FAIL: RMSNorm cols=%d, rows=%d, max_err=%f\n", n_cols, n_rows, max_err);
                g_failures++;
            }

            sycl::free(d_in, q);
            sycl::free(d_gamma, q);
            sycl::free(d_out, q);
        }
    }
    printf("      RMSNorm Dimension Boundary Fuzzing: %s\n", g_failures == 0 ? "PASSED" : "FAILED");
}

// ----------------------------------------------------------------------------
// 3. S-GEMV Split Reduction Parity Fuzzing
// ----------------------------------------------------------------------------
void test_s_gemv_split_fuzz(sycl::queue& q) {
    printf("[3/3] Fuzzing S-GEMV split reductions against naive baseline...\n");
    const int64_t n_in = 256;
    const int64_t n_out_cases[] = {32, 48, 64, 128, 256, 640};
    const int tpr_cases[] = {16, 32, 64, 128};

    strata::kernels::SForm form;
    form.code_bits = 4;
    form.code_bias = -8;
    form.group_elems = 64;
    form.has_offset = false;

    std::mt19937 rng(999);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);

    for (int64_t n_out : n_out_cases) {
        const size_t x_elems = (size_t)n_in;
        const size_t codes_bytes = (size_t)n_out * (n_in / 2);
        const size_t scales_elems = (size_t)n_out * (n_in / form.group_elems);

        std::vector<uint16_t> h_x(x_elems);
        for (auto& v : h_x) v = strata::kernels::f16_from_f32(f_dist(rng));

        std::vector<uint8_t> h_codes(codes_bytes);
        for (auto& v : h_codes) v = (uint8_t)byte_dist(rng);

        std::vector<float> h_scales(scales_elems);
        for (auto& v : h_scales) v = f_dist(rng) * 0.05f;

        auto* d_x = sycl::malloc_device<uint16_t>(x_elems, q);
        auto* d_codes = sycl::malloc_device<uint8_t>(codes_bytes, q);
        auto* d_scales = sycl::malloc_device<float>(scales_elems, q);
        auto* d_y_ref = sycl::malloc_device<float>((size_t)n_out, q);
        auto* d_y_split = sycl::malloc_device<float>((size_t)n_out, q);

        q.copy(h_x.data(), d_x, x_elems);
        q.copy(h_codes.data(), d_codes, codes_bytes);
        q.copy(h_scales.data(), d_scales, scales_elems);
        q.wait_and_throw();

        // Compute naive reference on GPU
        strata::kernels::s_gemv(d_x, d_codes, d_scales, nullptr, d_y_ref, n_in, n_out, form);
        q.wait_and_throw();

        std::vector<float> ref_y((size_t)n_out);
        q.copy(d_y_ref, ref_y.data(), ref_y.size()).wait();

        for (int tpr : tpr_cases) {
            strata::kernels::s_gemv_split(d_x, d_codes, d_scales, nullptr, d_y_split, n_in, n_out, form, tpr);
            q.wait_and_throw();

            std::vector<float> split_y((size_t)n_out);
            q.copy(d_y_split, split_y.data(), split_y.size()).wait();

            double max_diff = 0.0;
            for (size_t i = 0; i < (size_t)n_out; ++i) {
                double diff = std::fabs((double)split_y[i] - (double)ref_y[i]);
                if (diff > max_diff) max_diff = diff;
            }

            if (max_diff > 0.01) { // Order-of-summation rounding tolerance
                printf("FAIL: s_gemv_split n_out=%lld, tpr=%d, max_diff=%f\n", (long long)n_out, tpr, max_diff);
                g_failures++;
            }
        }

        sycl::free(d_x, q);
        sycl::free(d_codes, q);
        sycl::free(d_scales, q);
        sycl::free(d_y_ref, q);
        sycl::free(d_y_split, q);
    }
    printf("      S-GEMV Split Reduction Parity Fuzzing: %s\n", g_failures == 0 ? "PASSED" : "FAILED");
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    printf("====================================================\n");
    printf("SYCL Kernel Tuning Boundary & Robustness Fuzz Suite\n");
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    printf("====================================================\n");

    test_bf16_split_fuzz(q);
    test_rmsnorm_fuzz(q);
    test_s_gemv_split_fuzz(q);

    printf("====================================================\n");
    if (g_failures == 0) {
        printf("ALL KERNEL FUZZ TESTS PASSED (0 FAILURES)!\n");
        printf("====================================================\n");
        return 0;
    } else {
        printf("FAILED with %d total errors!\n", g_failures);
        printf("====================================================\n");
        return 1;
    }
}
