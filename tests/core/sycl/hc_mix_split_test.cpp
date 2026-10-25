// tests/core/sycl/hc_mix_split_test.cpp - Unit and numerical parity test for Fused HC Mix + Split on Intel Xe2
#include <sycl/sycl.hpp>
#include "strata/kernels/hc_mix_split.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// CPU reference implementation for validation
void cpu_hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes) {
    for (int t = 0; t < T; ++t) {
        const float* x = h + static_cast<size_t>(t) * hcdim;
        double ss = 0.0;
        for (int k = 0; k < hcdim; ++k) {
            ss += static_cast<double>(x[k]) * static_cast<double>(x[k]);
        }
        const float rms_scale = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(hcdim)) + eps);

        for (int m = 0; m < n_mix; ++m) {
            const float* w = fn + static_cast<size_t>(m) * hcdim;
            double acc = 0.0;
            for (int k = 0; k < hcdim; ++k) {
                acc += static_cast<double>(x[k]) * static_cast<double>(w[k]);
            }
            mixes[static_cast<size_t>(t) * 32 + m] = static_cast<float>(acc) * rms_scale;
        }
    }
}

void cpu_hc_split(const float* mixes, const float* scale, const float* base, int hc,
                  int iters, float eps, int T, float* pre, float* post, float* comb) {
    for (int t = 0; t < T; ++t) {
        const float* mx = mixes + static_cast<size_t>(t) * 32;
        float* p_pre = pre + static_cast<size_t>(t) * 8;
        float* p_post = post + static_cast<size_t>(t) * 8;
        float* p_comb = comb + static_cast<size_t>(t) * 64;

        for (int l = 0; l < hc; ++l) {
            p_pre[l] = 1.0f / (1.0f + std::exp(-(mx[l] * scale[0] + base[l]))) + eps;
            p_post[l] = 2.0f / (1.0f + std::exp(-(mx[hc + l] * scale[1] + base[hc + l])));
        }

        if (hc == 4) {
            float c[16];
            for (int l = 0; l < 16; ++l) {
                c[l] = mx[8 + l] * scale[2] + base[8 + l];
            }

            // Softmax over each row: rows are j = l / 4
            for (int j = 0; j < 4; ++j) {
                float m = -1e30f;
                for (int k = 0; k < 4; ++k) {
                    m = std::max(m, c[j * 4 + k]);
                }
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    c[j * 4 + k] = std::exp(c[j * 4 + k] - m);
                    sum += c[j * 4 + k];
                }
                for (int k = 0; k < 4; ++k) {
                    c[j * 4 + k] = (c[j * 4 + k] / sum) + eps;
                }
            }

            // Destination column normalization
            for (int k = 0; k < 4; ++k) {
                float col_sum = 0.0f;
                for (int j = 0; j < 4; ++j) {
                    col_sum += c[j * 4 + k];
                }
                for (int j = 0; j < 4; ++j) {
                    c[j * 4 + k] /= (col_sum + eps);
                }
            }

            for (int it = 1; it < iters; ++it) {
                for (int j = 0; j < 4; ++j) {
                    float row_sum = 0.0f;
                    for (int k = 0; k < 4; ++k) row_sum += c[j * 4 + k];
                    for (int k = 0; k < 4; ++k) c[j * 4 + k] = (c[j * 4 + k] / (row_sum + eps));
                }
                for (int k = 0; k < 4; ++k) {
                    float col_sum = 0.0f;
                    for (int j = 0; j < 4; ++j) col_sum += c[j * 4 + k];
                    for (int j = 0; j < 4; ++j) c[j * 4 + k] /= (col_sum + eps);
                }
            }

            for (int l = 0; l < 16; ++l) {
                p_comb[l] = c[l];
            }
        }
    }
}

} // namespace

int main() {
    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    const auto dev = q.get_device();
    std::printf("====================================================================================\n");
    std::printf(" Strata SYCL Fused HC Mix + Split (hc_mix_split) Unit & Parity Test\n");
    std::printf(" Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("====================================================================================\n\n");

    const std::vector<int> token_cases = {1, 2, 4, 8};
    const std::vector<int> hcdim_cases = {2560, 5120, 10240};
    const int n_mix = 24;
    const int hc = 4;
    const int iters = 2;
    const float eps = 1e-6f;
    const float hc_eps = 1e-6f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int passed_tests = 0;
    int total_tests = 0;

    for (int T : token_cases) {
        for (int hcdim : hcdim_cases) {
            ++total_tests;
            std::printf("[TEST %d] Token Count T = %d, Dimension hcdim = %d, n_mix = %d, hc = %d\n",
                        total_tests, T, hcdim, n_mix, hc);

            const size_t h_elems = static_cast<size_t>(T) * hcdim;
            const size_t fn_elems = static_cast<size_t>(n_mix) * hcdim;
            const size_t mixes_elems = static_cast<size_t>(T) * 32;
            const size_t pre_elems = static_cast<size_t>(T) * 8;
            const size_t post_elems = static_cast<size_t>(T) * 8;
            const size_t comb_elems = static_cast<size_t>(T) * 64;

            std::vector<float> h_h(h_elems);
            for (auto& v : h_h) v = dist(rng) * 0.5f;

            std::vector<float> h_fn(fn_elems);
            for (auto& v : h_fn) v = dist(rng) * 0.1f;

            std::vector<float> h_scale = {1.2f, 0.8f, 1.5f};
            std::vector<float> h_base(32);
            for (auto& v : h_base) v = dist(rng) * 0.2f;

            // Compute CPU reference
            std::vector<float> ref_mixes(mixes_elems, 0.0f);
            std::vector<float> ref_pre(pre_elems, 0.0f);
            std::vector<float> ref_post(post_elems, 0.0f);
            std::vector<float> ref_comb(comb_elems, 0.0f);

            cpu_hc_mix(h_h.data(), h_fn.data(), hcdim, n_mix, T, eps, ref_mixes.data());
            cpu_hc_split(ref_mixes.data(), h_scale.data(), h_base.data(), hc, iters, hc_eps, T,
                         ref_pre.data(), ref_post.data(), ref_comb.data());

            // Allocate device memory
            float* d_h = sycl::malloc_device<float>(h_elems, q);
            float* d_fn = sycl::malloc_device<float>(fn_elems, q);
            float* d_mixes_sep = sycl::malloc_device<float>(mixes_elems, q);
            float* d_mixes_fused = sycl::malloc_device<float>(mixes_elems, q);
            float* d_scale = sycl::malloc_device<float>(3, q);
            float* d_base = sycl::malloc_device<float>(32, q);
            float* d_pre = sycl::malloc_device<float>(pre_elems, q);
            float* d_post = sycl::malloc_device<float>(post_elems, q);
            float* d_comb = sycl::malloc_device<float>(comb_elems, q);
            uint32_t* d_cnt = sycl::malloc_device<uint32_t>(T, q);

            q.memcpy(d_h, h_h.data(), h_elems * sizeof(float));
            q.memcpy(d_fn, h_fn.data(), fn_elems * sizeof(float));
            q.memcpy(d_scale, h_scale.data(), 3 * sizeof(float));
            q.memcpy(d_base, h_base.data(), 32 * sizeof(float));
            q.memset(d_cnt, 0, T * sizeof(uint32_t));
            q.wait();

            // 1. Test separate hc_mix
            strata::kernels::hc_mix(d_h, d_fn, hcdim, n_mix, T, eps, d_mixes_sep, q);
            q.wait();

            std::vector<float> out_mixes(mixes_elems);
            q.memcpy(out_mixes.data(), d_mixes_sep, mixes_elems * sizeof(float)).wait();

            float max_diff_mix = 0.0f;
            for (size_t i = 0; i < mixes_elems; ++i) {
                max_diff_mix = std::max(max_diff_mix, std::abs(out_mixes[i] - ref_mixes[i]));
            }

            // 2. Test separate hc_split
            strata::kernels::hc_split(d_mixes_sep, d_scale, d_base, hc, iters, hc_eps, T,
                                      d_pre, d_post, d_comb, q);
            q.wait();

            std::vector<float> out_pre(pre_elems);
            std::vector<float> out_post(post_elems);
            std::vector<float> out_comb(comb_elems);
            q.memcpy(out_pre.data(), d_pre, pre_elems * sizeof(float));
            q.memcpy(out_post.data(), d_post, post_elems * sizeof(float));
            q.memcpy(out_comb.data(), d_comb, comb_elems * sizeof(float)).wait();

            float max_diff_pre = 0.0f;
            for (size_t i = 0; i < pre_elems; ++i) max_diff_pre = std::max(max_diff_pre, std::abs(out_pre[i] - ref_pre[i]));

            float max_diff_post = 0.0f;
            for (size_t i = 0; i < post_elems; ++i) max_diff_post = std::max(max_diff_post, std::abs(out_post[i] - ref_post[i]));

            float max_diff_comb = 0.0f;
            for (size_t i = 0; i < comb_elems; ++i) max_diff_comb = std::max(max_diff_comb, std::abs(out_comb[i] - ref_comb[i]));

            // 3. Test FUSED hc_mix_split
            q.memset(d_pre, 0, pre_elems * sizeof(float));
            q.memset(d_post, 0, post_elems * sizeof(float));
            q.memset(d_comb, 0, comb_elems * sizeof(float));
            q.memset(d_mixes_fused, 0, mixes_elems * sizeof(float));
            q.memset(d_cnt, 0, T * sizeof(uint32_t)).wait();

            strata::kernels::hc_mix_split(d_h, d_fn, hcdim, n_mix, T, eps, d_mixes_fused, d_cnt,
                                          d_scale, d_base, hc, iters, hc_eps, d_pre, d_post, d_comb, q);
            q.wait();

            std::vector<float> fused_mixes(mixes_elems);
            std::vector<float> fused_pre(pre_elems);
            std::vector<float> fused_post(post_elems);
            std::vector<float> fused_comb(comb_elems);
            std::vector<uint32_t> out_cnt(T);

            q.memcpy(fused_mixes.data(), d_mixes_fused, mixes_elems * sizeof(float));
            q.memcpy(fused_pre.data(), d_pre, pre_elems * sizeof(float));
            q.memcpy(fused_post.data(), d_post, post_elems * sizeof(float));
            q.memcpy(fused_comb.data(), d_comb, comb_elems * sizeof(float));
            q.memcpy(out_cnt.data(), d_cnt, T * sizeof(uint32_t)).wait();

            float fused_diff_mix = 0.0f;
            for (size_t i = 0; i < mixes_elems; ++i) fused_diff_mix = std::max(fused_diff_mix, std::abs(fused_mixes[i] - ref_mixes[i]));

            float fused_diff_pre = 0.0f;
            for (size_t i = 0; i < pre_elems; ++i) fused_diff_pre = std::max(fused_diff_pre, std::abs(fused_pre[i] - ref_pre[i]));

            float fused_diff_post = 0.0f;
            for (size_t i = 0; i < post_elems; ++i) fused_diff_post = std::max(fused_diff_post, std::abs(fused_post[i] - ref_post[i]));

            float fused_diff_comb = 0.0f;
            for (size_t i = 0; i < comb_elems; ++i) fused_diff_comb = std::max(fused_diff_comb, std::abs(fused_comb[i] - ref_comb[i]));

            // Verify atomic counter reset to zero
            bool cnt_reset_ok = true;
            for (int t = 0; t < T; ++t) {
                if (out_cnt[t] != 0u) cnt_reset_ok = false;
            }

            std::printf("   Separate: max_diff(mix)=%.2e, max_diff(pre)=%.2e, max_diff(post)=%.2e, max_diff(comb)=%.2e\n",
                        max_diff_mix, max_diff_pre, max_diff_post, max_diff_comb);
            std::printf("   Fused   : max_diff(mix)=%.2e, max_diff(pre)=%.2e, max_diff(post)=%.2e, max_diff(comb)=%.2e, cnt_reset=%s\n",
                        fused_diff_mix, fused_diff_pre, fused_diff_post, fused_diff_comb, cnt_reset_ok ? "OK" : "FAILED");

            const bool ok = (fused_diff_mix < 1e-4f) && (fused_diff_pre < 1e-4f) &&
                            (fused_diff_post < 1e-4f) && (fused_diff_comb < 1e-4f) && cnt_reset_ok;
            if (ok) {
                std::printf("   -> PASS\n\n");
                ++passed_tests;
            } else {
                std::printf("   -> FAIL\n\n");
            }

            sycl::free(d_h, q);
            sycl::free(d_fn, q);
            sycl::free(d_mixes_sep, q);
            sycl::free(d_mixes_fused, q);
            sycl::free(d_scale, q);
            sycl::free(d_base, q);
            sycl::free(d_pre, q);
            sycl::free(d_post, q);
            sycl::free(d_comb, q);
            sycl::free(d_cnt, q);
        }
    }

    std::printf("====================================================================================\n");
    std::printf(" Summary: %d / %d tests passed\n", passed_tests, total_tests);
    std::printf("====================================================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
