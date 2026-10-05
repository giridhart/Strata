#include <sycl/sycl.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/fused_gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running verify_kernels_test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int total_failures = 0;

    // 1. broadcast_streams test
    {
        constexpr int64_t n_embd = 2560;
        constexpr int hc = 4;
        constexpr int n_tok = 3;
        std::vector<float> h_x(n_tok * n_embd);
        for (auto& v : h_x)
            v = dist(rng);

        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        float* d_R = sycl::malloc_device<float>(n_tok * hc * n_embd, q);
        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float)).wait();

        strata::kernels::broadcast_streams(d_x, d_R, n_embd, hc, n_tok, &q);
        q.wait();

        std::vector<float> h_R(n_tok * hc * n_embd);
        q.memcpy(h_R.data(), d_R, h_R.size() * sizeof(float)).wait();

        int errs = 0;
        for (int t = 0; t < n_tok; ++t) {
            for (int c = 0; c < hc; ++c) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    float expected = h_x[t * n_embd + i];
                    float got = h_R[t * hc * n_embd + c * n_embd + i];
                    if (expected != got) errs++;
                }
            }
        }
        std::printf("broadcast_streams: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_x, q);
        sycl::free(d_R, q);
    }

    // 2. add_streams_broadcast test
    {
        constexpr int64_t n_embd = 2560;
        constexpr int hc = 4;
        constexpr int n_tok = 2;
        std::vector<float> h_h(n_tok * hc * n_embd);
        std::vector<float> h_e(n_tok * n_embd);
        for (auto& v : h_h)
            v = dist(rng);
        for (auto& v : h_e)
            v = dist(rng);

        float* d_h = sycl::malloc_device<float>(h_h.size(), q);
        float* d_e = sycl::malloc_device<float>(h_e.size(), q);
        float* d_R = sycl::malloc_device<float>(h_h.size(), q);
        q.memcpy(d_h, h_h.data(), h_h.size() * sizeof(float)).wait();
        q.memcpy(d_e, h_e.data(), h_e.size() * sizeof(float)).wait();

        strata::kernels::add_streams_broadcast(d_h, d_e, d_R, n_embd, hc, n_tok, &q);
        q.wait();

        std::vector<float> h_R(h_h.size());
        q.memcpy(h_R.data(), d_R, h_R.size() * sizeof(float)).wait();

        int errs = 0;
        for (int t = 0; t < n_tok; ++t) {
            for (int c = 0; c < hc; ++c) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    float expected = h_h[t * hc * n_embd + c * n_embd + i] + h_e[t * n_embd + i];
                    float got = h_R[t * hc * n_embd + c * n_embd + i];
                    if (expected != got) errs++;
                }
            }
        }
        std::printf("add_streams_broadcast: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_h, q);
        sycl::free(d_e, q);
        sycl::free(d_R, q);
    }

    // 3. copy_indexed test
    {
        constexpr int64_t stride = 512;
        constexpr int64_t n = 256;
        constexpr int rows = 8;
        std::vector<float> h_src(rows * stride);
        for (auto& v : h_src)
            v = dist(rng);
        int32_t index_val = 3;

        float* d_src = sycl::malloc_device<float>(h_src.size(), q);
        float* d_dst = sycl::malloc_device<float>(n, q);
        int32_t* d_idx = sycl::malloc_device<int32_t>(1, q);

        q.memcpy(d_src, h_src.data(), h_src.size() * sizeof(float)).wait();
        q.memcpy(d_idx, &index_val, sizeof(int32_t)).wait();

        strata::kernels::copy_indexed(d_dst, d_src, stride, d_idx, n, &q);
        q.wait();

        std::vector<float> h_dst(n);
        q.memcpy(h_dst.data(), d_dst, n * sizeof(float)).wait();

        int errs = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (h_dst[i] != h_src[index_val * stride + i]) errs++;
        }
        std::printf("copy_indexed: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_src, q);
        sycl::free(d_dst, q);
        sycl::free(d_idx, q);
    }

    // 4. ident_hits test
    {
        constexpr int n = 64;
        std::vector<int32_t> h_ids(n);
        for (int i = 0; i < n; ++i)
            h_ids[i] = i * 2;

        int32_t* d_ids = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_slot = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_dst = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_count = sycl::malloc_device<int32_t>(1, q);

        q.memcpy(d_ids, h_ids.data(), n * sizeof(int32_t)).wait();

        strata::kernels::ident_hits(d_ids, n, d_slot, d_dst, d_count, &q);
        q.wait();

        std::vector<int32_t> h_slot(n), h_dst(n);
        int32_t count = 0;
        q.memcpy(h_slot.data(), d_slot, n * sizeof(int32_t)).wait();
        q.memcpy(h_dst.data(), d_dst, n * sizeof(int32_t)).wait();
        q.memcpy(&count, d_count, sizeof(int32_t)).wait();

        int errs = (count == n) ? 0 : 1;
        for (int i = 0; i < n; ++i) {
            if (h_slot[i] != h_ids[i]) errs++;
            if (h_dst[i] != i) errs++;
        }
        std::printf("ident_hits: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_ids, q);
        sycl::free(d_slot, q);
        sycl::free(d_dst, q);
        sycl::free(d_count, q);
    }

    // 5. dense_steps test
    {
        constexpr int n = 8;
        std::vector<int32_t> h_cells = {0, 3, 7, 15, 20, 25, 31, 40};
        int32_t* d_cells = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_steps = sycl::malloc_device<int32_t>(n * 4, q);
        q.memcpy(d_cells, h_cells.data(), n * sizeof(int32_t)).wait();

        strata::kernels::dense_steps(d_cells, n, d_steps, &q);
        q.wait();

        std::vector<int32_t> h_steps(n * 4);
        q.memcpy(h_steps.data(), d_steps, n * 4 * sizeof(int32_t)).wait();

        int errs = 0;
        for (int i = 0; i < n; ++i) {
            int c = h_cells[i];
            if (h_steps[i * 4 + 0] != c) errs++;
            if (h_steps[i * 4 + 1] != c + 1) errs++;
            if (h_steps[i * 4 + 2] != (c + 1) / 4) errs++;
            if (h_steps[i * 4 + 3] != c + 1) errs++;
        }
        std::printf("dense_steps: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_cells, q);
        sycl::free(d_steps, q);
    }

    // 6. gdn_conv_l2_multi vs fused_gdn_conv_l2 parity (1 token)
    {
        constexpr int channels = 2048;
        constexpr int qk_heads = 4;
        constexpr float eps = 1e-5f;

        std::vector<float> h_hist(channels * 3);
        std::vector<float> h_qkv(channels);
        std::vector<float> h_w(channels * 4);
        for (auto& v : h_hist)
            v = dist(rng);
        for (auto& v : h_qkv)
            v = dist(rng);
        for (auto& v : h_w)
            v = dist(rng);

        float* d_hist1 = sycl::malloc_device<float>(h_hist.size(), q);
        float* d_hist2 = sycl::malloc_device<float>(h_hist.size(), q);
        float* d_qkv = sycl::malloc_device<float>(channels, q);
        float* d_w = sycl::malloc_device<float>(channels * 4, q);
        float* d_h_single = sycl::malloc_device<float>(channels, q);
        float* d_h_multi = sycl::malloc_device<float>(channels, q);

        q.memcpy(d_hist1, h_hist.data(), h_hist.size() * sizeof(float)).wait();
        q.memcpy(d_hist2, h_hist.data(), h_hist.size() * sizeof(float)).wait();
        q.memcpy(d_qkv, h_qkv.data(), channels * sizeof(float)).wait();
        q.memcpy(d_w, h_w.data(), channels * 4 * sizeof(float)).wait();

        // Run single-token oracle
        strata::kernels::fused_gdn_conv_l2(d_hist1, d_qkv, d_w, d_h_single, channels, qk_heads, eps, &q);

        // Run multi-token verify kernel (1 token)
        strata::kernels::gdn_conv_l2_multi(d_hist2, d_qkv, d_w, d_h_multi, channels, qk_heads, eps, 1, &q, 0);
        q.wait();

        std::vector<float> res_single(channels), res_multi(channels);
        q.memcpy(res_single.data(), d_h_single, channels * sizeof(float)).wait();
        q.memcpy(res_multi.data(), d_h_multi, channels * sizeof(float)).wait();

        int errs = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < channels; ++i) {
            float diff = std::fabs(res_single[i] - res_multi[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-6f) errs++;
        }
        std::printf("gdn_conv_l2_multi parity: %s (max_diff: %e, mismatches: %d)\n", errs == 0 ? "PASS" : "FAIL",
                    max_diff, errs);
        total_failures += errs;

        sycl::free(d_hist1, q);
        sycl::free(d_hist2, q);
        sycl::free(d_qkv, q);
        sycl::free(d_w, q);
        sycl::free(d_h_single, q);
        sycl::free(d_h_multi, q);
    }

    // 7. gdn_ab_multi vs fused_gdn_ab parity (1 token)
    {
        constexpr int n_embd = 2048;
        constexpr int h_v = 32;

        std::vector<float> h_x(n_embd);
        std::vector<uint16_t> h_wa(h_v * n_embd), h_wb(h_v * n_embd);
        std::vector<float> h_dt(h_v), h_ssm_a(h_v);
        for (auto& v : h_x)
            v = dist(rng);
        for (auto& v : h_wa)
            v = (uint16_t)(rng() & 0xFFFF);
        for (auto& v : h_wb)
            v = (uint16_t)(rng() & 0xFFFF);
        for (auto& v : h_dt)
            v = dist(rng);
        for (auto& v : h_ssm_a)
            v = -std::fabs(dist(rng));

        float* d_x = sycl::malloc_device<float>(n_embd, q);
        uint16_t* d_wa = sycl::malloc_device<uint16_t>(h_wa.size(), q);
        uint16_t* d_wb = sycl::malloc_device<uint16_t>(h_wb.size(), q);
        float* d_dt = sycl::malloc_device<float>(h_v, q);
        float* d_ssm_a = sycl::malloc_device<float>(h_v, q);

        float* d_gate_s = sycl::malloc_device<float>(h_v, q);
        float* d_beta_s = sycl::malloc_device<float>(h_v, q);
        float* d_gate_m = sycl::malloc_device<float>(h_v, q);
        float* d_beta_m = sycl::malloc_device<float>(h_v, q);

        q.memcpy(d_x, h_x.data(), n_embd * sizeof(float)).wait();
        q.memcpy(d_wa, h_wa.data(), h_wa.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_wb, h_wb.data(), h_wb.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_dt, h_dt.data(), h_v * sizeof(float)).wait();
        q.memcpy(d_ssm_a, h_ssm_a.data(), h_v * sizeof(float)).wait();

        // Run single-token oracle
        strata::kernels::fused_gdn_ab(d_x, d_wa, d_wb, d_dt, d_ssm_a, d_gate_s, d_beta_s, n_embd, h_v, &q);

        // Run multi-token verify kernel (1 token)
        strata::kernels::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_ssm_a, d_gate_m, d_beta_m, n_embd, h_v, 1, &q);
        q.wait();

        std::vector<float> gate_s(h_v), gate_m(h_v), beta_s(h_v), beta_m(h_v);
        q.memcpy(gate_s.data(), d_gate_s, h_v * sizeof(float)).wait();
        q.memcpy(gate_m.data(), d_gate_m, h_v * sizeof(float)).wait();
        q.memcpy(beta_s.data(), d_beta_s, h_v * sizeof(float)).wait();
        q.memcpy(beta_m.data(), d_beta_m, h_v * sizeof(float)).wait();

        int errs = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < h_v; ++i) {
            float dg = std::fabs(gate_s[i] - gate_m[i]);
            float db = std::fabs(beta_s[i] - beta_m[i]);
            max_diff = std::max(max_diff, std::max(dg, db));
            if (dg > 1e-6f || db > 1e-6f) errs++;
        }
        std::printf("gdn_ab_multi parity: %s (max_diff: %e, mismatches: %d)\n", errs == 0 ? "PASS" : "FAIL", max_diff,
                    errs);
        total_failures += errs;

        sycl::free(d_x, q);
        sycl::free(d_wa, q);
        sycl::free(d_wb, q);
        sycl::free(d_dt, q);
        sycl::free(d_ssm_a, q);
        sycl::free(d_gate_s, q);
        sycl::free(d_beta_s, q);
        sycl::free(d_gate_m, q);
        sycl::free(d_beta_m, q);
    }

    std::printf("\nverify_kernels_test: %s (total failures: %d)\n", total_failures == 0 ? "PASSED" : "FAILED",
                total_failures);
    return total_failures == 0 ? 0 : 1;
}
