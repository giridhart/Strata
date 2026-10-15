// tests/sycl/native_gr_norm_test.cpp - Standalone unit test for native GR RMSNorm on SYCL/Xe2.
#include <sycl/sycl.hpp>
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
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

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
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

    // Test fused_gr_quantize_q8_0 and fused_gr_dequant_q8_0
    {
        constexpr int64_t n_blocks = 128;
        constexpr size_t total_elems = (size_t)n_blocks * 32;
        std::vector<uint16_t> h_bf16(total_elems);
        for (size_t i = 0; i < total_elems; ++i) {
            h_bf16[i] = strata::kernels::bf16_from_f32(dist(rng));
        }

        // CPU reference quantization
        std::vector<uint8_t> ref_q8((size_t)n_blocks * 34);
        for (int64_t b = 0; b < n_blocks; ++b) {
            float amax = 0.0f;
            float vals[32];
            for (int l = 0; l < 32; ++l) {
                uint32_t u = (uint32_t)h_bf16[b * 32 + l] << 16;
                float v;
                std::memcpy(&v, &u, 4);
                vals[l] = v;
                amax = std::max(amax, std::fabs(v));
            }
            float d = amax / 127.0f;
            float id = d != 0.0f ? 1.0f / d : 0.0f;
            sycl::half h = (sycl::half)d;
            uint16_t hu;
            std::memcpy(&hu, &h, 2);
            uint8_t* o = ref_q8.data() + b * 34;
            o[0] = (uint8_t)(hu & 0xFF);
            o[1] = (uint8_t)(hu >> 8);
            for (int l = 0; l < 32; ++l) {
                float q_val = std::round(vals[l] * id);
                int q_clamped = std::max(-128, std::min(127, (int)q_val));
                o[2 + l] = (uint8_t)(int8_t)q_clamped;
            }
        }

        uint16_t* d_bf16 = sycl::malloc_device<uint16_t>(total_elems, q);
        uint8_t* d_q8 = sycl::malloc_device<uint8_t>((size_t)n_blocks * 34, q);
        uint16_t* d_dequant = sycl::malloc_device<uint16_t>(total_elems, q);

        q.memcpy(d_bf16, h_bf16.data(), total_elems * sizeof(uint16_t)).wait();

        strata::kernels::fused_gr_quantize_q8_0(d_bf16, d_q8, n_blocks, &q);
        q.wait();

        std::vector<uint8_t> act_q8((size_t)n_blocks * 34);
        q.memcpy(act_q8.data(), d_q8, act_q8.size()).wait();

        int bad_quant = 0;
        for (size_t i = 0; i < act_q8.size(); ++i) {
            if (act_q8[i] != ref_q8[i]) {
                size_t l = i % 34;
                if (l < 2) {
                    bad_quant++; // scale must be bit-identical
                } else {
                    int diff = std::abs((int)(int8_t)act_q8[i] - (int)(int8_t)ref_q8[i]);
                    if (diff > 1) bad_quant++;
                }
            }
        }
        printf("  fused_gr_quantize_q8_0: bad = %d / %zu\n", bad_quant, act_q8.size());
        total_failures += bad_quant;

        // Test dequant
        strata::kernels::fused_gr_dequant_q8_0(d_q8, d_dequant, n_blocks, &q);
        q.wait();

        std::vector<uint16_t> act_dequant(total_elems);
        q.memcpy(act_dequant.data(), d_dequant, total_elems * sizeof(uint16_t)).wait();

        // CPU reference dequant
        int bad_dequant = 0;
        for (int64_t b = 0; b < n_blocks; ++b) {
            const uint8_t* p = act_q8.data() + b * 34;
            uint16_t hu = (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
            sycl::half h;
            std::memcpy(&h, &hu, 2);
            float d = (float)h;
            for (int l = 0; l < 32; ++l) {
                float expected = d * (float)(int8_t)p[2 + l];
                uint16_t exp_bf16 = strata::kernels::bf16_from_f32(expected);
                if (act_dequant[b * 32 + l] != exp_bf16) bad_dequant++;
            }
        }
        printf("  fused_gr_dequant_q8_0: bad = %d / %zu\n", bad_dequant, total_elems);
        total_failures += bad_dequant;

        sycl::free(d_bf16, q);
        sycl::free(d_q8, q);
        sycl::free(d_dequant, q);
    }

    if (total_failures != 0) {
        printf("native_gr_norm SYCL test FAILED with %d errors\n", total_failures);
        return 1;
    }
    printf("native_gr_norm SYCL test OK (0 failures)\n");

    if (bench) {
        printf("\n--- Native GR Norm & Fused Q8_0 Microbenchmarks ---\n");
        constexpr int reps = 100;
        constexpr int n_cols = 10240, n_rows = 16;
        constexpr size_t total_elems = (size_t)n_cols * n_rows;
        constexpr int64_t n_blocks = (int64_t)total_elems / 32;

        float* d_in = sycl::malloc_device<float>(total_elems, q);
        float* d_gamma = sycl::malloc_device<float>(total_elems, q);
        float* d_out = sycl::malloc_device<float>(total_elems, q);
        uint16_t* d_bf16 = sycl::malloc_device<uint16_t>(total_elems, q);
        uint8_t* d_q8 = sycl::malloc_device<uint8_t>((size_t)n_blocks * 34, q);
        uint16_t* d_dequant = sycl::malloc_device<uint16_t>(total_elems, q);

        q.fill(d_in, 0.5f, total_elems);
        q.fill(d_gamma, 1.0f, total_elems);
        q.fill(d_bf16, (uint16_t)0x3F80, total_elems);
        q.wait();

        // 1. RMSNorm
        for (int i = 0; i < 5; ++i) {
            strata::kernels::native_gr_rms_norm_weighted(d_in, d_gamma, d_out, n_cols, n_rows, eps, &q);
        }
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            strata::kernels::native_gr_rms_norm_weighted(d_in, d_gamma, d_out, n_cols, n_rows, eps, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        double us_norm = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        double bytes_norm = (double)total_elems * sizeof(float) * 3; // in, gamma, out
        printf("BENCH RMSNorm [%5d x %2d]  : %6.2f us (%6.1f GB/s)\n", n_cols, n_rows, us_norm, bytes_norm / us_norm / 1e3);

        // 2. Fused GR Quantize Q8_0
        for (int i = 0; i < 5; ++i) {
            strata::kernels::fused_gr_quantize_q8_0(d_bf16, d_q8, n_blocks, &q);
        }
        q.wait();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            strata::kernels::fused_gr_quantize_q8_0(d_bf16, d_q8, n_blocks, &q);
        }
        q.wait();
        t1 = std::chrono::steady_clock::now();
        double us_quant = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        double bytes_quant = (double)total_elems * sizeof(uint16_t) + (double)n_blocks * 34;
        printf("BENCH Quantize Q8_0 [%6lld blk]: %6.2f us (%6.1f GB/s)\n", (long long)n_blocks, us_quant, bytes_quant / us_quant / 1e3);

        // 3. Fused GR Dequant Q8_0
        for (int i = 0; i < 5; ++i) {
            strata::kernels::fused_gr_dequant_q8_0(d_q8, d_dequant, n_blocks, &q);
        }
        q.wait();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            strata::kernels::fused_gr_dequant_q8_0(d_q8, d_dequant, n_blocks, &q);
        }
        q.wait();
        t1 = std::chrono::steady_clock::now();
        double us_dequant = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        double bytes_dequant = (double)total_elems * sizeof(uint16_t) + (double)n_blocks * 34;
        printf("BENCH Dequant Q8_0  [%6lld blk]: %6.2f us (%6.1f GB/s)\n", (long long)n_blocks, us_dequant, bytes_dequant / us_dequant / 1e3);

        sycl::free(d_in, q);
        sycl::free(d_gamma, q);
        sycl::free(d_out, q);
        sycl::free(d_bf16, q);
        sycl::free(d_q8, q);
        sycl::free(d_dequant, q);
    }
    return 0;
}
