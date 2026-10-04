#include <sycl/sycl.hpp>
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::bf16_gemv_fp32_mmvf;
using strata::kernels::bf16_gemv_fp32_mmvf_multi;

namespace {

void cpu_reference(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out) {
    for (int64_t o = 0; o < n_out; ++o) {
        double sum = 0.0;
        const uint16_t* row = w + (size_t)o * n_in;
        for (int64_t i = 0; i < n_in; ++i) {
            sum += (double)f32_from_bf16(row[i]) * (double)x[i];
        }
        y[o] = (float)sum;
    }
}

} // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    printf("Running native_bf16 SYCL test on device: %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    struct Shape {
        int64_t n_in;
        int64_t n_out;
        const char* desc;
    };

    const Shape shapes[] = {
        {128, 32, "tiny [128, 32]"},
        {512, 64, "small [512, 64]"},
        {2560, 48, "ssm_alpha [2560, 48]"},
        {2560, 128, "indexer.k_proj [2560, 128]"},
        {2560, 512, "indexer.q_proj [2560, 512]"},
        {2560, 2560, "ple_value [2560, 2560]"}
    };

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int total_bad = 0;

    for (const auto& s : shapes) {
        std::vector<float> h_x(s.n_in);
        std::vector<uint16_t> h_w(s.n_in * s.n_out);
        std::vector<float> h_y_ref(s.n_out);
        std::vector<float> h_y_dev(s.n_out);

        for (auto& val : h_x) val = dist(rng);
        for (auto& val : h_w) val = bf16_from_f32(dist(rng) * 0.05f);

        cpu_reference(h_x.data(), h_w.data(), h_y_ref.data(), s.n_in, s.n_out);

        float* d_x = sycl::malloc_device<float>(s.n_in, q);
        uint16_t* d_w = sycl::malloc_device<uint16_t>(s.n_in * s.n_out, q);
        float* d_y = sycl::malloc_device<float>(s.n_out, q);

        q.memcpy(d_x, h_x.data(), s.n_in * sizeof(float));
        q.memcpy(d_w, h_w.data(), s.n_in * s.n_out * sizeof(uint16_t));
        q.wait();

        // 1. Single-token test
        bf16_gemv_fp32_mmvf(d_x, d_w, d_y, s.n_in, s.n_out, &q);
        q.memcpy(h_y_dev.data(), d_y, s.n_out * sizeof(float)).wait();

        int bad_single = 0;
        float max_rel = 0.0f;
        for (int64_t o = 0; o < s.n_out; ++o) {
            float diff = std::fabs(h_y_dev[o] - h_y_ref[o]);
            float denom = std::max(std::fabs(h_y_ref[o]), 1e-6f);
            float rel = diff / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 5e-3f && diff > 1e-4f) bad_single++;
        }
        printf("  single %-25s: max_rel = %.3e, bad = %d / %ld\n",
               s.desc, max_rel, bad_single, (long)s.n_out);
        total_bad += bad_single;

        // 2. Multi-token test with exact bit-identical parity against single-token
        for (int n_tok : {1, 2, 4, 7, 8}) {
            int64_t ldx = s.n_in;
            int64_t ldy = s.n_out;

            std::vector<float> h_multi_x((size_t)n_tok * ldx);
            std::vector<float> h_multi_y((size_t)n_tok * ldy);
            std::vector<float> h_single_ys((size_t)n_tok * ldy);

            for (auto& val : h_multi_x) val = dist(rng);

            float* d_multi_x = sycl::malloc_device<float>((size_t)n_tok * ldx, q);
            float* d_multi_y = sycl::malloc_device<float>((size_t)n_tok * ldy, q);
            float* d_single_y = sycl::malloc_device<float>(s.n_out, q);

            q.memcpy(d_multi_x, h_multi_x.data(), (size_t)n_tok * ldx * sizeof(float)).wait();

            // Run multi-token launch
            bf16_gemv_fp32_mmvf_multi(d_multi_x, ldx, d_w, d_multi_y, ldy, s.n_in, s.n_out, n_tok, &q);
            q.memcpy(h_multi_y.data(), d_multi_y, (size_t)n_tok * ldy * sizeof(float)).wait();

            // Run each token individually
            int bit_mismatches = 0;
            for (int t = 0; t < n_tok; ++t) {
                const float* x_ptr = d_multi_x + (size_t)t * ldx;
                bf16_gemv_fp32_mmvf(x_ptr, d_w, d_single_y, s.n_in, s.n_out, &q);
                q.memcpy(h_single_ys.data() + (size_t)t * ldy, d_single_y, s.n_out * sizeof(float)).wait();

                for (int64_t o = 0; o < s.n_out; ++o) {
                    float m_val = h_multi_y[(size_t)t * ldy + o];
                    float s_val = h_single_ys[(size_t)t * ldy + o];
                    uint32_t m_bits, s_bits;
                    std::memcpy(&m_bits, &m_val, 4);
                    std::memcpy(&s_bits, &s_val, 4);
                    if (m_bits != s_bits) {
                        bit_mismatches++;
                    }
                }
            }
            if (bit_mismatches > 0) {
                printf("  FAIL: multi (n_tok=%d) vs single has %d bit mismatches!\n", n_tok, bit_mismatches);
                total_bad += bit_mismatches;
            }

            sycl::free(d_multi_x, q);
            sycl::free(d_multi_y, q);
            sycl::free(d_single_y, q);
        }

        sycl::free(d_x, q);
        sycl::free(d_w, q);
        sycl::free(d_y, q);
    }

    if (total_bad == 0) {
        printf("native_bf16 SYCL parity OK (0 failures)\n");
        return 0;
    } else {
        printf("native_bf16 SYCL parity FAILED (%d bad)\n", total_bad);
        return 1;
    }
}
