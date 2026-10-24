// tests/core/sycl/finish_test.cpp - Unit and regression test for Fused Finish Kernel on SYCL/Xe2
//
// Verifies numerical equivalence between unfused/CPU reference and the fused finish kernel:
//   a) RMSNorm only
//   b) RMSNorm + RoPE
//   c) RMSNorm + RoPE + FP8 Quantization / BF16 writeback
// Across vector lengths (128, 256, 512, 1024) and head counts (8, 16, 32).

#include <sycl/sycl.hpp>
#include "strata/kernels/finish.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>

namespace {

// Reference CPU RMSNorm
void cpu_rmsnorm(const float* x, const float* w, float* out, int n_vec, int len, float eps) {
    for (int v = 0; v < n_vec; ++v) {
        const float* iv = x + (size_t)v * len;
        float* ov = out + (size_t)v * len;
        double ss = 0.0;
        for (int i = 0; i < len; ++i) {
            ss += (double)iv[i] * (double)iv[i];
        }
        const float inv = 1.0f / (float)std::sqrt(ss / (double)len + (double)eps);
        for (int i = 0; i < len; ++i) {
            ov[i] = (w != nullptr) ? (iv[i] * inv * w[i]) : (iv[i] * inv);
        }
    }
}

// Reference CPU adjacent RoPE rotation
void cpu_rope(const float* in, float* out, int n_vec, int len, int rd, const float* freqs,
              int pos_base, int vpp = 1, int pos_step = 1, int inverse = 0) {
    for (int v = 0; v < n_vec; ++v) {
        const float* iv = in + (size_t)v * len;
        float* ov = out + (size_t)v * len;
        for (int i = 0; i < len; ++i) {
            ov[i] = iv[i];
        }
        if (rd <= 0 || freqs == nullptr) continue;
        const int r0 = len - rd;
        const int pos = pos_base + (int)(v / vpp) * pos_step;
        for (int i = 0; i < rd / 2; ++i) {
            const int e0 = r0 + 2 * i;
            const int e1 = e0 + 1;
            const float ang = (float)pos * freqs[i];
            const float c = std::cos(ang);
            const float sn = inverse ? -std::sin(ang) : std::sin(ang);
            const float re = iv[e0] * c - iv[e1] * sn;
            const float im = iv[e0] * sn + iv[e1] * c;
            ov[e0] = re;
            ov[e1] = im;
        }
    }
}

// Reference CPU FP8 E4M3 packed quantization (codes + group scales)
void cpu_quant_fp8(const float* in, uint8_t* pdst, int n_vec, int len, long long pdst_stride) {
    constexpr int g = 32;
    for (int v = 0; v < n_vec; ++v) {
        const float* iv = in + (size_t)v * len;
        uint8_t* pr = pdst + (size_t)v * pdst_stride;

        for (int grp = 0; grp < len / g; ++grp) {
            float a = 0.0f;
            for (int k = 0; k < g; ++k) {
                a = std::max(a, std::fabs(iv[grp * g + k]));
            }
            const float s = strata::kernels::pow2_ceil(std::max(a, 1e-4f) * (1.0f / 448.0f));
            for (int k = 0; k < g; ++k) {
                const int e = grp * g + k;
                const float q = strata::kernels::fp8_rt(std::clamp(iv[e] / s, -448.0f, 448.0f));
                pr[e] = (uint8_t)strata::kernels::e4m3_code(q);
            }
            pr[len + grp] = (uint8_t)(sycl::bit_cast<uint32_t>(s) >> 23);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
    std::printf("====================================================================================\n");
    std::printf(" Strata SYCL Xe2 Fused Finish Kernel Unit & Regression Test\n");
    std::printf(" Device: %s\n", dev_name.c_str());
    std::printf("====================================================================================\n\n");

    const std::vector<int> lens = {128, 256, 512, 1024};
    const std::vector<int> n_vecs = {8, 16, 32};
    const float eps = 1e-6f;
    const int pos = 42;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_x(-2.5f, 2.5f);
    std::uniform_real_distribution<float> dist_w(0.7f, 1.3f);

    int total_cases = 0;
    int passed_cases = 0;

    for (const int len : lens) {
        const int rd = (len >= 128) ? 128 : len;
        const int num_freqs = rd / 2;
        std::vector<float> h_freqs((size_t)num_freqs);
        for (int i = 0; i < num_freqs; ++i) {
            h_freqs[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)rd);
        }

        std::vector<float> h_w((size_t)len);
        for (auto& v : h_w) v = dist_w(rng);

        float* d_w = sycl::malloc_device<float>((size_t)len, q);
        float* d_freqs = sycl::malloc_device<float>((size_t)num_freqs, q);
        q.memcpy(d_w, h_w.data(), (size_t)len * sizeof(float)).wait();
        q.memcpy(d_freqs, h_freqs.data(), (size_t)num_freqs * sizeof(float)).wait();

        for (const int n_vec : n_vecs) {
            const size_t total_elems = (size_t)n_vec * (size_t)len;
            const long long stride = len;
            const long long pdst_stride = (long long)strata::kernels::finish_kv_row_bytes(strata::kernels::kKvFp8, len);
            const size_t pdst_bytes = (size_t)n_vec * (size_t)pdst_stride;

            std::vector<float> h_x(total_elems);
            for (auto& v : h_x) v = dist_x(rng);

            // Compute CPU references
            std::vector<float> h_ref_norm(total_elems);
            cpu_rmsnorm(h_x.data(), h_w.data(), h_ref_norm.data(), n_vec, len, eps);

            std::vector<float> h_ref_rope(total_elems);
            cpu_rope(h_ref_norm.data(), h_ref_rope.data(), n_vec, len, rd, h_freqs.data(), pos);

            std::vector<uint16_t> h_ref_bf16(total_elems);
            for (size_t i = 0; i < total_elems; ++i) {
                h_ref_bf16[i] = strata::kernels::bf16_from_f32(h_ref_rope[i]);
            }

            std::vector<uint8_t> h_ref_fp8(pdst_bytes, 0);
            cpu_quant_fp8(h_ref_rope.data(), h_ref_fp8.data(), n_vec, len, pdst_stride);

            // Device buffers
            float* d_x = sycl::malloc_device<float>(total_elems, q);
            float* d_out = sycl::malloc_device<float>(total_elems, q);
            uint16_t* d_dst = sycl::malloc_device<uint16_t>(total_elems, q);
            uint8_t* d_pdst = sycl::malloc_device<uint8_t>(pdst_bytes, q);

            // ----------------------------------------------------------------
            // Test a) RMSNorm only
            // ----------------------------------------------------------------
            total_cases++;
            q.memcpy(d_x, h_x.data(), total_elems * sizeof(float)).wait();
            strata::kernels::finish_norm_rope(d_x, d_out, n_vec, len, stride, d_w, eps,
                                              /*rd=*/0, nullptr, 0, 0, 1, 1, stride, &q);
            q.wait();

            std::vector<float> h_dev_norm(total_elems);
            q.memcpy(h_dev_norm.data(), d_out, total_elems * sizeof(float)).wait();

            float max_norm_diff = 0.0f;
            for (size_t i = 0; i < total_elems; ++i) {
                const float diff = std::abs(h_dev_norm[i] - h_ref_norm[i]);
                if (diff > max_norm_diff) max_norm_diff = diff;
            }

            bool ok_norm = (max_norm_diff <= 1e-4f);
            if (ok_norm) passed_cases++;
            std::printf("[%s] Shape (%2d x %4d) | Test A: RMSNorm only           | Max diff: %.2e\n",
                        ok_norm ? "PASS" : "FAIL", n_vec, len, max_norm_diff);

            // ----------------------------------------------------------------
            // Test b) RMSNorm + RoPE
            // ----------------------------------------------------------------
            total_cases++;
            q.memcpy(d_x, h_x.data(), total_elems * sizeof(float)).wait();
            strata::kernels::finish_norm_rope(d_x, d_out, n_vec, len, stride, d_w, eps,
                                              rd, d_freqs, pos, 0, 1, 1, stride, &q);
            q.wait();

            std::vector<float> h_dev_rope(total_elems);
            q.memcpy(h_dev_rope.data(), d_out, total_elems * sizeof(float)).wait();

            float max_rope_diff = 0.0f;
            for (size_t i = 0; i < total_elems; ++i) {
                const float diff = std::abs(h_dev_rope[i] - h_ref_rope[i]);
                if (diff > max_rope_diff) max_rope_diff = diff;
            }

            bool ok_rope = (max_rope_diff <= 1e-4f);
            if (ok_rope) passed_cases++;
            std::printf("[%s] Shape (%2d x %4d) | Test B: RMSNorm + RoPE         | Max diff: %.2e\n",
                        ok_rope ? "PASS" : "FAIL", n_vec, len, max_rope_diff);

            // ----------------------------------------------------------------
            // Test c1) RMSNorm + RoPE + BF16 Writeback
            // ----------------------------------------------------------------
            total_cases++;
            q.memcpy(d_x, h_x.data(), total_elems * sizeof(float)).wait();
            strata::kernels::finish_norm_rope(d_x, d_dst, n_vec, len, stride, stride, d_w, eps,
                                              rd, d_freqs, pos, 0, 1, 1, &q);
            q.wait();

            std::vector<uint16_t> h_dev_bf16(total_elems);
            q.memcpy(h_dev_bf16.data(), d_dst, total_elems * sizeof(uint16_t)).wait();

            float max_bf16_diff = 0.0f;
            int bf16_mismatches = 0;
            for (size_t i = 0; i < total_elems; ++i) {
                if (h_dev_bf16[i] != h_ref_bf16[i]) {
                    const float f_dev = strata::kernels::f32_from_bf16(h_dev_bf16[i]);
                    const float f_ref = strata::kernels::f32_from_bf16(h_ref_bf16[i]);
                    const float diff = std::abs(f_dev - f_ref);
                    if (diff > max_bf16_diff) max_bf16_diff = diff;
                    // Check if adjacent BF16 representation (1 ulp difference due to float summation order)
                    const int ulp_diff = std::abs((int)h_dev_bf16[i] - (int)h_ref_bf16[i]);
                    if (ulp_diff > 1) {
                        bf16_mismatches++;
                    }
                }
            }

            bool ok_bf16 = (bf16_mismatches == 0);
            if (ok_bf16) passed_cases++;
            std::printf("[%s] Shape (%2d x %4d) | Test C1: Norm + RoPE + BF16     | Max diff: %.2e (mismatches: %d)\n",
                        ok_bf16 ? "PASS" : "FAIL", n_vec, len, max_bf16_diff, bf16_mismatches);

            // ----------------------------------------------------------------
            // Test c2) RMSNorm + RoPE + FP8 Quantization packed writeback
            // ----------------------------------------------------------------
            total_cases++;
            q.memcpy(d_x, h_x.data(), total_elems * sizeof(float)).wait();
            q.memset(d_pdst, 0, pdst_bytes).wait();
            strata::kernels::finish_norm_rope_quant(d_x, d_pdst, n_vec, len, stride, pdst_stride,
                                                   d_w, eps, rd, d_freqs, pos, 0, 1, 1, &q);
            q.wait();

            std::vector<uint8_t> h_dev_fp8(pdst_bytes);
            q.memcpy(h_dev_fp8.data(), d_pdst, pdst_bytes).wait();

            int fp8_byte_diffs = 0;
            for (int v = 0; v < n_vec; ++v) {
                const size_t row_off = (size_t)v * pdst_stride;
                for (int e = 0; e < len; ++e) {
                    if (h_dev_fp8[row_off + e] != h_ref_fp8[row_off + e]) fp8_byte_diffs++;
                }
                for (int grp = 0; grp < len / 32; ++grp) {
                    if (h_dev_fp8[row_off + len + grp] != h_ref_fp8[row_off + len + grp]) fp8_byte_diffs++;
                }
            }

            bool ok_fp8 = (fp8_byte_diffs == 0);
            if (ok_fp8) passed_cases++;
            std::printf("[%s] Shape (%2d x %4d) | Test C2: Norm + RoPE + FP8 E4M3 | Byte mismatches: %d\n",
                        ok_fp8 ? "PASS" : "FAIL", n_vec, len, fp8_byte_diffs);

            sycl::free(d_x, q);
            sycl::free(d_out, q);
            sycl::free(d_dst, q);
            sycl::free(d_pdst, q);
        }

        sycl::free(d_w, q);
        sycl::free(d_freqs, q);
    }

    std::printf("\n====================================================================================\n");
    std::printf(" Result: %d / %d tests PASSED (%s)\n",
                passed_cases, total_cases, (passed_cases == total_cases) ? "ALL TESTS PASSED" : "FAILURES DETECTED");
    std::printf("====================================================================================\n");

    return (passed_cases == total_cases) ? 0 : 1;
}
