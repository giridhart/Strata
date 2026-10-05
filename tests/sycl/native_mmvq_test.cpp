#include <sycl/sycl.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
using strata::kernels::native_q4_0_f32;
using strata::kernels::native_q8_0_f32;
using strata::kernels::native_q8_1_bytes;
using strata::kernels::native_quantize_q8_1;

struct Q80TestBlock {
    uint16_t d;
    int8_t qs[32];
};

struct Q40TestBlock {
    uint16_t d;
    uint8_t qs[16];
};

struct Q81TestBlock {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
};

static std::vector<float> dequantize_q8_1(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t offset = 0; offset < x.size(); offset += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(x[offset + i]));
        if (amax == 0.0f) {
            std::fill(result.begin() + offset, result.begin() + offset + 32, 0.0f);
            continue;
        }
        const float scale_f32 = amax / 127.0f;
        const float scale_f16 = f32_from_f16(f16_from_f32(scale_f32));
        for (int i = 0; i < 32; ++i) {
            const int q = static_cast<int>(std::round(x[offset + i] / scale_f32));
            result[offset + i] = static_cast<float>(static_cast<int8_t>(q)) * scale_f16;
        }
    }
    return result;
}

static std::vector<Q81TestBlock> cpu_quantize_q8_1(const std::vector<float>& x) {
    std::vector<Q81TestBlock> result(x.size() / 32);
    for (size_t b = 0; b < x.size() / 32; ++b) {
        float amax = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(x[b * 32 + i]));
            sum += x[b * 32 + i];
        }
        float d = amax / 127.0f;
        result[b].d = f16_from_f32(d);
        result[b].s = f16_from_f32(sum);
        for (int i = 0; i < 32; ++i) {
            int q = amax == 0.0f ? 0 : static_cast<int>(std::round(x[b * 32 + i] / d));
            result[b].qs[i] = static_cast<int8_t>(q);
        }
    }
    return result;
}

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    printf("Running native_mmvq SYCL test on device: %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int total_bad = 0;

    // 1. Test native_quantize_q8_1
    {
        const int n_in = 2560;
        const int ncols = 2;
        std::vector<float> h_x(n_in * ncols);
        for (auto& val : h_x)
            val = dist(rng);

        size_t q8_bytes = native_q8_1_bytes(n_in, ncols);
        std::vector<Q81TestBlock> h_q8(q8_bytes / sizeof(Q81TestBlock));

        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        void* d_q8 = sycl::malloc_device<uint8_t>(q8_bytes, q);

        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float)).wait();
        native_quantize_q8_1(d_x, d_q8, n_in, ncols, &q);
        q.memcpy(h_q8.data(), d_q8, q8_bytes).wait();

        int quant_bad = 0;
        for (int col = 0; col < ncols; ++col) {
            for (int b = 0; b < n_in / 32; ++b) {
                const float* x_block = &h_x[col * n_in + b * 32];
                const auto& q_block = h_q8[col * (n_in / 32) + b];

                float amax = 0.0f;
                float sum = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    amax = std::max(amax, std::fabs(x_block[i]));
                    sum += x_block[i];
                }
                float ref_d = amax / 127.0f;
                float got_d = f32_from_f16(q_block.d);
                float got_s = f32_from_f16(q_block.s);

                if (std::fabs(got_d - ref_d) > 1e-3f && std::fabs(got_d - ref_d) / (ref_d + 1e-6f) > 0.05f) {
                    quant_bad++;
                }
                if (std::fabs(got_s - sum) > 0.1f && std::fabs(got_s - sum) / (std::fabs(sum) + 1e-6f) > 0.05f) {
                    quant_bad++;
                }
            }
        }
        printf("  quantize_q8_1: bad blocks = %d\n", quant_bad);
        total_bad += quant_bad;

        sycl::free(d_x, q);
        sycl::free(d_q8, q);
    }

    // 2. Test native_q8_0_f32
    {
        const int n_in = 256;
        const int n_out = 64;
        const int ncols = 1;

        std::vector<Q80TestBlock> h_w(n_out * (n_in / 32));
        std::vector<float> h_x(n_in);
        std::vector<float> ref_y(n_out, 0.0f);

        for (auto& val : h_x)
            val = dist(rng);
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < n_in / 32; ++b) {
                auto& blk = h_w[r * (n_in / 32) + b];
                float d_val = 0.01f + 0.01f * (rng() % 10);
                blk.d = f16_from_f32(d_val);
                for (int i = 0; i < 32; ++i) {
                    blk.qs[i] = (int8_t)((rng() % 255) - 127);
                }
            }
        }

        auto x_q8 = dequantize_q8_1(h_x);

        // CPU reference using dequantized Q8_1 activations
        for (int r = 0; r < n_out; ++r) {
            double sum = 0.0;
            for (int b = 0; b < n_in / 32; ++b) {
                const auto& blk = h_w[r * (n_in / 32) + b];
                float d = f32_from_f16(blk.d);
                for (int i = 0; i < 32; ++i) {
                    sum += (double)(d * (float)blk.qs[i]) * (double)x_q8[b * 32 + i];
                }
            }
            ref_y[r] = (float)sum;
        }

        void* d_w = sycl::malloc_device<uint8_t>(h_w.size() * sizeof(Q80TestBlock), q);
        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        float* d_y = sycl::malloc_device<float>(n_out, q);
        void* d_scratch = sycl::malloc_device<uint8_t>(native_q8_1_bytes(n_in, ncols), q);

        q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(Q80TestBlock));
        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float));
        q.wait();

        native_q8_0_f32(d_w, d_x, d_scratch, d_y, n_in, n_out, ncols, &q);

        std::vector<float> dev_y(n_out);
        q.memcpy(dev_y.data(), d_y, n_out * sizeof(float)).wait();

        int bad_q8_0 = 0;
        float max_rel = 0.0f;
        for (int r = 0; r < n_out; ++r) {
            float diff = std::fabs(dev_y[r] - ref_y[r]);
            float denom = std::max(std::fabs(ref_y[r]), 1e-4f);
            float rel = diff / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 0.02f && diff > 1e-3f) bad_q8_0++;
        }
        printf("  q8_0_f32 (single): max_rel = %.3e, bad = %d / %d\n", max_rel, bad_q8_0, n_out);
        total_bad += bad_q8_0;

        // 3. Multi-token exact parity check (ncols = 3)
        const int multi_cols = 3;
        std::vector<float> h_multi_x(n_in * multi_cols);
        for (auto& val : h_multi_x)
            val = dist(rng);

        float* d_multi_x = sycl::malloc_device<float>(h_multi_x.size(), q);
        float* d_multi_y = sycl::malloc_device<float>(n_out * multi_cols, q);
        void* d_multi_scratch = sycl::malloc_device<uint8_t>(native_q8_1_bytes(n_in, multi_cols), q);

        q.memcpy(d_multi_x, h_multi_x.data(), h_multi_x.size() * sizeof(float)).wait();

        native_q8_0_f32(d_w, d_multi_x, d_multi_scratch, d_multi_y, n_in, n_out, multi_cols, &q);

        std::vector<float> dev_multi_y(n_out * multi_cols);
        q.memcpy(dev_multi_y.data(), d_multi_y, dev_multi_y.size() * sizeof(float)).wait();

        int multi_mismatches = 0;
        for (int col = 0; col < multi_cols; ++col) {
            native_q8_0_f32(d_w, d_multi_x + col * n_in, d_scratch, d_y, n_in, n_out, 1, &q);
            std::vector<float> single_col_y(n_out);
            q.memcpy(single_col_y.data(), d_y, n_out * sizeof(float)).wait();

            for (int r = 0; r < n_out; ++r) {
                float mv = dev_multi_y[col * n_out + r];
                float sv = single_col_y[r];
                uint32_t mb, sb;
                std::memcpy(&mb, &mv, 4);
                std::memcpy(&sb, &sv, 4);
                if (mb != sb) multi_mismatches++;
            }
        }
        printf("  q8_0_f32 (multi_exact, ncols=3): bit mismatches = %d / %d\n", multi_mismatches, n_out * multi_cols);
        total_bad += multi_mismatches;

        sycl::free(d_w, q);
        sycl::free(d_x, q);
        sycl::free(d_y, q);
        sycl::free(d_scratch, q);
        sycl::free(d_multi_x, q);
        sycl::free(d_multi_y, q);
        sycl::free(d_multi_scratch, q);
    }

    // 4. Test native_q4_0_f32
    {
        const int n_in = 256;
        const int n_out = 64;
        const int ncols = 1;

        std::vector<Q40TestBlock> h_w(n_out * (n_in / 32));
        std::vector<float> h_x(n_in);
        std::vector<float> ref_y(n_out, 0.0f);

        for (auto& val : h_x)
            val = dist(rng);
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < n_in / 32; ++b) {
                auto& blk = h_w[r * (n_in / 32) + b];
                float d_val = 0.01f + 0.01f * (rng() % 10);
                blk.d = f16_from_f32(d_val);
                for (int i = 0; i < 16; ++i) {
                    blk.qs[i] = (uint8_t)(rng() % 256);
                }
            }
        }

        // CPU reference for Q4_0 using quantized Q8_1 activations
        auto q8_blocks = cpu_quantize_q8_1(h_x);
        for (int r = 0; r < n_out; ++r) {
            double sum = 0.0;
            for (int b = 0; b < n_in / 32; ++b) {
                const auto& blk = h_w[r * (n_in / 32) + b];
                const auto& qblk = q8_blocks[b];
                float d_w = f32_from_f16(blk.d);
                float d_x = f32_from_f16(qblk.d);
                float s_x = f32_from_f16(qblk.s);
                int sumi = 0;
                for (int i = 0; i < 16; ++i) {
                    int v0 = blk.qs[i] & 0x0f;
                    int v1 = blk.qs[i] >> 4;
                    sumi += v0 * qblk.qs[i];
                    sumi += v1 * qblk.qs[i + 16];
                }
                sum += (double)d_w * ((double)sumi * (double)d_x - 8.0 * (double)s_x);
            }
            ref_y[r] = (float)sum;
        }

        void* d_w = sycl::malloc_device<uint8_t>(h_w.size() * sizeof(Q40TestBlock), q);
        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        float* d_y = sycl::malloc_device<float>(n_out, q);
        void* d_scratch = sycl::malloc_device<uint8_t>(native_q8_1_bytes(n_in, ncols), q);

        q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(Q40TestBlock));
        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float));
        q.wait();

        native_q4_0_f32(d_w, d_x, d_scratch, d_y, n_in, n_out, ncols, &q);

        std::vector<float> dev_y(n_out);
        q.memcpy(dev_y.data(), d_y, n_out * sizeof(float)).wait();

        int bad_q4_0 = 0;
        float max_rel = 0.0f;
        for (int r = 0; r < n_out; ++r) {
            float diff = std::fabs(dev_y[r] - ref_y[r]);
            float denom = std::max(std::fabs(ref_y[r]), 1e-4f);
            float rel = diff / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 0.02f && diff > 1e-3f) bad_q4_0++;
        }
        printf("  q4_0_f32: max_rel = %.3e, bad = %d / %d\n", max_rel, bad_q4_0, n_out);
        total_bad += bad_q4_0;

        sycl::free(d_w, q);
        sycl::free(d_x, q);
        sycl::free(d_y, q);
        sycl::free(d_scratch, q);
    }

    if (total_bad == 0) {
        printf("native_mmvq SYCL parity OK (0 failures)\n");
        return 0;
    } else {
        printf("native_mmvq SYCL parity FAILED (%d bad)\n", total_bad);
        return 1;
    }
}
