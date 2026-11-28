#include <sycl/sycl.hpp>
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/qsa.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using strata::kernels::kNativeFlashAttnSuccess;
using strata::kernels::kNativeFlashAttnUnsupportedStep;
using strata::kernels::kStepCount;
using strata::kernels::kStepNBid;
using strata::kernels::kStepNKv;
using strata::kernels::kStepPos;
using strata::kernels::kStepWidth;
using strata::kernels::native_flash_attn_short_step;
using strata::kernels::qsa_real_shapes;
using strata::kernels::QsaShapes;

namespace {

uint16_t float_to_half_bits(float f) {
    sycl::half h = sycl::half(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

float half_bits_to_float(uint16_t b) {
    sycl::half h = *reinterpret_cast<sycl::half*>(&b);
    return float(h);
}

void cpu_reference_attend(const float* q, const uint16_t* k, const uint16_t* v, int width, const uint16_t* mask,
                          float* out) {
    const float scale = 0.0625f;  // 1/16
    for (int h = 0; h < 24; ++h) {
        const int kv = h / 12;
        std::vector<float> scores(width);
        float max_s = -1e30f;
        for (int cell = 0; cell < width; ++cell) {
            float dot = 0.0f;
            for (int d = 0; d < 256; ++d) {
                float q_val = q[h * 256 + d] * scale;
                float k_val = half_bits_to_float(k[(cell * 2 + kv) * 256 + d]);
                dot += q_val * k_val;
            }
            if (mask) dot += half_bits_to_float(mask[cell]);
            scores[cell] = dot;
            if (dot > max_s) max_s = dot;
        }

        float sum_exp = 0.0f;
        std::vector<float> weights(width);
        for (int cell = 0; cell < width; ++cell) {
            weights[cell] = std::exp(scores[cell] - max_s);
            sum_exp += weights[cell];
        }

        for (int d = 0; d < 256; ++d) {
            float acc = 0.0f;
            for (int cell = 0; cell < width; ++cell) {
                float v_val = half_bits_to_float(v[(cell * 2 + kv) * 256 + d]);
                acc += (weights[cell] / sum_exp) * v_val;
            }
            out[h * 256 + d] = acc;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    sycl::queue q{sycl::property::queue::in_order()};
    printf("Running native_flash_attn SYCL test on device: %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    if (!q.get_device().has(sycl::aspect::fp16)) {
        printf("Device does not support FP16, skipping.\n");
        return 0;
    }

    QsaShapes shapes;
    shapes.n_head = 24;
    shapes.n_head_kv = 2;
    shapes.head_dim = 256;
    shapes.idx_block = 4;
    shapes.idx_top_k = 256;

    const int64_t capacity = 256;
    const int max_context = 256;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int total_bad = 0;

    for (int width : {4, 16, 64, 128, 256}) {
        std::vector<float> h_q(24 * 256);
        std::vector<uint16_t> h_k(capacity * 2 * 256);
        std::vector<uint16_t> h_v(capacity * 2 * 256);
        std::vector<uint16_t> h_mask(256);
        std::vector<int32_t> h_step(kStepCount);

        for (auto& val : h_q)
            val = dist(rng) * 0.5f;
        for (auto& val : h_k)
            val = float_to_half_bits(dist(rng) * 0.5f);
        for (auto& val : h_v)
            val = float_to_half_bits(dist(rng) * 0.5f);
        for (int i = 0; i < 256; ++i)
            h_mask[i] = float_to_half_bits(0.0f);

        h_step[kStepPos] = width - 1;
        h_step[kStepNKv] = width;
        h_step[kStepNBid] = width / 4;
        h_step[kStepWidth] = width;

        std::vector<float> ref_out(24 * 256);
        cpu_reference_attend(h_q.data(), h_k.data(), h_v.data(), width, h_mask.data(), ref_out.data());

        float* d_q = sycl::malloc_device<float>(h_q.size(), q);
        uint16_t* d_k = sycl::malloc_device<uint16_t>(h_k.size(), q);
        uint16_t* d_v = sycl::malloc_device<uint16_t>(h_v.size(), q);
        uint16_t* d_mask = sycl::malloc_device<uint16_t>(h_mask.size(), q);
        int32_t* d_step = sycl::malloc_device<int32_t>(h_step.size(), q);
        float* d_out = sycl::malloc_device<float>(24 * 256, q);
        int32_t* d_status = sycl::malloc_device<int32_t>(1, q);

        q.memcpy(d_q, h_q.data(), h_q.size() * sizeof(float));
        q.memcpy(d_k, h_k.data(), h_k.size() * sizeof(uint16_t));
        q.memcpy(d_v, h_v.data(), h_v.size() * sizeof(uint16_t));
        q.memcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(uint16_t));
        q.memcpy(d_step, h_step.data(), h_step.size() * sizeof(int32_t));
        q.wait();

        native_flash_attn_short_step(d_q, d_k, d_v, d_step, capacity, max_context, shapes, d_out, d_status, d_mask, &q);

        std::vector<float> dev_out(24 * 256);
        int32_t dev_status = -1;
        q.memcpy(dev_out.data(), d_out, dev_out.size() * sizeof(float));
        q.memcpy(&dev_status, d_status, sizeof(int32_t));
        q.wait();

        if (dev_status != kNativeFlashAttnSuccess) {
            printf("  width=%d: status FAILED (%d)\n", width, dev_status);
            total_bad++;
        }

        int bad_elements = 0;
        float max_rel = 0.0f;
        for (size_t i = 0; i < dev_out.size(); ++i) {
            float diff = std::fabs(dev_out[i] - ref_out[i]);
            float denom = std::max(std::fabs(ref_out[i]), 1e-4f);
            float rel = diff / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-2f && diff > 1e-3f) bad_elements++;
        }

        printf("  width=%3d: max_rel = %.3e, bad = %d / %zu\n", width, max_rel, bad_elements, dev_out.size());
        total_bad += bad_elements;

        sycl::free(d_q, q);
        sycl::free(d_k, q);
        sycl::free(d_v, q);
        sycl::free(d_mask, q);
        sycl::free(d_step, q);
        sycl::free(d_out, q);
        sycl::free(d_status, q);
    }

    if (total_bad != 0) {
        printf("native_flash_attn SYCL parity FAILED (%d bad)\n", total_bad);
        return 1;
    }
    printf("native_flash_attn SYCL parity OK (0 failures)\n");

    if (bench) {
        printf("\n--- Native FlashAttention Microbenchmarks ---\n");
        constexpr int reps = 100;
        constexpr int64_t capacity = 256;
        constexpr int max_context = 256;
        const QsaShapes shapes = qsa_real_shapes();

        for (int width : {64, 128, 192, 256}) {
            const size_t q_sz = 24 * 256;
            const size_t kv_sz = capacity * 2 * 256;
            std::vector<int32_t> h_step(kStepCount);
            h_step[kStepPos] = width - 1;
            h_step[kStepNKv] = width;
            h_step[kStepNBid] = width / 4;
            h_step[kStepWidth] = width;

            float* d_q = sycl::malloc_device<float>(q_sz, q);
            uint16_t* d_k = sycl::malloc_device<uint16_t>(kv_sz, q);
            uint16_t* d_v = sycl::malloc_device<uint16_t>(kv_sz, q);
            int32_t* d_step = sycl::malloc_device<int32_t>(kStepCount, q);
            float* d_out = sycl::malloc_device<float>(q_sz, q);
            int32_t* d_status = sycl::malloc_device<int32_t>(1, q);

            q.fill(d_q, 0.05f, q_sz);
            q.fill(d_k, (uint16_t)0x3C00, kv_sz);
            q.fill(d_v, (uint16_t)0x3C00, kv_sz);
            q.memcpy(d_step, h_step.data(), kStepCount * sizeof(int32_t));
            q.wait();

            // Warmup
            for (int i = 0; i < 5; ++i) {
                native_flash_attn_short_step(d_q, d_k, d_v, d_step, capacity, max_context, shapes, d_out, d_status, nullptr, &q);
            }
            q.wait();

            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                native_flash_attn_short_step(d_q, d_k, d_v, d_step, capacity, max_context, shapes, d_out, d_status, nullptr, &q);
            }
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
            double bytes = (double)(q_sz * sizeof(float) + (size_t)width * 2 * 256 * 2 * sizeof(uint16_t) + q_sz * sizeof(float));
            printf("BENCH FlashAttn width=%3d: %6.2f us (%7.1f tok/s, %5.1f GB/s)\n",
                   width, us, (1.0 / us) * 1e6, bytes / us / 1e3);

            sycl::free(d_q, q);
            sycl::free(d_k, q);
            sycl::free(d_v, q);
            sycl::free(d_step, q);
            sycl::free(d_out, q);
            sycl::free(d_status, q);
        }
    }
    return 0;
}
