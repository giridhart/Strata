// tests/sycl/subgroup_divergence_test.cpp - Test for subgroup-level reduction safety across
// SIMD widths (Xe2 SIMD16 vs SIMD32) and default stream synchronization idempotency.
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_failures = 0;

template <typename T>
T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 256, q);
    q.memset(p, 0, (n + 256) * sizeof(T)).wait();
    return p;
}

template <typename T>
T* upload(const std::vector<T>& h, sycl::queue& q) {
    T* d = dalloc<T>(h.size(), q);
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

// 1. Direct micro-kernel verifying subgroup reduction with partial subgroup occupancy (num_sgs < subgroup_size)
void test_partial_subgroup_reduction(sycl::queue& q) {
    std::printf("Testing subgroup reduce_over_group with partial lane occupancy... ");
    constexpr int N_CASES = 10;
    const int test_counts[N_CASES] = {1, 2, 3, 5, 7, 8, 12, 15, 16, 31};

    float* d_in = dalloc<float>(32, q);
    float* d_out = dalloc<float>(N_CASES, q);

    std::vector<float> h_in(32);
    for (int i = 0; i < 32; ++i) h_in[i] = (float)(i + 1) * 0.5f;
    q.memcpy(d_in, h_in.data(), 32 * sizeof(float)).wait();

    bool all_ok = true;
    for (int c = 0; c < N_CASES; ++c) {
        const int num_active = test_counts[c];
        float expected_sum = 0.0f;
        for (int i = 0; i < num_active; ++i) expected_sum += h_in[i];

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), [=](sycl::nd_item<1> item) {
                auto sg = item.get_sub_group();
                const int tid = (int)item.get_local_id(0);

                // Safe pattern: Whole subgroup participates in reduce_over_group, inactive lanes contribute 0.0f
                float val = (tid < num_active) ? d_in[tid] : 0.0f;
                float sum = sycl::reduce_over_group(sg, val, sycl::plus<float>());

                if (tid == 0) {
                    d_out[c] = sum;
                }
            });
        }).wait();

        float got_sum = 0.0f;
        q.memcpy(&got_sum, d_out + c, sizeof(float)).wait();

        if (std::abs(got_sum - expected_sum) > 1e-4f) {
            std::printf("\n  Mismatch at num_active=%d: expected=%.4f, got=%.4f ", num_active, expected_sum, got_sum);
            all_ok = false;
        }
    }

    if (all_ok) {
        std::printf("PASS (%d test cases)\n", N_CASES);
    } else {
        std::printf("FAIL\n");
        g_failures++;
    }

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

// 2. CPU reference for s2_gemv_q8
void cpu_ref_s2(const uint8_t* x80, const uint8_t* codes, const float* scales, float* y,
                int64_t n_in, int64_t n_out) {
    const int64_t n_quads = n_in / 4;
    for (int64_t o = 0; o < n_out; ++o) {
        double acc = 0.0;
        const uint8_t* c = codes + o * n_quads;
        const float* s = scales + o * (n_in / 64);

        for (int64_t q = 0; q < n_quads; ++q) {
            const uint8_t byte = c[q];
            const float d = s[q >> 4];
            const int64_t ablk = (q * 4) / 32;
            const uint8_t* blk = x80 + ablk * 34;
            const float dx = k::f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
            const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
            const int off = (int)((q * 4) % 32);

            const int c0 = (int)(byte & 3) - 1;
            const int c1 = (int)((byte >> 2) & 3) - 1;
            const int c2 = (int)((byte >> 4) & 3) - 1;
            const int c3 = (int)((byte >> 6) & 3) - 1;
            const int idot0 = c0 * (int)xq[off + 0] + c1 * (int)xq[off + 1];
            const int idot1 = c2 * (int)xq[off + 2] + c3 * (int)xq[off + 3];
            const float scale = d * dx;
            acc += (double)idot0 * (double)scale + (double)idot1 * (double)scale;
        }
        y[o] = (float)acc;
    }
}

// 3. Test s2_gemv_q8 across all TPR configurations (16, 32, 64, 128, 256)
void test_s2_gemv_tpr_sweep(sycl::queue& q) {
    std::printf("Testing s2_gemv_q8 across TPR configurations (16, 32, 64, 128, 256)... ");
    const int64_t n_in = 2560;
    const int64_t n_out = 64;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> act_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.05f);

    std::vector<float> h_x((size_t)n_in);
    for (auto& v : h_x) v = act_dist(rng);

    const size_t q80_bytes = (size_t)(n_in / 32) * 34;
    float* d_xf = dalloc<float>((size_t)n_in, q);
    uint8_t* d_x80 = dalloc<uint8_t>(q80_bytes, q);

    q.memcpy(d_xf, h_x.data(), (size_t)n_in * sizeof(float)).wait();
    k::quantize_q8_0(d_xf, d_x80, n_in, &q);
    q.wait();

    std::vector<uint8_t> h_x80(q80_bytes);
    q.memcpy(h_x80.data(), d_x80, q80_bytes).wait();

    const size_t code_bytes = (size_t)n_out * (n_in / 4);
    const size_t scale_floats = (size_t)n_out * (n_in / 32);
    std::vector<uint8_t> h_codes(code_bytes);
    std::vector<float> h_scales(scale_floats);
    for (auto& v : h_codes) v = (uint8_t)rng();
    for (auto& v : h_scales) v = scale_dist(rng);

    uint8_t* d_codes = upload(h_codes, q);
    float* d_scales = upload(h_scales, q);
    float* d_y = dalloc<float>((size_t)n_out, q);

    std::vector<float> h_ref((size_t)n_out);
    cpu_ref_s2(h_x80.data(), h_codes.data(), h_scales.data(), h_ref.data(), n_in, n_out);

    bool all_tpr_ok = true;
    const int tpr_configs[] = {16, 32, 64, 128, 256};
    for (int tpr : tpr_configs) {
        q.memset(d_y, 0, n_out * sizeof(float)).wait();
        k::s2_gemv_q8(d_x80, d_codes, d_scales, d_y, n_in, n_out, tpr, &q);
        q.wait_and_throw();

        std::vector<float> h_got((size_t)n_out);
        q.memcpy(h_got.data(), d_y, n_out * sizeof(float)).wait();

        double max_err = 0.0;
        for (int64_t i = 0; i < n_out; ++i) {
            double err = std::abs(h_got[i] - h_ref[i]) / (std::abs(h_ref[i]) + 1e-4);
            if (err > max_err) max_err = err;
        }

        if (max_err > 5e-3) {
            std::printf("\n  TPR %d max error %.4e > threshold! ", tpr, max_err);
            all_tpr_ok = false;
        }
    }

    if (all_tpr_ok) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL\n");
        g_failures++;
    }

    sycl::free(d_xf, q); sycl::free(d_x80, q);
    sycl::free(d_codes, q); sycl::free(d_scales, q);
    sycl::free(d_y, q);
}

// 4. Default queue idempotency: calling with stream = nullptr repeatedly must produce byte-identical results
void test_default_stream_idempotency(sycl::queue& q) {
    std::printf("Testing default queue (stream = nullptr) idempotency (10 iterations)... ");
    const int64_t n_in = 2560;
    const int64_t n_out = 64;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> act_dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.05f);

    std::vector<float> h_x((size_t)n_in);
    for (auto& v : h_x) v = act_dist(rng);

    const size_t q80_bytes = (size_t)(n_in / 32) * 34;
    float* d_xf = dalloc<float>((size_t)n_in, q);
    uint8_t* d_x80 = dalloc<uint8_t>(q80_bytes, q);

    q.memcpy(d_xf, h_x.data(), (size_t)n_in * sizeof(float)).wait();
    k::quantize_q8_0(d_xf, d_x80, n_in, nullptr); // test quantize on default stream

    const size_t code_bytes = (size_t)n_out * (n_in / 4);
    const size_t scale_floats = (size_t)n_out * (n_in / 32);
    std::vector<uint8_t> h_codes(code_bytes);
    std::vector<float> h_scales(scale_floats);
    for (auto& v : h_codes) v = (uint8_t)rng();
    for (auto& v : h_scales) v = scale_dist(rng);

    uint8_t* d_codes = upload(h_codes, q);
    float* d_scales = upload(h_scales, q);
    float* d_y = dalloc<float>((size_t)n_out, q);

    // Initial run on default stream
    k::s2_gemv_q8(d_x80, d_codes, d_scales, d_y, n_in, n_out, 64, nullptr);
    std::vector<float> h_baseline((size_t)n_out);
    q.memcpy(h_baseline.data(), d_y, n_out * sizeof(float)).wait();

    bool identical = true;
    for (int iter = 0; iter < 10; ++iter) {
        q.memset(d_y, 0, n_out * sizeof(float)).wait();
        k::s2_gemv_q8(d_x80, d_codes, d_scales, d_y, n_in, n_out, 64, nullptr);

        std::vector<float> h_run((size_t)n_out);
        q.memcpy(h_run.data(), d_y, n_out * sizeof(float)).wait();

        if (std::memcmp(h_baseline.data(), h_run.data(), n_out * sizeof(float)) != 0) {
            identical = false;
            break;
        }
    }

    if (identical) {
        std::printf("PASS (bit-identical on all 10 runs)\n");
    } else {
        std::printf("FAIL (divergence across iterations on default stream)\n");
        g_failures++;
    }

    sycl::free(d_xf, q); sycl::free(d_x80, q);
    sycl::free(d_codes, q); sycl::free(d_scales, q);
    sycl::free(d_y, q);
}

} // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running Subgroup Reduction & Divergence test on: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    test_partial_subgroup_reduction(q);
    test_s2_gemv_tpr_sweep(q);
    test_default_stream_idempotency(q);

    if (g_failures == 0) {
        std::printf("subgroup_divergence_test: 0 failures -> PASS\n");
        return 0;
    } else {
        std::printf("subgroup_divergence_test: %d failures -> FAIL\n", g_failures);
        return 1;
    }
}
