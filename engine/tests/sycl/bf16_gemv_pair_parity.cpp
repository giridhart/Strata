// tests/sycl/bf16_gemv_pair_parity.cpp - Parity verification for fused bf16_gemv_pair vs standalone GEMVs on Intel Xe2.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() && b.empty()) return 0.0;
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double)a[i] - (double)b[i]);
        m += std::fabs((double)a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

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

void test_pair_case(int64_t n_in, int64_t n_out1, int64_t n_out2, sycl::queue& q, const char* label) {
    std::printf("Testing %-32s [n_in=%lld, n_out1=%lld, n_out2=%lld]... ", label,
                (long long)n_in, (long long)n_out1, (long long)n_out2);

    std::mt19937 rng(1337 + (unsigned)n_out1 * 31 + (unsigned)n_out2 * 17);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<uint16_t> h_x((size_t)n_in);
    for (auto& v : h_x) v = bf16_from_f32(dist(rng));

    std::vector<uint16_t> h_w1((size_t)(n_in * n_out1));
    for (auto& v : h_w1) v = bf16_from_f32(dist(rng) * 0.05f);

    std::vector<uint16_t> h_w2((size_t)(n_in * n_out2));
    for (auto& v : h_w2) v = bf16_from_f32(dist(rng) * 0.05f);

    uint16_t* d_x = sycl::malloc_device<uint16_t>((size_t)n_in, q);
    uint16_t* d_w1 = n_out1 > 0 ? sycl::malloc_device<uint16_t>((size_t)(n_in * n_out1), q) : nullptr;
    uint16_t* d_w2 = n_out2 > 0 ? sycl::malloc_device<uint16_t>((size_t)(n_in * n_out2), q) : nullptr;

    float* d_pair_y1 = n_out1 > 0 ? sycl::malloc_device<float>((size_t)n_out1, q) : nullptr;
    float* d_pair_y2 = n_out2 > 0 ? sycl::malloc_device<float>((size_t)n_out2, q) : nullptr;

    float* d_ref_y1 = n_out1 > 0 ? sycl::malloc_device<float>((size_t)n_out1, q) : nullptr;
    float* d_ref_y2 = n_out2 > 0 ? sycl::malloc_device<float>((size_t)n_out2, q) : nullptr;

    float* d_warp_y1 = n_out1 > 0 ? sycl::malloc_device<float>((size_t)n_out1, q) : nullptr;
    float* d_warp_y2 = n_out2 > 0 ? sycl::malloc_device<float>((size_t)n_out2, q) : nullptr;

    q.memcpy(d_x, h_x.data(), (size_t)n_in * sizeof(uint16_t));
    if (n_out1 > 0) q.memcpy(d_w1, h_w1.data(), (size_t)(n_in * n_out1) * sizeof(uint16_t));
    if (n_out2 > 0) q.memcpy(d_w2, h_w2.data(), (size_t)(n_in * n_out2) * sizeof(uint16_t));
    q.wait();

    // 1. Launch fused bf16_gemv_pair
    strata::kernels::bf16_gemv_pair(d_x, d_w1, d_pair_y1, n_out1, d_w2, d_pair_y2, n_out2, n_in, &q);

    // 2. Launch standalone bf16_gemv
    if (n_out1 > 0) strata::kernels::bf16_gemv(d_x, d_w1, d_ref_y1, n_in, n_out1, &q);
    if (n_out2 > 0) strata::kernels::bf16_gemv(d_x, d_w2, d_ref_y2, n_in, n_out2, &q);

    // 3. Launch subgroup-strided reference (split with TPR=32 / SG_SIZE)
    if (n_out1 > 0) strata::kernels::bf16_gemv_split(d_x, d_w1, d_warp_y1, n_in, n_out1, 32, &q);
    if (n_out2 > 0) strata::kernels::bf16_gemv_split(d_x, d_w2, d_warp_y2, n_in, n_out2, 32, &q);

    q.wait();

    std::vector<float> h_pair_y1((size_t)n_out1), h_pair_y2((size_t)n_out2);
    std::vector<float> h_ref_y1((size_t)n_out1), h_ref_y2((size_t)n_out2);
    std::vector<float> h_warp_y1((size_t)n_out1), h_warp_y2((size_t)n_out2);

    if (n_out1 > 0) {
        q.memcpy(h_pair_y1.data(), d_pair_y1, (size_t)n_out1 * sizeof(float));
        q.memcpy(h_ref_y1.data(), d_ref_y1, (size_t)n_out1 * sizeof(float));
        q.memcpy(h_warp_y1.data(), d_warp_y1, (size_t)n_out1 * sizeof(float));
    }
    if (n_out2 > 0) {
        q.memcpy(h_pair_y2.data(), d_pair_y2, (size_t)n_out2 * sizeof(float));
        q.memcpy(h_ref_y2.data(), d_ref_y2, (size_t)n_out2 * sizeof(float));
        q.memcpy(h_warp_y2.data(), d_warp_y2, (size_t)n_out2 * sizeof(float));
    }
    q.wait();

    // Check against double reference
    std::vector<float> dbl_want1, dbl_want2;
    ref_gemv(h_x, h_w1, n_in, n_out1, dbl_want1);
    ref_gemv(h_x, h_w2, n_in, n_out2, dbl_want2);

    double rel1_dbl = rel_l1(dbl_want1, h_pair_y1);
    double rel2_dbl = rel_l1(dbl_want2, h_pair_y2);

    // Check exact bitwise vs warp / subgroup kernel
    bool bitwise1 = true, bitwise2 = true;
    if (n_out1 > 0) bitwise1 = (std::memcmp(h_pair_y1.data(), h_warp_y1.data(), (size_t)n_out1 * sizeof(float)) == 0);
    if (n_out2 > 0) bitwise2 = (std::memcmp(h_pair_y2.data(), h_warp_y2.data(), (size_t)n_out2 * sizeof(float)) == 0);

    double rel1_ref = rel_l1(h_ref_y1, h_pair_y1);
    double rel2_ref = rel_l1(h_ref_y2, h_pair_y2);

    bool pass = (rel1_dbl <= 1e-5) && (rel2_dbl <= 1e-5) && (rel1_ref <= 1e-5) && (rel2_ref <= 1e-5);
    if (n_out1 >= 64 && !bitwise1) pass = false;
    if (n_out2 >= 64 && !bitwise2) pass = false;

    if (pass) {
        std::printf("PASS (rel1=%.2e, rel2=%.2e, bitwise=[%s, %s])\n",
                    rel1_ref, rel2_ref, bitwise1 ? "EXACT" : "diff", bitwise2 ? "EXACT" : "diff");
    } else {
        std::printf("FAIL! (rel1=%.2e, rel2=%.2e, bitwise=[%s, %s])\n",
                    rel1_ref, rel2_ref, bitwise1 ? "EXACT" : "diff", bitwise2 ? "EXACT" : "diff");
        g_failures++;
    }

    if (d_x) sycl::free(d_x, q);
    if (d_w1) sycl::free(d_w1, q);
    if (d_w2) sycl::free(d_w2, q);
    if (d_pair_y1) sycl::free(d_pair_y1, q);
    if (d_pair_y2) sycl::free(d_pair_y2, q);
    if (d_ref_y1) sycl::free(d_ref_y1, q);
    if (d_ref_y2) sycl::free(d_ref_y2, q);
    if (d_warp_y1) sycl::free(d_warp_y1, q);
    if (d_warp_y2) sycl::free(d_warp_y2, q);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::device dev = gpus.empty() ? sycl::device(sycl::default_selector_v) : gpus.front();
    sycl::queue q(dev, sycl::property::queue::in_order{});

    std::printf("=== Intel Xe2 bf16_gemv_pair Parity Verification ===\n");
    std::printf("Target Device: %s\n\n", dev.get_info<sycl::info::device::name>().c_str());

    // Shape A: ssm_alpha / ssm_beta [2560 x 48]
    test_pair_case(2560, 48, 48, q, "Shape A: ssm_alpha / ssm_beta");

    // Shape B: indexer.k_proj / indexer.q_proj [2560 x 128], [2560 x 512]
    test_pair_case(2560, 128, 512, q, "Shape B: indexer k / q proj");

    // Shape C: Asymmetric shapes [2560 x 64], [2560 x 256]
    test_pair_case(2560, 64, 256, q, "Shape C: Asymmetric 64 / 256");

    // Shape D: Reversed asymmetric shapes [2560 x 512], [2560 x 128]
    test_pair_case(2560, 512, 128, q, "Shape D: Reversed k / q proj");

    // Shape E: Large symmetric shapes [2560 x 2560], [2560 x 2560]
    test_pair_case(2560, 2560, 2560, q, "Shape E: Large symmetric 2560");

    // Shape F: One-sided Left [2560 x 128], [2560 x 0]
    test_pair_case(2560, 128, 0, q, "Shape F: One-sided (left only)");

    // Shape G: One-sided Right [2560 x 0], [2560 x 512]
    test_pair_case(2560, 0, 512, q, "Shape G: One-sided (right only)");

    // Shape H: Intermediate embedding [640 x 48], [640 x 48]
    test_pair_case(640, 48, 48, q, "Shape H: Intermediate 640 x 48");

    std::printf("\nParity verification completed: %d failures\n", g_failures);
    if (g_failures > 0) return 1;
    std::printf("bf16_gemv_pair_parity OK\n");
    return 0;
}
