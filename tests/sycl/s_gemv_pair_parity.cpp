// tests/sycl/s_gemv_pair_parity.cpp - Parity verification for fused s_gemv_pair_silu vs unfused GEMVs + SwiGLU.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"

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

void gemv_unfused(const strata::kernels::SForm& f, const uint8_t* codes, const float* scales, const float* off,
                  const uint8_t* x80, const uint8_t* x8k, float* y, int64_t n_in, int64_t n_out,
                  int tpr, sycl::queue& q) {
    if (f.code_bits == 2) {
        strata::kernels::s2_gemv_q8(x80, codes, scales, y, n_in, n_out, tpr, &q);
    } else if (f.act_kind == 1) {
        strata::kernels::s_gemv_q8k_split(x8k, codes, scales, off, y, n_in, n_out, f, &q);
    } else {
        strata::kernels::s_gemv_q8_0_split(x80, codes, scales, off, y, n_in, n_out, f, &q);
    }
}

void run_swiglu(const float* g, const float* u, float* out, int64_t n, int swilu_kind, sycl::queue& q) {
    const int threads = 128;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                   [=](sycl::nd_item<1> item) {
                       const int i = item.get_global_id(0);
                       if (i >= n) return;
                       if (swilu_kind == 0) {
                           const double x = (double)g[i];
                           out[i] = (float)(x / (1.0 + sycl::exp(-x))) * u[i];
                       } else {
                           out[i] = (g[i] / (1.0f + sycl::native::exp(-g[i]))) * u[i];
                       }
                   }).wait();
}

void run_test_case(int64_t n_in, int64_t n_out, const strata::kernels::SForm& fg, const strata::kernels::SForm& fu,
                   int swilu_kind, sycl::queue& q, const char* label) {
    std::mt19937 rng(42 + swilu_kind * 17);
    std::uniform_real_distribution<float> act_dist(-2.5f, 2.5f);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.05f);
    std::uniform_real_distribution<float> off_dist(-0.02f, 0.02f);

    std::vector<float> h_x((size_t)n_in);
    for (auto& v : h_x) v = act_dist(rng);

    const int per_byte_g = 8 / fg.code_bits;
    const int per_byte_u = 8 / fu.code_bits;
    const int64_t n_groups_g = n_in / fg.group_elems;
    const int64_t n_groups_u = n_in / fu.group_elems;

    std::vector<uint8_t> h_cg((size_t)(n_out * (n_in / per_byte_g)));
    std::vector<uint8_t> h_cu((size_t)(n_out * (n_in / per_byte_u)));
    for (auto& v : h_cg) v = (uint8_t)rng();
    for (auto& v : h_cu) v = (uint8_t)rng();

    std::vector<float> h_sg((size_t)(n_out * n_groups_g));
    std::vector<float> h_su((size_t)(n_out * n_groups_u));
    for (auto& v : h_sg) v = scale_dist(rng);
    for (auto& v : h_su) v = scale_dist(rng);

    std::vector<float> h_og(fg.has_offset ? (size_t)(n_out * n_groups_g) : 0);
    std::vector<float> h_ou(fu.has_offset ? (size_t)(n_out * n_groups_u) : 0);
    for (auto& v : h_og) v = off_dist(rng);
    for (auto& v : h_ou) v = off_dist(rng);

    // Quantize activations
    const size_t q80_bytes = (size_t)(n_in / 32) * 34;
    const size_t q8k_bytes = (size_t)(n_in / 256) * 292;
    float* d_xf = sycl::malloc_device<float>((size_t)n_in, q);
    uint8_t* d_x80 = sycl::malloc_device<uint8_t>(q80_bytes, q);
    uint8_t* d_x8k = (fg.act_kind == 1 || fu.act_kind == 1) ? sycl::malloc_device<uint8_t>(q8k_bytes, q) : nullptr;

    q.memcpy(d_xf, h_x.data(), (size_t)n_in * sizeof(float)).wait();
    strata::kernels::quantize_q8_0(d_xf, d_x80, n_in, &q);
    if (d_x8k) {
        strata::kernels::quantize_q8_K(d_xf, d_x8k, n_in, &q);
    }
    q.wait();

    // Weight buffers
    uint8_t* d_cg = sycl::malloc_device<uint8_t>(h_cg.size(), q);
    uint8_t* d_cu = sycl::malloc_device<uint8_t>(h_cu.size(), q);
    float* d_sg = sycl::malloc_device<float>(h_sg.size(), q);
    float* d_su = sycl::malloc_device<float>(h_su.size(), q);
    float* d_og = fg.has_offset ? sycl::malloc_device<float>(h_og.size(), q) : nullptr;
    float* d_ou = fu.has_offset ? sycl::malloc_device<float>(h_ou.size(), q) : nullptr;

    q.memcpy(d_cg, h_cg.data(), h_cg.size()).wait();
    q.memcpy(d_cu, h_cu.data(), h_cu.size()).wait();
    q.memcpy(d_sg, h_sg.data(), h_sg.size() * sizeof(float)).wait();
    q.memcpy(d_su, h_su.data(), h_su.size() * sizeof(float)).wait();
    if (d_og) q.memcpy(d_og, h_og.data(), h_og.size() * sizeof(float)).wait();
    if (d_ou) q.memcpy(d_ou, h_ou.data(), h_ou.size() * sizeof(float)).wait();

    // Output buffers
    float* d_gate = sycl::malloc_device<float>((size_t)n_out, q);
    float* d_up = sycl::malloc_device<float>((size_t)n_out, q);
    float* d_unfused = sycl::malloc_device<float>((size_t)n_out, q);
    float* d_fused = sycl::malloc_device<float>((size_t)n_out, q);

    const int tpr = strata::sycl_arch::SUB_GROUP_SIZE;
    // 1. Run unfused pipeline
    gemv_unfused(fg, d_cg, d_sg, d_og, d_x80, d_x8k, d_gate, n_in, n_out, tpr, q);
    gemv_unfused(fu, d_cu, d_su, d_ou, d_x80, d_x8k, d_up, n_in, n_out, tpr, q);
    run_swiglu(d_gate, d_up, d_unfused, n_out, swilu_kind, q);

    // 2. Run fused pipeline
    bool ok = strata::kernels::s_gemv_pair_silu(d_x80, d_x8k, fg, d_cg, d_sg, d_og,
                                                fu, d_cu, d_su, d_ou, d_fused,
                                                n_in, n_out, tpr, swilu_kind, &q);
    q.wait();

    if (!ok) {
        std::printf("  [FAIL] %-48s kernel returned false!\n", label);
        ++g_failures;
    } else {
        std::vector<float> h_unfused((size_t)n_out), h_fused((size_t)n_out);
        q.memcpy(h_unfused.data(), d_unfused, (size_t)n_out * sizeof(float)).wait();
        q.memcpy(h_fused.data(), d_fused, (size_t)n_out * sizeof(float)).wait();

        size_t bitwise_match = 0;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;
        for (int64_t i = 0; i < n_out; ++i) {
            uint32_t ub, fb;
            std::memcpy(&ub, &h_unfused[(size_t)i], 4);
            std::memcpy(&fb, &h_fused[(size_t)i], 4);
            if (ub == fb) ++bitwise_match;

            const double diff = std::fabs((double)h_unfused[(size_t)i] - (double)h_fused[(size_t)i]);
            const double denom = std::fabs((double)h_unfused[(size_t)i]);
            const double rel = diff / (denom > 1e-30 ? denom : 1e-30);
            if (diff > max_abs_diff) max_abs_diff = diff;
            if (rel > max_rel_diff) max_rel_diff = rel;
        }

        const bool pass = (max_abs_diff <= 1e-5) || (max_rel_diff <= 1e-5);
        if (bitwise_match == (size_t)n_out) {
            std::printf("  [PASS] %-48s BITWISE (%zu/%zu exact)\n", label, bitwise_match, (size_t)n_out);
        } else if (pass) {
            std::printf("  [PASS] %-48s MATCHES (%zu/%zu bitwise, max_diff=%.2e, max_rel=%.2e)\n",
                        label, bitwise_match, (size_t)n_out, max_abs_diff, max_rel_diff);
        } else {
            std::printf("  [FAIL] %-48s DIFFERS (max_diff=%.2e, max_rel=%.2e)\n",
                        label, max_abs_diff, max_rel_diff);
            ++g_failures;
        }
    }

    sycl::free(d_xf, q);
    sycl::free(d_x80, q);
    if (d_x8k) sycl::free(d_x8k, q);
    sycl::free(d_cg, q);
    sycl::free(d_cu, q);
    sycl::free(d_sg, q);
    sycl::free(d_su, q);
    if (d_og) sycl::free(d_og, q);
    if (d_ou) sycl::free(d_ou, q);
    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_unfused, q);
    sycl::free(d_fused, q);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
    }

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("s_gemv_pair_parity: Xe2 SIMD16 Fused Dual GEMV + SwiGLU Parity Suite\n");
    std::printf("  Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("  Native Subgroup Size: %d\n\n", strata::sycl_arch::SUB_GROUP_SIZE);

    using strata::kernels::Codebook;
    using strata::kernels::SForm;

    // SForm definitions
    const SForm s4_q80{4, -8, 64, Codebook::Affine, true, /*act_kind=*/0};
    const SForm s4_q8k{4, -8, 64, Codebook::Affine, true, /*act_kind=*/1};
    const SForm s8_q8k{8, -128, 64, Codebook::Affine, false, /*act_kind=*/1};
    const SForm s2_q80{2, -1, 64, Codebook::Affine, false, /*act_kind=*/0};

    // Test Suite 1: Shared expert decode shape (Nin = 2560, Nout = 640)
    std::printf("Suite 1: Shared Expert Decode Shape (Nin=2560, Nout=640)\n");
    run_test_case(2560, 640, s4_q80, s4_q80, 0, q, "S4/Q8_0 + S4/Q8_0 (legacy silu)");
    run_test_case(2560, 640, s4_q80, s4_q80, 1, q, "S4/Q8_0 + S4/Q8_0 (fast silu)");
    run_test_case(2560, 640, s4_q80, s4_q80, 2, q, "S4/Q8_0 + S4/Q8_0 (native silu)");
    run_test_case(2560, 640, s4_q8k, s4_q8k, 1, q, "S4/Q8_K + S4/Q8_K (fast silu)");
    run_test_case(2560, 640, s8_q8k, s8_q8k, 1, q, "S8/Q8_K + S8/Q8_K (fast silu)");
    run_test_case(2560, 640, s2_q80, s2_q80, 0, q, "S2/Q8_0 + S2/Q8_0 (legacy silu)");
    run_test_case(2560, 640, s2_q80, s2_q80, 1, q, "S2/Q8_0 + S2/Q8_0 (fast silu)");
    run_test_case(2560, 640, s4_q80, s2_q80, 1, q, "S4/Q8_0 + S2/Q8_0 Mixed (fast silu)");
    run_test_case(2560, 640, s2_q80, s4_q8k, 1, q, "S2/Q8_0 + S4/Q8_K Mixed (fast silu)");

    // Test Suite 2: Wide projection shape (Nin = 2560, Nout = 10240)
    std::printf("\nSuite 2: Wide Expert Projection Shape (Nin=2560, Nout=10240)\n");
    run_test_case(2560, 10240, s4_q80, s4_q80, 1, q, "S4/Q8_0 + S4/Q8_0 Wide (fast silu)");
    run_test_case(2560, 10240, s2_q80, s2_q80, 1, q, "S2/Q8_0 + S2/Q8_0 Wide (fast silu)");
    run_test_case(2560, 10240, s4_q8k, s4_q8k, 1, q, "S4/Q8_K + S4/Q8_K Wide (fast silu)");

    std::printf("\n===================================================\n");
    std::printf("s_gemv_pair_parity: %d failures\n", g_failures);
    if (g_failures > 0) return 1;
    if (selftest) std::printf("s_gemv_pair_parity OK\n");
    return 0;
}
