// tests/sycl/quantize_act_images_parity.cpp - Bitwise parity verification for Xe2 fused multi-format quantizers.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"

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

void run_swiglu_ref(const float* g, const float* u, float* out, int64_t n, int swilu_kind, sycl::queue& q) {
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

void test_quantize_act_images(int64_t n, bool with_q8_0, bool with_q8_K, bool with_bf16,
                              sycl::queue& q, const char* label) {
    std::mt19937 rng(1337 + (unsigned)n);
    std::uniform_real_distribution<float> dist(-4.5f, 4.5f);

    std::vector<float> h_x((size_t)n);
    for (auto& v : h_x) v = dist(rng);

    float* d_x = sycl::malloc_device<float>((size_t)n, q);
    q.memcpy(d_x, h_x.data(), (size_t)n * sizeof(float)).wait();

    const size_t nb_q8_0 = (size_t)(n / 32);
    const size_t nb_q8_K = (size_t)(n / 256);

    uint8_t* d_fused_q8_0 = with_q8_0 ? sycl::malloc_device<uint8_t>(nb_q8_0 * 34, q) : nullptr;
    uint8_t* d_fused_q8_K = with_q8_K ? sycl::malloc_device<uint8_t>(nb_q8_K * 292, q) : nullptr;
    uint16_t* d_fused_bf16 = with_bf16 ? sycl::malloc_device<uint16_t>((size_t)n, q) : nullptr;

    uint8_t* d_ref_q8_0 = with_q8_0 ? sycl::malloc_device<uint8_t>(nb_q8_0 * 34, q) : nullptr;
    uint8_t* d_ref_q8_K = with_q8_K ? sycl::malloc_device<uint8_t>(nb_q8_K * 292, q) : nullptr;
    uint16_t* d_ref_bf16 = with_bf16 ? sycl::malloc_device<uint16_t>((size_t)n, q) : nullptr;

    // Launch fused kernel
    strata::kernels::quantize_act_images(d_x, n, d_fused_q8_0, d_fused_q8_K, d_fused_bf16, &q);

    // Launch standalone reference kernels
    if (with_q8_0) strata::kernels::quantize_q8_0(d_x, d_ref_q8_0, n, &q);
    if (with_q8_K) strata::kernels::quantize_q8_K(d_x, d_ref_q8_K, n, &q);
    if (with_bf16) strata::kernels::f32_to_bf16_bulk(d_x, d_ref_bf16, n, &q);

    q.wait();

    bool ok_q8_0 = true;
    if (with_q8_0) {
        std::vector<uint8_t> h_fused(nb_q8_0 * 34), h_ref(nb_q8_0 * 34);
        q.memcpy(h_fused.data(), d_fused_q8_0, h_fused.size()).wait();
        q.memcpy(h_ref.data(), d_ref_q8_0, h_ref.size()).wait();
        if (std::memcmp(h_fused.data(), h_ref.data(), h_fused.size()) != 0) {
            ok_q8_0 = false;
        }
    }

    bool ok_q8_K = true;
    if (with_q8_K) {
        std::vector<uint8_t> h_fused(nb_q8_K * 292), h_ref(nb_q8_K * 292);
        q.memcpy(h_fused.data(), d_fused_q8_K, h_fused.size()).wait();
        q.memcpy(h_ref.data(), d_ref_q8_K, h_ref.size()).wait();
        if (std::memcmp(h_fused.data(), h_ref.data(), h_fused.size()) != 0) {
            ok_q8_K = false;
        }
    }

    bool ok_bf16 = true;
    if (with_bf16) {
        std::vector<uint16_t> h_fused((size_t)n), h_ref((size_t)n);
        q.memcpy(h_fused.data(), d_fused_bf16, (size_t)n * sizeof(uint16_t)).wait();
        q.memcpy(h_ref.data(), d_ref_bf16, (size_t)n * sizeof(uint16_t)).wait();
        if (std::memcmp(h_fused.data(), h_ref.data(), (size_t)n * sizeof(uint16_t)) != 0) {
            ok_bf16 = false;
        }
    }

    const bool all_ok = ok_q8_0 && ok_q8_K && ok_bf16;
    if (all_ok) {
        std::printf("  [PASS] %-50s BITWISE EXACT\n", label);
    } else {
        std::printf("  [FAIL] %-50s MISMATCH (q8_0=%d, q8_K=%d, bf16=%d)\n",
                    label, ok_q8_0 ? 1 : 0, ok_q8_K ? 1 : 0, ok_bf16 ? 1 : 0);
        ++g_failures;
    }

    if (d_x) sycl::free(d_x, q);
    if (d_fused_q8_0) sycl::free(d_fused_q8_0, q);
    if (d_fused_q8_K) sycl::free(d_fused_q8_K, q);
    if (d_fused_bf16) sycl::free(d_fused_bf16, q);
    if (d_ref_q8_0) sycl::free(d_ref_q8_0, q);
    if (d_ref_q8_K) sycl::free(d_ref_q8_K, q);
    if (d_ref_bf16) sycl::free(d_ref_bf16, q);
}

void test_swilu_quantize_q8_0(int64_t n, int swilu_kind, bool scaled, sycl::queue& q, const char* label) {
    std::mt19937 rng(4242 + (unsigned)n + (unsigned)swilu_kind * 100 + (unsigned)scaled);
    std::uniform_real_distribution<float> g_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> u_dist(-3.0f, 3.0f);

    std::vector<float> h_g((size_t)n), h_u((size_t)n);
    for (auto& v : h_g) v = g_dist(rng);
    for (auto& v : h_u) v = u_dist(rng);

    float* d_g = sycl::malloc_device<float>((size_t)n, q);
    float* d_u = sycl::malloc_device<float>((size_t)n, q);
    float* d_fused_out = sycl::malloc_device<float>((size_t)n, q);
    float* d_ref_out = sycl::malloc_device<float>((size_t)n, q);

    q.memcpy(d_g, h_g.data(), (size_t)n * sizeof(float)).wait();
    q.memcpy(d_u, h_u.data(), (size_t)n * sizeof(float)).wait();

    const size_t nb = (size_t)(n / 32);
    uint8_t* d_fused_blocks = sycl::malloc_device<uint8_t>(nb * 34, q);
    uint8_t* d_ref_blocks = sycl::malloc_device<uint8_t>(nb * 34, q);
    float* d_fused_scales = scaled ? sycl::malloc_device<float>(nb, q) : nullptr;
    float* d_ref_scales = scaled ? sycl::malloc_device<float>(nb, q) : nullptr;

    // Fused kernel
    if (scaled) {
        strata::kernels::swilu_quantize_q8_0_scaled(d_g, d_u, d_fused_out, n, swilu_kind,
                                                    d_fused_blocks, d_fused_scales, &q);
    } else {
        strata::kernels::swilu_quantize_q8_0(d_g, d_u, d_fused_out, n, swilu_kind,
                                             d_fused_blocks, &q);
    }

    // Unfused reference sequence: swiglu + quantize_q8_0
    run_swiglu_ref(d_g, d_u, d_ref_out, n, swilu_kind, q);
    if (scaled) {
        strata::kernels::quantize_q8_0_scaled(d_ref_out, d_ref_blocks, d_ref_scales, n, &q);
    } else {
        strata::kernels::quantize_q8_0(d_ref_out, d_ref_blocks, n, &q);
    }
    q.wait();

    // Verify products
    std::vector<float> h_fused_prod((size_t)n), h_ref_prod((size_t)n);
    q.memcpy(h_fused_prod.data(), d_fused_out, (size_t)n * sizeof(float)).wait();
    q.memcpy(h_ref_prod.data(), d_ref_out, (size_t)n * sizeof(float)).wait();
    const bool ok_prod = (std::memcmp(h_fused_prod.data(), h_ref_prod.data(), (size_t)n * sizeof(float)) == 0);

    // Verify blocks
    std::vector<uint8_t> h_fused_b(nb * 34), h_ref_b(nb * 34);
    q.memcpy(h_fused_b.data(), d_fused_blocks, h_fused_b.size()).wait();
    q.memcpy(h_ref_b.data(), d_ref_blocks, h_ref_b.size()).wait();
    const bool ok_blocks = (std::memcmp(h_fused_b.data(), h_ref_b.data(), h_fused_b.size()) == 0);

    // Verify scales if scaled
    bool ok_scales = true;
    if (scaled) {
        std::vector<float> h_fused_s(nb), h_ref_s(nb);
        q.memcpy(h_fused_s.data(), d_fused_scales, nb * sizeof(float)).wait();
        q.memcpy(h_ref_s.data(), d_ref_scales, nb * sizeof(float)).wait();
        ok_scales = (std::memcmp(h_fused_s.data(), h_ref_s.data(), nb * sizeof(float)) == 0);
    }

    const bool all_ok = ok_prod && ok_blocks && ok_scales;
    if (all_ok) {
        std::printf("  [PASS] %-50s BITWISE EXACT\n", label);
    } else {
        std::printf("  [FAIL] %-50s MISMATCH (prod=%d, blocks=%d, scales=%d)\n",
                    label, ok_prod ? 1 : 0, ok_blocks ? 1 : 0, ok_scales ? 1 : 0);
        ++g_failures;
    }

    sycl::free(d_g, q);
    sycl::free(d_u, q);
    sycl::free(d_fused_out, q);
    sycl::free(d_ref_out, q);
    sycl::free(d_fused_blocks, q);
    sycl::free(d_ref_blocks, q);
    if (d_fused_scales) sycl::free(d_fused_scales, q);
    if (d_ref_scales) sycl::free(d_ref_scales, q);
}

void test_swilu_quantize_q8_K(int64_t n, int swilu_kind, sycl::queue& q, const char* label) {
    std::mt19937 rng(9999 + (unsigned)n + (unsigned)swilu_kind * 10);
    std::uniform_real_distribution<float> g_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> u_dist(-3.0f, 3.0f);

    std::vector<float> h_g((size_t)n), h_u((size_t)n);
    for (auto& v : h_g) v = g_dist(rng);
    for (auto& v : h_u) v = u_dist(rng);

    float* d_g = sycl::malloc_device<float>((size_t)n, q);
    float* d_u = sycl::malloc_device<float>((size_t)n, q);
    float* d_fused_out = sycl::malloc_device<float>((size_t)n, q);
    float* d_ref_out = sycl::malloc_device<float>((size_t)n, q);

    q.memcpy(d_g, h_g.data(), (size_t)n * sizeof(float)).wait();
    q.memcpy(d_u, h_u.data(), (size_t)n * sizeof(float)).wait();

    const size_t nb = (size_t)(n / 256);
    uint8_t* d_fused_blocks = sycl::malloc_device<uint8_t>(nb * 292, q);
    uint8_t* d_ref_blocks = sycl::malloc_device<uint8_t>(nb * 292, q);

    strata::kernels::swilu_quantize_q8_K(d_g, d_u, d_fused_out, n, swilu_kind, d_fused_blocks, &q);

    run_swiglu_ref(d_g, d_u, d_ref_out, n, swilu_kind, q);
    strata::kernels::quantize_q8_K(d_ref_out, d_ref_blocks, n, &q);
    q.wait();

    std::vector<float> h_fused_prod((size_t)n), h_ref_prod((size_t)n);
    q.memcpy(h_fused_prod.data(), d_fused_out, (size_t)n * sizeof(float)).wait();
    q.memcpy(h_ref_prod.data(), d_ref_out, (size_t)n * sizeof(float)).wait();
    const bool ok_prod = (std::memcmp(h_fused_prod.data(), h_ref_prod.data(), (size_t)n * sizeof(float)) == 0);

    std::vector<uint8_t> h_fused_b(nb * 292), h_ref_b(nb * 292);
    q.memcpy(h_fused_b.data(), d_fused_blocks, h_fused_b.size()).wait();
    q.memcpy(h_ref_b.data(), d_ref_blocks, h_ref_b.size()).wait();
    const bool ok_blocks = (std::memcmp(h_fused_b.data(), h_ref_b.data(), h_fused_b.size()) == 0);

    const bool all_ok = ok_prod && ok_blocks;
    if (all_ok) {
        std::printf("  [PASS] %-50s BITWISE EXACT\n", label);
    } else {
        std::printf("  [FAIL] %-50s MISMATCH (prod=%d, blocks=%d)\n", label, ok_prod ? 1 : 0, ok_blocks ? 1 : 0);
        ++g_failures;
    }

    sycl::free(d_g, q);
    sycl::free(d_u, q);
    sycl::free(d_fused_out, q);
    sycl::free(d_ref_out, q);
    sycl::free(d_fused_blocks, q);
    sycl::free(d_ref_blocks, q);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const auto dev = q.get_device();
    std::printf("quantize_act_images_parity: Xe2 Multi-Format Activation Quantizer Parity\n");
    std::printf("  Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("  Native Subgroup Size: %d\n\n", strata::sycl_arch::SUB_GROUP_SIZE);

    std::printf("Suite 1: quantize_act_images Multi-Format Parity\n");
    test_quantize_act_images(2560, true, true, true, q, "quantize_act_images all 3 (N=2560)");
    test_quantize_act_images(2560, true, true, false, q, "quantize_act_images q8_0+q8_K (N=2560)");
    test_quantize_act_images(2560, true, false, true, q, "quantize_act_images q8_0+bf16 (N=2560)");
    test_quantize_act_images(2560, false, true, true, q, "quantize_act_images q8_K+bf16 (N=2560)");
    test_quantize_act_images(2560, true, false, false, q, "quantize_act_images q8_0 only (N=2560)");
    test_quantize_act_images(2560, false, true, false, q, "quantize_act_images q8_K only (N=2560)");
    test_quantize_act_images(2560, false, false, true, q, "quantize_act_images bf16 only (N=2560)");
    test_quantize_act_images(5120, true, true, true, q, "quantize_act_images all 3 (N=5120)");
    test_quantize_act_images(256, true, true, true, q, "quantize_act_images all 3 (N=256)");
    test_quantize_act_images(640, true, false, true, q, "quantize_act_images q8_0+bf16 (N=640)");
    test_quantize_act_images(640, true, false, false, q, "quantize_act_images q8_0 only (N=640)");

    std::printf("\nSuite 2: Fused swilu_quantize_q8_0 Parity\n");
    test_swilu_quantize_q8_0(2560, 0, false, q, "swilu_quantize_q8_0 (legacy silu, N=2560)");
    test_swilu_quantize_q8_0(2560, 1, false, q, "swilu_quantize_q8_0 (fast silu, N=2560)");
    test_swilu_quantize_q8_0(2560, 2, false, q, "swilu_quantize_q8_0 (native silu, N=2560)");
    test_swilu_quantize_q8_0(2560, 0, true, q, "swilu_quantize_q8_0_scaled (legacy silu, N=2560)");
    test_swilu_quantize_q8_0(2560, 1, true, q, "swilu_quantize_q8_0_scaled (fast silu, N=2560)");
    test_swilu_quantize_q8_0(2560, 2, true, q, "swilu_quantize_q8_0_scaled (native silu, N=2560)");
    test_swilu_quantize_q8_0(640, 1, false, q, "swilu_quantize_q8_0 (fast silu, N=640)");
    test_swilu_quantize_q8_0(640, 1, true, q, "swilu_quantize_q8_0_scaled (fast silu, N=640)");
    test_swilu_quantize_q8_0(6144, 1, false, q, "swilu_quantize_q8_0 (fast silu, N=6144)");
    test_swilu_quantize_q8_0(6144, 1, true, q, "swilu_quantize_q8_0_scaled (fast silu, N=6144)");

    std::printf("\nSuite 3: Fused swilu_quantize_q8_K Parity\n");
    test_swilu_quantize_q8_K(2560, 0, q, "swilu_quantize_q8_K (legacy silu, N=2560)");
    test_swilu_quantize_q8_K(2560, 1, q, "swilu_quantize_q8_K (fast silu, N=2560)");
    test_swilu_quantize_q8_K(2560, 2, q, "swilu_quantize_q8_K (native silu, N=2560)");

    std::printf("\n===================================================\n");
    std::printf("quantize_act_images_parity: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
