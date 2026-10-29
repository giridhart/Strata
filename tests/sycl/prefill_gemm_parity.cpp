// tests/sycl/prefill_gemm_parity.cpp - Numerical parity verification for Intel Xe2 Gemm (FP16/BF16 XMX DPAS).
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void ref_gemm_f16(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                  int64_t T, int64_t N, int64_t K, int64_t ldy, float beta, std::vector<float>& y) {
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                float fx = static_cast<float>(sycl::bit_cast<sycl::half>(x[(size_t)(t * K + k)]));
                float fw = static_cast<float>(sycl::bit_cast<sycl::half>(w[(size_t)(n * K + k)]));
                sum += (double)fx * (double)fw;
            }
            const size_t idx = (size_t)(t * ldy + n);
            if (beta == 0.0f) {
                y[idx] = (float)sum;
            } else {
                y[idx] = (float)sum + beta * y[idx];
            }
        }
    }
}

void ref_gemm_bf16(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                   int64_t T, int64_t N, int64_t K, int64_t ldy, float beta, std::vector<float>& y) {
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                float fx = strata::kernels::f32_from_bf16(x[(size_t)(t * K + k)]);
                float fw = strata::kernels::f32_from_bf16(w[(size_t)(n * K + k)]);
                sum += (double)fx * (double)fw;
            }
            const size_t idx = (size_t)(t * ldy + n);
            if (beta == 0.0f) {
                y[idx] = (float)sum;
            } else {
                y[idx] = (float)sum + beta * y[idx];
            }
        }
    }
}

void test_f16_case(strata::prefill::Gemm& gemm, sycl::queue& q, int64_t T, int64_t N, int64_t K, float beta) {
    const int64_t ldy = N + 16; // Test non-trivial stride
    std::printf("  Testing Gemm::f16 [T=%lld, N=%lld, K=%lld, ldy=%lld, beta=%.1f]... ",
                (long long)T, (long long)N, (long long)K, (long long)ldy, beta);

    std::mt19937 rng(42 + (unsigned)T * 17 + (unsigned)N * 31);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> h_x((size_t)(T * K));
    for (auto& v : h_x) v = sycl::bit_cast<uint16_t>(sycl::half(dist(rng) * 0.25f));

    std::vector<uint16_t> h_w((size_t)(N * K));
    for (auto& v : h_w) v = sycl::bit_cast<uint16_t>(sycl::half(dist(rng) * 0.25f));

    std::vector<float> h_y((size_t)(T * ldy), 0.0f);
    if (beta != 0.0f) {
        for (auto& v : h_y) v = dist(rng) * 0.5f;
    }

    std::vector<float> ref_y = h_y;
    ref_gemm_f16(h_x, h_w, T, N, K, ldy, beta, ref_y);

    auto* d_x = sycl::malloc_device<uint16_t>(h_x.size(), q);
    auto* d_w = sycl::malloc_device<uint16_t>(h_w.size(), q);
    auto* d_y = sycl::malloc_device<float>(h_y.size(), q);

    q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t));
    q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t));
    q.memcpy(d_y, h_y.data(), h_y.size() * sizeof(float));
    q.wait();

    gemm.f16(d_x, d_w, d_y, T, N, K, ldy, beta);
    q.wait();

    std::vector<float> gpu_y(h_y.size());
    q.memcpy(gpu_y.data(), d_y, gpu_y.size() * sizeof(float));
    q.wait();

    float max_diff = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            const size_t idx = (size_t)(t * ldy + n);
            max_diff = std::max(max_diff, std::abs(gpu_y[idx] - ref_y[idx]));
        }
    }

    if (max_diff <= 1e-2f) {
        std::printf("PASS (max_diff=%.2e)\n", max_diff);
    } else {
        std::printf("FAIL (max_diff=%.2e)\n", max_diff);
        g_failures++;
    }

    sycl::free(d_x, q);
    sycl::free(d_w, q);
    sycl::free(d_y, q);
}

void test_bf16_case(strata::prefill::Gemm& gemm, sycl::queue& q, int64_t T, int64_t N, int64_t K, float beta) {
    const int64_t ldy = N + 16;
    std::printf("  Testing Gemm::bf16 [T=%lld, N=%lld, K=%lld, ldy=%lld, beta=%.1f]... ",
                (long long)T, (long long)N, (long long)K, (long long)ldy, beta);

    std::mt19937 rng(1337 + (unsigned)T * 31 + (unsigned)N * 17);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> h_x((size_t)(T * K));
    for (auto& v : h_x) v = strata::kernels::bf16_from_f32(dist(rng) * 0.25f);

    std::vector<uint16_t> h_w((size_t)(N * K));
    for (auto& v : h_w) v = strata::kernels::bf16_from_f32(dist(rng) * 0.25f);

    std::vector<float> h_y((size_t)(T * ldy), 0.0f);
    if (beta != 0.0f) {
        for (auto& v : h_y) v = dist(rng) * 0.5f;
    }

    std::vector<float> ref_y = h_y;
    ref_gemm_bf16(h_x, h_w, T, N, K, ldy, beta, ref_y);

    auto* d_x = sycl::malloc_device<uint16_t>(h_x.size(), q);
    auto* d_w = sycl::malloc_device<uint16_t>(h_w.size(), q);
    auto* d_y = sycl::malloc_device<float>(h_y.size(), q);

    q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t));
    q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t));
    q.memcpy(d_y, h_y.data(), h_y.size() * sizeof(float));
    q.wait();

    gemm.bf16(d_x, d_w, d_y, T, N, K, ldy, beta);
    q.wait();

    std::vector<float> gpu_y(h_y.size());
    q.memcpy(gpu_y.data(), d_y, gpu_y.size() * sizeof(float));
    q.wait();

    float max_diff = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            const size_t idx = (size_t)(t * ldy + n);
            max_diff = std::max(max_diff, std::abs(gpu_y[idx] - ref_y[idx]));
        }
    }

    if (max_diff <= 1e-2f) {
        std::printf("PASS (max_diff=%.2e)\n", max_diff);
    } else {
        std::printf("FAIL (max_diff=%.2e)\n", max_diff);
        g_failures++;
    }

    sycl::free(d_x, q);
    sycl::free(d_w, q);
    sycl::free(d_y, q);
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::device dev = q.get_device();
    std::printf("=== Running Prefill Gemm XMX Parity Test on: %s ===\n", dev.get_info<sycl::info::device::name>().c_str());

    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(&q, 1024 * 1024, err)) {
        std::fprintf(stderr, "Failed to init Gemm: %s\n", err.c_str());
        return 1;
    }

    // Sweep across various prompt and expert chunk shapes
    std::vector<int64_t> Ts = {1, 5, 8, 17, 32, 64, 128, 512};
    for (int64_t T : Ts) {
        test_f16_case(gemm, q, T, 640, 1280, 0.0f);
        test_f16_case(gemm, q, T, 1280, 2560, 1.0f);
        test_bf16_case(gemm, q, T, 640, 1280, 0.0f);
        test_bf16_case(gemm, q, T, 1280, 2560, 1.0f);
    }

    if (g_failures == 0) {
        std::printf("All Prefill Gemm parity tests PASSED!\n");
        return 0;
    } else {
        std::printf("%d Prefill Gemm parity tests FAILED!\n", g_failures);
        return 1;
    }
}
