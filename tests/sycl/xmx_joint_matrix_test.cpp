// tests/sycl/xmx_joint_matrix_test.cpp - Standalone test for Intel XMX systolic matrix acceleration.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t TM = 8;
constexpr size_t TN = 16;
constexpr size_t TK = 16;

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::device dev = q.get_device();
    printf("Running XMX Joint Matrix test on: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Check device matrix combinations
    bool has_xmx = false;
    try {
        auto combos = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        if (!combos.empty()) {
            has_xmx = true;
            printf("  Device reports %zu hardware matrix combinations supported.\n", combos.size());
        }
    } catch (...) {
        printf("  Device matrix combinations query not supported.\n");
    }

    if (!has_xmx) {
        printf("  XMX matrix extensions not available on this device; skipping.\n");
        return 0;
    }

    // Test 1: Subgroup tile multiplication (8x16 x 16x16 -> 8x16)
    {
        std::vector<sycl::half> A(TM * TK, sycl::half(1.5f));
        std::vector<sycl::half> B(TK * TN, sycl::half(2.0f));
        std::vector<float> C(TM * TN, 0.0f);

        sycl::buffer<sycl::half, 2> bufA(A.data(), sycl::range<2>(TM, TK));
        sycl::buffer<sycl::half, 2> bufB(B.data(), sycl::range<2>(TK, TN));
        sycl::buffer<float, 2> bufC(C.data(), sycl::range<2>(TM, TN));

        q.submit([&](sycl::handler& cgh) {
            auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
            auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
            auto accC = bufC.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(1, 16), sycl::range<2>(1, 16)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = item.get_sub_group();
                    joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;
                    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;

                    joint_matrix_fill(sg, sub_c, 0.0f);
                    joint_matrix_load(sg, sub_a, accA.get_multi_ptr<sycl::access::decorated::no>(), TK);
                    joint_matrix_load(sg, sub_b, accB.get_multi_ptr<sycl::access::decorated::no>(), TN);
                    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                    joint_matrix_store(sg, sub_c, accC.get_multi_ptr<sycl::access::decorated::no>(), TN, layout::row_major);
                });
        }).wait();

        auto hostC = bufC.get_host_access();
        const float expected = (float)TK * 1.5f * 2.0f; // 16 * 3.0 = 48.0
        float diff = std::abs(hostC[0][0] - expected);
        printf("  Tile [8x16x16]: C[0][0] = %.1f (expected %.1f, diff=%.2e)\n",
               hostC[0][0], expected, diff);

        if (diff > 1e-3f) {
            printf("XMX Joint Matrix test FAILED on single tile!\n");
            return 1;
        }
    }

    // Test 2: Tiled Multi-step Matrix-Matrix Multiplication (32x64 with K=64)
    {
        constexpr size_t M = 32;
        constexpr size_t N = 64;
        constexpr size_t K = 64;

        std::vector<sycl::half> A(M * K);
        std::vector<sycl::half> B(K * N);
        std::vector<float> C(M * N, 0.0f);
        std::vector<float> ref(M * N, 0.0f);

        for (size_t i = 0; i < M * K; ++i) A[i] = sycl::half((float)((i % 7) - 3) * 0.25f);
        for (size_t i = 0; i < K * N; ++i) B[i] = sycl::half((float)((i % 5) - 2) * 0.25f);

        // CPU reference
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                double sum = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    sum += (double)(float)A[m * K + k] * (double)(float)B[k * N + n];
                }
                ref[m * N + n] = (float)sum;
            }
        }

        sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, q);
        sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, q);
        float* d_C = sycl::malloc_device<float>(M * N, q);

        q.memcpy(d_A, A.data(), M * K * sizeof(sycl::half)).wait();
        q.memcpy(d_B, B.data(), K * N * sizeof(sycl::half)).wait();
        q.memset(d_C, 0, M * N * sizeof(float)).wait();

        // Launch grid of (M/TM, N/TN) subgroups
        const size_t grid_m = M / TM;
        const size_t grid_n = N / TN;

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n * 16), sycl::range<2>(1, 16)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = item.get_sub_group();
                    const size_t tile_m = item.get_group(0) * TM;
                    const size_t tile_n = item.get_group(1) * TN;

                    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
                    joint_matrix_fill(sg, sub_c, 0.0f);

                    for (size_t k = 0; k < K; k += TK) {
                        joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
                        joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;

                        auto ptr_a = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_A + tile_m * K + k);
                        auto ptr_b = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_B + k * N + tile_n);
                        joint_matrix_load(sg, sub_a, ptr_a, K);
                        joint_matrix_load(sg, sub_b, ptr_b, N);
                        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                    }

                    auto ptr_c = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_C + tile_m * N + tile_n);
                    joint_matrix_store(sg, sub_c, ptr_c, N, layout::row_major);
                });
        }).wait();

        q.memcpy(C.data(), d_C, M * N * sizeof(float)).wait();

        double max_diff = 0.0;
        int bad = 0;
        for (size_t i = 0; i < M * N; ++i) {
            double diff = std::abs(C[i] - ref[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-2) ++bad;
        }

        printf("  Tiled [32x64x64]: max_diff = %.3e, bad = %d / %zu\n",
               max_diff, bad, M * N);

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

        if (bad > 0) {
            printf("XMX Joint Matrix test FAILED on tiled multiply!\n");
            return 1;
        }
    }

    printf("XMX Joint Matrix test OK (0 failures)\n");
    return 0;
}
