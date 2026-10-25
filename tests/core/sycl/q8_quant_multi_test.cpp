// tests/core/sycl/q8_quant_multi_test.cpp - Unit and parity test for Batched Multi-Activation Q8 Quantization on Intel Xe2
#include <sycl/sycl.hpp>
#include "strata/kernels/q8_quant_multi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// CPU reference implementation for Q8 block quantization
void cpu_q8_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d) {
    const int k_blocks = (k + 31) / 32;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k_blocks; ++b) {
            float am = 0.0f;
            for (int lane = 0; lane < 32; ++lane) {
                const int c = b * 32 + lane;
                if (c < k) {
                    am = std::max(am, std::abs(x[static_cast<size_t>(r) * ldx + c]));
                }
            }
            const float sc = am / 127.0f;
            d[static_cast<size_t>(r) * (k / 32) + b] = sc;

            for (int lane = 0; lane < 32; ++lane) {
                const int c = b * 32 + lane;
                if (c < k) {
                    const float v = x[static_cast<size_t>(r) * ldx + c];
                    int8_t qval = 0;
                    if (am > 0.0f) {
                        const int rounded = static_cast<int>(std::rint(v / sc));
                        qval = static_cast<int8_t>(std::clamp(rounded, -127, 127));
                    }
                    q[static_cast<size_t>(r) * k + c] = qval;
                }
            }
        }
    }
}

} // namespace

int main() {
    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    const auto dev = q.get_device();
    std::printf("====================================================================================\n");
    std::printf(" Strata SYCL Batched Multi-Activation Q8 Quantization (q8_quant_multi) Test\n");
    std::printf(" Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("====================================================================================\n\n");

    int passed_tests = 0;
    int total_tests = 0;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Test 1: Single item quantization against CPU reference
    {
        ++total_tests;
        std::printf("[TEST 1] Single Item Quantization vs CPU Reference (rows = 4, k = 2560)\n");

        const int rows = 4;
        const int k = 2560;
        const size_t ldx = k;
        const size_t num_elems = rows * k;
        const size_t num_scales = rows * (k / 32);

        std::vector<float> h_x(num_elems);
        for (auto& v : h_x) v = dist(rng);

        std::vector<int8_t> ref_q(num_elems);
        std::vector<float> ref_d(num_scales);
        cpu_q8_quant(h_x.data(), ldx, rows, k, ref_q.data(), ref_d.data());

        float* d_x = sycl::malloc_device<float>(num_elems, q);
        int8_t* d_q = sycl::malloc_device<int8_t>(num_elems, q);
        float* d_d = sycl::malloc_device<float>(num_scales, q);

        q.memcpy(d_x, h_x.data(), num_elems * sizeof(float)).wait();

        strata::kernels::q8_quant_single(d_x, ldx, rows, k, d_q, d_d, q);
        q.wait();

        std::vector<int8_t> out_q(num_elems);
        std::vector<float> out_d(num_scales);
        q.memcpy(out_q.data(), d_q, num_elems * sizeof(int8_t));
        q.memcpy(out_d.data(), d_d, num_scales * sizeof(float)).wait();

        int diff_q = 0;
        float max_diff_d = 0.0f;
        for (size_t i = 0; i < num_elems; ++i) {
            if (out_q[i] != ref_q[i]) ++diff_q;
        }
        for (size_t i = 0; i < num_scales; ++i) {
            max_diff_d = std::max(max_diff_d, std::abs(out_d[i] - ref_d[i]));
        }

        std::printf("   Single-item results: mismatched codes = %d / %zu, max_scale_diff = %.2e\n",
                    diff_q, num_elems, max_diff_d);

        if (diff_q == 0 && max_diff_d < 1e-5f) {
            std::printf("   -> PASS\n\n");
            ++passed_tests;
        } else {
            std::printf("   -> FAIL\n\n");
        }

        sycl::free(d_x, q);
        sycl::free(d_q, q);
        sycl::free(d_d, q);
    }

    // Test 2: Batched multi-item quantization with heterogeneous shapes
    {
        ++total_tests;
        std::printf("[TEST 2] Multi-Item Batch (N = 6 items with varying dimensions)\n");

        struct ItemSpec {
            int rows;
            int k;
        };
        std::vector<ItemSpec> specs = {
            {1, 512},
            {2, 1024},
            {4, 2048},
            {1, 2560},
            {8, 640},
            {2, 4096}
        };

        const int num_items = static_cast<int>(specs.size());
        std::vector<std::vector<float>> h_x(num_items);
        std::vector<std::vector<int8_t>> ref_q(num_items);
        std::vector<std::vector<float>> ref_d(num_items);
        std::vector<float*> d_x(num_items);
        std::vector<int8_t*> d_q(num_items);
        std::vector<float*> d_d(num_items);

        strata::kernels::QuantItems batch;

        for (int i = 0; i < num_items; ++i) {
            const int r = specs[i].rows;
            const int k = specs[i].k;
            const size_t ne = r * k;
            const size_t ns = r * (k / 32);

            h_x[i].resize(ne);
            for (auto& v : h_x[i]) v = dist(rng);

            ref_q[i].resize(ne);
            ref_d[i].resize(ns);
            cpu_q8_quant(h_x[i].data(), k, r, k, ref_q[i].data(), ref_d[i].data());

            d_x[i] = sycl::malloc_device<float>(ne, q);
            d_q[i] = sycl::malloc_device<int8_t>(ne, q);
            d_d[i] = sycl::malloc_device<float>(ns, q);

            q.memcpy(d_x[i], h_x[i].data(), ne * sizeof(float)).wait();

            int item_id = strata::kernels::quant_add(batch, d_x[i], k, r, k, d_q[i], d_d[i]);
            assert(item_id == i);
        }

        // Launch batched kernel
        strata::kernels::q8_quant_multi(batch, q);
        q.wait();

        bool all_ok = true;
        for (int i = 0; i < num_items; ++i) {
            const size_t ne = specs[i].rows * specs[i].k;
            const size_t ns = specs[i].rows * (specs[i].k / 32);

            std::vector<int8_t> out_q(ne);
            std::vector<float> out_d(ns);
            q.memcpy(out_q.data(), d_q[i], ne * sizeof(int8_t));
            q.memcpy(out_d.data(), d_d[i], ns * sizeof(float)).wait();

            int diff_q = 0;
            float max_diff_d = 0.0f;
            for (size_t j = 0; j < ne; ++j) {
                if (out_q[j] != ref_q[i][j]) ++diff_q;
            }
            for (size_t j = 0; j < ns; ++j) {
                max_diff_d = std::max(max_diff_d, std::abs(out_d[j] - ref_d[i][j]));
            }

            if (diff_q != 0 || max_diff_d > 1e-5f) {
                std::printf("   Item %d mismatch: diff_q = %d, max_diff_d = %.2e\n", i, diff_q, max_diff_d);
                all_ok = false;
            }
        }

        if (all_ok) {
            std::printf("   All %d items bit-exact and matching CPU reference!\n", num_items);
            std::printf("   -> PASS\n\n");
            ++passed_tests;
        } else {
            std::printf("   -> FAIL\n\n");
        }

        for (int i = 0; i < num_items; ++i) {
            sycl::free(d_x[i], q);
            sycl::free(d_q[i], q);
            sycl::free(d_d[i], q);
        }
    }

    // Test 3: Deduplication in quant_add
    {
        ++total_tests;
        std::printf("[TEST 3] Input Activation Deduplication via quant_add\n");

        const int rows = 2;
        const int k = 2048;
        const size_t ne = rows * k;
        const size_t ns = rows * (k / 32);

        float* d_x1 = sycl::malloc_device<float>(ne, q);
        float* d_x2 = sycl::malloc_device<float>(ne, q);
        int8_t* d_q1 = sycl::malloc_device<int8_t>(ne, q);
        int8_t* d_q2 = sycl::malloc_device<int8_t>(ne, q);
        int8_t* d_q3 = sycl::malloc_device<int8_t>(ne, q);
        float* d_d1 = sycl::malloc_device<float>(ns, q);
        float* d_d2 = sycl::malloc_device<float>(ns, q);
        float* d_d3 = sycl::malloc_device<float>(ns, q);

        strata::kernels::QuantItems batch;
        int id1 = strata::kernels::quant_add(batch, d_x1, k, rows, k, d_q1, d_d1);
        int id2 = strata::kernels::quant_add(batch, d_x2, k, rows, k, d_q2, d_d2);
        // Duplicate d_x1 (e.g. gate and up sharing projection input)
        int id3 = strata::kernels::quant_add(batch, d_x1, k, rows, k, d_q3, d_d3);

        std::printf("   Queued jobs: job 1 id=%d, job 2 id=%d, job 3 (duplicate of 1) returned id=%d (batch size = %d)\n",
                    id1, id2, id3, batch.n);

        const bool dedup_ok = (id1 == 0) && (id2 == 1) && (id3 == 0) && (batch.n == 2);
        if (dedup_ok) {
            std::printf("   Deduplication correctly reused slot 0, avoiding redundant kernel execution!\n");
            std::printf("   -> PASS\n\n");
            ++passed_tests;
        } else {
            std::printf("   -> FAIL: Expected id3 == 0 and batch.n == 2\n\n");
        }

        sycl::free(d_x1, q);
        sycl::free(d_x2, q);
        sycl::free(d_q1, q);
        sycl::free(d_q2, q);
        sycl::free(d_q3, q);
        sycl::free(d_d1, q);
        sycl::free(d_d2, q);
        sycl::free(d_d3, q);
    }

    std::printf("====================================================================================\n");
    std::printf(" Summary: %d / %d tests passed\n", passed_tests, total_tests);
    std::printf("====================================================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
