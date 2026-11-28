// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <sycl/sycl.hpp>
#include "strata/kernels/draft_head.hpp"
#include "strata/core/draft_head.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::printf("Running draft_head_test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int total_failures = 0;

    // Test 1: int2_row_bytes
    {
        assert(strata::kernels::kInt2Group == 128);
        assert(strata::kernels::kTopKMax == 64);
        size_t b128 = strata::kernels::int2_row_bytes(128);
        // 128/4 = 32 bytes codes, 1 group = 4 bytes scales -> 36 bytes
        if (b128 != 36) {
            std::printf("FAIL: int2_row_bytes(128) expected 36, got %zu\n", b128);
            total_failures++;
        }
        size_t b2560 = strata::kernels::int2_row_bytes(2560);
        // 2560/4 = 640 bytes codes, 20 groups = 80 bytes scales -> 720 bytes
        if (b2560 != 720) {
            std::printf("FAIL: int2_row_bytes(2560) expected 720, got %zu\n", b2560);
            total_failures++;
        }
        std::printf("int2_row_bytes test: PASS\n");
    }

    // Test 2: int2_pack_rows and int2_gemv
    {
        constexpr int n_in = 256;
        constexpr int64_t n_out = 64;
        const size_t rb = strata::kernels::int2_row_bytes(n_in);

        std::vector<float> host_w(n_out * n_in);
        for (size_t i = 0; i < host_w.size(); ++i) {
            host_w[i] = ((float)(i % 17) - 8.0f) * 0.125f;
        }
        std::vector<uint8_t> packed_w(n_out * rb);
        strata::kernels::int2_pack_rows(host_w.data(), n_out, n_in, packed_w.data());

        std::vector<float> host_x(n_in);
        for (int i = 0; i < n_in; ++i) {
            host_x[i] = ((float)(i % 11) - 5.0f) * 0.2f;
        }

        uint8_t* dev_w = sycl::malloc_device<uint8_t>(packed_w.size(), q);
        float* dev_x = sycl::malloc_device<float>(host_x.size(), q);
        float* dev_y = sycl::malloc_device<float>(n_out, q);

        q.memcpy(dev_w, packed_w.data(), packed_w.size()).wait();
        q.memcpy(dev_x, host_x.data(), host_x.size() * sizeof(float)).wait();

        strata::kernels::int2_gemv(dev_w, dev_x, dev_y, n_in, n_out, &q);
        q.wait();

        std::vector<float> host_y(n_out);
        q.memcpy(host_y.data(), dev_y, n_out * sizeof(float)).wait();

        // Verify that unpacked INT2 dot products match dev_y closely
        for (int64_t r = 0; r < n_out; ++r) {
            const uint8_t* row = packed_w.data() + (size_t)r * rb;
            const auto* scales = reinterpret_cast<const sycl::half*>(row + n_in / 4);
            float expected = 0.0f;
            for (int g = 0; g < n_in / strata::kernels::kInt2Group; ++g) {
                float m = (float)scales[2 * g];
                float d = (float)scales[2 * g + 1];
                for (int i = 0; i < strata::kernels::kInt2Group; ++i) {
                    int at = g * strata::kernels::kInt2Group + i;
                    uint8_t code_byte = row[at / 4];
                    int qval = (code_byte >> (2 * (at % 4))) & 3;
                    float val = m + (float)qval * d;
                    expected += val * host_x[at];
                }
            }
            if (std::abs(host_y[r] - expected) > 1e-2f) {
                std::printf("FAIL: int2_gemv row %lld got %f, expected %f\n", (long long)r, host_y[r], expected);
                total_failures++;
                break;
            }
        }
        std::printf("int2_gemv test: %s\n", total_failures == 0 ? "PASS" : "FAIL");

        sycl::free(dev_w, q);
        sycl::free(dev_x, q);
        sycl::free(dev_y, q);
    }

    // Test 3: topk
    {
        constexpr int64_t n = 4096;
        constexpr int k = 16;
        std::vector<float> v(n);
        for (int64_t i = 0; i < n; ++i) {
            v[i] = (float)i * 0.1f;
        }
        // Shuffle slightly so largest are known
        // Set 16 specific indices to very high values
        std::vector<int32_t> expected_top(k);
        for (int j = 0; j < k; ++j) {
            int32_t idx = (int32_t)(j * 200 + 42);
            v[idx] = 10000.0f + (float)j * 10.0f;
            expected_top[j] = idx;
        }

        float* dev_v = sycl::malloc_device<float>(n, q);
        int32_t* dev_ids = sycl::malloc_device<int32_t>(k, q);
        size_t scratch_sz = strata::kernels::topk_scratch_bytes(n, k);
        void* dev_scratch = sycl::malloc_device(scratch_sz, q);

        q.memcpy(dev_v, v.data(), n * sizeof(float)).wait();

        strata::kernels::topk(dev_v, n, k, dev_ids, dev_scratch, &q);
        q.wait();

        std::vector<int32_t> got_ids(k);
        q.memcpy(got_ids.data(), dev_ids, k * sizeof(int32_t)).wait();

        std::sort(got_ids.begin(), got_ids.end());
        std::sort(expected_top.begin(), expected_top.end());

        bool match = true;
        for (int j = 0; j < k; ++j) {
            if (got_ids[j] != expected_top[j]) {
                std::printf("FAIL: topk[%d] got %d, expected %d\n", j, got_ids[j], expected_top[j]);
                match = false;
                total_failures++;
                break;
            }
        }
        if (match) {
            std::printf("topk test: PASS\n");
        }

        sycl::free(dev_v, q);
        sycl::free(dev_ids, q);
        sycl::free(dev_scratch, q);
    }

    // Test 4: int2_pad_from and scatter_values
    {
        constexpr int64_t n = 128;
        float* dev_y = sycl::malloc_device<float>(n, q);
        int32_t* dev_pos = sycl::malloc_device<int32_t>(1, q);

        std::vector<float> init_y(n, 5.0f);
        q.memcpy(dev_y, init_y.data(), n * sizeof(float)).wait();

        int32_t pos_val = 1000;
        q.memcpy(dev_pos, &pos_val, sizeof(int32_t)).wait();

        strata::kernels::int2_pad_from(dev_pos, 2000, dev_y, n, &q);
        q.wait();

        std::vector<float> check_y(n);
        q.memcpy(check_y.data(), dev_y, n * sizeof(float)).wait();
        if (check_y[0] != 5.0f) {
            std::printf("FAIL: int2_pad_from should not pad when pos < from\n");
            total_failures++;
        }

        pos_val = 2500;
        q.memcpy(dev_pos, &pos_val, sizeof(int32_t)).wait();
        strata::kernels::int2_pad_from(dev_pos, 2000, dev_y, n, &q);
        q.wait();
        q.memcpy(check_y.data(), dev_y, n * sizeof(float)).wait();
        if (check_y[0] != -1.7e38f) {
            std::printf("FAIL: int2_pad_from should pad when pos >= from\n");
            total_failures++;
        }

        // scatter_values
        constexpr int k = 4;
        int32_t scat_ids[k] = {3, 10, 42, 99};
        float scat_vals[k] = {1.1f, 2.2f, 3.3f, 4.4f};
        int32_t* dev_scat_ids = sycl::malloc_device<int32_t>(k, q);
        float* dev_scat_vals = sycl::malloc_device<float>(k, q);
        q.memcpy(dev_scat_ids, scat_ids, k * sizeof(int32_t)).wait();
        q.memcpy(dev_scat_vals, scat_vals, k * sizeof(float)).wait();

        strata::kernels::scatter_values(dev_scat_vals, dev_scat_ids, k, dev_y, &q);
        q.wait();

        q.memcpy(check_y.data(), dev_y, n * sizeof(float)).wait();
        if (check_y[3] != 1.1f || check_y[10] != 2.2f || check_y[42] != 3.3f || check_y[99] != 4.4f) {
            std::printf("FAIL: scatter_values did not match\n");
            total_failures++;
        } else {
            std::printf("int2_pad_from & scatter_values test: PASS\n");
        }

        sycl::free(dev_y, q);
        sycl::free(dev_pos, q);
        sycl::free(dev_scat_ids, q);
        sycl::free(dev_scat_vals, q);
    }

    // Test 5: DraftHeadInt2 build & logits
    {
        constexpr int64_t n_rows = 128;
        constexpr int n_in = 256;
        // Float rows (type GGML_TYPE_F32 = 0)
        std::vector<float> host_data(n_rows * n_in);
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = ((float)(i % 13) - 6.0f) * 0.1f;
        }

        strata::core::DraftHeadInt2 dh;
        std::string err;
        // row bytes for F32 is n_in * sizeof(float)
        size_t exact_rb = n_in * sizeof(float);
        bool built = dh.build(0 /* GGML_TYPE_F32 */, n_rows, n_in,
                              [&](int64_t r) { return reinterpret_cast<const uint8_t*>(host_data.data() + r * n_in); },
                              exact_rb, err);
        if (!built) {
            std::printf("FAIL: DraftHeadInt2::build failed: %s\n", err.c_str());
            total_failures++;
        } else {
            std::printf("DraftHeadInt2::build test: PASS (holds %llu bytes)\n", (unsigned long long)dh.bytes());

            // Run logits
            float* dev_x = sycl::malloc_device<float>(n_in, q);
            uint8_t* dev_head_rows = sycl::malloc_device<uint8_t>(n_rows * exact_rb, q);
            float* dev_out = sycl::malloc_device<float>(n_rows, q);

            std::vector<float> x(n_in, 0.5f);
            q.memcpy(dev_x, x.data(), n_in * sizeof(float)).wait();
            q.memcpy(dev_head_rows, host_data.data(), n_rows * exact_rb).wait();

            const strata::core::DraftHeadInt2::Exact exact = [n_in, dev_x](const uint8_t* r, int n_cands, float* y, void* s) {
                // Exact dot product of candidates
                sycl::queue* qp = s ? static_cast<sycl::queue*>(s) : nullptr;
                const auto* rf = reinterpret_cast<const float*>(r);
                qp->parallel_for(sycl::range<1>(n_cands), [=](sycl::id<1> idx) {
                    float dot = 0.0f;
                    for (int i = 0; i < n_in; ++i) {
                        dot += rf[idx * n_in + i] * dev_x[i];
                    }
                    y[idx] = dot;
                });
            };

            dh.logits(dev_x, dev_head_rows, exact, dev_out, &q);
            q.wait();

            std::vector<float> logits_out(n_rows);
            q.memcpy(logits_out.data(), dev_out, n_rows * sizeof(float)).wait();

            // Check that output has valid finite numbers
            bool all_finite = true;
            for (float v : logits_out) {
                if (!std::isfinite(v)) all_finite = false;
            }
            if (!all_finite) {
                std::printf("FAIL: DraftHeadInt2::logits output has non-finite values\n");
                total_failures++;
            } else {
                std::printf("DraftHeadInt2::logits test: PASS\n");
            }

            sycl::free(dev_x, q);
            sycl::free(dev_head_rows, q);
            sycl::free(dev_out, q);
        }
    }

    std::printf("\ndraft_head_test: %s (failures: %d)\n", total_failures == 0 ? "PASSED" : "FAILED", total_failures);
    return total_failures == 0 ? 0 : 1;
}
