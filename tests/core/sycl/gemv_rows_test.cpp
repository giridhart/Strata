// tests/core/sycl/gemv_rows_test.cpp - Unit and regression test for Multi-Row GEMV on SYCL/Xe2
//
// Verifies numerical equivalence between standard GEMV, multi-row GEMV (gemv_f16_rows<4> and <2>),
// and engine dispatch (gemv_f16_dispatch) across token counts T in {1, 2, 4, 8} and
// shapes K, N in {128, 256, 512, 1024, 2048, 2560}.

#include <sycl/sycl.hpp>
#include "strata/kernels/gemv_f16_rows.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

// CPU reference GEMV
void cpu_gemv_f16(const sycl::half* W, const float* x, int n, int k, int T, float alpha,
                  float* y, size_t ldx = 0, size_t ldy = 0) {
    if (ldx == 0) ldx = k;
    if (ldy == 0) ldy = n;

    for (int t = 0; t < T; ++t) {
        const float* xt = x + static_cast<size_t>(t) * ldx;
        float* yt = y + static_cast<size_t>(t) * ldy;
        for (int r = 0; r < n; ++r) {
            const sycl::half* wr = W + static_cast<size_t>(r) * k;
            double sum = 0.0;
            for (int c = 0; c < k; ++c) {
                sum += static_cast<double>(static_cast<float>(wr[c])) * static_cast<double>(xt[c]);
            }
            yt[r] = static_cast<float>(alpha * sum);
        }
    }
}

struct ShapeTest {
    int K;
    int N;
};

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});

    const auto dev = q.get_device();
    std::printf("====================================================================================\n");
    std::printf(" Strata SYCL Multi-Row GEMV (gemv_f16_rows) Unit & Regression Test\n");
    std::printf(" Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf(" Driver: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("====================================================================================\n\n");

    const std::vector<ShapeTest> test_shapes = {
        // Square shapes spanning required {128, 256, 512, 1024, 2048, 2560}
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {2560, 2560},
        // Non-square projections
        {2560, 512},
        {512, 2560},
        {2048, 1024},
        {1024, 2048},
        {2560, 128},
        {128, 2560}
    };

    const std::vector<int> token_counts = {1, 2, 4, 8};

    int passed_cases = 0;
    int total_cases = 0;
    constexpr float kTolerance = 1e-4f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (const auto& shape : test_shapes) {
        const int K = shape.K;
        const int N = shape.N;
        const size_t w_elems = static_cast<size_t>(N) * static_cast<size_t>(K);

        // Host weight tensor
        std::vector<sycl::half> h_w(w_elems);
        for (size_t i = 0; i < w_elems; ++i) {
            h_w[i] = sycl::half(dist(rng) * 0.1f);
        }

        sycl::half* d_w = sycl::malloc_device<sycl::half>(w_elems, q);
        q.memcpy(d_w, h_w.data(), w_elems * sizeof(sycl::half)).wait();

        for (int T : token_counts) {
            const size_t x_elems = static_cast<size_t>(T) * static_cast<size_t>(K);
            const size_t y_elems = static_cast<size_t>(T) * static_cast<size_t>(N);

            std::vector<float> h_x(x_elems);
            for (size_t i = 0; i < x_elems; ++i) {
                h_x[i] = dist(rng) * 0.2f;
            }

            std::vector<float> h_y_ref(y_elems, 0.0f);
            cpu_gemv_f16(h_w.data(), h_x.data(), N, K, T, 1.0f, h_y_ref.data());

            float* d_x = sycl::malloc_device<float>(x_elems, q);
            float* d_y_base = sycl::malloc_device<float>(y_elems, q);
            float* d_y_row4 = sycl::malloc_device<float>(y_elems, q);
            float* d_y_row2 = sycl::malloc_device<float>(y_elems, q);
            float* d_y_disp = sycl::malloc_device<float>(y_elems, q);

            q.memcpy(d_x, h_x.data(), x_elems * sizeof(float)).wait();

            // 1. Baseline standard single-row GEMV
            q.fill(d_y_base, 0.0f, y_elems);
            strata::kernels::gemv_f16(d_w, d_x, N, K, T, 1.0f, d_y_base, q);
            q.wait();

            // 2. Multi-row GEMV (ROWS_T = 4)
            q.fill(d_y_row4, 0.0f, y_elems);
            strata::kernels::gemv_f16_rows<4>(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y_row4, (size_t)N, q);
            q.wait();

            // 3. Multi-row GEMV (ROWS_T = 2)
            q.fill(d_y_row2, 0.0f, y_elems);
            strata::kernels::gemv_f16_rows<2>(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y_row2, (size_t)N, q);
            q.wait();

            // 4. Engine dispatch (gemv_f16_dispatch)
            q.fill(d_y_disp, 0.0f, y_elems);
            strata::kernels::set_multi_row_gemv_enabled(true);
            strata::kernels::gemv_f16_dispatch(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y_disp, (size_t)N, q);
            q.wait();

            // Readback device outputs
            std::vector<float> h_dev_base(y_elems);
            std::vector<float> h_dev_row4(y_elems);
            std::vector<float> h_dev_row2(y_elems);
            std::vector<float> h_dev_disp(y_elems);

            q.memcpy(h_dev_base.data(), d_y_base, y_elems * sizeof(float)).wait();
            q.memcpy(h_dev_row4.data(), d_y_row4, y_elems * sizeof(float)).wait();
            q.memcpy(h_dev_row2.data(), d_y_row2, y_elems * sizeof(float)).wait();
            q.memcpy(h_dev_disp.data(), d_y_disp, y_elems * sizeof(float)).wait();

            // Check diff between gemv_f16_rows<4> and baseline gemv_f16
            float max_diff_base_row4 = 0.0f;
            float max_diff_base_row2 = 0.0f;
            float max_diff_base_disp = 0.0f;
            float max_diff_ref_row4  = 0.0f;

            for (size_t i = 0; i < y_elems; ++i) {
                const float diff4 = std::abs(h_dev_row4[i] - h_dev_base[i]);
                if (diff4 > max_diff_base_row4) max_diff_base_row4 = diff4;

                const float diff2 = std::abs(h_dev_row2[i] - h_dev_base[i]);
                if (diff2 > max_diff_base_row2) max_diff_base_row2 = diff2;

                const float diffd = std::abs(h_dev_disp[i] - h_dev_base[i]);
                if (diffd > max_diff_base_disp) max_diff_base_disp = diffd;

                const float diffr = std::abs(h_dev_row4[i] - h_y_ref[i]);
                if (diffr > max_diff_ref_row4) max_diff_ref_row4 = diffr;
            }

            total_cases++;
            const bool ok = (max_diff_base_row4 <= kTolerance) &&
                            (max_diff_base_row2 <= kTolerance) &&
                            (max_diff_base_disp <= kTolerance) &&
                            (max_diff_ref_row4  <= kTolerance);
            if (ok) passed_cases++;

            std::printf("[%s] Shape [K=%4d, N=%4d] | T=%-2d | diff(Base vs Row4): %.2e | diff(Dispatch): %.2e | diff(CPU Ref): %.2e\n",
                        ok ? "PASS" : "FAIL", K, N, T, max_diff_base_row4, max_diff_base_disp, max_diff_ref_row4);

            sycl::free(d_x, q);
            sycl::free(d_y_base, q);
            sycl::free(d_y_row4, q);
            sycl::free(d_y_row2, q);
            sycl::free(d_y_disp, q);
        }

        sycl::free(d_w, q);
    }

    // -------------------------------------------------------------------------
    // Test 5: Verify Stream Interoperability (void* stream, queue*, queue&)
    // -------------------------------------------------------------------------
    {
        const int K = 512, N = 512, T = 4;
        const size_t w_elems = (size_t)N * K;
        const size_t x_elems = (size_t)T * K;
        const size_t y_elems = (size_t)T * N;

        std::vector<sycl::half> h_w(w_elems, sycl::half(0.01f));
        std::vector<float> h_x(x_elems, 0.05f);

        sycl::half* d_w = sycl::malloc_device<sycl::half>(w_elems, q);
        float* d_x = sycl::malloc_device<float>(x_elems, q);
        float* d_y1 = sycl::malloc_device<float>(y_elems, q);
        float* d_y2 = sycl::malloc_device<float>(y_elems, q);
        float* d_y3 = sycl::malloc_device<float>(y_elems, q);

        q.memcpy(d_w, h_w.data(), w_elems * sizeof(sycl::half)).wait();
        q.memcpy(d_x, h_x.data(), x_elems * sizeof(float)).wait();

        // Overload 1: sycl::queue&
        strata::kernels::gemv_f16_dispatch(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y1, (size_t)N, q);
        // Overload 2: sycl::queue*
        strata::kernels::gemv_f16_dispatch(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y2, (size_t)N, &q);
        // Overload 3: void* stream
        void* stream_ptr = static_cast<void*>(&q);
        strata::kernels::gemv_f16_dispatch(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y3, (size_t)N, stream_ptr);
        q.wait();

        std::vector<float> h_y1(y_elems), h_y2(y_elems), h_y3(y_elems);
        q.memcpy(h_y1.data(), d_y1, y_elems * sizeof(float)).wait();
        q.memcpy(h_y2.data(), d_y2, y_elems * sizeof(float)).wait();
        q.memcpy(h_y3.data(), d_y3, y_elems * sizeof(float)).wait();

        float max_diff_stream = 0.0f;
        for (size_t i = 0; i < y_elems; ++i) {
            max_diff_stream = std::max(max_diff_stream, std::abs(h_y1[i] - h_y2[i]));
            max_diff_stream = std::max(max_diff_stream, std::abs(h_y1[i] - h_y3[i]));
        }

        total_cases++;
        bool ok_stream = (max_diff_stream == 0.0f);
        if (ok_stream) passed_cases++;
        std::printf("[%s] Stream Interoperability (queue&, queue*, void*) | Max diff: %.2e\n",
                    ok_stream ? "PASS" : "FAIL", max_diff_stream);

        sycl::free(d_w, q);
        sycl::free(d_x, q);
        sycl::free(d_y1, q);
        sycl::free(d_y2, q);
        sycl::free(d_y3, q);
    }

    // -------------------------------------------------------------------------
    // Test 6: Runtime Toggle (set_multi_row_gemv_enabled & fallback behavior)
    // -------------------------------------------------------------------------
    {
        const int K = 256, N = 256, T = 2;
        const size_t w_elems = (size_t)N * K;
        const size_t x_elems = (size_t)T * K;
        const size_t y_elems = (size_t)T * N;

        sycl::half* d_w = sycl::malloc_device<sycl::half>(w_elems, q);
        float* d_x = sycl::malloc_device<float>(x_elems, q);
        float* d_y_on = sycl::malloc_device<float>(y_elems, q);
        float* d_y_off = sycl::malloc_device<float>(y_elems, q);

        q.fill(d_w, sycl::half(0.02f), w_elems);
        q.fill(d_x, 0.1f, x_elems);
        q.wait();

        // Enabled
        strata::kernels::set_multi_row_gemv_enabled(true);
        strata::kernels::gemv_f16_dispatch(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y_on, (size_t)N, q);

        // Disabled (falls back to single-row gemv_f16)
        strata::kernels::set_multi_row_gemv_enabled(false);
        strata::kernels::gemv_f16_dispatch(d_w, d_x, (size_t)K, N, K, T, 1.0f, d_y_off, (size_t)N, q);
        q.wait();

        std::vector<float> h_on(y_elems), h_off(y_elems);
        q.memcpy(h_on.data(), d_y_on, y_elems * sizeof(float)).wait();
        q.memcpy(h_off.data(), d_y_off, y_elems * sizeof(float)).wait();

        float max_diff_toggle = 0.0f;
        for (size_t i = 0; i < y_elems; ++i) {
            max_diff_toggle = std::max(max_diff_toggle, std::abs(h_on[i] - h_off[i]));
        }

        total_cases++;
        bool ok_toggle = (max_diff_toggle <= kTolerance);
        if (ok_toggle) passed_cases++;
        std::printf("[%s] Runtime Toggle (STRATA_MULTI_ROW_GEMV On vs Off fallback) | Max diff: %.2e\n",
                    ok_toggle ? "PASS" : "FAIL", max_diff_toggle);

        // Restore default enabled
        strata::kernels::set_multi_row_gemv_enabled(true);

        sycl::free(d_w, q);
        sycl::free(d_x, q);
        sycl::free(d_y_on, q);
        sycl::free(d_y_off, q);
    }

    std::printf("\n====================================================================================\n");
    std::printf(" Result: %d / %d test suites PASSED (%s)\n",
                passed_cases, total_cases, (passed_cases == total_cases) ? "ALL TESTS PASSED" : "FAILURES DETECTED");
    std::printf("====================================================================================\n");

    return (passed_cases == total_cases) ? 0 : 1;
}
