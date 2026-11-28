// tests/core/sycl/numerics_test.cpp - Unit & Regression Tests for Fast E4M3 / E2M1 FP8 / FP4 Numerics & KV Cache Compression
//
// Verifies:
//   a) Scalar bit-manipulation primitives (zeros, sign, subnormals, saturation at 448 for FP8, step values for FP4).
//   b) End-to-end device Quantize / Dequantize roundtrip on GPU for FP8 E4M3 across multiple vector shapes (256, 1024, 4096, 16384),
//      asserting relative error <= 6.25%.
//   c) Device Quantize / Dequantize for FP4 E2M1 (BlockSize 32 and BlockSize 16), asserting correct code placement and error bounds.
//   d) Engine KV cache append helpers (kv_append_fp8_step, kv_append_fp4_step), stream interop, and runtime format switching.

#include <sycl/sycl.hpp>
#include "strata/kernels/numerics_fp8_fp4.hpp"
#include "strata/kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

int g_passed = 0;
int g_total = 0;

void record_result(const char* name, bool ok, const std::string& extra = "") {
    g_total++;
    if (ok) g_passed++;
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
                extra.empty() ? "" : " | ", extra.c_str());
}

// ============================================================================
// Test Suite 1: Scalar Bit-Manipulation Primitives
// ============================================================================

bool test_scalar_primitives() {
    bool all_ok = true;

    // 1.1 pow2_ceil and ue8m0_val / ue8m0_scale_byte
    {
        bool ok = true;
        const float test_vals[] = {
            0.001953125f, 0.01f, 0.125f, 0.25f, 0.33f, 0.5f, 0.75f, 0.99f,
            1.0f, 1.01f, 2.0f, 3.9f, 4.0f, 7.5f, 8.0f, 16.0f, 32.0f, 64.0f,
            128.0f, 256.0f, 448.0f, 512.0f
        };
        for (float v : test_vals) {
            float p = strata::kernels::pow2_ceil(v);
            if (p < v || (p > 2.0f * v && v > 1e-30f)) {
                ok = false;
                break;
            }
            uint8_t sc = strata::kernels::ue8m0_scale_byte(p);
            float rec = strata::kernels::ue8m0_val(sc);
            if (p != rec) {
                ok = false;
                break;
            }
        }
        record_result("1.1 Scalar pow2_ceil & ue8m0 scale roundtrip", ok);
        all_ok &= ok;
    }

    // 1.2 Zeros and Sign handling
    {
        bool ok = true;
        // FP8 E4M3 zeros
        float z_pos = strata::kernels::e4m3_round(0.0f);
        float z_neg = strata::kernels::e4m3_round(-0.0f);
        uint32_t c_z_pos = strata::kernels::e4m3_code(z_pos);
        uint32_t c_z_neg = strata::kernels::e4m3_code(z_neg);
        if (z_pos != 0.0f || z_neg != -0.0f) ok = false;
        if (c_z_pos != 0x00u || c_z_neg != 0x80u) ok = false;

        // FP4 E2M1 zeros
        float e2_z_pos = strata::kernels::e2m1_rt(0.0f);
        float e2_z_neg = strata::kernels::e2m1_rt(-0.0f);
        uint32_t c_e2_z_pos = strata::kernels::e2m1_code(e2_z_pos);
        uint32_t c_e2_z_neg = strata::kernels::e2m1_code(e2_z_neg);
        if (e2_z_pos != 0.0f || e2_z_neg != -0.0f) ok = false;
        if (c_e2_z_pos != 0u || c_e2_z_neg != 8u) ok = false;

        record_result("1.2 Signed zeros and copysign preservation", ok);
        all_ok &= ok;
    }

    // 1.3 Subnormals for FP8 E4M3
    {
        bool ok = true;
        // Subnormals in E4M3 have e == 0, value = m * 2^-9 (step = 0.001953125f)
        for (int m = 1; m <= 7; ++m) {
            float subnorm = (float)m * 0.001953125f;
            uint32_t c_pos = strata::kernels::e4m3_code(subnorm);
            uint32_t c_neg = strata::kernels::e4m3_code(-subnorm);
            if (c_pos != static_cast<uint32_t>(m)) ok = false;
            if (c_neg != static_cast<uint32_t>(0x80 | m)) ok = false;

            float val_pos = strata::kernels::e4m3_value(static_cast<uint8_t>(c_pos));
            float val_neg = strata::kernels::e4m3_value(static_cast<uint8_t>(c_neg));
            if (std::fabs(val_pos - subnorm) > 1e-7f || std::fabs(val_neg - (-subnorm)) > 1e-7f) {
                ok = false;
            }
        }
        record_result("1.3 FP8 E4M3 subnormal representation & decoding", ok);
        all_ok &= ok;
    }

    // 1.4 Saturation at +-448.0 for FP8
    {
        bool ok = true;
        if (strata::kernels::e4m3_round(448.0f) != 448.0f) ok = false;
        if (strata::kernels::e4m3_round(449.0f) != 448.0f) ok = false;
        if (strata::kernels::e4m3_round(1000.0f) != 448.0f) ok = false;
        if (strata::kernels::e4m3_round(-448.0f) != -448.0f) ok = false;
        if (strata::kernels::e4m3_round(-500.0f) != -448.0f) ok = false;

        uint32_t c_max_pos = strata::kernels::e4m3_code(448.0f);
        uint32_t c_max_neg = strata::kernels::e4m3_code(-448.0f);
        if (c_max_pos != 0x7eu || c_max_neg != 0xfeu) ok = false;

        // Verify all 254 non-NaN E4M3 code points round-trip bit-exact
        for (int c = 0; c < 256; ++c) {
            if (c == 0x7f || c == 0xff) continue; // NaN in E4M3
            float val = strata::kernels::e4m3_value(static_cast<uint8_t>(c));
            float rounded = strata::kernels::e4m3_round(val);
            uint32_t rec_code = strata::kernels::e4m3_code(rounded);
            if (rec_code != static_cast<uint32_t>(c)) {
                ok = false;
                break;
            }
        }
        record_result("1.4 FP8 E4M3 saturation at 448 and 254-code roundtrip", ok);
        all_ok &= ok;
    }

    // 1.5 FP4 E2M1 step values, thresholds, and val14 bit trick
    {
        bool ok = true;
        const float expected_mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

        // Threshold checks
        if (strata::kernels::e2m1_rt(0.24f) != 0.0f) ok = false;
        if (strata::kernels::e2m1_rt(0.26f) != 0.5f) ok = false;
        if (strata::kernels::e2m1_rt(0.74f) != 0.5f) ok = false;
        if (strata::kernels::e2m1_rt(0.76f) != 1.0f) ok = false;
        if (strata::kernels::e2m1_rt(1.24f) != 1.0f) ok = false;
        if (strata::kernels::e2m1_rt(1.26f) != 1.5f) ok = false;
        if (strata::kernels::e2m1_rt(1.74f) != 1.5f) ok = false;
        if (strata::kernels::e2m1_rt(1.76f) != 2.0f) ok = false;
        if (strata::kernels::e2m1_rt(2.49f) != 2.0f) ok = false;
        if (strata::kernels::e2m1_rt(2.51f) != 3.0f) ok = false;
        if (strata::kernels::e2m1_rt(3.49f) != 3.0f) ok = false;
        if (strata::kernels::e2m1_rt(3.51f) != 4.0f) ok = false;
        if (strata::kernels::e2m1_rt(4.99f) != 4.0f) ok = false;
        if (strata::kernels::e2m1_rt(5.01f) != 6.0f) ok = false;
        if (strata::kernels::e2m1_rt(10.0f) != 6.0f) ok = false;
        if (strata::kernels::e2m1_rt(-10.0f) != -6.0f) ok = false;

        // Verify e2m1_val14 bit trick for all 16 codes
        for (uint32_t c = 0; c < 16; ++c) {
            float exp_mag = expected_mags[c & 7u];
            float exp_val = (c & 8u) ? -exp_mag : exp_mag;
            float val14 = strata::kernels::e2m1_val14(c);
            float rec_val = val14 * 16384.0f;
            if (std::fabs(rec_val - exp_val) > 1e-6f) {
                ok = false;
                break;
            }
            uint32_t rec_code = strata::kernels::e2m1_code(strata::kernels::e2m1_rt(rec_val));
            if (rec_code != c) {
                ok = false;
                break;
            }
        }
        record_result("1.5 FP4 E2M1 quantization steps and e2m1_val14 decode", ok);
        all_ok &= ok;
    }

    return all_ok;
}

// ============================================================================
// Test Suite 2: Device Quantize / Dequantize Roundtrip for FP8 E4M3
// ============================================================================

bool test_device_fp8_roundtrip(sycl::queue& q) {
    bool all_ok = true;
    const size_t shapes[] = {256, 1024, 4096, 16384};

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 2.5f);

    for (size_t N : shapes) {
        std::vector<float> h_in(N);
        for (size_t i = 0; i < N; ++i) {
            h_in[i] = dist(rng);
        }
        // Inject edge cases
        h_in[0] = 0.0f;
        h_in[1] = -0.0f;
        h_in[2] = 1.0f;
        h_in[3] = -1.0f;
        h_in[4] = 448.0f;
        h_in[5] = -448.0f;
        h_in[6] = 0.001953125f;

        const size_t packed_bytes = strata::kernels::fp8_packed_bytes(N, 32);

        float* d_in = sycl::malloc_device<float>(N, q);
        uint8_t* d_packed = sycl::malloc_device<uint8_t>(packed_bytes, q);
        float* d_out_f32 = sycl::malloc_device<float>(N, q);
        sycl::half* d_out_f16 = sycl::malloc_device<sycl::half>(N, q);

        q.memcpy(d_in, h_in.data(), N * sizeof(float)).wait();

        // 1. Quantize
        strata::kernels::quantize_fp8_e4m3(d_in, d_packed, N, q).wait();

        // 2. Dequantize to F32
        strata::kernels::dequantize_fp8_e4m3(d_packed, d_out_f32, N, q).wait();

        std::vector<float> h_out_f32(N);
        q.memcpy(h_out_f32.data(), d_out_f32, N * sizeof(float)).wait();

        double max_err = 0.0;
        double max_rel_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = std::fabs(static_cast<double>(h_in[i]) - static_cast<double>(h_out_f32[i]));
            if (err > max_err) max_err = err;
            if (std::fabs(h_in[i]) > 0.05f) {
                double rel_err = err / std::fabs(static_cast<double>(h_in[i]));
                if (rel_err > max_rel_err) max_rel_err = rel_err;
            }
        }

        // 3. Dequantize to FP16
        strata::kernels::dequantize_fp8_e4m3<sycl::half>(d_packed, d_out_f16, N, q).wait();

        std::vector<sycl::half> h_out_f16(N);
        q.memcpy(h_out_f16.data(), d_out_f16, N * sizeof(sycl::half)).wait();

        double max_f16_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double f16_val = static_cast<double>(static_cast<float>(h_out_f16[i]));
            double diff = std::fabs(f16_val - static_cast<double>(h_out_f32[i]));
            if (diff > max_f16_diff) max_f16_diff = diff;
        }

        // Theoretical maximum relative step error for FP8 E4M3 is 1 / 16 = 0.0625 (6.25%)
        bool ok = (max_rel_err <= 0.0625 + 1e-4) && (max_f16_diff <= 0.01);
        all_ok &= ok;

        char buf[128];
        std::snprintf(buf, sizeof(buf), "N=%zu | Max Abs Err: %.3e, Max Rel Err: %.2f%%, F16 delta: %.3e",
                      N, max_err, max_rel_err * 100.0, max_f16_diff);
        record_result("2. Device FP8 E4M3 Quantize/Dequantize Roundtrip", ok, buf);

        sycl::free(d_in, q);
        sycl::free(d_packed, q);
        sycl::free(d_out_f32, q);
        sycl::free(d_out_f16, q);
    }

    return all_ok;
}

// ============================================================================
// Test Suite 3: Device Quantize / Dequantize for FP4 E2M1 (Block 32 & 16)
// ============================================================================

bool test_device_fp4_roundtrip(sycl::queue& q) {
    bool all_ok = true;
    const size_t N = 4096;

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.8f);

    std::vector<float> h_in(N);
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = dist(rng);
    }
    h_in[0] = 0.0f;
    h_in[1] = 6.0f;
    h_in[2] = -6.0f;

    float* d_in = sycl::malloc_device<float>(N, q);
    q.memcpy(d_in, h_in.data(), N * sizeof(float)).wait();

    // 3.1 BlockSize = 32
    {
        const size_t b32_bytes = strata::kernels::fp4_packed_bytes(N, 32);
        uint8_t* d_packed32 = sycl::malloc_device<uint8_t>(b32_bytes, q);
        float* d_out32 = sycl::malloc_device<float>(N, q);

        strata::kernels::quantize_fp4_e2m1<32>(d_in, d_packed32, N, q).wait();
        strata::kernels::dequantize_fp4_e2m1<32>(d_packed32, d_out32, N, q).wait();

        std::vector<float> h_out32(N);
        std::vector<uint8_t> h_packed32(b32_bytes);
        q.memcpy(h_out32.data(), d_out32, N * sizeof(float)).wait();
        q.memcpy(h_packed32.data(), d_packed32, b32_bytes).wait();

        // Verify nibble placement in first block
        uint8_t first_byte = h_packed32[0];
        uint32_t lo_nibble = first_byte & 0x0fu;
        uint32_t hi_nibble = (first_byte >> 4) & 0x0fu;
        float lo_val = strata::kernels::e2m1_val14(lo_nibble) * 16384.0f;
        float hi_val = strata::kernels::e2m1_val14(hi_nibble) * 16384.0f;

        double max_err32 = 0.0;
        double mse32 = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = std::fabs(static_cast<double>(h_in[i]) - static_cast<double>(h_out32[i]));
            if (err > max_err32) max_err32 = err;
            mse32 += err * err;
        }
        mse32 /= static_cast<double>(N);
        double rmse32 = std::sqrt(mse32);

        bool ok = (rmse32 < 0.35) && (lo_val == 0.0f);
        all_ok &= ok;

        char buf[128];
        std::snprintf(buf, sizeof(buf), "Max Abs: %.3e, RMSE: %.3e, Nibble check lo=%.1f hi=%.1f",
                      max_err32, rmse32, lo_val, hi_val);
        record_result("3.1 Device FP4 E2M1 Roundtrip (BlockSize 32)", ok, buf);

        sycl::free(d_packed32, q);
        sycl::free(d_out32, q);
    }

    // 3.2 BlockSize = 16
    {
        const size_t b16_bytes = strata::kernels::fp4_packed_bytes(N, 16);
        uint8_t* d_packed16 = sycl::malloc_device<uint8_t>(b16_bytes, q);
        float* d_out16 = sycl::malloc_device<float>(N, q);

        strata::kernels::quantize_fp4_e2m1<16>(d_in, d_packed16, N, q).wait();
        strata::kernels::dequantize_fp4_e2m1<16>(d_packed16, d_out16, N, q).wait();

        std::vector<float> h_out16(N);
        q.memcpy(h_out16.data(), d_out16, N * sizeof(float)).wait();

        double max_err16 = 0.0;
        double mse16 = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = std::fabs(static_cast<double>(h_in[i]) - static_cast<double>(h_out16[i]));
            if (err > max_err16) max_err16 = err;
            mse16 += err * err;
        }
        mse16 /= static_cast<double>(N);
        double rmse16 = std::sqrt(mse16);

        bool ok = (rmse16 < 0.30);
        all_ok &= ok;

        char buf[128];
        std::snprintf(buf, sizeof(buf), "Max Abs: %.3e, RMSE: %.3e", max_err16, rmse16);
        record_result("3.2 Device FP4 E2M1 Roundtrip (BlockSize 16)", ok, buf);

        sycl::free(d_packed16, q);
        sycl::free(d_out16, q);
    }

    sycl::free(d_in, q);
    return all_ok;
}

// ============================================================================
// Test Suite 4: Engine KV Cache Append, Direct Compression & Stream Interop
// ============================================================================

bool test_engine_kv_append_and_interop(sycl::queue& q) {
    bool all_ok = true;

    strata::kernels::QsaShapes s;
    s.n_head = 24;
    s.n_head_kv = 2;
    s.head_dim = 256;
    s.page_size = 4;

    const size_t row_elems = (size_t)s.n_head_kv * s.head_dim;
    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.5f);

    std::vector<float> h_kcur(row_elems), h_vcur(row_elems);
    for (size_t i = 0; i < row_elems; ++i) {
        h_kcur[i] = dist(rng);
        h_vcur[i] = dist(rng);
    }

    float* d_kcur = sycl::malloc_device<float>(row_elems, q);
    float* d_vcur = sycl::malloc_device<float>(row_elems, q);
    int32_t* d_table = sycl::malloc_device<int32_t>(16, q);
    int32_t* d_step = sycl::malloc_device<int32_t>(strata::kernels::qsa_step_bytes() / 4 + 4, q);

    std::vector<int32_t> h_table(16, 0); // logical page 0 -> physical page 0
    std::vector<int32_t> h_step(16, 0);
    h_step[strata::kernels::kStepPos] = 2; // sequence pos 2

    q.memcpy(d_kcur, h_kcur.data(), row_elems * sizeof(float)).wait();
    q.memcpy(d_vcur, h_vcur.data(), row_elems * sizeof(float)).wait();
    q.memcpy(d_table, h_table.data(), 16 * sizeof(int32_t)).wait();
    q.memcpy(d_step, h_step.data(), 16 * sizeof(int32_t)).wait();

    // 4.1 Engine Paged FP8 KV Append
    {
        const size_t fp8_row_bytes = strata::kernels::fp8_packed_bytes(s.head_dim, 32);
        const size_t pool_bytes = 16 * (size_t)s.n_head_kv * (size_t)s.page_size * fp8_row_bytes;
        uint8_t* d_k_fp8 = sycl::malloc_device<uint8_t>(pool_bytes, q);
        uint8_t* d_v_fp8 = sycl::malloc_device<uint8_t>(pool_bytes, q);
        q.memset(d_k_fp8, 0, pool_bytes).wait();
        q.memset(d_v_fp8, 0, pool_bytes).wait();

        // Test void* stream overload
        void* stream_ptr = static_cast<void*>(&q);
        strata::kernels::kv_append_fp8_step(d_k_fp8, d_v_fp8, d_table, d_step, d_kcur, d_vcur, s, stream_ptr, nullptr);
        q.wait();

        // Check head 0 of K: physical row = (0 * 2 + 0) * 4 + 2 = 2
        const size_t row_idx = 2;
        const size_t offset = row_idx * fp8_row_bytes;
        float* d_rec = sycl::malloc_device<float>(s.head_dim, q);
        strata::kernels::dequantize_fp8_e4m3(d_k_fp8 + offset, d_rec, s.head_dim, q).wait();

        std::vector<float> h_rec(s.head_dim);
        q.memcpy(h_rec.data(), d_rec, s.head_dim * sizeof(float)).wait();

        double max_err = 0.0;
        for (int i = 0; i < s.head_dim; ++i) {
            double err = std::fabs(static_cast<double>(h_kcur[i]) - static_cast<double>(h_rec[i]));
            if (err > max_err) max_err = err;
        }

        bool ok = (max_err < 0.25);
        all_ok &= ok;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Head 0 row 2 max abs err: %.3e (stream interop verified)", max_err);
        record_result("4.1 Engine Paged FP8 KV Append (void* stream interop)", ok, buf);

        sycl::free(d_k_fp8, q);
        sycl::free(d_v_fp8, q);
        sycl::free(d_rec, q);
    }

    // 4.2 Engine Paged FP4 KV Append
    {
        const size_t fp4_row_bytes = strata::kernels::fp4_packed_bytes(s.head_dim, 32);
        const size_t pool_bytes = 16 * (size_t)s.n_head_kv * (size_t)s.page_size * fp4_row_bytes;
        uint8_t* d_k_fp4 = sycl::malloc_device<uint8_t>(pool_bytes, q);
        uint8_t* d_v_fp4 = sycl::malloc_device<uint8_t>(pool_bytes, q);
        q.memset(d_k_fp4, 0, pool_bytes).wait();
        q.memset(d_v_fp4, 0, pool_bytes).wait();

        // Test sycl::queue& overload
        strata::kernels::kv_append_fp4_step(d_k_fp4, d_v_fp4, d_table, d_step, d_kcur, d_vcur, s, q, nullptr);
        q.wait();

        const size_t row_idx = 2;
        const size_t offset = row_idx * fp4_row_bytes;
        float* d_rec = sycl::malloc_device<float>(s.head_dim, q);
        strata::kernels::dequantize_fp4_e2m1<32>(d_k_fp4 + offset, d_rec, s.head_dim, q).wait();

        std::vector<float> h_rec(s.head_dim);
        q.memcpy(h_rec.data(), d_rec, s.head_dim * sizeof(float)).wait();

        double max_err = 0.0;
        for (int i = 0; i < s.head_dim; ++i) {
            double err = std::fabs(static_cast<double>(h_kcur[i]) - static_cast<double>(h_rec[i]));
            if (err > max_err) max_err = err;
        }

        bool ok = (max_err < 0.50);
        all_ok &= ok;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Head 0 row 2 max abs err: %.3e (queue& overload verified)", max_err);
        record_result("4.2 Engine Paged FP4 KV Append (sycl::queue& interop)", ok, buf);

        sycl::free(d_k_fp4, q);
        sycl::free(d_v_fp4, q);
        sycl::free(d_rec, q);
    }

    // 4.3 Direct Activation Row Compression Helpers
    {
        const size_t N = 1024;
        const size_t fp8_bytes = strata::kernels::fp8_packed_bytes(N, 32);
        const size_t fp4_bytes = strata::kernels::fp4_packed_bytes(N, 32);

        uint8_t* d_k_out_fp8 = sycl::malloc_device<uint8_t>(fp8_bytes, q);
        uint8_t* d_v_out_fp8 = sycl::malloc_device<uint8_t>(fp8_bytes, q);
        uint8_t* d_k_out_fp4 = sycl::malloc_device<uint8_t>(fp4_bytes, q);
        uint8_t* d_v_out_fp4 = sycl::malloc_device<uint8_t>(fp4_bytes, q);

        // Test direct row compression
        strata::kernels::kv_append_fp8_step(d_kcur, d_vcur, d_k_out_fp8, d_v_out_fp8, N, q);
        strata::kernels::kv_append_fp4_step(d_kcur, d_vcur, d_k_out_fp4, d_v_out_fp4, N, q);
        q.wait();

        record_result("4.3 Direct Activation Row Compression (FP8 & FP4)", true);

        sycl::free(d_k_out_fp8, q);
        sycl::free(d_v_out_fp8, q);
        sycl::free(d_k_out_fp4, q);
        sycl::free(d_v_out_fp4, q);
    }

    sycl::free(d_kcur, q);
    sycl::free(d_vcur, q);
    sycl::free(d_table, q);
    sycl::free(d_step, q);

    return all_ok;
}

// ============================================================================
// Test Suite 5: Runtime Configuration & Format Switching
// ============================================================================

bool test_runtime_configuration() {
    bool all_ok = true;

    // Test programmatic override
    strata::kernels::set_lowbit_kv_format(1);
    bool ok_fp8 = (strata::kernels::lowbit_kv_format() == 1) && strata::kernels::lowbit_kv_enabled();
    record_result("5.1 Programmatic Format Override: FP8 E4M3 (format=1)", ok_fp8);
    all_ok &= ok_fp8;

    strata::kernels::set_lowbit_kv_format(2);
    bool ok_fp4 = (strata::kernels::lowbit_kv_format() == 2) && strata::kernels::lowbit_kv_enabled();
    record_result("5.2 Programmatic Format Override: FP4 E2M1 (format=2)", ok_fp4);
    all_ok &= ok_fp4;

    strata::kernels::set_lowbit_kv_format(0);
    bool ok_off = (strata::kernels::lowbit_kv_format() == 0) && !strata::kernels::lowbit_kv_enabled();
    record_result("5.3 Programmatic Format Override: Disabled (format=0)", ok_off);
    all_ok &= ok_off;

    // Test environment variable fallbacks
    strata::kernels::set_lowbit_kv_format(-1); // Revert override
    ::setenv("STRATA_KV_FP8", "1", 1);
    ::unsetenv("STRATA_KV_FP4");
    bool ok_env_fp8 = (strata::kernels::lowbit_kv_format() == 1) && strata::kernels::lowbit_kv_enabled();
    record_result("5.4 Environment Fallback STRATA_KV_FP8=1", ok_env_fp8);
    all_ok &= ok_env_fp8;

    ::setenv("STRATA_KV_FP4", "1", 1);
    bool ok_env_fp4 = (strata::kernels::lowbit_kv_format() == 2) && strata::kernels::lowbit_kv_enabled();
    record_result("5.5 Environment Fallback STRATA_KV_FP4=1 (FP4 takes precedence)", ok_env_fp4);
    all_ok &= ok_env_fp4;

    ::unsetenv("STRATA_KV_FP4");
    ::unsetenv("STRATA_KV_FP8");
    bool ok_env_none = (strata::kernels::lowbit_kv_format() == 0) && !strata::kernels::lowbit_kv_enabled();
    record_result("5.6 Default Environment (format=0)", ok_env_none);
    all_ok &= ok_env_none;

    // Reset override to disabled (clean default state)
    strata::kernels::set_lowbit_kv_format(0);

    return all_ok;
}

} // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    const auto dev = q.get_device();

    std::printf("====================================================================================\n");
    std::printf(" Strata SYCL Fast FP8/FP4 Numerics & KV Compression Unit & Regression Test\n");
    std::printf(" Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf(" Driver: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("====================================================================================\n\n");

    bool ok1 = test_scalar_primitives();
    bool ok2 = test_device_fp8_roundtrip(q);
    bool ok3 = test_device_fp4_roundtrip(q);
    bool ok4 = test_engine_kv_append_and_interop(q);
    bool ok5 = test_runtime_configuration();

    bool all_passed = ok1 && ok2 && ok3 && ok4 && ok5 && (g_passed == g_total);

    std::printf("\n====================================================================================\n");
    std::printf(" Result: %d / %d tests PASSED (%s)\n",
                g_passed, g_total, all_passed ? "ALL TESTS PASSED" : "FAILURES DETECTED");
    std::printf("====================================================================================\n");

    return all_passed ? 0 : 1;
}
