// tests/sycl/trellis_parity.cpp - Parity verification suite for Gyro Trellis Quantization (TQ) decoders,
// FWHT Hadamard transforms, and SIMD16 GEMV on Intel Xe2.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/tq_gemv.hpp"

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

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() && b.empty()) return 0.0;
    double num = 0.0, denom = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        denom += std::fabs(static_cast<double>(b[i]));
    }
    return num / (denom > 1e-30 ? denom : 1e-30);
}

// =============================================================================
// Test 1: Fast Walsh-Hadamard Transform (hadamard_rows)
// =============================================================================

void cpu_hadamard_rotate(float* x, int64_t n, int block, const float* signs) {
    if (signs != nullptr) {
        for (int64_t i = 0; i < n; ++i) {
            x[i] *= signs[i];
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(block));
    for (int64_t b0 = 0; b0 < n; b0 += block) {
        float* v = x + b0;
        for (int len = 1; len < block; len <<= 1) {
            for (int i = 0; i < block; i += len << 1) {
                for (int j = i; j < i + len; ++j) {
                    const float a = v[j];
                    const float c = v[j + len];
                    v[j] = a + c;
                    v[j + len] = a - c;
                }
            }
        }
        for (int i = 0; i < block; ++i) {
            v[i] *= scale;
        }
    }
}

void test_hadamard_case(int64_t rows, int64_t cols, int block, bool use_signs, bool in_place,
                        sycl::queue& q, const char* label) {
    std::printf("Testing Hadamard %-28s [rows=%2lld, cols=%5lld, block=%3d, signs=%d, in_place=%d]... ",
                label, static_cast<long long>(rows), static_cast<long long>(cols), block,
                use_signs ? 1 : 0, in_place ? 1 : 0);

    std::mt19937 rng(42 + static_cast<unsigned>(rows * 17 + cols * 31 + block));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    const size_t total_elems = static_cast<size_t>(rows * cols);
    std::vector<float> h_in(total_elems);
    for (auto& v : h_in) v = dist(rng);

    std::vector<float> h_signs;
    if (use_signs) {
        h_signs.resize(cols);
        for (auto& s : h_signs) s = (rng() & 1) ? 1.0f : -1.0f;
    }

    // CPU reference computation
    std::vector<float> h_ref = h_in;
    for (int64_t r = 0; r < rows; ++r) {
        cpu_hadamard_rotate(h_ref.data() + r * cols, cols, block, use_signs ? h_signs.data() : nullptr);
    }

    // GPU allocation and execution
    float* d_in = sycl::malloc_device<float>(total_elems, q);
    float* d_out = in_place ? d_in : sycl::malloc_device<float>(total_elems, q);
    float* d_signs = use_signs ? sycl::malloc_device<float>(cols, q) : nullptr;

    q.memcpy(d_in, h_in.data(), total_elems * sizeof(float));
    if (use_signs) {
        q.memcpy(d_signs, h_signs.data(), cols * sizeof(float));
    }
    q.wait();

    strata::kernels::hadamard_rows(d_in, d_out, rows, cols, block, d_signs, &q);
    q.wait();

    std::vector<float> h_gpu(total_elems);
    q.memcpy(h_gpu.data(), d_out, total_elems * sizeof(float)).wait();

    const double err = rel_l1(h_gpu, h_ref);
    if (err < 1e-6) {
        std::printf("[PASS] (rel_l1=%.2e)\n", err);
    } else {
        std::printf("[FAIL] (rel_l1=%.2e >= 1e-6)\n", err);
        ++g_failures;
    }

    sycl::free(d_in, q);
    if (!in_place) sycl::free(d_out, q);
    if (d_signs) sycl::free(d_signs, q);
}

void run_hadamard_tests(sycl::queue& q) {
    std::printf("\n--- Test Suite 1: Fast Walsh-Hadamard Transform (hadamard_rows) ---\n");
    test_hadamard_case(1, 128, 128, false, false, q, "block128_single_row");
    test_hadamard_case(1, 128, 128, true,  true,  q, "block128_single_in_place");
    test_hadamard_case(4, 256, 128, true,  false, q, "block128_multi_row");
    test_hadamard_case(8, 640, 128, true,  true,  q, "block128_640_in_place");
    test_hadamard_case(16, 2560, 128, true, false, q, "block128_2560_out_of_place");
    test_hadamard_case(1, 256, 256, false, false, q, "block256_single_row");
    test_hadamard_case(4, 512, 256, true,  true,  q, "block256_512_in_place");
    test_hadamard_case(8, 2560, 256, true, false, q, "block256_2560_out_of_place");
}

// =============================================================================
// Test 2: TQ Dequantization (FP16 codebook)
// =============================================================================

void cpu_dequantize_tq(int type, const uint8_t* vx, float* y, int64_t n) {
    const int64_t n_blocks = n / 128;
    const size_t bsize = strata::kernels::tq_block_size(type);

    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk_ptr = vx + b * bsize;
        const uint16_t d_raw = *reinterpret_cast<const uint16_t*>(blk_ptr);
        const float d = strata::kernels::tq_half_to_float(d_raw);
        const uint8_t* qs = blk_ptr + 2;

        for (int l = 0; l < 8; ++l) {
            uint32_t s[4];
            if (type == 144) strata::kernels::tq_states4<144>(qs, l, s);
            else if (type == 145) strata::kernels::tq_states4<145>(qs, l, s);
            else if (type == 146) strata::kernels::tq_states4<146>(qs, l, s);

            const int64_t base_out = b * 128 + l * 16;
            for (int j = 0; j < 4; ++j) {
                const uint32_t x = (s[j] & strata::kernels::TQ_STATE_MASK) * 0x9E3779B1u;
                const uint32_t pa = x >> 21;
                const uint32_t pb = (x >> 10) & 2047u;

                y[base_out + j * 4 + 0] = d * strata::kernels::tq_half_to_float(strata::kernels::tq_lut_f16[2 * pa + 0]);
                y[base_out + j * 4 + 1] = d * strata::kernels::tq_half_to_float(strata::kernels::tq_lut_f16[2 * pa + 1]);
                y[base_out + j * 4 + 2] = d * strata::kernels::tq_half_to_float(strata::kernels::tq_lut_f16[2 * pb + 0]);
                y[base_out + j * 4 + 3] = d * strata::kernels::tq_half_to_float(strata::kernels::tq_lut_f16[2 * pb + 1]);
            }
        }
    }
}

void test_dequantize_case(int type, int64_t n, sycl::queue& q, const char* label) {
    const char* type_str = (type == 144 ? "TQ2_T (144)" : (type == 145 ? "TQK6  (145)" : "TQK7  (146)"));
    std::printf("Testing Dequant %-15s [type=%s, n=%5lld]... ", label, type_str, static_cast<long long>(n));

    std::mt19937 rng(1337 + static_cast<unsigned>(type * 101 + n));
    const int64_t n_blocks = n / 128;
    const size_t bsize = strata::kernels::tq_block_size(type);
    const size_t total_bytes = static_cast<size_t>(n_blocks * bsize);

    std::vector<uint8_t> h_bytes(total_bytes);
    for (size_t b = 0; b < static_cast<size_t>(n_blocks); ++b) {
        uint8_t* blk = h_bytes.data() + b * bsize;
        // Random fp16 scale in [0.05, 1.5]
        float scale = 0.05f + (rng() % 1000) * 0.00145f;
        sycl::half sh(scale);
        *reinterpret_cast<uint16_t*>(blk) = *reinterpret_cast<uint16_t*>(&sh);
        // Random trellis bytes
        for (size_t k = 2; k < bsize; ++k) {
            blk[k] = static_cast<uint8_t>(rng() & 0xFF);
        }
    }

    // CPU reference dequantization
    std::vector<float> h_ref(n);
    cpu_dequantize_tq(type, h_bytes.data(), h_ref.data(), n);

    // GPU execution
    uint8_t* d_vx = sycl::malloc_device<uint8_t>(total_bytes, q);
    float* d_y = sycl::malloc_device<float>(n, q);

    q.memcpy(d_vx, h_bytes.data(), total_bytes).wait();

    strata::kernels::dequantize_tq(type, d_vx, d_y, n, &q);
    q.wait();

    std::vector<float> h_gpu(n);
    q.memcpy(h_gpu.data(), d_y, n * sizeof(float)).wait();

    const double err = rel_l1(h_gpu, h_ref);
    if (err < 1e-6) {
        std::printf("[PASS] (rel_l1=%.2e)\n", err);
    } else {
        std::printf("[FAIL] (rel_l1=%.2e >= 1e-6)\n", err);
        ++g_failures;
    }

    sycl::free(d_vx, q);
    sycl::free(d_y, q);
}

void run_dequant_tests(sycl::queue& q) {
    std::printf("\n--- Test Suite 2: TQ Dequantization (dequantize_tq) ---\n");
    for (int ty : {144, 145, 146}) {
        test_dequantize_case(ty, 128, q, "single_block");
        test_dequantize_case(ty, 640, q, "640_weights");
        test_dequantize_case(ty, 2560, q, "2560_weights");
        test_dequantize_case(ty, 10240, q, "10240_weights");
    }
}

// =============================================================================
// Test 3: TQ GEMV (tq_gemv)
// =============================================================================

void cpu_tq_gemv_ref(int type, const uint8_t* w, const float* x, float* y,
                     int64_t n_out, int64_t n_in) {
    const size_t bsize = strata::kernels::tq_block_size(type);
    const int64_t n_blocks = n_in / 128;
    const size_t row_bytes = static_cast<size_t>(n_blocks * bsize);

    std::vector<float> dequant_row(n_in);
    for (int64_t r = 0; r < n_out; ++r) {
        const uint8_t* row_w = w + r * row_bytes;
        cpu_dequantize_tq(type, row_w, dequant_row.data(), n_in);

        double acc = 0.0;
        for (int64_t c = 0; c < n_in; ++c) {
            acc += static_cast<double>(dequant_row[c]) * static_cast<double>(x[c]);
        }
        y[r] = static_cast<float>(acc);
    }
}

void test_gemv_case(int type, int64_t n_out, int64_t n_in, sycl::queue& q, const char* label) {
    const char* type_str = (type == 144 ? "TQ2_T (144)" : (type == 145 ? "TQK6  (145)" : "TQK7  (146)"));
    std::printf("Testing GEMV %-14s [type=%s, K=%lld, N=%5lld]... ",
                label, type_str, static_cast<long long>(n_in), static_cast<long long>(n_out));

    std::mt19937 rng(777 + static_cast<unsigned>(type * 37 + n_out * 13 + n_in));
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);

    const int64_t n_blocks = n_in / 128;
    const size_t bsize = strata::kernels::tq_block_size(type);
    const size_t row_bytes = static_cast<size_t>(n_blocks * bsize);
    const size_t total_w_bytes = static_cast<size_t>(n_out * row_bytes);

    std::vector<uint8_t> h_w(total_w_bytes);
    for (size_t r = 0; r < static_cast<size_t>(n_out); ++r) {
        uint8_t* row = h_w.data() + r * row_bytes;
        for (size_t b = 0; b < static_cast<size_t>(n_blocks); ++b) {
            uint8_t* blk = row + b * bsize;
            float scale = 0.02f + (rng() % 500) * 0.0001f;
            sycl::half sh(scale);
            *reinterpret_cast<uint16_t*>(blk) = *reinterpret_cast<uint16_t*>(&sh);
            for (size_t k = 2; k < bsize; ++k) {
                blk[k] = static_cast<uint8_t>(rng() & 0xFF);
            }
        }
    }

    std::vector<float> h_x(n_in);
    for (auto& v : h_x) v = dist_x(rng);

    // CPU FP32 reference (dequantize via FP16 codebook + exact dot product)
    std::vector<float> h_ref_y(n_out);
    cpu_tq_gemv_ref(type, h_w.data(), h_x.data(), h_ref_y.data(), n_out, n_in);

    // Device allocations
    uint8_t* d_w = sycl::malloc_device<uint8_t>(total_w_bytes, q);
    float* d_x = sycl::malloc_device<float>(n_in, q);
    float* d_y_slm = sycl::malloc_device<float>(n_out, q);
    float* d_y_glob = sycl::malloc_device<float>(n_out, q);

    q.memcpy(d_w, h_w.data(), total_w_bytes);
    q.memcpy(d_x, h_x.data(), n_in * sizeof(float));
    q.wait();

    // 1. Launch TQ GEMV with SLM staging
    strata::kernels::tq_gemv_ext(type, d_w, d_x, d_y_slm, n_out, n_in, /*stage_slm=*/true, &q);
    // 2. Launch TQ GEMV with global memory lookup
    strata::kernels::tq_gemv_ext(type, d_w, d_x, d_y_glob, n_out, n_in, /*stage_slm=*/false, &q);
    q.wait();

    std::vector<float> h_gpu_slm(n_out);
    std::vector<float> h_gpu_glob(n_out);
    q.memcpy(h_gpu_slm.data(), d_y_slm, n_out * sizeof(float));
    q.memcpy(h_gpu_glob.data(), d_y_glob, n_out * sizeof(float));
    q.wait();

    // Check 1: SLM staging vs global memory lookup must be bitwise identical
    const double diff_slm_glob = rel_l1(h_gpu_slm, h_gpu_glob);
    // Check 2: SLM GEMV vs theoretical unquantized codebook reference must be < 2.5e-2
    const double err_vs_ref = rel_l1(h_gpu_slm, h_ref_y);

    const bool pass = (diff_slm_glob == 0.0) && (err_vs_ref < 2.5e-2);
    if (pass) {
        std::printf("[PASS] (err_vs_ref=%.2e < 2.5e-2, slm_vs_glob=%.2e [bitwise])\n",
                    err_vs_ref, diff_slm_glob);
    } else {
        std::printf("[FAIL] (err_vs_ref=%.2e, slm_vs_glob=%.2e)\n",
                    err_vs_ref, diff_slm_glob);
        ++g_failures;
    }

    sycl::free(d_w, q);
    sycl::free(d_x, q);
    sycl::free(d_y_slm, q);
    sycl::free(d_y_glob, q);
}

void run_gemv_tests(sycl::queue& q) {
    std::printf("\n--- Test Suite 3: TQ GEMV (tq_gemv) ---\n");
    for (int ty : {144, 145, 146}) {
        test_gemv_case(ty, 640, 2560, q, "down_decode");
        test_gemv_case(ty, 10240, 2560, q, "gate_up_decode");
    }
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
    std::printf("================================================================\n");
    std::printf("Intel Xe2 Gyro Trellis Quantization (TQ) Parity Verification\n");
    std::printf("================================================================\n");

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    run_hadamard_tests(q);
    run_dequant_tests(q);
    run_gemv_tests(q);

    std::printf("\n================================================================\n");
    if (g_failures == 0) {
        std::printf("All Trellis Quantization tests PASSED (0 failures)!\n");
        std::printf("================================================================\n");
        return 0;
    } else {
        std::printf("Trellis Quantization tests FAILED (%d failures).\n", g_failures);
        std::printf("================================================================\n");
        return 1;
    }
}
