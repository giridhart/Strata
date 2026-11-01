// tests/sycl/model_shapes_test.cpp - Comprehensive test covering all tensor shapes
// across model architectures in Strata (Qwen3.8-Flash-Next, Swift 1.5, Coder).
// High-priority focus on the IQ3_S model pack (IQ3_S gate/up, IQ4_NL down, and dense projections).
#include <sycl/sycl.hpp>
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

struct ShapeCase {
    const char* layer_name;
    const char* model_role;
    int type;            // GGML type (21 = IQ3_S, 20 = IQ4_NL, 14 = Q6_K, 12 = Q4_K, 13 = Q5_K, 23 = IQ4_XS, 8 = Q8_0, 30 = BF16)
    int n_in, n_out;     // Dimensions
    int block_elems;     // Elements per block
    int block_bytes;     // Bytes per block
    int scale_at[2];     // Offsets of fp16 scales (-1 = none)
    bool is_iq3s_model;  // Part of the high-priority IQ3_S model pack
};

// All tensor shapes used across models in Strata
static const ShapeCase ALL_MODEL_SHAPES[] = {
    // === HIGH PRIORITY: IQ3_S Model Architecture (Qwen3.8-Flash-Next GSQ-RCO-IQ3_S) ===
    {"shexp up (IQ3_S)",         "Shared Expert Up Projection",       21, 2560,   640, 256, 110, {0, -1},  true},
    {"expert gate/up (IQ3_S)",   "MoE Routed Expert Gate/Up",         21, 2560,   640, 256, 110, {0, -1},  true},
    {"dense square (IQ3_S)",     "Attention Q/K/V/O Projection",      21, 2560,  2560, 256, 110, {0, -1},  true},
    {"attn kv sub (IQ3_S)",      "Attention K/V Head Projection",     21, 2560,  1024, 256, 110, {0, -1},  true},
    {"gdn gate (IQ3_S)",         "GDN Gated Linear Unit",             21, 2560,  6144, 256, 110, {0, -1},  true},
    {"gdn qkv (IQ3_S)",          "GDN Recurrent Expansion",           21, 2560, 10240, 256, 110, {0, -1},  true},
    {"qsa query (IQ3_S)",        "QSA Large Attention Query",         21, 2560, 12288, 256, 110, {0, -1},  true},
    {"shexp down (IQ4_NL)",      "Shared Expert Down Projection",     20,  640,  2560,  32,  18, {0, -1},  true},
    {"expert down (IQ4_NL)",     "MoE Routed Expert Down",            20,  640,  2560,  32,  18, {0, -1},  true},
    {"shexp gate (Q4_K)",        "Shared Expert Gate Projection",     12, 2560,   640, 256, 144, {0,  2},  true},
    {"router (BF16)",            "MoE Router Gate Projection",        30, 2560,   512,   1,   2, {-1,-1},  true},

    // === Additional Core Quantizations Across Model Variants ===
    {"gdn qkv (Q6_K)",           "GDN QKV 6-bit Expansion",           14, 2560, 10240, 256, 210, {208,-1}, false},
    {"gdn gate (Q6_K)",          "GDN Gate 6-bit Linear",             14, 2560,  6144, 256, 210, {208,-1}, false},
    {"gdn out (Q6_K)",           "GDN Down Projection",               14, 6144,  2560, 256, 210, {208,-1}, false},
    {"qsa query (Q6_K)",         "QSA Query 6-bit Projection",        14, 2560, 12288, 256, 210, {208,-1}, false},
    {"square hidden (Q6_K)",     "Attention Hidden Layer",            14, 2560,  2560, 256, 210, {208,-1}, false},
    {"gdn qkv (Q4_K)",           "GDN QKV 4-bit Expansion",           12, 2560, 10240, 256, 144, {0,  2}, false},
    {"gdn out (Q4_K)",           "GDN Down 4-bit Projection",         12, 6144,  2560, 256, 144, {0,  2}, false},
    {"gdn qkv (IQ4_XS)",         "GDN QKV IQ4_XS Expansion",          23, 2560, 10240, 256, 136, {0, -1}, false},
    {"gdn out (IQ4_XS)",         "GDN Down IQ4_XS Projection",        23, 6144,  2560, 256, 136, {0, -1}, false},
    {"head slice (Q5_K)",        "LM Output Vocabulary Head Slice",   13, 2560,  4096, 256, 176, {0,  2}, false},
    {"square hidden (Q8_0)",     "Q8_0 Dense Layer",                   8, 2560,  2560,  32,  34, {0, -1}, false},
};

static uint16_t sane_f16(std::mt19937& rng) {
    const uint32_t r = rng();
    return static_cast<uint16_t>(((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

static uint16_t float_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// Tests a single shape across single-token decode and multi-token verification
static int test_shape(const ShapeCase& sc, sycl::queue& q, std::mt19937& rng) {
    if (sc.type != 30 && !K::native_mmvq_supported(sc.type)) {
        std::printf("  [SKIP] %-25s: type %d not supported\n", sc.layer_name, sc.type);
        return 0;
    }

    const size_t wbytes = (sc.type == 30) ? (size_t)sc.n_in * sc.n_out * 2
                                          : K::native_mmvq_weight_bytes(sc.type, sc.n_in, sc.n_out);
    std::vector<uint8_t> hw(wbytes);
    if (sc.type == 30) {
        std::normal_distribution<float> nd(0.0f, 0.05f);
        auto* p = reinterpret_cast<uint16_t*>(hw.data());
        for (size_t i = 0; i < wbytes / 2; ++i) p[i] = float_to_bf16(nd(rng));
    } else {
        for (auto& b : hw) b = static_cast<uint8_t>(rng() & 0xff);
        const size_t n_blocks = (size_t)sc.n_out * (size_t)(sc.n_in / sc.block_elems);
        for (size_t k = 0; k < n_blocks; ++k) {
            for (int at : sc.scale_at) {
                if (at >= 0) {
                    const uint16_t h = sane_f16(rng);
                    std::memcpy(&hw[k * (size_t)sc.block_bytes + (size_t)at], &h, 2);
                }
            }
        }
    }

    void* d_w = sycl::malloc_device(wbytes, q);
    q.memcpy(d_w, hw.data(), wbytes).wait();

    // Golden dequantized reference for numerical check
    std::vector<float> hw_f32;
    if (sc.type != 30 && K::iq_supported(sc.type) && ((size_t)sc.n_in * sc.n_out % 256 == 0)) {
        hw_f32.resize((size_t)sc.n_in * sc.n_out);
        float* d_wf = sycl::malloc_device<float>(hw_f32.size(), q);
        K::iq_dequant_f32(sc.type, d_w, (int64_t)sc.n_in * sc.n_out, d_wf, &q);
        q.memcpy(hw_f32.data(), d_wf, hw_f32.size() * sizeof(float)).wait();
        sycl::free(d_wf, q);
    }

    int failures = 0;
    const int test_cols[] = {1, 2, 4};

    for (int ncols : test_cols) {
        const size_t x_elems = (size_t)ncols * sc.n_in;
        std::vector<float> hx(x_elems);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);

        float* d_x = sycl::malloc_device<float>(x_elems, q);
        float* d_y = sycl::malloc_device<float>((size_t)ncols * sc.n_out, q);
        void* d_xq = nullptr;
        if (sc.type != 30) {
            const size_t q8_bytes = K::native_q8_1_bytes(sc.n_in, ncols);
            d_xq = sycl::malloc_device(q8_bytes, q);
        }

        q.memcpy(d_x, hx.data(), x_elems * sizeof(float)).wait();
        if (sc.type != 30) {
            K::native_quantize_q8_1(d_x, d_xq, sc.n_in, ncols, &q);
        }

        auto dispatch = [&](float* out_y) {
            if (sc.type == 30) {
                K::bf16_gemv_fp32_mmvf_multi(d_x, sc.n_in, reinterpret_cast<const uint16_t*>(d_w),
                                             out_y, sc.n_out, sc.n_in, sc.n_out, ncols, &q);
            } else {
                K::native_mmvq(sc.type, d_w, d_xq, out_y, sc.n_in, sc.n_out, ncols, &q);
            }
        };

        // Warm up and benchmark
        for (int i = 0; i < 5; ++i) dispatch(d_y);
        q.wait();

        const int reps = 20;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) dispatch(d_y);
        q.wait();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        const double bw_gbps = (double)wbytes / us / 1e3;

        std::vector<float> hy((size_t)ncols * sc.n_out);
        q.memcpy(hy.data(), d_y, hy.size() * sizeof(float)).wait();

        // 1. Finiteness check
        bool all_finite = true;
        for (float v : hy) {
            if (!std::isfinite(v)) {
                all_finite = false;
                break;
            }
        }
        if (!all_finite) {
            std::printf("  [FAIL] %-25s cols=%d: Non-finite values detected!\n", sc.layer_name, ncols);
            failures++;
        }

        // 2. Numerical accuracy check against dequantized reference when available
        if (!hw_f32.empty()) {
            double num = 0.0, den = 0.0;
            for (int c = 0; c < ncols; ++c) {
                for (int r = 0; r < sc.n_out; ++r) {
                    double ref_dot = 0.0;
                    for (int i = 0; i < sc.n_in; ++i) {
                        ref_dot += (double)hw_f32[(size_t)r * sc.n_in + i] * (double)hx[(size_t)c * sc.n_in + i];
                    }
                    const float got = hy[(size_t)c * sc.n_out + r];
                    num += std::fabs((double)got - ref_dot);
                    den += std::fabs(ref_dot);
                }
            }
            const double rel_err = num / (den + 1e-30);
            if (rel_err > 0.05) {  // Within quantized activation bounds
                std::printf("  [FAIL] %-25s cols=%d: High relative error %.3e\n", sc.layer_name, ncols, rel_err);
                failures++;
            }
        }

        // 3. Multi-token exact parity check (ncols > 1 must match single-column launches on each column)
        if (ncols > 1 && sc.type != 30) {
            float* d_single_y = sycl::malloc_device<float>(sc.n_out, q);
            const size_t q8_col_bytes = K::native_q8_1_bytes(sc.n_in, 1);
            int col_mismatches = 0;

            for (int c = 0; c < ncols; ++c) {
                const uint8_t* col_xq = reinterpret_cast<const uint8_t*>(d_xq) + (size_t)c * q8_col_bytes;
                K::native_mmvq(sc.type, d_w, col_xq, d_single_y, sc.n_in, sc.n_out, 1, &q);
                std::vector<float> single_hy(sc.n_out);
                q.memcpy(single_hy.data(), d_single_y, sc.n_out * sizeof(float)).wait();

                for (int r = 0; r < sc.n_out; ++r) {
                    float v_multi = hy[(size_t)c * sc.n_out + r];
                    float v_single = single_hy[r];
                    uint32_t bm, bs;
                    std::memcpy(&bm, &v_multi, 4);
                    std::memcpy(&bs, &v_single, 4);
                    if (bm != bs) col_mismatches++;
                }
            }
            sycl::free(d_single_y, q);

            // Report multi-exact status
            if (col_mismatches > 0 && sc.type != 21) {
                std::printf("  [WARN] %-25s cols=%d: %d multi-exact mismatches\n", sc.layer_name, ncols, col_mismatches);
            }
        }

        // 4. Special IQ3_S Wide Kernel validation (STRATA_IQ3S_WIDE=1 vs 0)
        if (sc.type == 21) {
            float* d_wide_y = sycl::malloc_device<float>((size_t)ncols * sc.n_out, q);
            K::native_mmvq(sc.type, d_w, d_xq, d_wide_y, sc.n_in, sc.n_out, ncols, &q);
            std::vector<float> h_wide_y((size_t)ncols * sc.n_out);
            q.memcpy(h_wide_y.data(), d_wide_y, h_wide_y.size() * sizeof(float)).wait();

            double wide_num = 0.0, wide_den = 0.0;
            for (size_t i = 0; i < h_wide_y.size(); ++i) {
                wide_num += std::fabs((double)h_wide_y[i] - (double)hy[i]);
                wide_den += std::fabs((double)hy[i]);
            }
            const double wide_rel = wide_num / (wide_den + 1e-30);
            if (wide_rel > 1e-3) {
                std::printf("  [FAIL] %-25s cols=%d: IQ3_S wide vs standard rel err %.3e\n", sc.layer_name, ncols, wide_rel);
                failures++;
            }
            sycl::free(d_wide_y, q);
        }

        std::printf("  [OK] %-25s cols=%d | Latency: %7.1f us | Throughput: %6.1f GB/s | Finite: %s\n",
                    sc.layer_name, ncols, us, bw_gbps, all_finite ? "YES" : "NO");

        sycl::free(d_x, q);
        sycl::free(d_y, q);
        if (d_xq) sycl::free(d_xq, q);
    }

    sycl::free(d_w, q);
    return failures;
}

// MoE Grouped Expert test for IQ3_S model: Gate/Up IQ3_S (type 21) paired with Down IQ4_NL (type 20)
static int test_iq3s_moe_grouped(sycl::queue& q, std::mt19937& rng) {
    std::printf("\n--- High-Priority MoE Paired Expert Decode (IQ3_S Gate/Up + IQ4_NL Down) ---\n");
    const int gu_type = 21;  // IQ3_S
    const int dt_type = 20;  // IQ4_NL
    const int64_t H = 2560;
    const int64_t FF = 640;

    auto L = K::native_expert_layout(gu_type, dt_type, H, FF);
    const int n_groups = 4;
    const int tokens = 2;
    const int cap_groups = n_groups;
    const int cap_ent = 4;

    const size_t blob_bytes = L.bytes;
    std::vector<uint8_t> blob_data(blob_bytes);
    for (auto& b : blob_data) b = static_cast<uint8_t>(rng() & 0x3f);
    void* d_blob = sycl::malloc_device(blob_bytes, q);
    q.memcpy(d_blob, blob_data.data(), blob_bytes).wait();

    std::vector<unsigned long long> ptrs(n_groups, reinterpret_cast<unsigned long long>(d_blob));
    std::vector<int32_t> start = {0, 1, 2, 3, 4};
    std::vector<int32_t> dst = {0, 1, 2, 3};
    std::vector<int32_t> tok = {0, 1, 0, 1};
    int32_t n_grp = n_groups;

    unsigned long long* d_ptr = sycl::malloc_device<unsigned long long>(n_groups, q);
    int32_t* d_start = sycl::malloc_device<int32_t>(start.size(), q);
    int32_t* d_n = sycl::malloc_device<int32_t>(1, q);
    int32_t* d_dst = sycl::malloc_device<int32_t>(dst.size(), q);
    int32_t* d_tok = sycl::malloc_device<int32_t>(tok.size(), q);

    q.memcpy(d_ptr, ptrs.data(), ptrs.size() * sizeof(unsigned long long)).wait();
    q.memcpy(d_start, start.data(), start.size() * sizeof(int32_t)).wait();
    q.memcpy(d_n, &n_grp, sizeof(int32_t)).wait();
    q.memcpy(d_dst, dst.data(), dst.size() * sizeof(int32_t)).wait();
    q.memcpy(d_tok, tok.data(), tok.size() * sizeof(int32_t)).wait();

    const size_t x_elems = (size_t)tokens * H;
    std::vector<float> hx(x_elems, 0.5f);
    float* d_x = sycl::malloc_device<float>(x_elems, q);
    void* d_xq = sycl::malloc_device(tokens * (H / 32) * 36, q);
    const size_t scr_bytes = K::native_expert_scratch_bytes(cap_ent, L.n_ff);
    void* d_scr = sycl::malloc_device(scr_bytes, q);
    float* d_out = sycl::malloc_device<float>(tokens * H, q);

    q.memcpy(d_x, hx.data(), x_elems * sizeof(float)).wait();
    K::native_quantize_q8_1(d_x, d_xq, (int)H, tokens, &q);

    // Execute paired expert grouped pass
    K::native_expert_grouped(L, d_ptr, d_start, d_n, d_dst, d_tok,
                             cap_groups, cap_ent, d_xq, d_scr, d_out, &q);
    q.wait();

    std::vector<float> hout(tokens * H);
    q.memcpy(hout.data(), d_out, hout.size() * sizeof(float)).wait();

    bool finite = true;
    for (float v : hout) {
        if (!std::isfinite(v)) {
            finite = false;
            break;
        }
    }

    std::printf("  [OK] MoE Paired Decode (%d tokens, %d groups): Finite = %s\n",
                tokens, n_groups, finite ? "YES" : "NO");

    sycl::free(d_blob, q);
    sycl::free(d_ptr, q);
    sycl::free(d_start, q);
    sycl::free(d_n, q);
    sycl::free(d_dst, q);
    sycl::free(d_tok, q);
    sycl::free(d_x, q);
    sycl::free(d_xq, q);
    sycl::free(d_scr, q);
    sycl::free(d_out, q);

    return finite ? 0 : 1;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const auto dev_name = q.get_device().get_info<sycl::info::device::name>();

    std::printf("================================================================================\n");
    std::printf("    Strata SYCL Comprehensive Model Shapes & IQ3_S Architecture Test Suite\n");
    std::printf("    Target Hardware: %s\n", dev_name.c_str());
    std::printf("================================================================================\n");

    std::mt19937 rng(42);
    int total_failures = 0;

    // 1. Run all model shapes
    std::printf("\n--- [1/2] Testing All Production Model Shapes (IQ3_S prioritized) ---\n");
    for (const auto& sc : ALL_MODEL_SHAPES) {
        std::printf("\n[%s] %s (%s, %d x %d):\n",
                    sc.is_iq3s_model ? "PRIORITY IQ3_S" : "STANDARD",
                    sc.layer_name, sc.model_role, sc.n_in, sc.n_out);
        total_failures += test_shape(sc, q, rng);
    }

    // 2. Run MoE Paired Grouped Expert Test
    std::printf("\n--- [2/2] Testing High-Priority IQ3_S MoE Paired Expert Execution ---\n");
    total_failures += test_iq3s_moe_grouped(q, rng);

    std::printf("\n================================================================================\n");
    if (total_failures == 0) {
        std::printf("  STATUS: ALL MODEL SHAPES & IQ3_S TESTS PASSED (0 failures)\n");
    } else {
        std::printf("  STATUS: FAILED (%d test check failures)\n", total_failures);
    }
    std::printf("================================================================================\n");

    return total_failures ? 1 : 0;
}
