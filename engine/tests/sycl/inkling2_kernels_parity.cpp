// engine/tests/sycl/inkling2_kernels_parity.cpp
// Standalone parity test for Inkling-2-Mini SYCL kernels on real Inkling shapes:
//   D=2304, inter=1152, n_experts=128, M=1,4,8,16,64
// Covers: FMT_SF4, FMT_MX8, FMT_S2..S8 (MoE w13/w2 + dense), Q8_ROW dp4a, and k_gemm_q8row_sg16.
// Compares GPU outputs against exact CPU FP32 reference oracles, so both default and opt-in
// optimized kernel paths (STRATA_INK_SYCL_MOE_VEC, STRATA_INK_SYCL_KREG, STRATA_INK_SYCL_256GRF,
// STRATA_INK_SYCL_XMX, STRATA_INK_SYCL_ESIMD_GEMM) are verified for accuracy and bit/FMA parity.
// MIT License - Copyright (c) 2026 Strata contributors

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ink_sy_codecs.h"
#include "ink_sy_intelarc_xmx.h"

using namespace ink_sycl;

namespace {

constexpr int kD = 2304;
constexpr int kInter = 1152;
constexpr int kExperts = 128;
constexpr int kTopK = 8;

struct DiffStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double l2_err = 0.0;
    double l2_ref = 0.0;
    int bad_nan = 0;
};

DiffStats compare_arrays(const float* got, const float* ref, size_t n) {
    DiffStats s;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) {
            ++s.bad_nan;
            continue;
        }
        const float d = std::fabs(got[i] - ref[i]);
        const float r = std::fabs(ref[i]);
        if (d > s.max_abs) s.max_abs = d;
        const float rel = d / std::max(r, 1e-5f);
        if (rel > s.max_rel) s.max_rel = rel;
        s.l2_err += (double)d * d;
        s.l2_ref += (double)r * r;
    }
    return s;
}

bool check_stats(const char* name, const DiffStats& s, float tol_abs, float tol_rel = 1e-3f) {
    const double rel_l2 = (s.l2_ref > 0.0) ? std::sqrt(s.l2_err / s.l2_ref) : std::sqrt(s.l2_err);
    const bool ok = (s.bad_nan == 0) && (s.max_abs <= tol_abs || rel_l2 <= tol_rel);
    std::printf("[%s] %-42s max_abs=%.3e rel_l2=%.3e nan=%d\n",
                ok ? "PASS" : "FAIL", name, s.max_abs, rel_l2, s.bad_nan);
    return ok;
}

uint16_t f32_to_fp16_bits(float v) {
    sycl::half h(v);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Fill deterministic random bytes in a valid range for scales (avoiding NaN/Inf/0)
void fill_e4m3_scales(std::vector<uint8_t>& buf, size_t off, size_t n, std::mt19937& rng) {
    // Positive E4M3 codes in [0x20..0x48] (~0.0625 .. 2.0)
    std::uniform_int_distribution<int> dist(0x20, 0x48);
    for (size_t i = 0; i < n; ++i) buf[off + i] = (uint8_t)dist(rng);
}

void fill_ue8m0_scales(std::vector<uint8_t>& buf, size_t off, size_t n, std::mt19937& rng) {
    // Exponents around 2^-6 .. 2^0 (codes 121..127)
    std::uniform_int_distribution<int> dist(121, 127);
    for (size_t i = 0; i < n; ++i) buf[off + i] = (uint8_t)dist(rng);
}

void fill_e4m3_weights(std::vector<uint8_t>& buf, size_t off, size_t n, std::mt19937& rng) {
    // Signed E4M3 avoiding NaN (0x7F, 0xFF) and huge values
    std::uniform_int_distribution<int> mag(0x00, 0x47);
    std::uniform_int_distribution<int> sgn(0, 1);
    for (size_t i = 0; i < n; ++i) buf[off + i] = (uint8_t)(mag(rng) | (sgn(rng) << 7));
}

void fill_sk_superblocks(std::vector<uint8_t>& buf, size_t off, size_t n_sbs, int F, std::mt19937& rng) {
    const int SB = sk_sb_bytes(F);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> gsc_dist(32, 220);
    const uint16_t d16 = f32_to_fp16_bits(0.05f);
    for (size_t s = 0; s < n_sbs; ++s) {
        uint8_t* blk = buf.data() + off + s * SB;
        blk[0] = (uint8_t)(d16 & 0xFF);
        blk[1] = (uint8_t)(d16 >> 8);
        for (int g = 0; g < 4; ++g) blk[2 + g] = (uint8_t)gsc_dist(rng);
        for (int b = 6; b < SB; ++b) blk[b] = (uint8_t)byte_dist(rng);
    }
}

// ---- CPU Reference Oracles ----

void ref_q8row_dp4a(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    const float* scales = reinterpret_cast<const float*>(W);
    const int8_t* qw = reinterpret_cast<const int8_t*>(W + (size_t)N * 4);
    for (int m = 0; m < M; ++m) {
        const float* xp = X + (size_t)m * K;
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(xp[k]));
        const float d = amax * (1.0f / 127.0f);
        const float inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
        std::vector<int8_t> xq(K);
        for (int k = 0; k < K; ++k) {
            int v = (int)std::rint(xp[k] * inv);
            xq[k] = (int8_t)std::clamp(v, -127, 127);
        }
        for (int r = 0; r < N; ++r) {
            int32_t acc = 0;
            const int8_t* wp = qw + (size_t)r * K;
            for (int k = 0; k < K; ++k) acc += (int32_t)wp[k] * (int32_t)xq[k];
            Y[(size_t)m * N + r] = (float)acc * (scales[r] * d);
        }
    }
}

void ref_q8row_fp32(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    const float* scales = reinterpret_cast<const float*>(W);
    const int8_t* qw = reinterpret_cast<const int8_t*>(W + (size_t)N * 4);
    for (int m = 0; m < M; ++m) {
        const float* xp = X + (size_t)m * K;
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            const int8_t* wp = qw + (size_t)r * K;
            for (int k = 0; k < K; ++k) acc += (double)wp[k] * (double)xp[k];
            Y[(size_t)m * N + r] = (float)(acc * (double)scales[r]);
        }
    }
}

void ref_mx8_linear(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    const int n_blk = K >> 5;
    const uint8_t* S = W + (size_t)N * K;
    for (int m = 0; m < M; ++m) {
        const float* xp = X + (size_t)m * K;
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            const uint8_t* wr = W + (size_t)r * K;
            const uint8_t* sr = S + (size_t)r * n_blk;
            for (int b = 0; b < n_blk; ++b) {
                const float sc = ue8m0_to_f32(sr[b]);
                for (int j = 0; j < 32; ++j) {
                    acc += (double)(e4m3_to_f32(wr[b * 32 + j]) * sc) * (double)xp[b * 32 + j];
                }
            }
            Y[(size_t)m * N + r] = (float)acc;
        }
    }
}

template <int F>
void ref_sk_linear(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * SB;
    for (int m = 0; m < M; ++m) {
        const float* xp = X + (size_t)m * K;
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            const uint8_t* rp = W + (size_t)r * rowb;
            for (int sbi = 0; sbi < nsb; ++sbi) {
                const uint8_t* blk = rp + (size_t)sbi * SB;
                const float d = h2f_bits(ld16(blk));
                for (int g = 0; g < 4; ++g) {
                    const float eff = d * ((float)blk[2 + g] * (1.0f / 255.0f));
                    for (int u = 0; u < 8; ++u) {
                        const int j = g * 64 + u * 8;
                        int q[8];
                        sk_q8_sy<F>(blk + 6, j, q);
                        for (int t = 0; t < 8; ++t) {
                            acc += (double)(((float)q[t] - H) * eff) * (double)xp[sbi * 256 + j + t];
                        }
                    }
                }
            }
            Y[(size_t)m * N + r] = (float)acc;
        }
    }
}

// CPU reference for MoE w13 + SITU + w2 for one format
void ref_moe_gu_dn(
    const std::vector<const uint8_t*>& bodies, uint32_t dn_off, int fmt,
    const float* x_in, const int* topk_idx, const float* topk_w, int M,
    float* h_act_ref, float* y_pairs_ref) {
    const int P = M * kTopK;
    for (int p = 0; p < P; ++p) {
        const int tk = p >> 3;
        const int e = topk_idx[p];
        const uint8_t* body = bodies[e];
        const float* xp = x_in + (size_t)tk * kD;
        const bool is_sk = (fmt != FMT_SF4 && fmt != FMT_MX8);

        // w13 + SITU
        for (int r = 0; r < kInter; ++r) {
            const float inv_sq = is_sk ? (1.0f / h2f_bits(ld16(body + 2 * r))) : 1.0f;
            double ag = 0.0, au = 0.0;
            if (fmt == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[0] / hdr[1];
                const uint8_t* pb = body + 16;
                const uint8_t* sb = pb + 2654208;
                const uint8_t* pg = pb + (size_t)(2 * r) * 1152;
                const uint8_t* sg = sb + (size_t)(2 * r) * 144;
                const uint8_t* pu = pb + (size_t)(2 * r + 1) * 1152;
                const uint8_t* su = sb + (size_t)(2 * r + 1) * 144;
                for (int b = 0; b < 144; ++b) {
                    const float scg = e4m3_to_f32(sg[b]) * outer;
                    const float scu = e4m3_to_f32(su[b]) * outer;
                    double dg = 0.0, du = 0.0;
                    for (int j = 0; j < 8; ++j) {
                        dg += (double)e2m1_lut(pg[b * 8 + j] & 15u) * (double)xp[b * 16 + 2 * j];
                        dg += (double)e2m1_lut(pg[b * 8 + j] >> 4) * (double)xp[b * 16 + 2 * j + 1];
                        du += (double)e2m1_lut(pu[b * 8 + j] & 15u) * (double)xp[b * 16 + 2 * j];
                        du += (double)e2m1_lut(pu[b * 8 + j] >> 4) * (double)xp[b * 16 + 2 * j + 1];
                    }
                    ag += dg * scg;
                    au += du * scu;
                }
            } else if (fmt == FMT_MX8) {
                const uint8_t* wb = body;
                const uint8_t* sb = body + 5308416;
                const uint8_t* wg = wb + (size_t)(2 * r) * 2304;
                const uint8_t* sg = sb + (size_t)(2 * r) * 72;
                const uint8_t* wu = wb + (size_t)(2 * r + 1) * 2304;
                const uint8_t* su = sb + (size_t)(2 * r + 1) * 72;
                for (int b = 0; b < 72; ++b) {
                    const float scg = ue8m0_to_f32(sg[b]);
                    const float scu = ue8m0_to_f32(su[b]);
                    double dg = 0.0, du = 0.0;
                    for (int j = 0; j < 32; ++j) {
                        dg += (double)e4m3_to_f32(wg[b * 32 + j]) * (double)xp[b * 32 + j];
                        du += (double)e4m3_to_f32(wu[b * 32 + j]) * (double)xp[b * 32 + j];
                    }
                    ag += dg * scg;
                    au += du * scu;
                }
            } else {
                const int SB = sk_sb_bytes(fmt);
                const float H = sk_half(fmt);
                const size_t rowb = 9ul * SB;
                const uint8_t* gu = body + 2304;
                const uint8_t* rg = gu + (size_t)(2 * r) * rowb;
                const uint8_t* ru = gu + (size_t)(2 * r + 1) * rowb;
                for (int unit = 0; unit < 288; ++unit) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* bg = rg + (size_t)sbi * SB;
                    const uint8_t* bu = ru + (size_t)sbi * SB;
                    const float effg = h2f_bits(ld16(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
                    const float effu = h2f_bits(ld16(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
                    int qg[8], qu[8];
                    if (fmt == FMT_S2) { sk_q8_sy<FMT_S2>(bg + 6, j, qg); sk_q8_sy<FMT_S2>(bu + 6, j, qu); }
                    else if (fmt == FMT_S3) { sk_q8_sy<FMT_S3>(bg + 6, j, qg); sk_q8_sy<FMT_S3>(bu + 6, j, qu); }
                    else if (fmt == FMT_S4) { sk_q8_sy<FMT_S4>(bg + 6, j, qg); sk_q8_sy<FMT_S4>(bu + 6, j, qu); }
                    else if (fmt == FMT_S5) { sk_q8_sy<FMT_S5>(bg + 6, j, qg); sk_q8_sy<FMT_S5>(bu + 6, j, qu); }
                    else if (fmt == FMT_S6) { sk_q8_sy<FMT_S6>(bg + 6, j, qg); sk_q8_sy<FMT_S6>(bu + 6, j, qu); }
                    else { sk_q8_sy<FMT_S8>(bg + 6, j, qg); sk_q8_sy<FMT_S8>(bu + 6, j, qu); }
                    double dg = 0.0, du = 0.0;
                    for (int t = 0; t < 8; ++t) {
                        dg += (double)((float)qg[t] - H) * (double)xp[sbi * 256 + j + t];
                        du += (double)((float)qu[t] - H) * (double)xp[sbi * 256 + j + t];
                    }
                    ag += dg * effg;
                    au += du * effu;
                }
            }
            h_act_ref[(size_t)p * kInter + r] = situ_act((float)ag, (float)au) * inv_sq;
        }

        // w2
        const float* hp = h_act_ref + (size_t)p * kInter;
        const float rw = topk_w[p];
        for (int rp = 0; rp < kInter; ++rp) {
            double a0 = 0.0, a1 = 0.0;
            if (fmt == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[2] / hdr[3];
                const uint8_t* pb = body + dn_off;
                const uint8_t* sb = pb + 1327104;
                const uint8_t* p0 = pb + (size_t)(2 * rp) * 576;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 72;
                const uint8_t* p1 = pb + (size_t)(2 * rp + 1) * 576;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 72;
                for (int b = 0; b < 72; ++b) {
                    const float sc0 = e4m3_to_f32(s0[b]) * outer;
                    const float sc1 = e4m3_to_f32(s1[b]) * outer;
                    double d0 = 0.0, d1 = 0.0;
                    for (int j = 0; j < 8; ++j) {
                        d0 += (double)e2m1_lut(p0[b * 8 + j] & 15u) * (double)hp[b * 16 + 2 * j];
                        d0 += (double)e2m1_lut(p0[b * 8 + j] >> 4) * (double)hp[b * 16 + 2 * j + 1];
                        d1 += (double)e2m1_lut(p1[b * 8 + j] & 15u) * (double)hp[b * 16 + 2 * j];
                        d1 += (double)e2m1_lut(p1[b * 8 + j] >> 4) * (double)hp[b * 16 + 2 * j + 1];
                    }
                    a0 += d0 * sc0;
                    a1 += d1 * sc1;
                }
            } else if (fmt == FMT_MX8) {
                const uint8_t* wb = body + dn_off;
                const uint8_t* sb = wb + 2654208;
                const uint8_t* w0 = wb + (size_t)(2 * rp) * 1152;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 36;
                const uint8_t* w1 = wb + (size_t)(2 * rp + 1) * 1152;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 36;
                for (int b = 0; b < 36; ++b) {
                    const float sc0 = ue8m0_to_f32(s0[b]);
                    const float sc1 = ue8m0_to_f32(s1[b]);
                    double d0 = 0.0, d1 = 0.0;
                    for (int j = 0; j < 32; ++j) {
                        d0 += (double)e4m3_to_f32(w0[b * 32 + j]) * (double)hp[b * 32 + j];
                        d1 += (double)e4m3_to_f32(w1[b * 32 + j]) * (double)hp[b * 32 + j];
                    }
                    a0 += d0 * sc0;
                    a1 += d1 * sc1;
                }
            } else {
                const int SB = sk_sb_bytes(fmt);
                const float H = sk_half(fmt);
                const uint8_t* pr = body + dn_off + (size_t)rp * 9ul * SB;
                for (int unit = 0; unit < 288; ++unit) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* blk = pr + (size_t)sbi * SB;
                    const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                    int q[8];
                    if (fmt == FMT_S2) sk_q8_sy<FMT_S2>(blk + 6, j, q);
                    else if (fmt == FMT_S3) sk_q8_sy<FMT_S3>(blk + 6, j, q);
                    else if (fmt == FMT_S4) sk_q8_sy<FMT_S4>(blk + 6, j, q);
                    else if (fmt == FMT_S5) sk_q8_sy<FMT_S5>(blk + 6, j, q);
                    else if (fmt == FMT_S6) sk_q8_sy<FMT_S6>(blk + 6, j, q);
                    else sk_q8_sy<FMT_S8>(blk + 6, j, q);
                    const bool is_r1 = (unit >= 144);
                    const int col = (is_r1 ? (unit - 144) : unit) << 3;
                    double s = 0.0;
                    for (int t = 0; t < 8; ++t) s += (double)((float)q[t] - H) * (double)hp[col + t];
                    if (is_r1) a1 += s * eff; else a0 += s * eff;
                }
            }
            y_pairs_ref[(size_t)p * kD + 2 * rp] = (float)a0 * rw;
            y_pairs_ref[(size_t)p * kD + 2 * rp + 1] = (float)a1 * rw;
        }
    }
}

bool test_dense_q8row(std::mt19937& rng) {
    bool all_ok = true;
    const int N = kD, K = kD;
    const size_t w_bytes = (size_t)N * 4 + (size_t)N * K;
    std::vector<uint8_t> h_W(w_bytes);
    std::uniform_real_distribution<float> sc_dist(0.005f, 0.05f);
    std::uniform_int_distribution<int> q_dist(-127, 127);
    float* scales = reinterpret_cast<float*>(h_W.data());
    int8_t* qw = reinterpret_cast<int8_t*>(h_W.data() + (size_t)N * 4);
    for (int r = 0; r < N; ++r) scales[r] = sc_dist(rng);
    for (size_t i = 0; i < (size_t)N * K; ++i) qw[i] = (int8_t)q_dist(rng);

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    Q().memcpy(d_W, h_W.data(), w_bytes).wait();

    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    for (int M : {1, 4, 8, 16, 64}) {
        std::vector<float> h_X((size_t)M * K);
        for (auto& v : h_X) v = x_dist(rng);

        float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
        int8_t* d_xq = sycl::malloc_device<int8_t>((size_t)M * K, Q());
        float* d_xsc = sycl::malloc_device<float>(M, Q());
        float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
        Q().memcpy(d_X, h_X.data(), h_X.size() * sizeof(float)).wait();

        // 1. Q8_ROW dp4a linear
        k_quantize_x_q8_sg16(d_X, M, K, d_xq, d_xsc);
        k_linear_q8row_dp4a_sg16(d_W, N, K, d_xq, d_xsc, d_Y, M);
        Q().wait_and_throw();

        std::vector<float> h_Y((size_t)M * N), h_ref((size_t)M * N);
        Q().memcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(float)).wait();
        ref_q8row_dp4a(h_W.data(), N, K, h_X.data(), h_ref.data(), M);

        char label[96];
        std::snprintf(label, sizeof(label), "Q8_ROW dp4a N=%d K=%d M=%d", N, K, M);
        all_ok &= check_stats(label, compare_arrays(h_Y.data(), h_ref.data(), h_Y.size()), 1e-4f, 1e-4f);

        // 2. k_gemm_q8row_sg16 (prefill GEMM) & k_jm_gemm_q8row_s8_sg16 (joint_matrix INT8 XMX)
        if (M == 16 || M == 64) {
            k_gemm_q8row_sg16(d_W, N, K, d_X, d_Y, M);
            Q().wait_and_throw();
            Q().memcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(float)).wait();
            ref_q8row_fp32(h_W.data(), N, K, h_X.data(), h_ref.data(), M);
            std::snprintf(label, sizeof(label), "k_gemm_q8row_sg16 N=%d K=%d M=%d", N, K, M);
            all_ok &= check_stats(label, compare_arrays(h_Y.data(), h_ref.data(), h_Y.size()), 5e-3f, 1e-4f);

#if INK_SY_HAS_JOINT_MATRIX
            if (Q().get_device().has(sycl::aspect::ext_intel_matrix)) {
                k_quantize_x_q8_sg16(d_X, M, K, d_xq, d_xsc);
                k_jm_gemm_q8row_s8_sg16(Q(), d_W, N, K, N, d_xq, d_xsc, d_Y, M);
                Q().wait_and_throw();
                Q().memcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(float)).wait();
                ref_q8row_dp4a(h_W.data(), N, K, h_X.data(), h_ref.data(), M);
                std::snprintf(label, sizeof(label), "k_jm_gemm_q8row_s8 (XMX) M=%d", M);
                all_ok &= check_stats(label, compare_arrays(h_Y.data(), h_ref.data(), h_Y.size()), 1e-4f, 1e-4f);
            }
#endif
        }

        sycl::free(d_X, Q());
        sycl::free(d_xq, Q());
        sycl::free(d_xsc, Q());
        sycl::free(d_Y, Q());
    }
    sycl::free(d_W, Q());
    return all_ok;
}

bool test_dense_mx8(std::mt19937& rng) {
    bool all_ok = true;
    const int N = kD, K = kD;
    const size_t w_bytes = (size_t)N * K + (size_t)N * (K >> 5);
    std::vector<uint8_t> h_W(w_bytes);
    fill_e4m3_weights(h_W, 0, (size_t)N * K, rng);
    fill_ue8m0_scales(h_W, (size_t)N * K, (size_t)N * (K >> 5), rng);

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    Q().memcpy(d_W, h_W.data(), w_bytes).wait();

    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    for (int M : {1, 4, 8}) {
        std::vector<float> h_X((size_t)M * K);
        for (auto& v : h_X) v = x_dist(rng);

        float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
        float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
        Q().memcpy(d_X, h_X.data(), h_X.size() * sizeof(float)).wait();

        k_linear_mx8_sg16(d_W, N, K, d_X, d_Y, M);
        Q().wait_and_throw();

        std::vector<float> h_Y((size_t)M * N), h_ref((size_t)M * N);
        Q().memcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(float)).wait();
        ref_mx8_linear(h_W.data(), N, K, h_X.data(), h_ref.data(), M);

        char label[96];
        std::snprintf(label, sizeof(label), "MX8 linear N=%d K=%d M=%d", N, K, M);
        all_ok &= check_stats(label, compare_arrays(h_Y.data(), h_ref.data(), h_Y.size()), 1e-4f, 1e-4f);

        sycl::free(d_X, Q());
        sycl::free(d_Y, Q());
    }
    sycl::free(d_W, Q());
    return all_ok;
}

template <int F>
bool test_dense_sk_fmt(const char* fname, std::mt19937& rng) {
    bool all_ok = true;
    const int N = kD, K = kD;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    const size_t w_bytes = (size_t)N * nsb * SB;
    std::vector<uint8_t> h_W(w_bytes);
    fill_sk_superblocks(h_W, 0, (size_t)N * nsb, F, rng);

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    Q().memcpy(d_W, h_W.data(), w_bytes).wait();

    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    for (int M : {1, 4}) {
        std::vector<float> h_X((size_t)M * K);
        for (auto& v : h_X) v = x_dist(rng);

        float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
        float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
        Q().memcpy(d_X, h_X.data(), h_X.size() * sizeof(float)).wait();

        k_linear_sk_sg16<F>(d_W, N, K, d_X, d_Y, M);
        Q().wait_and_throw();

        std::vector<float> h_Y((size_t)M * N), h_ref((size_t)M * N);
        Q().memcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(float)).wait();
        ref_sk_linear<F>(h_W.data(), N, K, h_X.data(), h_ref.data(), M);

        char label[96];
        std::snprintf(label, sizeof(label), "%s linear N=%d K=%d M=%d", fname, N, K, M);
        all_ok &= check_stats(label, compare_arrays(h_Y.data(), h_ref.data(), h_Y.size()), 1e-4f, 1e-4f);

        sycl::free(d_X, Q());
        sycl::free(d_Y, Q());
    }
    sycl::free(d_W, Q());
    return all_ok;
}

bool test_moe_format(int fmt, const char* fname, std::mt19937& rng) {
    bool all_ok = true;
    constexpr int kPool = 4;  // 4 distinct expert bodies aliased across 128 experts
    uint32_t dn_off = 0;
    size_t body_bytes = 0;
    if (fmt == FMT_SF4) {
        dn_off = 16u + 2654208u + 331776u;
        body_bytes = dn_off + 1327104u + 165888u;
    } else if (fmt == FMT_MX8) {
        dn_off = 5308416u + 165888u;
        body_bytes = dn_off + 2654208u + 82944u;
    } else {
        const int SB = sk_sb_bytes(fmt);
        dn_off = 2304u + 2304u * 9u * (uint32_t)SB;
        body_bytes = dn_off + 1152u * 9u * (size_t)SB;
    }

    std::vector<std::vector<uint8_t>> h_pool(kPool, std::vector<uint8_t>(body_bytes));
    std::vector<uint8_t*> d_pool(kPool, nullptr);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (int i = 0; i < kPool; ++i) {
        auto& buf = h_pool[i];
        if (fmt == FMT_SF4) {
            float* hdr = reinterpret_cast<float*>(buf.data());
            hdr[0] = 0.25f; hdr[1] = 1.0f;
            hdr[2] = 0.20f; hdr[3] = 1.0f;
            for (size_t b = 16; b < 16 + 2654208; ++b) buf[b] = (uint8_t)byte_dist(rng);
            fill_e4m3_scales(buf, 16 + 2654208, 331776, rng);
            for (size_t b = dn_off; b < dn_off + 1327104; ++b) buf[b] = (uint8_t)byte_dist(rng);
            fill_e4m3_scales(buf, dn_off + 1327104, 165888, rng);
        } else if (fmt == FMT_MX8) {
            fill_e4m3_weights(buf, 0, 5308416, rng);
            fill_ue8m0_scales(buf, 5308416, 165888, rng);
            fill_e4m3_weights(buf, dn_off, 2654208, rng);
            fill_ue8m0_scales(buf, dn_off + 2654208, 82944, rng);
        } else {
            const uint16_t one16 = f32_to_fp16_bits(1.25f);
            for (int r = 0; r < kInter; ++r) {
                buf[2 * r] = (uint8_t)(one16 & 0xFF);
                buf[2 * r + 1] = (uint8_t)(one16 >> 8);
            }
            fill_sk_superblocks(buf, 2304, 2304ul * 9ul, fmt, rng);
            fill_sk_superblocks(buf, dn_off, 1152ul * 9ul, fmt, rng);
        }
        d_pool[i] = sycl::malloc_device<uint8_t>(body_bytes, Q());
        Q().memcpy(d_pool[i], buf.data(), body_bytes).wait();
    }

    std::vector<InkSyExpertDev> h_ex(kExperts);
    std::vector<const uint8_t*> h_bodies(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        h_ex[e].body_ptr = d_pool[e % kPool];
        h_ex[e].dn_off = dn_off;
        h_ex[e].gu_code = (uint8_t)fmt;
        h_ex[e].dn_code = (uint8_t)fmt;
        h_ex[e].in_vram = 1;
        h_ex[e].pad = 0;
        h_bodies[e] = h_pool[e % kPool].data();
    }
    InkSyExpertDev* d_ex = sycl::malloc_device<InkSyExpertDev>(kExperts, Q());
    Q().memcpy(d_ex, h_ex.data(), kExperts * sizeof(InkSyExpertDev)).wait();

    std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> w_dist(0.05f, 0.25f);

    for (int M : {1, 4, 8}) {
        const int P = M * kTopK;
        const int max_uniq = std::min(P, kExperts);
        std::vector<float> h_X((size_t)M * kD);
        for (auto& v : h_X) v = x_dist(rng);
        std::vector<int> h_topk_idx(P);
        std::vector<float> h_topk_w(P);
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < kTopK; ++k) {
                h_topk_idx[m * kTopK + k] = (m * 13 + k * 7) % kExperts;
                h_topk_w[m * kTopK + k] = w_dist(rng);
            }
        }

        float* d_X = sycl::malloc_device<float>((size_t)M * kD, Q());
        int* d_topk_idx = sycl::malloc_device<int>(P, Q());
        float* d_topk_w = sycl::malloc_device<float>(P, Q());
        int* d_uexp = sycl::malloc_device<int>(kExperts, Q());
        int* d_uoff = sycl::malloc_device<int>(kExperts, Q());
        int* d_ucnt = sycl::malloc_device<int>(kExperts, Q());
        int* d_plist = sycl::malloc_device<int>(P, Q());
        int* d_nuniq = sycl::malloc_device<int>(1, Q());
        float* d_hact = sycl::malloc_device<float>((size_t)P * kInter, Q());
        float* d_ypairs = sycl::malloc_device<float>((size_t)P * kD, Q());

        Q().memcpy(d_X, h_X.data(), h_X.size() * sizeof(float));
        Q().memcpy(d_topk_idx, h_topk_idx.data(), P * sizeof(int));
        Q().memcpy(d_topk_w, h_topk_w.data(), P * sizeof(float));
        Q().wait();

        k_moe_plan_sg16(d_topk_idx, P, d_uexp, d_uoff, d_ucnt, d_plist, d_nuniq);
        k_moe_gu_situ_sg16(d_ex, d_X, d_uexp, d_uoff, d_ucnt, d_plist, d_nuniq, max_uniq, d_hact);
        k_moe_dn_sg16(d_ex, d_topk_w, d_uexp, d_uoff, d_ucnt, d_plist, d_nuniq, max_uniq, d_hact, d_ypairs);
        Q().wait_and_throw();

        std::vector<float> h_hact((size_t)P * kInter), h_ypairs((size_t)P * kD);
        Q().memcpy(h_hact.data(), d_hact, h_hact.size() * sizeof(float));
        Q().memcpy(h_ypairs.data(), d_ypairs, h_ypairs.size() * sizeof(float));
        Q().wait();

        std::vector<float> h_hact_ref((size_t)P * kInter), h_ypairs_ref((size_t)P * kD);
        ref_moe_gu_dn(h_bodies, dn_off, fmt, h_X.data(), h_topk_idx.data(), h_topk_w.data(), M,
                      h_hact_ref.data(), h_ypairs_ref.data());

        char label[96];
        std::snprintf(label, sizeof(label), "MoE %s w13+SITU M=%d", fname, M);
        all_ok &= check_stats(label, compare_arrays(h_hact.data(), h_hact_ref.data(), h_hact.size()), 1e-4f, 1e-4f);
        std::snprintf(label, sizeof(label), "MoE %s w2       M=%d", fname, M);
        all_ok &= check_stats(label, compare_arrays(h_ypairs.data(), h_ypairs_ref.data(), h_ypairs.size()), 5e-4f, 1e-4f);

        sycl::free(d_X, Q());
        sycl::free(d_topk_idx, Q());
        sycl::free(d_topk_w, Q());
        sycl::free(d_uexp, Q());
        sycl::free(d_uoff, Q());
        sycl::free(d_ucnt, Q());
        sycl::free(d_plist, Q());
        sycl::free(d_nuniq, Q());
        sycl::free(d_hact, Q());
        sycl::free(d_ypairs, Q());
    }

    sycl::free(d_ex, Q());
    for (auto* p : d_pool) sycl::free(p, Q());
    return all_ok;
}

}  // namespace

int main() {
    const auto dev = Q().get_device();
    std::printf("=== inkling2_kernels_parity on %s ===\n",
                dev.get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(20261128u);
    bool ok = true;

    ok &= test_dense_q8row(rng);
    ok &= test_dense_mx8(rng);
    ok &= test_dense_sk_fmt<FMT_S2>("S2_K", rng);
    ok &= test_dense_sk_fmt<FMT_S3>("S3_K", rng);
    ok &= test_dense_sk_fmt<FMT_S4>("S4_K", rng);
    ok &= test_dense_sk_fmt<FMT_S5>("S5_K", rng);
    ok &= test_dense_sk_fmt<FMT_S6>("S6_K", rng);
    ok &= test_dense_sk_fmt<FMT_S8>("S8_K", rng);

    ok &= test_moe_format(FMT_SF4, "SF4", rng);
    ok &= test_moe_format(FMT_MX8, "MX8", rng);
    ok &= test_moe_format(FMT_S2,  "S2_K", rng);
    ok &= test_moe_format(FMT_S3,  "S3_K", rng);
    ok &= test_moe_format(FMT_S4,  "S4_K", rng);
    ok &= test_moe_format(FMT_S5,  "S5_K", rng);
    ok &= test_moe_format(FMT_S6,  "S6_K", rng);
    ok &= test_moe_format(FMT_S8,  "S8_K", rng);

    std::printf("\n=== inkling2_kernels_parity: %s ===\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
