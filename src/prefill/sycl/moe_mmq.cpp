// src/prefill/sycl/moe_mmq.cpp - Pure SYCL 2020 MMQ & Direct Pointer-Table Expert MMQ implementation
// for Intel Arc B580 / Xe2 microarchitecture.
#include <sycl/sycl.hpp>
#include "strata/prefill/moe_mmq.hpp"
#include "strata/core/sycl_arch.hpp"

#include "ggml.h"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace strata::prefill::mmq {
namespace {

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

inline int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }
inline unsigned blocks(int64_t n) { return (unsigned)((n + 255) / 256); }

inline int dp4a(int a, int b, int c) {
    auto va = sycl::vec<int, 1>(a).as<sycl::vec<int8_t, 4>>().convert<int>();
    auto vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>().convert<int>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

template <typename VecT, class BinaryOperation>
inline unsigned int vectorized_binary(unsigned int a, unsigned int b, const BinaryOperation binary_op) {
    sycl::vec<unsigned int, 1> v0{a}, v1{b};
    auto v2 = v0.as<VecT>();
    auto v3 = v1.as<VecT>();
    auto res = binary_op(v2, v3);
    return res.template as<VecT>().template as<sycl::vec<unsigned int, 1>>()[0];
}

inline int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

inline int get_int_b4(const void* x, const int& i32) { return ((const int*)x)[i32]; }

inline float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx) {
    const block_q8_0* bq8_0 = (const block_q8_0*)vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sumi = dp4a(((const int*)bq8_0->qs)[i], ((const int*)bq8_1->qs)[i], sumi);
    }
    const float d =
        sycl::vec<sycl::half, 1>(bq8_0->d).convert<float, sycl::rounding_mode::automatic>()[0] *
        sycl::vec<sycl::half, 1>(bq8_1->ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}

inline float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*)vbq + kbx;
    const int* q8 = (const int*)bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        int v0 = (int)kvalues_iq4nl[aux_q4 & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 4) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 8) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 12) & 0x0F] << 24);
        int v1 = (int)kvalues_iq4nl[(aux_q4 >> 16) & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 20) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 24) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 28) & 0x0F] << 24);
        sumi = dp4a(v0, q8[l + 0], sumi);
        sumi = dp4a(v1, q8[l + 4], sumi);
    }
    const float d =
        sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0] *
        sycl::vec<sycl::half, 1>(bq8_1->ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}

inline float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*)vbq + kbx;
    const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*)&qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos = sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                               iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = vectorized_binary<sycl::uchar4>(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                                                               ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                                           0x00000000, std::not_equal_to<>());
        const int signs1 = vectorized_binary<sycl::uchar4>(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                                                               ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                                           0x00000000, std::not_equal_to<>());
        const int grid_l = vectorized_binary<sycl::uchar4>(grid_pos.x() ^ signs0, signs0, std::minus<>());
        const int grid_h = vectorized_binary<sycl::uchar4>(grid_pos.y() ^ signs1, signs1, std::minus<>());
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = dp4a(grid_l, u0, sumi);
        sumi = dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d =
        sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0] *
        sycl::vec<sycl::half, 1>(bq8_1[iqs / 2].ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}

inline float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q2_0* bq2 = (const block_q2_0*)vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t c = bq2->qs[i];
        int q0 = ((c >> 0) & 3) - 1;
        int q1 = ((c >> 2) & 3) - 1;
        int q2 = ((c >> 4) & 3) - 1;
        int q3 = ((c >> 6) & 3) - 1;
        sumi += q0 * (int)bq8_1->qs[i * 4 + 0] + q1 * (int)bq8_1->qs[i * 4 + 1] +
                q2 * (int)bq8_1->qs[i * 4 + 2] + q3 * (int)bq8_1->qs[i * 4 + 3];
    }
    const float d =
        sycl::vec<sycl::half, 1>(bq2->d).convert<float, sycl::rounding_mode::automatic>()[0] *
        sycl::vec<sycl::half, 1>(bq8_1->ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}

inline float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*)vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        int v0 = (int)kvalues_iq4nl[aux_q4 & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 4) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 8) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 12) & 0x0F] << 24);
        int v1 = (int)kvalues_iq4nl[(aux_q4 >> 16) & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 20) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 24) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 28) & 0x0F] << 24);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = dp4a(v0, u0, sumi);
        sumi = dp4a(v1, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d =
        sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0] *
        sycl::vec<sycl::half, 1>(bq8_1[iqs / 4].ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}

// ---------------- Copy and Utility Kernels ----------------

void copy16_kernel(const sycl::uint4* __restrict__ a, int64_t na, const sycl::uint4* __restrict__ b, int64_t nb,
                   sycl::uint4* __restrict__ ab_dst, const sycl::uint4* __restrict__ c, int64_t nc,
                   sycl::uint4* __restrict__ c_dst, const sycl::uint4* __restrict__ tail,
                   sycl::uint4* __restrict__ tail_dst, int64_t nz) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    const int64_t n = na + nb + nc;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < n) c_dst[i - na - nb] = c[i - na - nb];
    else if (i < n + nz) ab_dst[na + nb + (i - n)] = sycl::uint4(0, 0, 0, 0);
    else if (i < n + 2 * nz) c_dst[nc + (i - n - nz)] = sycl::uint4(0, 0, 0, 0);
    else if (i == n + 2 * nz && tail != nullptr) *tail_dst = *tail;
}

void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                  uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc,
                  uint8_t* __restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride, tail_off;
};

void copy16_group_kernel(GroupArgs ga, int first, int n, int64_t na, int64_t nc, int64_t nz,
                         sycl::uint4* __restrict__ gu_dst, sycl::uint4* __restrict__ d_dst,
                         sycl::uint4* __restrict__ tail_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    const int q = first + (int)item_ct1.get_group(0);
    const sycl::uint4* src = (const sycl::uint4*)ga.blob[q];
    sycl::uint4* ab = gu_dst + (int64_t)q * ga.gu_stride;
    sycl::uint4* cd = d_dst + (int64_t)q * ga.d_stride;
    const int64_t i = (int64_t)item_ct1.get_global_id(1);
    const int64_t m = 2 * na + nc;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < m) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
    else if (i < m + nz) { if (q == n - 1) ab[ga.gu_stride + (i - m)] = sycl::uint4(0, 0, 0, 0); }
    else if (i < m + 2 * nz) { if (q == n - 1) cd[ga.d_stride + (i - m - nz)] = sycl::uint4(0, 0, 0, 0); }
    else if (i == m + 2 * nz && tail_dst != nullptr) tail_dst[q] = src[ga.tail_off];
}

void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    constexpr size_t O_D_CODES = (size_t)1280 * 640, O_GU_SC = O_D_CODES + (size_t)2560 * 160,
                     O_D_SC = O_GU_SC + (size_t)1280 * 40 * 2;
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*)(blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*)(blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const sycl::uint4 q = *(const sycl::uint4*)codes;
    const uint16_t* qh = (const uint16_t*)&q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff, bool interleaved) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + sycl::native::exp(-g)) * u;
}

void swiglu_scaled_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                          const int32_t* __restrict__ bounds, int n, const float* __restrict__ tails,
                          int64_t row0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff, ra = row0 + r;
    int q = 0;
    while (q + 1 < n && ra >= bounds[q + 1]) ++q;
    const float* row = gu + r * 2 * n_ff;
    const float g = row[k] * tails[4 * q], u = row[n_ff + k] * tails[4 * q + 1];
    h[i] = g / (1.0f + sycl::native::exp(-g)) * u;
}

void down_row_scales_kernel(float* __restrict__ sd, const int32_t* __restrict__ bounds, int n,
                            const float* __restrict__ tails, int64_t nrows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t r = (int64_t)item_ct1.get_global_id(0);
    if (r >= nrows) return;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    sd[r] = tails[4 * q + 2];
}

void iota_kernel(int32_t* dst, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    if (i < n) dst[i] = (int32_t)i;
}

void f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item_ct1.get_global_id(0);
    if (i < n) {
        y[i] = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half>(x[i])).convert<float, sycl::rounding_mode::automatic>()[0];
    }
}

// ---------------- MMQ Matrix Multiply Kernel ----------------

void mmq_kernel(const int64_t total_rows, const int64_t w_rows, const int64_t w_cols,
                const int type, const bool has_tab, const WeightTab tab, const void* w,
                const size_t expert_bytes, const void* xq, const int32_t* bounds,
                const int32_t* ids, float* dst, const int64_t ld_dst, const int n_experts,
                sycl::nd_item<2> it) {
    const int64_t r = (int64_t)it.get_global_id(0);
    const int64_t c = (int64_t)it.get_global_id(1);
    if (r >= total_rows || c >= w_rows) return;

    int e = 0;
    while (e + 1 < n_experts && r >= bounds[e + 1]) ++e;
    if (r < bounds[e] || r >= bounds[e + 1]) return;

    const uint8_t* w_base = has_tab ? (const uint8_t*)tab.p[e] : ((const uint8_t*)w + (size_t)e * expert_bytes);
    if (!w_base) return;

    const size_t row_bytes = matrix_bytes(type, 1, w_cols);
    const uint8_t* w_row = w_base + (size_t)c * row_bytes;

    const size_t act_row_stride = pad512(w_cols) * sizeof(block_q8_1) / 32;
    const block_q8_1* x_row = reinterpret_cast<const block_q8_1*>((const uint8_t*)xq + (size_t)r * act_row_stride);

    float dot = 0.0f;
    switch (type) {
        case GGML_TYPE_Q8_0: {
            const int n_blocks = ((int)(w_cols + 255) / 256) * 8;
            for (int b = 0; b < n_blocks; ++b) {
                dot += vec_dot_q8_0_q8_1(w_row, x_row + b, b);
            }
            break;
        }
        case GGML_TYPE_IQ4_NL: {
            const int n_blocks = ((int)(w_cols + 255) / 256) * 8;
            for (int b = 0; b < n_blocks; ++b) {
                dot += vec_dot_iq4_nl_q8_1(w_row, x_row + b, b, 0);
            }
            break;
        }
        case GGML_TYPE_IQ3_S: {
            const int n_super = (int)w_cols / 256;
            for (int sb = 0; sb < n_super; ++sb) {
                for (int sub = 0; sub < 8; ++sub) {
                    dot += vec_dot_iq3_s_q8_1(w_row, x_row + sb * 8, sb, sub * 2);
                }
            }
            break;
        }
        case GGML_TYPE_Q2_0: {
            const int n_blocks = (int)(w_cols / 32);
            for (int b = 0; b < n_blocks; ++b) {
                dot += vec_dot_q2_0_q8_1(w_row, x_row + b, b, 0);
            }
            break;
        }
        case GGML_TYPE_IQ4_XS: {
            const int n_super = (int)w_cols / 256;
            for (int sb = 0; sb < n_super; ++sb) {
                for (int sub = 0; sub < 8; ++sub) {
                    dot += vec_dot_iq4_xs_q8_1(w_row, x_row + sb * 8, sb, sub * 4);
                }
            }
            break;
        }
        default:
            break;
    }

    const int dst_r = ids ? ids[r] : (int)r;
    dst[(size_t)dst_r * ld_dst + c] = dot;
}

}  // namespace

// ---------------- Public Interface Implementation ----------------

bool built() { return true; }

Nvfp4Mode nvfp4_mode() { return Nvfp4Mode::FP16; }

bool supported(int t) {
    switch ((ggml_type)t) {
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    return supported(t) && (w_rows > 0);
}

inline int64_t type_blck_size(int t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_IQ3_S: return 256;
        case GGML_TYPE_Q2_0: return 32;
        case GGML_TYPE_IQ2_XXS: return 256;
        case GGML_TYPE_IQ2_XS: return 256;
        case GGML_TYPE_IQ2_S: return 256;
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ4_XS: return 256;
        default: return 32;
    }
}

inline size_t type_size(int t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ3_S: return sizeof(block_iq3_s);
        case GGML_TYPE_Q2_0: return sizeof(block_q2_0);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS: return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        default: return 0;
    }
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    const int qk = type_blck_size(t);
    const size_t ts = type_size(t);
    return (size_t)rows * (size_t)(cols / qk) * ts;
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t)rows * (size_t)pad512(cols) * sizeof(block_q8_1) / 32 + 128 * sizeof(block_q8_1);
}

bool fp4_activations(int) { return false; }

void scale_gu_rows(float*, int64_t, int64_t, const int32_t*, int, const float*, int64_t, int64_t, void*) {}

void scale_down_rows(float*, int64_t, const int32_t*, int, const float*, int64_t, void*) {}

void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int ggml_type, int64_t cols,
                      int64_t ld, int64_t tokens, int k_used, void* stream) {
    (void)slot;
    quantize(x, src, xq, ggml_type, cols, ld, tokens * k_used, stream, nullptr);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows,
              void* stream, float* yscale) {
    (void)t; (void)yscale;
    if (rows <= 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t act_row_stride = pad512(cols) * sizeof(block_q8_1) / 32;
    const int64_t blocks_per_row = pad512(cols) / 32;

    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t)rows, (size_t)blocks_per_row), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) {
            const int64_t r = (int64_t)it.get_global_id(0);
            const int64_t b = (int64_t)it.get_global_id(1);
            if (r >= rows || b >= blocks_per_row) return;

            block_q8_1* dst_row = reinterpret_cast<block_q8_1*>((uint8_t*)xq + (size_t)r * act_row_stride);
            block_q8_1* bq = dst_row + b;

            const int64_t src_row_idx = ids ? (int64_t)ids[r] : r;
            const float* src_row = x + src_row_idx * ld;

            if (b * 32 >= cols) {
                bq->ds[0] = sycl::vec<float, 1>(0.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                bq->ds[1] = sycl::vec<float, 1>(0.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                for (int i = 0; i < 32; ++i) bq->qs[i] = 0;
                return;
            }

            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const int64_t col = b * 32 + i;
                const float val = col < cols ? src_row[col] : 0.0f;
                amax = sycl::fmax(amax, sycl::fabs(val));
            }

            if (amax == 0.0f) {
                bq->ds[0] = sycl::vec<float, 1>(0.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                bq->ds[1] = sycl::vec<float, 1>(0.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                for (int i = 0; i < 32; ++i) bq->qs[i] = 0;
                return;
            }

            const float d = amax / 127.0f;
            const float id = 1.0f / d;
            float sum = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const int64_t col = b * 32 + i;
                const float val = col < cols ? src_row[col] : 0.0f;
                int q0 = (int)sycl::round(val * id);
                q0 = sycl::clamp(q0, -128, 127);
                bq->qs[i] = (int8_t)q0;
                sum += (float)q0;
            }
            bq->ds[0] = sycl::vec<float, 1>(d).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            bq->ds[1] = sycl::vec<float, 1>(d * sum).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        });
}

Context::Context() {}
Context::~Context() {}

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0 || p.total_rows <= 0 || p.w_rows <= 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();

    const WeightTab tab = p.tab ? *p.tab : WeightTab{};
    const bool has_tab = (p.tab != nullptr);

    const int64_t total_rows = p.total_rows;
    const int64_t w_rows = p.w_rows;
    const int64_t w_cols = p.w_cols;
    const int type = p.type;
    const void* w = p.w;
    const size_t expert_bytes = p.expert_bytes;
    const void* xq = p.xq;
    const int32_t* bounds = p.bounds;
    const int32_t* ids = p.ids;
    float* dst = p.dst;
    const int64_t ld_dst = p.ld_dst;
    const int n_experts = p.n;

    const size_t row_dim = (size_t)total_rows;
    const size_t col_dim = ((size_t)w_rows + 15) / 16 * 16;

    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(row_dim, col_dim), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            mmq_kernel(total_rows, w_rows, w_cols, type, has_tab, tab, w, expert_bytes,
                       xq, bounds, ids, dst, ld_dst, n_experts, it);
        });
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail, void* tail_dst, size_t zero_bytes) {
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const bool a16 = ((uintptr_t)gate | (uintptr_t)up | (uintptr_t)down | (uintptr_t)gu_dst | (uintptr_t)d_dst |
                      (uintptr_t)tail | (uintptr_t)tail_dst | gu_half_bytes | d_bytes | zero_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t)gu_half_bytes / 16, nc = (int64_t)d_bytes / 16, nz = (int64_t)zero_bytes / 16;
        const int64_t total = 2 * na + nc + 2 * nz + (tail_dst ? 1 : 0);
        s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(total) * 256), sycl::range<1>(256)),
                        [=](sycl::nd_item<1> it) {
                            copy16_kernel((const sycl::uint4*)gate, na, (const sycl::uint4*)up, na,
                                          (sycl::uint4*)gu_dst, (const sycl::uint4*)down, nc, (sycl::uint4*)d_dst,
                                          (const sycl::uint4*)tail, (sycl::uint4*)tail_dst, nz);
                        });
    } else {
        if (tail && tail_dst) s->memcpy(tail_dst, tail, 16);
        if (zero_bytes) {
            s->memset((uint8_t*)gu_dst + 2 * gu_half_bytes, 0, zero_bytes);
            s->memset((uint8_t*)d_dst + d_bytes, 0, zero_bytes);
        }
        const int64_t na = (int64_t)gu_half_bytes, nc = (int64_t)d_bytes;
        s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(2 * na + nc) * 256), sycl::range<1>(256)),
                        [=](sycl::nd_item<1> it) {
                            copy1_kernel((const uint8_t*)gate, na, (const uint8_t*)up, na, (uint8_t*)gu_dst,
                                         (const uint8_t*)down, nc, (uint8_t*)d_dst);
                        });
    }
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream,
                         size_t tail_off, void* tail_dst, size_t zero_bytes) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    if (tail_off == 0) tail_dst = nullptr;
    uintptr_t a = (uintptr_t)gu_dst | (uintptr_t)d_dst | up_off | gu_half_bytes | down_off | d_bytes |
                  gu_stride | d_stride | tail_off | (uintptr_t)tail_dst | zero_bytes;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t)g.blob[q];
    if (a % 16 != 0) return false;

    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t)up_off / 16;
    ga.down_off = (int64_t)down_off / 16;
    ga.gu_stride = (int64_t)gu_stride / 16;
    ga.d_stride = (int64_t)d_stride / 16;
    ga.tail_off = (int64_t)tail_off / 16;

    const int64_t na = (int64_t)gu_half_bytes / 16, nc = (int64_t)d_bytes / 16, nz = (int64_t)zero_bytes / 16;
    const int64_t work = 2 * na + nc + 2 * nz + (tail_dst != nullptr ? 1 : 0);

    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t ny = (size_t)(g.n - g.first);
    s->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(ny, blocks(work) * 256), sycl::range<2>(1, 256)),
        [=](sycl::nd_item<2> it) {
            copy16_group_kernel(ga, g.first, g.n, na, nc, nz, (sycl::uint4*)gu_dst, (sycl::uint4*)d_dst,
                                (sycl::uint4*)tail_dst);
        });
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(1280LL * 40 + 2560LL * 10) * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> it) { strata_q2_kernel(blob, (uint16_t*)gu_dst, (uint16_t*)d_dst); });
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(rows * n_ff) * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> it) { swiglu_kernel(gu, h, rows, n_ff, interleaved); });
}

void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream) {
    if (rows <= 0) return;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(rows * n_ff) * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> it) { swiglu_scaled_kernel(gu, h, rows, n_ff, bounds, n, tails, row0); });
}

void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(nrows) * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> it) { down_row_scales_kernel(sd, bounds, n, tails, nrows); });
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(n) * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> it) { iota_kernel(dst, n); });
}

void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks(n) * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> it) { f16_to_f32_kernel(x, y, n); });
}

void set_bounds(int32_t* dst, int32_t rows, void* stream) {
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s->single_task([=]() {
        dst[0] = 0;
        dst[1] = rows;
    });
}

// ---- STRATA_MMQ_DIRECT ----

bool tab_built() { return true; }

bool tab_fits(int t, int64_t w_rows) {
    return tab_built() && (w_rows % 128 == 0) && supported(t) && fits(t, w_rows);
}

size_t overread_bytes(int t, int64_t cols) {
    if (!supported(t)) return 0;
    const int64_t qk = type_blck_size(t);
    const int64_t bpn = cols / qk;
    auto over = [&](int64_t k_vram) -> int64_t {
        const int64_t bpi = k_vram / qk;
        return bpi > 0 ? (bpi - bpn % bpi) % bpi : 0;
    };
    int64_t blocks = over(256);
    return (size_t)blocks * type_size(t);
}

bool whole_tiles(int t, int64_t w_rows, int n_experts) {
    return supported(t) && (w_rows % 128 == 0) && (n_experts > 0);
}

int direct_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_MMQ_DIRECT");
        const int m = v ? std::atoi(v) : 0;
        if (m != 1 && m != 2) return 0;
        if (!tab_built()) {
            std::fprintf(stderr, "strata: STRATA_MMQ_DIRECT=%d is ignored: this build has no pointer-table MMQ\n", m);
            return 0;
        }
        return m;
    }();
    return mode;
}

size_t direct_slot_guard() {
    return direct_mode() != 0 ? kSlotGuard : 0;
}

void prep_group(const GuardTab& g, size_t guard_bytes, void* tail_gu, void* tail_d, size_t tail_bytes, void* stream) {
    const int n = g.n > 0 ? std::min(g.n, kGatherGroupMax) : 0;
    const size_t guard = n > 0 ? guard_bytes : 0;
    const size_t tail = (tail_gu || tail_d) ? tail_bytes : 0;
    if (guard == 0 && tail == 0) return;

    GuardTab gt = g;
    gt.n = guard > 0 ? n : 0;
    const bool tail16 = ((uintptr_t)tail_gu | (uintptr_t)tail_d | tail_bytes) % 16 == 0;
    const size_t span = std::max(guard, tail);
    if (span == 0) return;

    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t wg_size = 256;
    const size_t n_blocks = (span + wg_size - 1) / wg_size;
    const size_t ny = (size_t)(gt.n + 2);

    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(ny, n_blocks * wg_size), sycl::range<2>(1, wg_size)),
        [=](sycl::nd_item<2> it) {
            const int y = (int)it.get_group(0);
            const int64_t i = (int64_t)it.get_global_id(1);
            if (y < gt.n) {
                if (i < (int64_t)guard && gt.p[y] != nullptr) {
                    gt.p[y][i] = 0;
                }
                return;
            }
            uint8_t* t = (y == gt.n) ? (uint8_t*)tail_gu : (uint8_t*)tail_d;
            if (t == nullptr) return;
            if (tail16) {
                if (i < (int64_t)(tail / 16)) {
                    reinterpret_cast<sycl::uint4*>(t)[i] = sycl::uint4(0, 0, 0, 0);
                }
            } else if (i < (int64_t)tail) {
                t[i] = 0;
            }
        });
}

}  // namespace strata::prefill::mmq
