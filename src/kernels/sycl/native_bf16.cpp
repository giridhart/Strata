#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/core/sycl_arch.hpp"

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

constexpr int SG_SZ = strata::sycl_arch::SUB_GROUP_SIZE;

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
// Optimized for Intel Xe2 microarchitecture (Battlemage B580 / Lunar Lake) native SIMD16 execution.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

template <int BLOCK_SIZE>
void bf16_f32_mmvf_kernel(sycl::nd_item<1> item, const float* __restrict__ x, const uint16_t* __restrict__ w,
                          float* __restrict__ y, int n_in, sycl::local_accessor<float, 1> partials) {
    const int t = item.get_local_id(0);
    const int row_idx = item.get_group(0);
    const uint16_t* row = w + (size_t)row_idx * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const sycl::float2* inputs2 = reinterpret_cast<const sycl::float2*>(x);

    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const sycl::float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = sycl::fma(f32_from_bf16((uint16_t)weight), input.x(), acc);
        acc = sycl::fma(f32_from_bf16((uint16_t)(weight >> 16)), input.y(), acc);
    }
    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();

    if constexpr (BLOCK_SIZE > SG_SZ) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if (lane == 0) partials[sg_id] = acc;
        sycl::group_barrier(item.get_group());
        acc = 0.0f;
        constexpr int num_sgs = BLOCK_SIZE / SG_SZ;
        for (int s = lane; s < num_sgs; s += SG_SZ) {
            acc += partials[s];
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    }
    if (t == 0) y[row_idx] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT>
void bf16_f32_mmvf_multi_kernel(sycl::nd_item<1> item, const float* __restrict__ x, int64_t ldx,
                                const uint16_t* __restrict__ w, float* __restrict__ y, int64_t ldy, int n_in, int n_tok,
                                sycl::local_accessor<float, 2> partials) {
    const int t = item.get_local_id(0);
    const int row_idx = item.get_group(0);
    const uint16_t* row = w + (size_t)row_idx * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k)
        acc[k] = 0.0f;

    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t)weight);
        const float w1 = f32_from_bf16((uint16_t)(weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const sycl::float2 input = reinterpret_cast<const sycl::float2*>(x + (size_t)k * ldx)[pair];
                acc[k] = sycl::fma(w0, input.x(), acc[k]);
                acc[k] = sycl::fma(w1, input.y(), acc[k]);
            }
        }
    }
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
#pragma unroll
    for (int k = 0; k < NT; ++k)
        acc[k] = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());

    if constexpr (BLOCK_SIZE > SG_SZ) {
        if (lane == 0) {
#pragma unroll
            for (int k = 0; k < NT; ++k)
                partials[k][sg_id] = acc[k];
        }
        sycl::group_barrier(item.get_group());
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            acc[k] = 0.0f;
            constexpr int num_sgs = BLOCK_SIZE / SG_SZ;
            for (int s = lane; s < num_sgs; s += SG_SZ) {
                acc[k] += partials[k][s];
            }
            acc[k] = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
        }
    }
    if (t == 0) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) y[(size_t)k * ldy + row_idx] = acc[k];
        }
    }
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

// S25 (STRATA_MMVF_ROWS=1): RPB output rows per block. Each block reads the activations once for RPB rows.
// Per output, thread t walks pairs t, t + BLOCK_SIZE, ... with the same two ordered FMAs and the same warp
// and block reductions: bit-identical to bf16_f32_mmvf_multi_kernel.
// S26 STRATA_LFUSE=1 (AUX): one more block computes row 0 of w_aux into y_aux[k * ldy_aux]
template <int BLOCK_SIZE, int NT, int RPB, bool AUX = false>
void bf16_f32_mmvf_rows_kernel(sycl::nd_item<1> item, const float* __restrict__ x, int64_t ldx,
                               const uint16_t* __restrict__ w, float* __restrict__ y, int64_t ldy,
                               int n_in, int n_out, int n_tok, sycl::local_accessor<float, 3> partials,
                               const uint16_t* __restrict__ w_aux = nullptr,
                               float* __restrict__ y_aux = nullptr, int64_t ldy_aux = 0) {
    const int t = item.get_local_id(0);
    int o0 = item.get_group(0) * RPB;
    if constexpr (AUX) {
        if (item.get_group(0) == item.get_group_range(0) - 1) {
            w = w_aux;
            y = y_aux;
            ldy = ldy_aux;
            n_out = 1;
            o0 = 0;
        }
    }
    float acc[RPB][NT];
#pragma unroll
    for (int r = 0; r < RPB; ++r) {
#pragma unroll
        for (int k = 0; k < NT; ++k)
            acc[r][k] = 0.0f;
    }
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        sycl::float2 in[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                in[k] = reinterpret_cast<const sycl::float2*>(x + (size_t)k * ldx)[pair];
            }
        }
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            if (o0 + r >= n_out) break;
            const uint16_t* row = w + (size_t)(o0 + r) * n_in;
            const uint32_t weight = reinterpret_cast<const uint32_t*>(row)[pair];
            const float w0 = f32_from_bf16((uint16_t)weight);
            const float w1 = f32_from_bf16((uint16_t)(weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k < n_tok) {
                    acc[r][k] = sycl::fma(w0, in[k].x(), acc[r][k]);
                    acc[r][k] = sycl::fma(w1, in[k].y(), acc[r][k]);
                }
            }
        }
    }
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
#pragma unroll
    for (int r = 0; r < RPB; ++r) {
#pragma unroll
        for (int k = 0; k < NT; ++k)
            acc[r][k] = sycl::reduce_over_group(sg, acc[r][k], sycl::plus<float>());
    }

    if constexpr (BLOCK_SIZE > SG_SZ) {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < RPB; ++r) {
#pragma unroll
                for (int k = 0; k < NT; ++k)
                    partials[r][k][sg_id] = acc[r][k];
            }
        }
        sycl::group_barrier(item.get_group());
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                acc[r][k] = 0.0f;
                constexpr int num_sgs = BLOCK_SIZE / SG_SZ;
                for (int s = lane; s < num_sgs; s += SG_SZ) {
                    acc[r][k] += partials[r][k][s];
                }
                acc[r][k] = sycl::reduce_over_group(sg, acc[r][k], sycl::plus<float>());
            }
        }
    }
    if (t == 0) {
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            if (o0 + r < n_out) {
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    if (k < n_tok) y[(size_t)k * ldy + o0 + r] = acc[r][k];
                }
            }
        }
    }
}

template <int B>
void launch_rows(sycl::queue* q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                 int n_in, int n_out, int n_tok, const uint16_t* w_aux = nullptr, float* y_aux = nullptr,
                 int64_t ldy_aux = 0) {
    constexpr int RPB = 4;
    const unsigned nb = (unsigned)((n_out + RPB - 1) / RPB);
    const unsigned total_blocks = w_aux != nullptr ? (nb + 1) : nb;
    constexpr int num_sgs = (B + SG_SZ - 1) / SG_SZ;

    if (w_aux != nullptr) {
        if (n_tok <= 4) {
            q->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 3> partials(sycl::range<3>(RPB, 4, num_sgs), cgh);
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)total_blocks * B), sycl::range<1>(B)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                                     bf16_f32_mmvf_rows_kernel<B, 4, RPB, true>(
                                         item, x, ldx, w, y, ldy, n_in, n_out, n_tok, partials, w_aux, y_aux, ldy_aux);
                                 });
            });
        } else {
            q->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 3> partials(sycl::range<3>(RPB, 8, num_sgs), cgh);
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)total_blocks * B), sycl::range<1>(B)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                                     bf16_f32_mmvf_rows_kernel<B, 8, RPB, true>(
                                         item, x, ldx, w, y, ldy, n_in, n_out, n_tok, partials, w_aux, y_aux, ldy_aux);
                                 });
            });
        }
    } else {
        if (n_tok <= 4) {
            q->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 3> partials(sycl::range<3>(RPB, 4, num_sgs), cgh);
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)total_blocks * B), sycl::range<1>(B)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                                     bf16_f32_mmvf_rows_kernel<B, 4, RPB, false>(
                                         item, x, ldx, w, y, ldy, n_in, n_out, n_tok, partials);
                                 });
            });
        } else {
            q->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 3> partials(sycl::range<3>(RPB, 8, num_sgs), cgh);
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)total_blocks * B), sycl::range<1>(B)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                                     bf16_f32_mmvf_rows_kernel<B, 8, RPB, false>(
                                         item, x, ldx, w, y, ldy, n_in, n_out, n_tok, partials);
                                 });
            });
        }
    }
}

}  // namespace

bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream) {
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (!rows || n_out < 64 || n_tok < 2 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || (ldx & 1) != 0 ||
        x == nullptr || w == nullptr || y == nullptr || w_aux == nullptr || y_aux == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        return false;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const int ni = (int)n_in, no = (int)n_out;
    switch (mmvf_block_size(n_in)) {
        case 32: launch_rows<32>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        case 64: launch_rows<64>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        case 96: launch_rows<96>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        case 128: launch_rows<128>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        case 160: launch_rows<160>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        case 192: launch_rows<192>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        case 224: launch_rows<224>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
        default: launch_rows<256>(q, x, ldx, w, y, ldy, ni, no, n_tok, w_aux, y_aux, ldy_aux); break;
    }
    return true;
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in,
                               int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) {
        bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream);
        return;
    }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();

    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (rows && n_out >= 64) {
        const int ni = (int)n_in, no = (int)n_out;
        switch (mmvf_block_size(n_in)) {
            case 32: launch_rows<32>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            case 64: launch_rows<64>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            case 96: launch_rows<96>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            case 128: launch_rows<128>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            case 160: launch_rows<160>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            case 192: launch_rows<192>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            case 224: launch_rows<224>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
            default: launch_rows<256>(q, x, ldx, w, y, ldy, ni, no, n_tok); break;
        }
        return;
    }

#define STRATA_MMVF_M(N)                                                                                         \
    case N:                                                                                                      \
        if (n_tok <= 4) {                                                                                        \
            q->submit([&](sycl::handler& cgh) {                                                                  \
                sycl::local_accessor<float, 2> partials(sycl::range<2>(4, 32), cgh);                             \
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * (N)), sycl::range<1>(N)),      \
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {               \
                                     bf16_f32_mmvf_multi_kernel<N, 4>(item, x, ldx, w, y, ldy, (int)n_in, n_tok, \
                                                                      partials);                                 \
                                 });                                                                             \
            });                                                                                                  \
        } else {                                                                                                 \
            q->submit([&](sycl::handler& cgh) {                                                                  \
                sycl::local_accessor<float, 2> partials(sycl::range<2>(8, 32), cgh);                             \
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * (N)), sycl::range<1>(N)),      \
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {               \
                                     bf16_f32_mmvf_multi_kernel<N, 8>(item, x, ldx, w, y, ldy, (int)n_in, n_tok, \
                                                                      partials);                                 \
                                 });                                                                             \
            });                                                                                                  \
        }                                                                                                        \
        break

    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32);
        STRATA_MMVF_M(64);
        STRATA_MMVF_M(96);
        STRATA_MMVF_M(128);
        STRATA_MMVF_M(160);
        STRATA_MMVF_M(192);
        STRATA_MMVF_M(224);
        STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() || n_out <= 0 ||
        n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 || (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();

#define STRATA_MMVF_CASE(N)                                                                             \
    case N:                                                                                             \
        q->submit([&](sycl::handler& cgh) {                                                             \
            sycl::local_accessor<float, 1> partials(sycl::range<1>(32), cgh);                           \
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * (N)), sycl::range<1>(N)), \
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {          \
                                 bf16_f32_mmvf_kernel<N>(item, x, w, y, (int)n_in, partials);           \
                             });                                                                        \
        });                                                                                             \
        break

    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
}

void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream) {
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1)
            bf16_gemv_fp32_mmvf(x + (size_t)c0 * n_in, w, y + (size_t)c0 * n_out, n_in, n_out, stream);
        else
            bf16_gemv_fp32_mmvf_multi(x + (size_t)c0 * n_in, n_in, w, y + (size_t)c0 * n_out, n_out, n_in, n_out, nc,
                                      stream);
    }
}

}  // namespace strata::kernels
