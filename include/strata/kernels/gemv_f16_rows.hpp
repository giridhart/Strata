// include/strata/kernels/gemv_f16_rows.hpp - Multi-Row Matrix-Vector Multiplication for SYCL 2020 on Intel Xe2.
//
// Ports the DeepSeek-V4.1 multi-row GEMV technique (src/dsv41/ds4_kernels.cu lines 370-435) to pure SYCL 2020.
// In speculative decoding (MTP) and small-batch inference, evaluating T draft tokens by launching standard GEMV
// re-reads the entire weight matrix T times from VRAM, saturating memory bandwidth.
// gemv_f16_rows reads each row of weights from memory ONCE into registers, amortizing memory traffic across
// all T activation rows simultaneously.
#pragma once

#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Standard single-row GEMV kernel: evaluates row of weights against one activation row t.
/// Each sub-group processes one row using 16-byte vector loads (sycl::half8 and sycl::float4)
/// and reduces across lanes using sycl::reduce_over_group.
template <int SG_SIZE = 16, int WG_SIZE = 256>
void gemv_f16_kernel(sycl::nd_item<1> item,
                     const sycl::half* __restrict__ W,
                     const float* __restrict__ x,
                     int n, int k, float alpha,
                     float* __restrict__ y,
                     int t, size_t ldx, size_t ldy) {
    auto sg = item.get_sub_group();
    constexpr int sgs_per_wg = WG_SIZE / SG_SIZE;
    const int gid = item.get_group(0);
    const int lid = item.get_local_id(0);
    const int sg_in_wg = lid / SG_SIZE;
    const int row = gid * sgs_per_wg + sg_in_wg;
    const int lane = sg.get_local_linear_id();

    if (row >= n) return;

    const sycl::half* w = W + static_cast<size_t>(row) * k;
    const float* xv = x + static_cast<size_t>(t) * ldx;
    float acc = 0.0f;

    for (int c = lane * 8; c < k; c += SG_SIZE * 8) {
        const sycl::vec<sycl::half, 8> q = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(w + c);
        const float* xp = xv + c;
        const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(xp);
        const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(xp + 4);

        const float w0 = static_cast<float>(q[0]);
        const float w1 = static_cast<float>(q[1]);
        const float w2 = static_cast<float>(q[2]);
        const float w3 = static_cast<float>(q[3]);
        const float w4 = static_cast<float>(q[4]);
        const float w5 = static_cast<float>(q[5]);
        const float w6 = static_cast<float>(q[6]);
        const float w7 = static_cast<float>(q[7]);

        acc += (w0 * a[0] + w1 * a[1]) + (w2 * a[2] + w3 * a[3]) +
               (w4 * b[0] + w5 * b[1]) + (w6 * b[2] + w7 * b[3]);
    }

    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) {
        y[static_cast<size_t>(t) * ldy + row] = alpha * acc;
    }
}

/// Multi-row GEMV kernel: evaluates row of weights across ROWS_T activation rows simultaneously.
/// Reads each 16-byte weight chunk ONCE into registers, amortizing weight bandwidth across all ROWS_T tokens.
/// Performs sub-group reductions for all active rows and writes outputs.
template <int ROWS_T, int SG_SIZE = 16, int WG_SIZE = 256>
void gemv_f16_rows_kernel(sycl::nd_item<1> item,
                          const sycl::half* __restrict__ W,
                          const float* __restrict__ x, size_t ldx,
                          int n, int k, int T, float alpha,
                          float* __restrict__ y, size_t ldy) {
    auto sg = item.get_sub_group();
    constexpr int sgs_per_wg = WG_SIZE / SG_SIZE;
    const int gid = item.get_group(0);
    const int lid = item.get_local_id(0);
    const int sg_in_wg = lid / SG_SIZE;
    const int task_id = gid * sgs_per_wg + sg_in_wg;

    const int total_tiles_t = (T + ROWS_T - 1) / ROWS_T;
    const int total_tasks = n * total_tiles_t;
    if (task_id >= total_tasks) return;

    const int row = task_id % n;
    const int tile_idx = task_id / n;
    const int t0 = tile_idx * ROWS_T;
    const int nt = sycl::min(ROWS_T, T - t0);
    const int lane = sg.get_local_linear_id();

    const sycl::half* w = W + static_cast<size_t>(row) * k;
    float acc[ROWS_T];
    #pragma unroll
    for (int t = 0; t < ROWS_T; ++t) acc[t] = 0.0f;

    for (int c = lane * 8; c < k; c += SG_SIZE * 8) {
        const sycl::vec<sycl::half, 8> q = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(w + c);
        const float w0 = static_cast<float>(q[0]);
        const float w1 = static_cast<float>(q[1]);
        const float w2 = static_cast<float>(q[2]);
        const float w3 = static_cast<float>(q[3]);
        const float w4 = static_cast<float>(q[4]);
        const float w5 = static_cast<float>(q[5]);
        const float w6 = static_cast<float>(q[6]);
        const float w7 = static_cast<float>(q[7]);

        #pragma unroll
        for (int t = 0; t < ROWS_T; ++t) {
            if (t < nt) {
                const float* xp = x + static_cast<size_t>(t0 + t) * ldx + c;
                const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(xp);
                const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(xp + 4);
                acc[t] += (w0 * a[0] + w1 * a[1]) + (w2 * a[2] + w3 * a[3]) +
                          (w4 * b[0] + w5 * b[1]) + (w6 * b[2] + w7 * b[3]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < ROWS_T; ++t) {
        if (t < nt) {
            float sum = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
            if (lane == 0) {
                y[static_cast<size_t>(t0 + t) * ldy + row] = alpha * sum;
            }
        }
    }
}

/// Baseline GEMV: launches standard single-row GEMV across T tokens (re-reading weights per token).
inline void gemv_f16(const sycl::half* W, const float* x, int n, int k, int T, float alpha,
                     float* y, sycl::queue& q, size_t ldx = 0, size_t ldy = 0) {
    if (T <= 0 || n <= 0 || k <= 0) return;
    if (ldx == 0) ldx = k;
    if (ldy == 0) ldy = n;

    constexpr int SG_SIZE = 16;
    constexpr int WG_SIZE = 256;
    constexpr int sgs_per_wg = WG_SIZE / SG_SIZE;
    const int num_wgs = (n + sgs_per_wg - 1) / sgs_per_wg;

    for (int t = 0; t < T; ++t) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 gemv_f16_kernel<SG_SIZE, WG_SIZE>(item, W, x, n, k, alpha, y, t, ldx, ldy);
                             });
        });
    }
}

inline void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha,
                     float* y, sycl::queue& q, size_t ldx = 0, size_t ldy = 0) {
    gemv_f16(reinterpret_cast<const sycl::half*>(W), x, n, k, T, alpha, y, q, ldx, ldy);
}

/// Multi-row GEMV: evaluates T tokens tiled over ROWS_T (default 4 on Xe2).
/// Each sub-group evaluates a row of weights across ROWS_T activation vectors simultaneously.
template <int ROWS_T = 4>
inline void gemv_f16_rows(const sycl::half* W, const float* x, size_t ldx, int n, int k, int T,
                          float alpha, float* y, size_t ldy, sycl::queue& q) {
    if (T <= 0 || n <= 0 || k <= 0) return;
    if (ldx == 0) ldx = k;
    if (ldy == 0) ldy = n;

    constexpr int SG_SIZE = 16;
    constexpr int WG_SIZE = 256;
    constexpr int sgs_per_wg = WG_SIZE / SG_SIZE;

    const int total_tiles_t = (T + ROWS_T - 1) / ROWS_T;
    const int total_tasks = n * total_tiles_t;
    const int num_wgs = (total_tasks + sgs_per_wg - 1) / sgs_per_wg;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                             gemv_f16_rows_kernel<ROWS_T, SG_SIZE, WG_SIZE>(
                                 item, W, x, ldx, n, k, T, alpha, y, ldy);
                         });
    });
}

template <int ROWS_T = 4>
inline void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T,
                          float alpha, float* y, size_t ldy, sycl::queue& q) {
    gemv_f16_rows<ROWS_T>(reinterpret_cast<const sycl::half*>(W), x, ldx, n, k, T, alpha, y, ldy, q);
}

}  // namespace strata::kernels
