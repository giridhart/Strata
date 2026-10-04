// src/kernels/sycl/bf16_gemv.dp.cpp - Pure SYCL 2020 BF16 GEMV kernels.
// Optimized for Intel Xe2 microarchitecture (Battlemage B580 and Lunar Lake) native SIMD16 execution.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

/// One thread per output row, walking it contiguously. Kept as the naive reference the split version is
/// checked against, and used directly for the shapes where the output width is already large.
void bf16_gemv_naive_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                            float* __restrict__ y, long long n_in, long long n_out) {
    const long long o =
        (long long)item.get_group(0) * item.get_local_range(0) +
        item.get_local_id(0);
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = 0; i < n_in; ++i)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    y[o] = acc;
}

/// One SUBGROUP per output row, lanes striding the reduction axis. On Xe2 (B580/Lunar Lake),
/// this executes natively in SIMD16 with single-cycle hardware group reduction.
template <int SUB_SIZE = SG_SIZE>
void bf16_gemv_subgroup_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                               float* __restrict__ y, long long n_in, long long n_out) {
    const auto sg = item.get_sub_group();
    const int sgs_per_block = (int)(item.get_local_range(0) / SUB_SIZE);
    const long long o = (long long)item.get_group(0) * sgs_per_block +
                        (long long)(item.get_local_id(0) / SUB_SIZE);
    if (o >= n_out) return;
    const int lane = sg.get_local_linear_id();
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = lane; i < n_in; i += SUB_SIZE)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[o] = acc;
}

/// TPR lanes cooperate on ONE row's reduction: `threads_per_row` threads each take a strided slice and the
/// block reduces through shared memory. Used when the output width is too small to fill the machine.
void bf16_gemv_split_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                            float* __restrict__ y, long long n_in, long long n_out, int tpr,
                            sycl::local_accessor<float, 1> scratch) {
    const long long o = item.get_group(0);
    if (o >= n_out) return;
    const int t = item.get_local_id(0); // 0 .. tpr-1
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = t; i < n_in; i += tpr)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    // block reduction; `tpr` is at most a few hundred, so a tree in shared is enough
    scratch[t] = acc;
    item.barrier(sycl::access::fence_space::local_space);
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        item.barrier(sycl::access::fence_space::local_space);
    }
    if (t == 0) y[o] = scratch[0];
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || n_out <= 0) return;
    if (n_out >= 64) {
        const int sgs = THREADS / SG_SIZE;
        const unsigned grid = (unsigned) ((n_out + sgs - 1) / sgs);
        q->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                bf16_gemv_subgroup_kernel<SG_SIZE>(item, x, w, y, n_in, n_out);
            });
        return;
    }
    const unsigned grid = (unsigned) ((n_out + THREADS - 1) / THREADS);
    q->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            bf16_gemv_naive_kernel(item, x, w, y, n_in, n_out);
        });
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row == 32 || threads_per_row == SG_SIZE) {
        const int sgs = THREADS / SG_SIZE;
        const unsigned grid = (unsigned) ((n_out + sgs - 1) / sgs);
        q->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                bf16_gemv_subgroup_kernel<SG_SIZE>(item, x, w, y, n_in, n_out);
            });
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                             "warp-per-row path)\n", threads_per_row);
        std::exit(1);
    }
    const unsigned grid = (unsigned) n_out;
    q->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> scratch_acc(sycl::range<1>(threads_per_row), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((grid) * (threads_per_row)), sycl::range<1>(threads_per_row)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                bf16_gemv_split_kernel(
                    item, x, w, y, n_in, n_out, threads_per_row,
                    scratch_acc);
            });
    });
}

}  // namespace strata::kernels
