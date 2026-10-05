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
[[maybe_unused]] void bf16_gemv_naive_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x,
                                             const uint16_t* __restrict__ w, float* __restrict__ y, long long n_in,
                                             long long n_out) {
    const long long o = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
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
    const long long o = (long long)item.get_group(0) * sgs_per_block + (long long)(item.get_local_id(0) / SUB_SIZE);
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
/// block reduces through shared memory. On Xe2, we first reduce within each SIMD16 subgroup via
/// single-cycle hardware reduction, then reduce the subgroup sums in local memory.
template <int SUB_SIZE = SG_SIZE>
void bf16_gemv_split_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                            float* __restrict__ y, long long n_in, long long n_out, int tpr,
                            sycl::local_accessor<float, 1> scratch) {
    const long long o = item.get_group(0);
    if (o >= n_out) return;
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int t = item.get_local_id(0);  // 0 .. tpr-1
    const int sg_id = t / SUB_SIZE;
    const int num_sgs = (tpr + SUB_SIZE - 1) / SUB_SIZE;

    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = t; i < n_in; i += tpr)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);

    // Subgroup reduction first (single-cycle hardware reduction on Xe2)
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) scratch[sg_id] = acc;
    item.barrier(sycl::access::fence_space::local_space);

    // First subgroup reduces the per-subgroup partials
    if (sg_id == 0) {
        float final_acc = 0.0f;
        for (int k = lane; k < num_sgs; k += SUB_SIZE)
            final_acc += scratch[k];
        final_acc = sycl::reduce_over_group(sg, final_acc, sycl::plus<float>());
        if (lane == 0) y[o] = final_acc;
    }
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || n_out <= 0) return;
    if (n_out >= 64) {
        const int sgs = THREADS / SG_SIZE;
        const unsigned grid = (unsigned)((n_out + sgs - 1) / sgs);
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            bf16_gemv_subgroup_kernel<SG_SIZE>(item, x, w, y, n_in, n_out);
                        });
        return;
    }
    // For narrow shapes (n_out < 64, e.g. ssm_alpha [2560, 48]), dispatch the SLM split reduction
    // with 128 threads per row to fully saturate Xe2 Vector Engines (160 engines).
    bf16_gemv_split(x, w, y, n_in, n_out, 128, stream);
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int threads_per_row,
                     void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row == 32 || threads_per_row == SG_SIZE) {
        const int sgs = THREADS / SG_SIZE;
        const unsigned grid = (unsigned)((n_out + sgs - 1) / sgs);
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            bf16_gemv_subgroup_kernel<SG_SIZE>(item, x, w, y, n_in, n_out);
                        });
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr,
                     "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                     "warp-per-row path)\n",
                     threads_per_row);
        std::exit(1);
    }
    const unsigned grid = (unsigned)n_out;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch_acc(sycl::range<1>(threads_per_row), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads_per_row)), sycl::range<1>(threads_per_row)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                             bf16_gemv_split_kernel<SG_SIZE>(item, x, w, y, n_in, n_out, threads_per_row, scratch_acc);
                         });
    });
}

}  // namespace strata::kernels
