// src/kernels/sycl/s2_gemv_q8.dp.cpp - Pure SYCL 2020 S2 GEMV with Q8_0 activations.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

void s2_gemv_q8_kernel(sycl::nd_item<1> item,
                       const uint8_t* __restrict__ act, const uint8_t* __restrict__ codes,
                       const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                       long long n_out, int threads_per_row,
                       sycl::local_accessor<float, 1> partial) {
    const long long o = item.get_group(0);
    if (o >= n_out) return;
    const int tid = item.get_local_id(0);

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;             // one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];                      // (q*4) >> 6, the S2 group index as a shift
        // the activation quad: four int8 in one 32-element block, so one block scale
        const long long ablk = (q * 4) / QK8_0;
        const uint8_t* blk = act + ablk * 34;
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        sycl::half h;
        *reinterpret_cast<uint16_t*>(&h) = dbits;
        const float dx = static_cast<float>(h);
        const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
        const int off = (int) ((q * 4) % QK8_0);

        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * ((float) xq[off + 0] * dx);
        a1 += w1 * ((float) xq[off + 1] * dx);
        a2 += w2 * ((float) xq[off + 2] * dx);
        a3 += w3 * ((float) xq[off + 3] * dx);
    }
    float acc_sum = (a0 + a1) + (a2 + a3);
    const auto sg = item.get_sub_group();
    if (threads_per_row <= SG_SIZE) {
        acc_sum = sycl::reduce_over_group(sg, acc_sum, sycl::plus<float>());
        if (tid == 0) y[o] = acc_sum;
    } else {
        acc_sum = sycl::reduce_over_group(sg, acc_sum, sycl::plus<float>());
        if (sg.get_local_linear_id() == 0) {
            partial[tid / SG_SIZE] = acc_sum;
        }
        item.barrier(sycl::access::fence_space::local_space);
        const int num_sgs = threads_per_row / SG_SIZE;
        if (tid < num_sgs) {
            float p_val = partial[tid];
            p_val = sycl::reduce_over_group(sg, p_val, sycl::plus<float>());
            if (tid == 0) y[o] = p_val;
        }
    }
}

}  // namespace

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();

    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }

    q->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> partial_acc(sycl::range<1>(threads_per_row), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(((unsigned)n_out) * (threads_per_row)), sycl::range<1>(threads_per_row)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                s2_gemv_q8_kernel(
                    item, act, codes, scales, y, n_in, n_out, threads_per_row,
                    partial_acc);
            });
    });
}

}  // namespace strata::kernels
