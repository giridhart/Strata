// src/kernels/sycl/s2_gemv_fast.dp.cpp - Pure SYCL 2020 refactoring
// Two levers on the hottest kernel, BOTH MEASURED NEGATIVE OR NEUTRAL.
//
// *** THIS FILE IS NAMED "fast" AND IS NOT FASTER.  DO NOT ADOPT IT AS THE FAST PATH. ***
// It is kept because the negative result is worth more than the file costs: it closes two plausible
// optimisations and it is the experiment that established the measurement floor.
//
// L5 got the S2 GEMV from 10.5 to 86.2 G weights/s by amortising the loads, and left the instruction-count
// argument confirmed but unfinished.  Two further reductions were obvious, so both were built and measured:
//
// LEVER 1 - STAGE `x` IN SHARED MEMORY.  `x` is n_in halves (5 KB at n_embd 2560) and every output row reads
// all of it, so the generic kernel issues n_out * n_in activation loads per GEMV.  Staging it once per block
// measured WORSE in every single paired comparison (59-76 G w/s staged against 64-87 G w/s unstaged, across
// four sweep passes and three thread counts).  The __syncthreads and the copy cost more than the L1/L2 hits
// they replace - the same 5 KB is already resident and already broadcast.
//
// LEVER 2 - A CONSTANT-MEMORY CODE TABLE.  Turn the per-element shifts, masks and int-to-float convert into
// one broadcast `float4` load per four elements.  Measured NEUTRAL: the two orderings swap between passes
// with no consistent winner.
//
// WHY THE NEGATIVE RESULT IS THE REAL FINDING: run-to-run variance on this machine is now ~20% within a
// single configuration (quads tpr=32 measured 64.3 to 79.5 G w/s across four passes), because `verify` holds
// ~97% of six CPU cores.  **The noise floor now EXCEEDS the effect sizes being chased**, so further
// micro-optimisation of this kernel cannot be evaluated here and should wait for a quiet machine rather than
// produce more numbers that cannot be distinguished from noise.
//
// Both changes alter the summation order or the expression, so this kernel is checked against the naive
// reference like every other one - being faster is never a reason to be trusted, and being neutral is not
// either.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;
constexpr int MAX_SHARED_HALVES = 4096;      // 8 KB of shared for x; n_embd 2560 fits with room

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

// Direct-to-register dequantization computes the four code values with the -1 bias
// directly in registers from bits 0, 2, 4, 6 without constant-memory table overhead.

// One workgroup per output row. `STAGE_X` says whether x was copied to local memory.
template <bool STAGE_X>
void s2_gemv_fast_kernel(const uint16_t *__restrict__ x,
                         const uint8_t *__restrict__ codes,
                         const float *__restrict__ scales,
                         float *__restrict__ y, long long n_in, long long n_out,
                         int threads_per_row,
                         sycl::nd_item<1> item_ct1,
                         float* smem) {
    sycl::half *sx = reinterpret_cast<sycl::half *>(smem);
    float* partial = smem + (STAGE_X ? (MAX_SHARED_HALVES / 2) : 0);

    const long long o = item_ct1.get_group(0);
    if (o >= n_out) return;
    const int tid = item_ct1.get_local_id(0);

    if (STAGE_X) {
        const sycl::half *xh = reinterpret_cast<const sycl::half *>(x);
        for (long long i = tid; i < n_in; i += threads_per_row) sx[i] = xh[i];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                          // ONE load for four codes
        const float d = s[q >> 4];                          // (q*4) >> 6
        // Direct-to-register dequantization: unpack codes without memory or table lookup
        const float cv0 = static_cast<float>((int)(byte & 3) - 1);
        const float cv1 = static_cast<float>((int)((byte >> 2) & 3) - 1);
        const float cv2 = static_cast<float>((int)((byte >> 4) & 3) - 1);
        const float cv3 = static_cast<float>((int)((byte >> 6) & 3) - 1);
        if (STAGE_X) {
            const sycl::half2 h01 =
                *reinterpret_cast<const sycl::half2 *>(&sx[q * 4]);
            const sycl::half2 h23 =
                *reinterpret_cast<const sycl::half2 *>(&sx[q * 4 + 2]);
            a0 += cv0 * d * h01[0];
            a1 += cv1 * d * h01[1];
            a2 += cv2 * d * h23[0];
            a3 += cv3 * d * h23[1];
        } else {
            const sycl::uint2 xw2 = *reinterpret_cast<const sycl::uint2 *>(x + q * 4);
            const sycl::half2 h01 = *reinterpret_cast<const sycl::half2 *>(&xw2.x());
            const sycl::half2 h23 = *reinterpret_cast<const sycl::half2 *>(&xw2.y());
            a0 += cv0 * d * h01[0];
            a1 += cv1 * d * h01[1];
            a2 += cv2 * d * h23[0];
            a3 += cv3 * d * h23[1];
        }
    }
    float acc_sum = (a0 + a1) + (a2 + a3);
    const auto sg = item_ct1.get_sub_group();
    if (threads_per_row <= SG_SIZE) {
        acc_sum = sycl::reduce_over_group(sg, acc_sum, sycl::plus<float>());
        if (tid == 0) y[o] = acc_sum;
    } else {
        acc_sum = sycl::reduce_over_group(sg, acc_sum, sycl::plus<float>());
        if (sg.get_local_linear_id() == 0) {
            partial[tid / SG_SIZE] = acc_sum;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const int num_sgs = threads_per_row / SG_SIZE;
        if (tid < num_sgs) {
            float p_val = partial[tid];
            p_val = sycl::reduce_over_group(sg, p_val, sycl::plus<float>());
            if (tid == 0) y[o] = p_val;
        }
    }
}

}  // namespace

void s2_gemv_fast(const uint16_t *x, const uint8_t *codes, const float *scales,
                  float *y, int64_t n_in, int64_t n_out, int threads_per_row,
                  bool stage_x) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    if (stage_x && n_in > MAX_SHARED_HALVES) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld exceeds the %d-half shared staging limit\n",
                     (long long) n_in, MAX_SHARED_HALVES);
        std::exit(1);
    }
    const size_t smem_floats = (stage_x ? (MAX_SHARED_HALVES / 2) : 0) + (size_t)threads_per_row;
    auto& q = default_queue();
    const sycl::nd_range<1> range(sycl::range<1>((size_t)n_out * threads_per_row),
                                  sycl::range<1>(threads_per_row));
    if (stage_x) {
        q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(smem_floats), cgh);
            cgh.parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                s2_gemv_fast_kernel<true>(
                    x, codes, scales, y, n_in, n_out, threads_per_row,
                    item_ct1,
                    smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    } else {
        q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(smem_floats), cgh);
            cgh.parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                s2_gemv_fast_kernel<false>(
                    x, codes, scales, y, n_in, n_out, threads_per_row,
                    item_ct1,
                    smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    }
    q.wait();
}

}  // namespace strata::kernels
