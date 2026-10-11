// src/kernels/sycl/s2_gemv_quads.dp.cpp - Pure SYCL 2020 refactoring
// S2 with the code load amortised over four elements.
//
// THE HYPOTHESIS THIS TESTS.  L4 measured 66 G weights/s = 0.55 weights/cycle/SM, and at an estimated ~13
// instructions per weight that is ~7.2 instructions/cycle/SM against an SM's 4/cycle issue ceiling.  Compute
// (0.4% of FP32 peak) and bandwidth (2%) are excluded by measurement and the dependency chain was excluded by
// experiment in L4, so instruction ISSUE is what remains.  If that is right, cutting instructions per weight
// must move the rate; if the rate does not move, the hypothesis is wrong and that is worth knowing too.
//
// WHERE THE INSTRUCTIONS GO.  For S2 four consecutive elements share ONE byte (element i at byte i/4, bits
// (i%4)*2), and four halves of x are eight contiguous bytes.  The generic kernel walks with a thread stride
// and loads one byte and one half PER ELEMENT; a thread that takes a contiguous quad instead loads one byte
// and one 64-bit word for all four.  Memory instructions per element drop from 2 to 0.5, and the address
// arithmetic is computed once rather than four times.
//
// S2 IS SPECIALISED DELIBERATELY: it is 31.64 GiB of the 38 GiB pack, so it is the kernel that matters.  The
// generic `s_gemv_split` still serves S4 and S8.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// QK = 64 elements per group; a quad of 4 elements is 1/16 of a group, so
//     group = (quad * 4) >> 6 = quad >> 4
constexpr int QK_S2 = 64;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

void s2_gemv_quads_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                          const float* __restrict__ scales, float* __restrict__ y, long long n_in, long long n_out,
                          int threads_per_row, sycl::nd_item<1> item_ct1, float* partial) {
    const long long o = item_ct1.get_group(0);
    if (o >= n_out) return;
    const int tid = item_ct1.get_local_id(0);

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;  // exactly one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];  // ONE load for four codes
        const float d = s[q >> 4];  // (q*4) >> 6, the group index as a shift
        // ONE 64-bit load for four halves.  `x` is 256-byte aligned and four halves are eight bytes, so
        // this is an aligned uint2 - if it were not, the misaligned access would fault rather than be slow.
        const sycl::uint2 xw = *reinterpret_cast<const sycl::uint2*>(x + q * 4);
        const sycl::half2 h01 = *reinterpret_cast<const sycl::half2*>(&xw.x());
        const sycl::half2 h23 = *reinterpret_cast<const sycl::half2*>(&xw.y());
        const float q0 = (float)((int)(byte & 3) - 1) * (float)h01[0] +
                         (float)((int)((byte >> 2) & 3) - 1) * (float)h01[1];
        const float q1 = (float)((int)((byte >> 4) & 3) - 1) * (float)h23[0] +
                         (float)((int)((byte >> 6) & 3) - 1) * (float)h23[1];
        a0 += q0 * d;
        a1 += q1 * d;
    }
    float acc_sum = a0 + a1;
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

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out,
                   int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long)n_in);
        std::exit(1);
    }
    auto& q = default_queue();
    const sycl::nd_range<1> range(sycl::range<1>((size_t)n_out * threads_per_row), sycl::range<1>(threads_per_row));
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> partial_acc(sycl::range<1>(threads_per_row), cgh);
        cgh.parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            s2_gemv_quads_kernel(x, codes, scales, y, n_in, n_out, threads_per_row, item_ct1,
                                 partial_acc.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
    q.wait();
}

}  // namespace strata::kernels
