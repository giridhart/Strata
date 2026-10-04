// src/kernels/cuda/s2_gemv.cu - P2.S2: the S2 GEMV, one thread per output row.
//
// Naive per the phase rule: dequantize on the fly, FP32 accumulation inside the row, no shared memory, no
// vector loads, no __ldg hints.  Phase 3 changes this file; the parity test is what says whether a change is
// still right.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/s2_gemv.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 64;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

void s2_gemv_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                    const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                    long long n_out) {
    const long long o =
        (long long)item.get_group(0) * item.get_local_range(0) +
        item.get_local_id(0);
    if (o >= n_out) return;

    const long long nb = n_in / QK;
    const uint8_t* c = codes + o * nb * (QK / 4);
    const float* s = scales + o * nb;

    float acc = 0.0f;
    for (long long b = 0; b < nb; ++b) {
        const float d = s[b];
        const uint8_t* cb = c + b * (QK / 4);
        const uint16_t* xb = x + b * QK;
        for (int j = 0; j < QK; ++j) {
            const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
            acc += (float)(code - 1) * d *
                   sycl::vec<sycl::half, 1>(
                       sycl::bit_cast<sycl::half, unsigned short>(xb[j]))
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        }
    }
    y[o] = acc;
}

template <int SUB_SIZE = SG_SIZE>
void s2_gemv_subgroup_kernel(sycl::nd_item<1> item,
                             const uint16_t* __restrict__ x,
                             const uint8_t* __restrict__ codes,
                             const float* __restrict__ scales,
                             float* __restrict__ y,
                             long long n_in,
                             long long n_out) {
    const auto sg = item.get_sub_group();
    const int sgs_per_block = (int)(item.get_local_range(0) / SUB_SIZE);
    const long long o = (long long)item.get_group(0) * sgs_per_block +
                        (long long)(item.get_local_id(0) / SUB_SIZE);
    if (o >= n_out) return;

    const int lane = sg.get_local_linear_id();
    const long long nb = n_in / QK;
    const uint8_t* c = codes + o * nb * (QK / 4);
    const float* s = scales + o * nb;

    double acc = 0.0;
    for (long long b = 0; b < nb; ++b) {
        const double d = (double)s[b];
        const uint8_t* cb = c + b * (QK / 4);
        const uint16_t* xb = x + b * QK;

        for (int j_byte = lane; j_byte < QK / 4; j_byte += SUB_SIZE) {
            const uint8_t byte = cb[j_byte];
            const int base_j = j_byte * 4;
            const int c0 = (byte >> 0) & 0x03;
            const int c1 = (byte >> 2) & 0x03;
            const int c2 = (byte >> 4) & 0x03;
            const int c3 = (byte >> 6) & 0x03;

            const sycl::half h0 = sycl::bit_cast<sycl::half, unsigned short>(xb[base_j + 0]);
            const sycl::half h1 = sycl::bit_cast<sycl::half, unsigned short>(xb[base_j + 1]);
            const sycl::half h2 = sycl::bit_cast<sycl::half, unsigned short>(xb[base_j + 2]);
            const sycl::half h3 = sycl::bit_cast<sycl::half, unsigned short>(xb[base_j + 3]);

            acc += (double)(c0 - 1) * d * (double)static_cast<float>(h0);
            acc += (double)(c1 - 1) * d * (double)static_cast<float>(h1);
            acc += (double)(c2 - 1) * d * (double)static_cast<float>(h2);
            acc += (double)(c3 - 1) * d * (double)static_cast<float>(h3);
        }
    }
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<double>());
    if (lane == 0) y[o] = (float)acc;
}

}  // namespace

void s2_gemv(const uint16_t *x, const uint8_t *codes, const float *scales,
             float *y, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long) n_in, QK);
        std::exit(1);
    }
    auto& q = default_queue();
    if (n_out >= 64) {
        constexpr int threads = 256;
        const int sgs = threads / SG_SIZE;
        const unsigned blocks = (unsigned) ((n_out + sgs - 1) / sgs);
        q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                s2_gemv_subgroup_kernel<SG_SIZE>(item, x, codes, scales, y, n_in, n_out);
            });
        return;
    }
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((unsigned)blocks) * (threads)), sycl::range<1>(threads)),
        [=](sycl::nd_item<1> item) {
            s2_gemv_kernel(item, x, codes, scales, y, n_in, n_out);
        });
}

}  // namespace strata::kernels
