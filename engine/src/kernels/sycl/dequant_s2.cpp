// src/kernels/sycl/dequant_s2.cpp - S2 dequantization kernel for Intel Xe2 microarchitecture.
//
// S2 is Q2_0's canonical form and 31.64 GiB of the artifact is Q2_0, so this is the hottest decode in the engine.
// Optimized for Intel Xe2 (Arc B580 / Lunar Lake) native SIMD16 execution units:
// 16 lanes per subgroup process one 64-element S2 block with fully coalesced 16-byte code loads and
// coalesced 256-byte vector float4 stores.
#include <sycl/sycl.hpp>
#include <cstdint>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/dequant_s2.hpp"

namespace strata::kernels {
namespace {

constexpr int QK = 64;  // S2 group = Q2_0 block = 64 elements
constexpr int THREADS = 256;
constexpr int SG_SIZE = 16;  // Native SIMD16 subgroup size on Xe2

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

void dequant_s2_subgroup_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                float* __restrict__ out, long long n_blocks, sycl::nd_item<1> item) {
    const int sgs_per_wg = item.get_local_range(0) / SG_SIZE;
    const long long b = (long long)item.get_group(0) * sgs_per_wg + (item.get_local_id(0) / SG_SIZE);
    if (b >= n_blocks) return;

    const int lane = item.get_local_id(0) % SG_SIZE;
    const float d = scales[b];
    const uint8_t byte = codes[b * 16 + lane];

    const int c0 = (byte >> 0) & 0x03;
    const int c1 = (byte >> 2) & 0x03;
    const int c2 = (byte >> 4) & 0x03;
    const int c3 = (byte >> 6) & 0x03;

    reinterpret_cast<sycl::float4*>(out + b * QK)[lane] = sycl::float4(
        (float)(c0 - 1) * d,
        (float)(c1 - 1) * d,
        (float)(c2 - 1) * d,
        (float)(c3 - 1) * d
    );
}

}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    constexpr int sgs_per_wg = THREADS / SG_SIZE;
    const long long num_wgs = (n_blocks + sgs_per_wg - 1) / sgs_per_wg;
    auto& q = default_queue();
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)num_wgs * THREADS), sycl::range<1>(THREADS)),
                   [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                       dequant_s2_subgroup_kernel(codes, scales, out, n_blocks, item);
                   })
        .wait();
}

}  // namespace strata::kernels
