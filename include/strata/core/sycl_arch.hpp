#pragma once
#include <sycl/sycl.hpp>

namespace strata::sycl_arch {

#if defined(STRATA_TARGET_BMG) || defined(STRATA_TARGET_XE2) || defined(__INTEL_XE2__)
// Intel Arc Battlemage (Xe2-HPG, B580) and Lunar Lake (Xe2-LPG) native SIMD16 execution units
constexpr int SUB_GROUP_SIZE = 16;
constexpr bool IS_XE2 = true;
#else
// Generic fallback / Xe1 dual-issue SIMD
constexpr int SUB_GROUP_SIZE = 32;
constexpr bool IS_XE2 = false;
#endif

/// Standard SYCL subgroup reduction primitive mapping natively to Xe2 hardware reduction
template <typename T, typename BinaryOp = sycl::plus<T>>
inline T sub_group_reduce(const sycl::sub_group& sg, T val, BinaryOp op = BinaryOp{}) {
    return sycl::reduce_over_group(sg, val, op);
}

} // namespace strata::sycl_arch
