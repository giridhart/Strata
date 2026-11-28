#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>

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

/// Dynamic Tuning Profile for Intel Xe2 microarchitecture targets (Arc B580 vs Lunar Lake)
struct Xe2TuningProfile {
    int split_k_factor;
    int slm_pad_elements;
    bool enable_dual_column;
    bool enable_streaming_hints;
    bool enable_xmx_moe;
};

inline Xe2TuningProfile get_xe2_tuning_profile(const sycl::device& dev) {
    const uint32_t cu_count = dev.get_info<sycl::info::device::max_compute_units>();
    const std::string dev_name = dev.get_info<sycl::info::device::name>();
    const bool is_bmg = (cu_count >= 128) || (dev_name.find("B580") != std::string::npos) ||
                        (dev_name.find("BMG") != std::string::npos);
    Xe2TuningProfile p;
    if (is_bmg) {
        // Desktop Intel Arc B580 (20 Xe-cores, 160 VEs)
        p.split_k_factor = 32;
        p.slm_pad_elements = 8;
        p.enable_dual_column = false;  // Opt-in via STRATA_XE2_DUAL_COLUMN=1 (preserves byte-exact parity)
        p.enable_streaming_hints = true;
        p.enable_xmx_moe = true;
    } else {
        // Integrated Lunar Lake (8 Xe-cores, 64 VEs) - default to verified baseline
        p.split_k_factor = 1;
        p.slm_pad_elements = 0;
        p.enable_dual_column = false;
        p.enable_streaming_hints = false;
        p.enable_xmx_moe = false;
    }

    // Tier 1: Dynamic Environment Variable Control (Zero-Recompile Toggling)
    if (const char* v = std::getenv("STRATA_XE2_SPLIT_K")) {
        p.split_k_factor = std::atoi(v);
    }
    if (const char* v = std::getenv("STRATA_XE2_SLM_PAD")) {
        const int pad = std::atoi(v);
        if (pad == 1) p.slm_pad_elements = 4;
        else if (pad == 2) p.slm_pad_elements = 8;
        else p.slm_pad_elements = pad;
    }
    if (const char* v = std::getenv("STRATA_XE2_DUAL_COLUMN")) {
        p.enable_dual_column = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("STRATA_XE2_CACHE_HINT")) {
        p.enable_streaming_hints = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("STRATA_XE2_XMX_MOE")) {
        p.enable_xmx_moe = (std::atoi(v) != 0);
    }
    return p;
}

}  // namespace strata::sycl_arch
