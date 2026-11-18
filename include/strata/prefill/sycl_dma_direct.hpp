#pragma once
// SYCL direct-DMA USM check for expert blobs on the prompt path.
// Kept in a small header so prefill and its SYCL unit tests share the exact predicate.

#include <sycl/sycl.hpp>

#include "strata/core/expert_source.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <unordered_map>

namespace strata::prefill {

/// Returns true when `[blob, blob + bytes)` can be copied directly with `queue::memcpy` without staging.
/// Interior 2 MiB buckets that are wholly `sycl::usm::alloc::host` are cached in `usm_bucket` for fast
/// repeated lookups; edge buckets of non-2 MiB-aligned `sycl::malloc_host` slices fall back to checking the
/// exact blob range `[blob, blob + bytes - 1]` so valid USM slices are never misclassified.
inline bool sycl_dma_direct_check(const core::ExpertSource* src, int64_t layer, int64_t expert,
                                  const void* blob, size_t bytes, const sycl::context& ctx,
                                  bool pageable_ok) {
    if (!src || !src->pinned(layer, expert) || blob == nullptr) return false;
    if (pageable_ok) return true;

    static thread_local std::optional<sycl::context> cached_ctx;
    static thread_local std::unordered_map<uintptr_t, bool> usm_bucket;
    if (!cached_ctx.has_value() || *cached_ctx != ctx) {
        usm_bucket.clear();
        cached_ctx.emplace(ctx);
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(blob);
    const uintptr_t end_addr = bytes > 1 ? addr + bytes - 1 : addr;
    if (end_addr < addr) return false;

    const uintptr_t k0 = addr >> 21;
    const uintptr_t k1 = end_addr >> 21;

    auto bucket_is_host = [&](uintptr_t k) -> bool {
        const auto it = usm_bucket.find(k);
        if (it != usm_bucket.end()) return it->second;
        const bool full =
            sycl::get_pointer_type(reinterpret_cast<const void*>(k << 21), ctx) == sycl::usm::alloc::host &&
            sycl::get_pointer_type(reinterpret_cast<const void*>(((k + 1) << 21) - 1), ctx) == sycl::usm::alloc::host;
        // Cache only full-bucket USM hits; unaligned allocation edges fall through to the exact span check.
        if (full) usm_bucket.emplace(k, true);
        return full;
    };

    bool all_buckets_host = bucket_is_host(k0);
    for (uintptr_t k = k0 + 1; all_buckets_host && k <= k1; ++k) {
        all_buckets_host = bucket_is_host(k);
    }
    if (all_buckets_host) return true;

    if (sycl::get_pointer_type(blob, ctx) != sycl::usm::alloc::host) return false;
    if (end_addr != addr && sycl::get_pointer_type(reinterpret_cast<const void*>(end_addr), ctx) != sycl::usm::alloc::host) {
        return false;
    }
    for (uintptr_t k = k0 + 1; k < k1; ++k) {
        if (!bucket_is_host(k)) return false;
    }
    return true;
}

}  // namespace strata::prefill
