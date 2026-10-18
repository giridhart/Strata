#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <string>

#if defined(STRATA_HAVE_ZE) || (defined(__has_include) && __has_include(<level_zero/ze_api.h>) && __has_include(<sycl/ext/oneapi/backend/level_zero.hpp>))
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#ifndef STRATA_HAVE_ZE
#define STRATA_HAVE_ZE 1
#endif
#endif

namespace strata {

using doorbell_uncached_read = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1, sycl::ext::oneapi::experimental::cache_level::L3>>));

inline uint32_t sys_load(const volatile uint32_t* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(const_cast<uint32_t*>(p));
    return u[0];
}

using doorbell_uncached_write = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::write_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1, sycl::ext::oneapi::experimental::cache_level::L3>>));

inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_write> u(const_cast<uint32_t*>(p));
    u[0] = v;
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

template<class T> inline void sys_store_mapped(T* p, T v) {
    sycl::ext::oneapi::experimental::annotated_ptr<T, doorbell_uncached_write> u(p);
    u[0] = v;
}

inline uint32_t spin_max(const sycl::queue& q) {
    (void)q;
    if (const char* e = std::getenv("STRATA_SPIN_MAX"); e && std::atol(e) > 0) {
        return (uint32_t)std::atol(e);
    }
    // Default 2,000,000 reads gives CPU expert pool ample time (several seconds)
    // while guaranteeing the GPU never spins forever on a stalled or released engine.
    return 2000000u;
}

namespace detail {
inline std::unordered_set<void*>& uncached_set() { static std::unordered_set<void*> s; return s; }
inline std::mutex& uncached_mu() { static std::mutex m; return m; }
}  // namespace detail

inline void* host_malloc_polled(size_t bytes, sycl::queue& q) {
#ifdef STRATA_HAVE_ZE
    static const bool on = [] { const char* v = std::getenv("STRATA_HOST_UNCACHED"); return !(v && v[0] == '0'); }();
    if (on && q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ze_host_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* p = nullptr;
        if (zeMemAllocHost(ctx, &d, bytes, 64, &p) == ZE_RESULT_SUCCESS && p != nullptr) {
            std::lock_guard<std::mutex> lk(detail::uncached_mu());
            detail::uncached_set().insert(p);
            return p;
        }
    }
#endif
    return sycl::malloc_host(bytes, q);
}

inline void host_free_polled(void* p, sycl::queue& q) {
    if (p == nullptr) return;
#ifdef STRATA_HAVE_ZE
    {
        std::lock_guard<std::mutex> lk(detail::uncached_mu());
        if (detail::uncached_set().erase(p)) {
            zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
            return;
        }
    }
#endif
    sycl::free(p, q);
}

enum class IntelGpuGen { Unknown, Alchemist, Battlemage };
inline IntelGpuGen intel_gpu_gen(const sycl::device& d) {
    if (!d.is_gpu()) return IntelGpuGen::Unknown;
    try {
        namespace exp = sycl::ext::oneapi::experimental;
        const auto a = d.get_info<exp::info::device::architecture>();
        if (a == exp::architecture::intel_gpu_bmg_g21 || a == exp::architecture::intel_gpu_bmg_g31)
            return IntelGpuGen::Battlemage;
        if (a == exp::architecture::intel_gpu_acm_g10 || a == exp::architecture::intel_gpu_acm_g11 ||
            a == exp::architecture::intel_gpu_acm_g12)
            return IntelGpuGen::Alchemist;
    } catch (...) {
    }
    std::string n;
    try { n = d.get_info<sycl::info::device::name>(); } catch (...) { return IntelGpuGen::Unknown; }
    if (n.find("Intel") == std::string::npos) return IntelGpuGen::Unknown;
    if (const size_t b = n.find("[0x"); b != std::string::npos) {
        const unsigned long id = std::strtoul(n.c_str() + b + 1, nullptr, 16);
        if ((id >= 0xE200 && id <= 0xE2FF) || (id >= 0x6400 && id <= 0x64FF)) return IntelGpuGen::Battlemage;
        if ((id >= 0x5690 && id <= 0x56BF) || id == 0x56C0 || id == 0x56C1) return IntelGpuGen::Alchemist;
    }
    if (n.find("Battlemage") != std::string::npos) return IntelGpuGen::Battlemage;
    if (n.find("Alchemist") != std::string::npos) return IntelGpuGen::Alchemist;
    for (size_t i = 0; i + 2 < n.size(); ++i) {
        if ((i > 0 && n[i - 1] != ' ') || (n[i] != 'A' && n[i] != 'B')) continue;
        size_t j = i + 1;
        while (j < n.size() && n[j] >= '0' && n[j] <= '9') ++j;
        if (j - i - 1 < 2 || j - i - 1 > 3 || (j < n.size() && n[j] != ' ' && n[j] != '(')) continue;
        return n[i] == 'B' ? IntelGpuGen::Battlemage : IntelGpuGen::Alchemist;
    }
    return IntelGpuGen::Unknown;
}

}  // namespace strata
