// src/core/device.cu - P2.S1: the SYCL side of the runtime core.
#include <sycl/sycl.hpp>
#include "strata/core/device.hpp"

#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <mutex>

namespace strata::core {

namespace {

inline std::vector<sycl::device> get_devices_list() {
    std::vector<sycl::device> res;
    try {
        sycl::device def = sycl::device(sycl::default_selector_v);
        res.push_back(def);
        auto all = sycl::device::get_devices(sycl::info::device_type::all);
        for (auto& d : all) {
            if (d != def) {
                res.push_back(d);
            }
        }
    } catch (...) {
    }
    return res;
}

inline sycl::queue& get_queue(int ordinal = 0) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    static std::map<int, std::unique_ptr<sycl::queue>> queues;
    auto it = queues.find(ordinal);
    if (it != queues.end()) {
        return *it->second;
    }
    auto devs = get_devices_list();
    if (ordinal >= 0 && ordinal < static_cast<int>(devs.size())) {
        auto q = std::make_unique<sycl::queue>(devs[ordinal], sycl::property::queue::in_order());
        auto& ref = *q;
        queues[ordinal] = std::move(q);
        return ref;
    }
    static sycl::queue fallback{sycl::default_selector_v, sycl::property::queue::in_order()};
    return fallback;
}

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

int device_count() {
    return static_cast<int>(get_devices_list().size());
}

bool device_summary(int ordinal, std::string &name, std::string &detail) {
    auto devs = get_devices_list();
    if (ordinal < 0 || ordinal >= static_cast<int>(devs.size())) {
        return false;
    }
    try {
        auto dev = devs[ordinal];
        name = dev.get_info<sycl::info::device::name>();
        char buf[160];
        auto total_mem = dev.get_info<sycl::info::device::global_mem_size>();
        std::snprintf(buf, sizeof(buf), "compute capability 20.4, %.1f GiB",
                      (double)total_mem / (1024.0 * 1024 * 1024));
        detail = buf;
        return true;
    } catch (...) {
        return false;
    }
}

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

std::string device_code_error() {
    return "";
}

DeviceInfo device_info(int ordinal) {
    auto devs = get_devices_list();
    if (devs.empty()) {
        throw CudaError("no SYCL accelerator device is present", -1);
    }
    if (ordinal < 0 || ordinal >= static_cast<int>(devs.size())) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(devs.size()) + ")",
                        -1);
    }
    auto dev = devs[ordinal];
    DeviceInfo d;
    d.ordinal = ordinal;
    d.name = dev.get_info<sycl::info::device::name>();
    d.cc_major = 20;
    d.cc_minor = 4;
    d.multi_processor_count = static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());
    d.total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
    d.free_bytes = 0;
#if defined(SYCL_EXT_INTEL_FREE_MEMORY)
    try {
        if (dev.has(sycl::aspect::ext_intel_free_memory)) {
            d.free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
        }
    } catch (...) {}
#endif
    if (d.free_bytes == 0) {
        // Fallback for iGPU / unified memory: allow budget up to total VRAM
        d.free_bytes = d.total_bytes;
    }
    d.driver_version = 20;
    d.runtime_version = 20;

#if defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#endif
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    auto& q = get_queue(ordinal);
    base_ = sycl::malloc_device((size_t)bytes, q);
    if (!base_) {
        throw CudaError("sycl::malloc_device failed for DeviceArena of " + std::to_string(bytes) + " bytes", -1);
    }
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            float* float_base = (float*)base_ + b * threads;
            const uint64_t n_chunk = n - b * threads;
            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)chunk * threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item) {
                        const uint64_t i = (uint64_t)item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
                        if (i < n_chunk) float_base[i] = sycl::bit_cast<float>(0x7fc00000);
                    });
            });
        }
        q.wait_and_throw();
    }
}

DeviceArena::~DeviceArena() {
    if (base_) {
        try {
            auto& q = get_queue(ordinal_);
            sycl::free(base_, q);
        } catch (...) {
        }
    }
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
