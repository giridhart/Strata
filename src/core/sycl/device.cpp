// src/core/device.cu - P2.S1: the SYCL side of the runtime core.
#include <sycl/sycl.hpp>
#include "strata/core/device.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <sys/ioctl.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<drm/drm.h>) && __has_include(<drm/xe_drm.h>)
#include <drm/drm.h>
#include <drm/xe_drm.h>
#define STRATA_XE_QUERY 1
#endif
#endif
#endif

namespace strata::core {

namespace {

// IGC 2.27 to 2.30 (Ubuntu 26.04's and Debian's 2.28.4 among them) miscompile, for Xe2 and Xe3 GPUs, reads from one
// pointer at a runtime index N and at constant multiples of it: the byte offsets of x[2N] and x[3N] are taken for
// element indices, and the kernel reads x[8N] and x[9N] (a fault past the buffer, or wrong values inside it).  Their
// GEP simplification does it, and IGC_EnableGEPSimplification=0 turns that pass off: on the B70 the decode of a 1K
// prompt reads 5 to 10% slower with it off and 32K the same, so it is off only where both are true: the installed IGC
// is one of those (2.24 and 2.32 on compile the reads right) and a GPU the xe driver runs is Xe2 or later (graphics IP
// 20 and on; Alchemist, IP 12.55, compiles the reads right; AGENTS.md, "Intel Graphics Compiler").  IGC reads its
// IGC_ variables when Level Zero loads it, as the SYCL runtime lists the devices, so the GPUs are asked of the kernel
// driver here, before that; a value the user set stays.  Linux only: Windows keeps its driver's compiler as it is.
struct CompilerWorkaround {
    bool set = false;      // the engine set IGC_EnableGEPSimplification=0
    std::string version;   // IGC's, when known
};
std::string g_workaround_version;   // set by get_devices_list when the workaround applies to a listed GPU

#if !defined(_WIN32)
// Whether a GPU of the xe kernel driver has a graphics IP of 20 or later (DRM_XE_DEVICE_QUERY_GT_LIST, the main GT's
// ip_ver_major; 0 on a GPU without GMD_ID, Alchemist and older, which is not one). False when no xe GPU can be opened.
bool xe_gpu_from_xe2() {
#if defined(STRATA_XE_QUERY)
    bool any_intel_opened = false;
    for (int minor = 128; minor < 192; ++minor) {
        const std::string node = "/dev/dri/renderD" + std::to_string(minor);
        const int fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        char name[16] = {};
        drm_version ver{};
        ver.name_len = sizeof(name) - 1;
        ver.name = name;
        bool found = false;
        if (ioctl(fd, DRM_IOCTL_VERSION, &ver) == 0) {
            if (std::strcmp(name, "xe") == 0) {
                any_intel_opened = true;
                drm_xe_device_query q{};
                q.query = DRM_XE_DEVICE_QUERY_GT_LIST;
                if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) == 0 && q.size >= sizeof(drm_xe_query_gt_list)) {
                    std::vector<uint64_t> buf((q.size + 7) / 8);
                    q.data = (uint64_t) (uintptr_t) buf.data();
                    if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) == 0) {
                        const auto* list = reinterpret_cast<const drm_xe_query_gt_list*>(buf.data());
                        const uint8_t* gt_ptr = reinterpret_cast<const uint8_t*>(list->gt_list);
                        for (uint32_t i = 0; i < list->num_gt; ++i) {
                            const uint8_t* cur = gt_ptr + i * sizeof(drm_xe_gt);
                            uint16_t gt_type = *reinterpret_cast<const uint16_t*>(cur);
                            uint16_t ip_ver_major = *reinterpret_cast<const uint16_t*>(cur + 32);
                            if (gt_type == DRM_XE_QUERY_GT_TYPE_MAIN && ip_ver_major >= 20)
                                found = true;
                        }
                    }
                }
            } else if (std::strcmp(name, "i915") == 0) {
                any_intel_opened = true;
            }
        }
        close(fd);
        if (found) return true;
    }
    if (any_intel_opened) return false;
#endif
#if defined(STRATA_TARGET_XE2) || defined(STRATA_TARGET_BMG)
    return true;
#else
    return false;
#endif
}
#endif

CompilerWorkaround intel_compiler_workaround() {
    static const CompilerWorkaround done = [] {
        CompilerWorkaround w;
#if !defined(_WIN32)
        if (std::getenv("IGC_EnableGEPSimplification") != nullptr || !xe_gpu_from_xe2()) return w;
        void* lib = dlopen("libigc.so.2", RTLD_LAZY | RTLD_LOCAL);   // the driver loads the same one later
        if (lib == nullptr) return w;                                 // no Intel compiler: nothing to set
        bool affected = true;                                         // (a file name with no version: as affected)
        link_map* map = nullptr;
        char path[PATH_MAX];
        if (dlinfo(lib, RTLD_DI_LINKMAP, static_cast<void*>(&map)) == 0 && map != nullptr && map->l_name != nullptr &&
            realpath(map->l_name, path) != nullptr) {
            // libigc.so.2.28.4+0: the version after "libigc.so.", up to the build number
            const char* name = std::strrchr(path, '/');
            name = name ? name + 1 : path;
            if (std::strncmp(name, "libigc.so.", 10) == 0) {
                const char* v = name + 10;
                size_t n = 0;
                while (std::isdigit((unsigned char) v[n]) || v[n] == '.') ++n;
                w.version.assign(v, n);
                char* rest = nullptr;
                const long major = std::strtol(w.version.c_str(), &rest, 10);
                const long minor = rest != nullptr && *rest == '.' ? std::strtol(rest + 1, nullptr, 10) : -1;
                if (minor >= 0) affected = major == 2 && minor >= 25 && minor < 32;
            }
        }
        if (affected) setenv("IGC_EnableGEPSimplification", "0", 0);
        w.set = affected;
#endif
        return w;
    }();
    return done;
}

// The SYCL runtime's on-disk cache of the device code it builds (SYCL_CACHE_PERSISTENT), on unless the user set it:
// without it every start builds each kernel module it uses from the SPIR-V again, linking the device libraries in, and
// the GPU driver's own cache does not spare that.
static void sycl_code_cache() {
    if (std::getenv("SYCL_CACHE_PERSISTENT") != nullptr) return;
#if defined(_WIN32)
    _putenv_s("SYCL_CACHE_PERSISTENT", "1");
#else
    setenv("SYCL_CACHE_PERSISTENT", "1", 0);
#endif
}

inline const std::vector<sycl::device>& get_devices_list() {
    static const std::vector<sycl::device> devices = [] {
        sycl_code_cache();
        const CompilerWorkaround w = intel_compiler_workaround();
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
        if (w.set && std::any_of(res.begin(), res.end(), [](const sycl::device& d) {
                return d.get_backend() == sycl::backend::ext_oneapi_level_zero;
            })) {
            g_workaround_version = w.version.empty() ? "unknown" : w.version;
            std::fprintf(stderr, "Xe runtime: IGC's GEP simplification is off (IGC_EnableGEPSimplification=0; IGC %s): "
                                 "2.27 to 2.30 miscompile some reads for Xe2 and later GPUs. The decode of short "
                                 "prompts is up to 10%% slower. For full speed, use Fedora 44 (its IGC is 2.32 or "
                                 "later), or install with Podman (Fedora 44 inside)\n", g_workaround_version.c_str());
        }
        return res;
    }();
    return devices;
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

int device_count() { return static_cast<int>(get_devices_list().size()); }

bool device_summary(int ordinal, std::string& name, std::string& detail) {
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
    (void)ordinal;
    return "";
#endif
}

std::string device_code_error() { return ""; }

std::string compiler_workaround() {
    (void) get_devices_list();
    return g_workaround_version;
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
    } catch (...) {
    }
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
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)chunk * threads), sycl::range<3>(1, 1, threads)),
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
                      (unsigned long long)bytes, (unsigned long long)start, (unsigned long long)align,
                      (unsigned long long)capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*)base_ + start;
}

}  // namespace strata::core
