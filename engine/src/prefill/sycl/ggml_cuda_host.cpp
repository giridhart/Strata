// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its
// MMQ and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include <sycl/sycl.hpp>
#include "strata/core/emulate.hpp"
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace {
inline sycl::queue& get_mmq_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{},
                                                                       sycl::property::queue::enable_profiling{}}};
    return q;
}
}  // namespace

[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() { return 0; }

#if defined(GGML_USE_HIP)
namespace {

unsigned parse_hex(const std::string& value) {
    if (value.empty()) return 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 16);
    return end == value.c_str() || *end != '\0' ? 0 : static_cast<unsigned>(parsed);
}

// Match GGML's AMD cc encoding: OFFSET_AMD + gfx major * 0x100 + minor/stepping.
// For example, gfx1100 maps to GGML_CUDA_CC_RDNA3 (0x01001100).
int ggml_cuda_parse_amd_id(const char* device_name) {
    if (device_name == nullptr) return GGML_CUDA_CC_OFFSET_AMD;

    std::string arch(device_name);
    if (arch.compare(0, 3, "gfx") == 0) arch.erase(0, 3);
    const size_t suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    const size_t generic = arch.rfind("-generic");
    if (generic != std::string::npos && generic + 8 == arch.size()) arch.resize(generic);

    unsigned major = 0;
    unsigned minor = 0;
    const size_t separator = arch.find('-');
    if (separator != std::string::npos) {
        major = parse_hex(arch.substr(0, separator));
        minor = parse_hex(arch.substr(separator + 1)) * 0x10;
    } else if (arch.size() >= 3) {
        minor = parse_hex(arch.substr(arch.size() - 2));
        major = parse_hex(arch.substr(0, arch.size() - 2));
    }
    return GGML_CUDA_CC_OFFSET_AMD + static_cast<int>(major * 0x100 + minor);
}

}  // namespace
#endif

const ggml_cuda_device_info& ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        std::vector<sycl::device> gpus;
        try {
            gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        } catch (...) {
        }
        if (gpus.empty()) {
            try {
                gpus.push_back(sycl::device(sycl::default_selector_v));
            } catch (...) {
            }
        }
        int n = static_cast<int>(gpus.size());
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            const auto& dev = gpus[id];
            auto& d = in.devices[id];
#if defined(GGML_USE_HIP)
            std::string dname = dev.get_info<sycl::info::device::name>();
            d.cc = ggml_cuda_parse_amd_id(dname.c_str());
            d.smpbo = dev.get_info<sycl::info::device::local_mem_size>();
            d.integrated = false;
            d.supports_cooperative_launch = false;
#else
            d.cc = 100 * strata::cc_major_of(20) + 10 * strata::cc_minor_of(4);
            d.smpbo = strata::smem_optin_of((int)dev.get_info<sycl::info::device::local_mem_size>());
            d.integrated = false;
            d.supports_cooperative_launch = false;
#endif
            d.nsm(dev.get_info<sycl::info::device::max_compute_units>());
            d.smpb(dev.get_info<sycl::info::device::local_mem_size>());
            d.vmm = false;
            d.total_vram(dev.get_info<sycl::info::device::global_mem_size>());
            auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
            d.warp_size(!sgs.empty() ? sgs.back() : 32);
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf {
        void* p;
        size_t size;
        bool used;
    };
    std::vector<Buf> bufs;
    std::mutex mu;
    void* alloc(size_t size, size_t* actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& b : bufs)
            if (!b.used && b.size >= size) {
                b.used = true;
                *actual_size = b.size;
                return b.p;
            }
        void* p = sycl::malloc_device(size, get_mmq_queue());
        if (!p) {
            ggml_cuda_error("malloc_device", "alloc", __FILE__, __LINE__, "out of memory");
        }
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void* ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& b : bufs)
            if (b.p == ptr) {
                b.used = false;
                return;
            }
    }
    ~CachingPool() override {
        for (auto& b : bufs) {
            if (b.p) {
                try {
                    sycl::free(b.p, get_mmq_queue());
                } catch (...) {
                }
            }
        }
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
