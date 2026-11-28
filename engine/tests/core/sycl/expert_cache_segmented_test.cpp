// tests/core/sycl/expert_cache_segmented_test.cpp
// #533: the segmented expert cache (--vram-elastic).
// Tests segmented cache support in SYCL. If segmented cache (--vram-elastic / virtual memory management)
// is unsupported on the platform, exits cleanly with 77 (SKIP).
#include <sycl/sycl.hpp>
#include "strata/core/expert_cache.hpp"
#include "strata/core/device.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}

size_t free_vram() {
    try {
        strata::core::DeviceInfo d = strata::core::device_info(0);
        return (size_t)d.free_bytes;
    } catch (...) {
        return 0;
    }
}

// slot i holds the byte (i * 7 + 1) everywhere
bool fill(sycl::queue& q, strata::core::ExpertCache& c, int64_t from, int64_t to, int64_t blob) {
    std::vector<uint8_t> b((size_t)blob);
    for (int64_t i = from; i < to; ++i) {
        std::fill(b.begin(), b.end(), (uint8_t)(i * 7 + 1));
        uint8_t* dst = c.device_slot((int32_t)i);
        if (!dst) return false;
        q.memcpy(dst, b.data(), (size_t)blob).wait();
    }
    return true;
}

bool check(sycl::queue& q, const strata::core::ExpertCache& c, int64_t to, int64_t blob) {
    std::vector<uint8_t> b((size_t)blob);
    for (int64_t i = 0; i < to; ++i) {
        const uint8_t* src = c.device_slot((int32_t)i);
        if (!src) return false;
        q.memcpy(b.data(), src, (size_t)blob).wait();
        for (const uint8_t x : b)
            if (x != (uint8_t)(i * 7 + 1)) return false;
    }
    return true;
}
}  // namespace

int main() try {
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::puts("no GPU device: skipped");
        return 77;
    }
    sycl::queue q{gpus[0], sycl::property::queue::in_order()};

    constexpr int64_t kBlob = 1 << 20, kSlots = 96, kSeg = 8ll << 20;
    for (int sized = 0; sized < 2; ++sized) {
        strata::core::ExpertCache c;
        c.set_segment_bytes(kSeg);
        std::string err;
        const bool ok = sized ? c.open_sized(std::vector<int64_t>(kSlots, kBlob), 4, 64, err)
                              : c.open(kSlots, 4, 64, kBlob, err);
        if (!ok) {
            if (err.find("virtual memory management") != std::string::npos ||
                err.find("CUDA-only") != std::string::npos ||
                err.find("not supported") != std::string::npos) {
                std::printf("%s: skipped\n", err.c_str());
                return 77;
            }
            return fail("open", err);
        }
        if (!c.segmented() || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob) return fail("open sizes");
        uint8_t* const first = c.device_slot(0);
        if (!fill(q, c, 0, kSlots, kBlob)) return fail("fill");
        const size_t before = free_vram();
        // keep 30 MiB: rounded up to whole 8 MiB segments = 32 MiB = 32 slots
        if (!c.shrink(30ll << 20, err)) return fail("shrink", err);
        if (c.slots() != 32 || c.mapped_bytes() != 32ll << 20 || c.bytes() != 32ll << 20)
            return fail("shrunk sizes");
        if (c.full_slots() != kSlots || c.full_bytes() != kSlots * kBlob) return fail("full sizes");
        const size_t freed = free_vram() > before ? free_vram() - before : 0;
        if (freed > 0 && freed + (8u << 20) < (size_t)(64ll << 20)) {  // ~64 MiB back to the driver
            std::fprintf(stderr, "freed %zu bytes\n", freed);
            return fail("VRAM not given back");
        }
        if (c.device_slot(0) != first || !check(q, c, 32, kBlob)) return fail("slots kept after the shrink");
        // grow to 70 MiB: whole segments only, 64 MiB = 64 slots
        if (!c.grow(70ll << 20, err) || c.slots() != 64) return fail("grow part", err);
        if (!c.grow(1ll << 40, err) || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob)
            return fail("grow all", err);
        if (c.device_slot(0) != first || !check(q, c, 32, kBlob)) return fail("slots kept after the grow");
        if (!fill(q, c, 32, kSlots, kBlob) || !check(q, c, kSlots, kBlob)) return fail("the grown slots");
        if (c.slots_within(0) != 0 || c.slots_within(kBlob * 3 + 5) != 3 || c.bytes_of(5) != 5 * kBlob)
            return fail("slots_within / bytes_of");
        c.close();
        if (c.valid() || c.slots() != 0) return fail("close");
    }
    // the default (no segment size): one allocation, shrink refused
    strata::core::ExpertCache d;
    std::string err;
    if (!d.open(8, 2, 4, kBlob, err) || d.segmented() || d.shrink(0, err)) return fail("unsegmented", err);
    std::puts("expert cache segments: shrink gives VRAM back and keeps the slots, grow maps them again");
    return 0;
} catch (const std::exception& exc) {
    std::fprintf(stderr, "exception: %s\n", exc.what());
    return 1;
}
