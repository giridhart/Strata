// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#include <sycl/sycl.hpp>
#include <sycl/usm.hpp>
#include "strata/core/device.hpp"
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

namespace strata::core {
namespace {
inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}
}  // namespace
}  // namespace strata::core

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

// Loader fix: `fseek`/`ftell` are 32-bit on Windows by default (and the pack is 42.9 GB), and the 64-bit
// spelling is not the same on the two platforms the engine builds for.
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long)(o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t)(o), SEEK_SET)
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#endif

namespace strata::core {

namespace {

constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "STRATA-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);

#if !defined(_WIN32)
// bytes in free blocks of 2 MiB or more (/proc/buddyinfo, order >= 9): what MADV_HUGEPAGE can get without compaction
uint64_t free_huge_bytes() {
    uint64_t total = 0;
    if (std::FILE* f = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            const char* q = z + 4;
            while (*q == ' ') ++q;
            while (*q != '\0' && *q != ' ') ++q;   // the zone's name
            unsigned long long c[11] = {0};
            int n = 0;
            for (; n < 11; ++n) {
                char* e = nullptr;
                const unsigned long long v = std::strtoull(q, &e, 10);
                if (e == q) break;
                c[n] = v;
                q = e;
            }
            for (int o = 9; o < n; ++o) total += c[o] * (4096ull << o);
        }
        std::fclose(f);
    }
    return total;
}

std::string thp_request_skipped(uint64_t bytes) {
    if (std::getenv("STRATA_NO_ARENA_THP") != nullptr) return "transparent huge pages skipped (STRATA_NO_ARENA_THP)";
    std::string mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    if (mode == "always")
        return "transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    if (std::getenv("STRATA_ARENA_THP") == nullptr) {
        const uint64_t free_huge = free_huge_bytes();
        if (free_huge < bytes / 2) {
            std::string why = "transparent huge pages NOT requested: only " + std::to_string(free_huge >> 30) +
                              " GiB of free 2 MiB blocks for a " + std::to_string(bytes >> 30) +
                              " GiB arena, compaction would stall the load (STRATA_ARENA_THP=1 requests them anyway)";
            std::fprintf(stderr, "strata: expert arena: %s\n", why.c_str());
            return why;
        }
    }
    return std::string();
}
#endif

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
// A non-empty shared_file instead maps one file whose first 4 KiB identify the pack and whose remaining bytes
// are the resident arena.  This shared-file layout is intended for tmpfs (/dev/shm); hugetlbfs would need
// hugepage-aligned file size and arena offset rather than the 4 KiB header layout used here.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note, const std::string& shared_file,
              uint64_t shared_pack_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
#ifdef _WIN32
    if (!shared_file.empty()) {
        note = "shared-file arena backing is not implemented on Windows";
        return nullptr;
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege.  Having it assigned to the account is not enough: the
    // PROCESS must enable it in its own token (AdjustTokenPrivileges) before VirtualAlloc, or the call fails.
    // An account without the assignment, or a failure to enable, leaves the process as it was: VirtualAlloc
    // then refuses and the 4 KB fallback below runs - that is the EXPECTED outcome on a desktop.
    {
        HANDLE tok = nullptr;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp{};
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                if (!AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) &&
                    GetLastError() != ERROR_NOT_ALL_ASSIGNED)
                    (void)0;  // nothing actionable: the large-page attempt below reports the outcome
            }
            CloseHandle(tok);
        }
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege; a normal account does not have it and VirtualAlloc then
    // fails with ERROR_PRIVILEGE_NOT_HELD.  That is the EXPECTED outcome on a desktop, not an error.
    SIZE_T large = GetLargePageMinimum();
    // A/B switch: STRATA_NO_LARGEPAGES=1 skips the large-page attempt, same run, same boot.
    if (large > 0 && std::getenv("STRATA_NO_LARGEPAGES") == nullptr) {
        // MEM_LARGE_PAGES requires the allocation size to be an exact multiple of the large page size -
        // anything else is ERROR_INVALID_PARAMETER (87), which reads like a privilege problem but is not.
        // Round up: the slack is under 2 MB and the tail stays unused.
        const SIZE_T lbytes = (SIZE_T)(((SIZE_T)bytes + large - 1) / large * large);
        void* p = VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (p) {
            got = PageBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long)large) + " B)";
            return p;
        }
        // 1450 (ERROR_NO_SYSTEM_RESOURCES) is the large-page pool saying no, 87 is a size that is not a
        // multiple of the minimum, 1314 is the privilege: without the byte count the three read as one bug.
        note = "large pages refused for " + std::to_string((unsigned long long)lbytes) +
               " B (GetLargePageMinimum=" + std::to_string((unsigned long long)large) + ", VirtualAlloc error " +
               std::to_string((unsigned long long)GetLastError()) + "); using 4 KB pages";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T)bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = PageBacking::NormalPages;
    return p;
#else
    if (!shared_file.empty()) {
        if (shared_pack_hash == 0) {
            note = "shared arena requires a nonzero pack hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t)file_bytes) != 0) {
                const int e = errno;
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t)kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.pack_hash = shared_pack_hash;
            const ssize_t written = pwrite(fd, &hdr, sizeof hdr, 0);
            if (written != (ssize_t)sizeof hdr) {
                const int e = errno;
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (written < 0 ? std::string(std::strerror(e)) : std::string("short write"));
                return nullptr;
            }
        } else {
            if (st.st_size < 0 || (uint64_t)st.st_size != file_bytes) {
                close(fd);
                note = "shared arena " + shared_file + " is " + std::to_string((unsigned long long)st.st_size) +
                       " B, expected " + std::to_string((unsigned long long)file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            const ssize_t got_header = pread(fd, &hdr, sizeof hdr, 0);
            if (got_header != (ssize_t)sizeof hdr || std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.pack_hash != shared_pack_hash) {
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b, "shared arena %s was written for pack hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long)hdr.pack_hash,
                              (unsigned long long)shared_pack_hash);
                note = b;
                return nullptr;
            }
        }

        void* map = mmap(nullptr, (size_t)file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = PageBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (pack hash checked)";
        return (uint8_t*)map + kSharedArenaHeaderBytes;
    }

    constexpr uint64_t kHugePage = 2ull << 20;
    // Over-map by one huge page so the arena starts on a 2 MB boundary and THP can back it from the first byte.
    mapping_bytes = (size_t) bytes + kHugePage;
    mapping_base = mmap(nullptr, (size_t) mapping_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mapping_base == MAP_FAILED) {
        mapping_base = nullptr;
        mapping_bytes = 0;
        note = "mmap failed for " + std::to_string(bytes) + " bytes: " + std::string(std::strerror(errno));
        return nullptr;
    }
    void* base = (void*) (((uintptr_t) mapping_base + kHugePage - 1) & ~(uintptr_t) (kHugePage - 1));
    const std::string thp_skip = thp_request_skipped(bytes);
    const bool huge = thp_skip.empty() && madvise(base, (size_t) bytes, MADV_HUGEPAGE) == 0;
    got = huge ? PageBacking::LargePages : PageBacking::NormalPages;
    note = !thp_skip.empty() ? thp_skip
                             : (huge ? "transparent huge pages requested (madvise)"
                                     : "madvise(MADV_HUGEPAGE) refused: 4 KB pages");
    return base;
#endif
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
#ifdef _WIN32
    (void)bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int arena_pin_cap_gib() {
    const char* e = std::getenv("STRATA_ARENA_PIN_GIB");
    if (e == nullptr || *e == '\0') return -1;
    if (std::string(e) == "auto") return -2;  // #243: the Windows shared-memory budget sets the sliced pin's cap
    const int v = std::atoi(e);
    return v < 0 ? -1 : v;
}

namespace {
#ifdef _WIN32
// #243: how much of the arena the sliced registration may pin on Windows when the whole arena was refused.
// Page-locked memory the GPU maps is charged to its shared (non-local) WDDM segment; pinned slice by slice until
// the driver refused one (28 GiB of a 63 GB PC), that segment was left full and every later cudaMalloc failed
// "out of memory".  4 GiB of the budget stay free for what the engine allocates after the arena; without the DXGI
// numbers, RAM/2 - 8 GiB (the budget is about half the RAM).  False: no limit could be worked out.
bool sliced_pin_limit(uint64_t& limit, std::string& why) {
    constexpr uint64_t GiB = 1ull << 30;
    char buf[256];
    int dev = 0;
    cudaDeviceProp p{};
    uint64_t budget = 0, usage = 0;
    std::string err = "no CUDA device properties";
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
        strata::platform::gpu_shared_memory_budget(p.luid, budget, usage, err)) {
        limit = budget > usage + 4 * GiB ? budget - usage - 4 * GiB : 0;
        std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget %.1f GiB - %.1f GiB in use - 4 GiB",
                      (double)budget / GiB, (double)usage / GiB);
        why = buf;
        return true;
    }
    (void)cudaGetLastError();
    const uint64_t ram = strata::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "%s: RAM/2 - 8 GiB of %.1f GiB", err.c_str(), (double)ram / GiB);
    why = buf;
    return true;
}
#endif
}  // namespace

namespace {
std::vector<int> g_copy_devices;
}

void arena_copy_devices(const std::vector<int>& devices) { g_copy_devices = devices; }

uint64_t max_alloc_bytes(int ordinal) {
    if (const char* v = std::getenv("STRATA_MAX_ALLOC_MIB")) {
        uint64_t mib = std::strtoull(v, nullptr, 10);
        if (mib > 0) return mib << 20;
    }
    try {
        auto devs = get_devices_list();
        if (ordinal >= 0 && ordinal < static_cast<int>(devs.size())) {
            return devs[ordinal].get_info<sycl::info::device::max_mem_alloc_size>();
        }
        sycl::device d{sycl::default_selector_v};
        return d.get_info<sycl::info::device::max_mem_alloc_size>();
    } catch (...) {
        return 1ull << 30; // 1 GiB fallback
    }
}

namespace {
bool pin_parts_enabled() {
    const char* v = std::getenv("STRATA_PIN_PARTS");
    if (v != nullptr && std::strcmp(v, "0") == 0) return false;
    return true; // enabled by default (STRATA_PIN_PARTS=1)
}

uint64_t register_pieces(uint8_t* base, uint64_t bytes, const std::vector<uint64_t>& cuts,
                         std::vector<void*>& reg, std::vector<std::pair<void*, size_t>>& pieces, uint64_t cap) {
    uint64_t limit = bytes;
    if (pin_parts_enabled()) {
        limit = max_alloc_bytes();
        for (int d : g_copy_devices) {
            limit = std::min(limit, max_alloc_bytes(d));
        }
    }
    std::vector<uint64_t> ends, c;
    for (uint64_t x : cuts)
        if (x > 0 && x < bytes) c.push_back(x);
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    c.push_back(bytes);

    for (uint64_t start = 0; start < bytes;) {
        size_t i = 0;
        while (i < c.size() && c[i] <= start) ++i;
        uint64_t prev = start;
        while (i < c.size() && c[i] - start <= limit) prev = c[i++];
        const uint64_t end = prev > start ? prev : std::min(bytes, start + limit);
        ends.push_back(end);
        start = end;
    }

    uint64_t start = 0;
    sycl::context ctx = get_device_context(0);
    for (uint64_t end : ends) {
        if (cap > 0 && end > cap) break;
        try {
            sycl::ext::oneapi::experimental::prepare_for_device_copy(base + start, (size_t) (end - start), ctx);
            reg.push_back(base + start);
            pieces.emplace_back(base + start, (size_t) (end - start));
            start = end;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata: prepare_for_device_copy failed for piece [%llu, %llu): %s\n",
                         (unsigned long long)start, (unsigned long long)end, e.what());
            break;
        }
    }
    return start;
}

std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice)
        b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice)
    : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;
}

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice, const std::vector<uint64_t>& cuts)
    : PinnedArena(bytes, cuts.empty() ? uniform_bounds(bytes, slice) : cuts) {
    if (slice_bytes) slice_bytes = slice;
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t pad, bool host_usm,
                         const std::vector<uint64_t>& cuts)
    : capacity(bytes), bounds_(bounds) {
    if (!host_usm) {
        std::vector<uint64_t> cuts_to_use = cuts;
        if (cuts_to_use.empty() && bounds.size() > 1) {
            cuts_to_use.assign(bounds.begin() + 1, bounds.end());
        }
        base = reserve(bytes, backing, note, {}, 0, mapping_base, mapping_bytes);
        if (base) {
            registered_bytes = register_pieces((uint8_t*) base, bytes, cuts_to_use, reg_, pieces_, 0);
            registered_slices = static_cast<int>(pieces_.size());
            slice_starts.clear();
            for (const auto& p : pieces_)
                slice_starts.push_back(static_cast<uint8_t*>(p.first) - static_cast<uint8_t*>(base));
            note = std::string("mapping registered for device copies") +
                   (reg_.size() > 1 ? " in " + std::to_string(reg_.size()) + " pieces (max_mem_alloc_size)" : "") +
                   "; " + note;
        }
        return;
    }
    if (bounds.size() < 2 || bounds.back() > bytes) throw std::runtime_error("expert arena bounds invalid");
    auto ctx = get_device_context(0);
    for (size_t l = 0; l + 1 < bounds.size(); ++l) {
        const uint64_t n = bounds[l + 1] - bounds[l] + pad;
        void* p = sycl::malloc_host((size_t) n, ctx);
        if (!p) {
            for (uint8_t* q : layer_base) sycl::free(q, ctx);
            layer_base.clear();
            throw std::runtime_error("host USM allocation failed");
        }
        layer_base.push_back((uint8_t*) p);
        pieces_.emplace_back(p, (size_t) n);
        slice_starts.push_back(bounds[l]);
    }
    base = layer_base[0];
    backing = PageBacking::NormalPages;
    registered_bytes = bytes;
    registered_slices = static_cast<int>(layer_base.size());
    note = "host USM, one block per layer (" + std::to_string(layer_base.size()) + ")";
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t max_pinned_bytes,
                         const std::string& shared_file, uint64_t shared_pack_hash) try
    : capacity(bytes), bounds_(bounds) {
    if (bytes == 0) return;
    base = reserve(bytes, backing, note, shared_file, shared_pack_hash, mapping_base, mapping_bytes);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }

    if (base) {
        const int env_gib = arena_pin_cap_gib();
        uint64_t cap = max_pinned_bytes;
        if (cap == 0 && env_gib > 0) {
            cap = (uint64_t)env_gib << 30;
        }
        std::vector<uint64_t> cuts_to_use;
        if (bounds.size() > 1) cuts_to_use.assign(bounds.begin() + 1, bounds.end());
        registered_bytes = register_pieces((uint8_t*) base, bytes, cuts_to_use, reg_, pieces_, cap);
        registered_slices = static_cast<int>(pieces_.size());
        slice_starts.clear();
        for (const auto& p : pieces_)
            slice_starts.push_back(static_cast<uint8_t*>(p.first) - static_cast<uint8_t*>(base));

        note = std::string("mapping registered for device copies") +
               (reg_.size() > 1 ? " in " + std::to_string(reg_.size()) + " pieces (max_mem_alloc_size)" : "") +
               "; " + note;
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        } else {
            note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
        }
    }
} catch (const std::exception& exc) {
    std::cerr << exc.what() << " Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
}

PinnedArena::PinnedArena(uint64_t bytes, ByNeed, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                         uint64_t max_pinned_bytes, const std::string& shared_file, uint64_t shared_pack_hash)
    : PinnedArena(bytes, {}, max_pinned_bytes, shared_file, shared_pack_hash) {
    by_need = true;
    pin_ranges = ranges;
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred)
    : PinnedArena(bytes, bounds) {}

void PinnedArena::register_slices(std::atomic<int>& ready) {
    ready.store(INT_MAX);
}

uint8_t* PinnedArena::at(uint64_t off) const {
    if (layer_base.empty()) return (uint8_t*) base + off;
    size_t l = (size_t) (std::upper_bound(bounds_.begin(), bounds_.end() - 1, off) - bounds_.begin());
    l = l == 0 ? 0 : l - 1;
    return layer_base[l] + (off - bounds_[l]);
}

PinnedArena::~PinnedArena() {
    if (base || mapping_base) {
        for (void* r : reg_) {
            try {
                sycl::ext::oneapi::experimental::release_from_device_copy(r, get_device_context(0));
            } catch (...) {}
        }
        reg_.clear();
        pieces_.clear();
        for (uint8_t* q : layer_base) {
            try {
                sycl::free(q, get_device_context(0));
            } catch (...) {}
        }
        layer_base.clear();

        if (locked_bytes)
            strata::platform::unlock_resident((uint8_t*)base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t)layers), n((size_t)layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L)
        off[(size_t)L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    (void)ready;
    LoadStats st;
    const uint64_t layers = (uint64_t)layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes)
        st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t)layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};  // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t)chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: see the header - MSVC's `basic_filebuf::xsgetn` splits any request larger than
        // `_INTERNAL_BUFSIZ - 1` into 4095-byte freads, which turned one 8 MiB chunk into ~2048 4 KiB reads.
        // `fread` sees a request bigger than the stream buffer and passes it to `_read()`/
        // `ReadFile()` unchanged, so the chunk size reaches the disk.  Buffered, not `FILE_FLAG_NO_BUFFERING`:
        // the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() {
                if (f != nullptr) std::fclose(f);
            }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            const uint64_t off = layer_off[(size_t)L];
            uint64_t remaining = layer_bytes[(size_t)L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t)n, f);
                read_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read)
                               .count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t)n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of " +
                          std::to_string(n) + " B at offset " + std::to_string(off + pos) +
                          (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                std::memcpy(dst + off + pos, buf.data(), (size_t)n);
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy)
                               .count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t)L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i)
        pool.emplace_back(worker);
    worker();
    for (auto& t : pool)
        t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double)read_ns_sum.load() / 1e9;
    st.copy_seconds = (double)copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t)iters;
    st.chunk = chunk;
    try {
        auto& q = default_queue();
        uint8_t* dst = (uint8_t*)sycl::malloc_device((size_t)chunk, q);
        if (!dst) {
            std::fprintf(stderr, "stream_bandwidth: malloc_device failed for %llu B\n", (unsigned long long)chunk);
            st.seconds = -1.0;
            return st;
        }

        // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            q.memcpy(dst, src + off, (size_t)chunk);
        }
        q.wait();

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
                q.memcpy(dst, src + off, (size_t)chunk);
            }
        }
        q.wait();
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        sycl::free(dst, q);
        return st;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << " Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        st.seconds = -1.0;
        return st;
    }
}

}  // namespace strata::core
