// tests/sycl/level_zero_memory_test.cpp - Unit test for Level Zero infrastructure & memory subsystem:
// 1. Dedicated copy engine queue (create_copy_stream, STRATA_COPY_ENGINE)
// 2. Event profiling vs barrier auto-calibration (measure_marks, STRATA_EVENT_MARK)
// 3. Device status query (device_status)
// 4. Chunked device memset (device_memset)
// 5. Multi-part pinned arena (PinnedArena, 2MB alignment, layer-bounded pieces)
// 6. THP settle check (settle_page_migration, settle_before_ready)

#include <sycl/sycl.hpp>
#include "strata/core/device.hpp"
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("FAIL: %s (line %d): %s\n", __FUNCTION__, __LINE__, msg); \
            g_failures++; \
        } \
    } while (0)

void test_copy_engine_queue() {
    std::printf("[TEST] Dedicated Copy Engine Queue...\n");

    // Test 1: Standard copy queue creation
    sycl::queue* copy_q = strata::core::create_copy_stream(0);
    TEST_CHECK(copy_q != nullptr, "create_copy_stream(0) returned nullptr");
    if (!copy_q) return;

    std::printf("  Copy queue created on device: %s\n",
                copy_q->get_device().get_info<sycl::info::device::name>().c_str());

    constexpr size_t N = 1024 * 1024; // 4 MB
    uint32_t* d_buf = sycl::malloc_device<uint32_t>(N, *copy_q);
    TEST_CHECK(d_buf != nullptr, "sycl::malloc_device failed");

    std::vector<uint32_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint32_t>(i ^ 0x5a5a5a5a);

    // Test H2D and D2H on copy stream
    copy_q->memcpy(d_buf, h_in.data(), N * sizeof(uint32_t)).wait();

    std::vector<uint32_t> h_out(N, 0);
    copy_q->memcpy(h_out.data(), d_buf, N * sizeof(uint32_t)).wait();

    bool data_ok = (std::memcmp(h_in.data(), h_out.data(), N * sizeof(uint32_t)) == 0);
    TEST_CHECK(data_ok, "Copy queue H2D -> D2H data mismatch");
    sycl::free(d_buf, *copy_q);

    // Test 2: STRATA_COPY_ENGINE=0 fallback
    setenv("STRATA_COPY_ENGINE", "0", 1);
    sycl::queue* disabled_q = strata::core::create_copy_stream(0);
    TEST_CHECK(disabled_q != nullptr, "create_copy_stream(0) with STRATA_COPY_ENGINE=0 returned nullptr");
    if (disabled_q) {
        uint32_t* d_test = sycl::malloc_device<uint32_t>(1024, *disabled_q);
        TEST_CHECK(d_test != nullptr, "malloc on fallback queue failed");
        if (d_test) {
            uint32_t val = 42;
            disabled_q->memcpy(d_test, &val, sizeof(uint32_t)).wait();
            uint32_t back = 0;
            disabled_q->memcpy(&back, d_test, sizeof(uint32_t)).wait();
            TEST_CHECK(back == 42, "Fallback queue memcpy mismatch");
            sycl::free(d_test, *disabled_q);
        }
    }
    unsetenv("STRATA_COPY_ENGINE");

    std::printf("  Dedicated Copy Engine Queue: OK\n");
}

void test_mark_calibration() {
    std::printf("[TEST] Event Profiling vs Barrier Auto-Calibration...\n");

    bool auto_barrier = strata::core::measure_marks(0);
    std::printf("  Auto-calibration result: barrier=%s\n", auto_barrier ? "true" : "false");

    bool current = strata::core::marks_by_barrier(0);
    TEST_CHECK(current == auto_barrier, "marks_by_barrier does not match auto_barrier result");

    // Test STRATA_EVENT_MARK overrides
    setenv("STRATA_EVENT_MARK", "barrier", 1);
    TEST_CHECK(strata::core::marks_by_barrier(0) == true, "STRATA_EVENT_MARK=barrier not respected");

    setenv("STRATA_EVENT_MARK", "tag", 1);
    TEST_CHECK(strata::core::marks_by_barrier(0) == false, "STRATA_EVENT_MARK=tag not respected");

    unsetenv("STRATA_EVENT_MARK");
    TEST_CHECK(strata::core::marks_by_barrier(0) == auto_barrier, "Default not restored after unsetenv");

    std::printf("  Event Profiling vs Barrier Auto-Calibration: OK\n");
}

void test_device_status() {
    std::printf("[TEST] Device Status Query...\n");

    std::string status = strata::core::device_status(0);
    std::printf("  Device status: %s\n", status.c_str());
    TEST_CHECK(!status.empty(), "device_status returned empty string");
    TEST_CHECK(status.find("answers") != std::string::npos || status.find("lost") != std::string::npos,
               "device_status did not contain expected status keywords");

    std::printf("  Device Status Query: OK\n");
}

void test_chunked_device_memset() {
    std::printf("[TEST] Chunked Device Memset...\n");

    sycl::queue& q = strata::core::get_queue(0);
    constexpr size_t N = 16 * 1024 * 1024; // 16 MB
    uint8_t* d_buf = sycl::malloc_device<uint8_t>(N, q);
    TEST_CHECK(d_buf != nullptr, "malloc_device failed for chunked memset test");
    if (!d_buf) return;

    // Set to 0x7e using device_memset
    strata::core::device_memset(d_buf, 0x7e, N, q).wait();

    std::vector<uint8_t> h_buf(N, 0);
    q.memcpy(h_buf.data(), d_buf, N).wait();

    bool all_match = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_buf[i] != 0x7e) {
            all_match = false;
            break;
        }
    }
    TEST_CHECK(all_match, "device_memset data verification failed");
    sycl::free(d_buf, q);

    std::printf("  Chunked Device Memset: OK\n");
}

void test_pinned_arena_alignment_and_pieces() {
    std::printf("[TEST] Multi-Part Pinned Arena & Alignment...\n");

    constexpr uint64_t arena_sz = 64 * 1024 * 1024; // 64 MB
    std::vector<uint64_t> cuts = {16 * 1024 * 1024, 32 * 1024 * 1024, 48 * 1024 * 1024};

    strata::core::PinnedArena arena(arena_sz, 0, cuts);
    TEST_CHECK(arena.valid(), "PinnedArena::valid() returned false");
    TEST_CHECK(arena.data() != nullptr, "PinnedArena::data() returned nullptr");

    uintptr_t addr = reinterpret_cast<uintptr_t>(arena.data());
    TEST_CHECK((addr % (2 * 1024 * 1024)) == 0, "PinnedArena::data() is NOT 2 MB-aligned");

    // Test at(offset)
    TEST_CHECK(arena.at(0) == arena.data(), "arena.at(0) != arena.data()");
    TEST_CHECK(arena.at(16 * 1024 * 1024) == arena.data() + 16 * 1024 * 1024,
               "arena.at(16MB) mismatch");

    // Test writing and reading data
    std::memset(arena.data(), 0xa5, arena_sz);
    TEST_CHECK(*arena.at(32 * 1024 * 1024) == 0xa5, "arena memory readback mismatch");

    std::printf("  Arena base: %p (2MB aligned: %s), registered %zu pieces, %zu bytes\n",
                arena.data(), ((addr % (2 * 1024 * 1024)) == 0) ? "yes" : "no",
                arena.pieces().size(), (size_t)arena.registered_bytes);

    TEST_CHECK(!arena.pieces().empty(), "arena.pieces() is empty");

    // Test simulated multi-part cut with STRATA_MAX_ALLOC_MIB=16 and STRATA_PIN_PARTS=1
    setenv("STRATA_MAX_ALLOC_MIB", "16", 1);
    setenv("STRATA_PIN_PARTS", "1", 1);
    {
        strata::core::PinnedArena multipart_arena(arena_sz, 0, cuts);
        TEST_CHECK(multipart_arena.valid(), "multipart_arena.valid() returned false");
        std::printf("  Multipart arena with 16 MiB max alloc: registered %zu pieces\n",
                    multipart_arena.pieces().size());
        TEST_CHECK(multipart_arena.pieces().size() >= 4,
                   "Expected at least 4 pieces with 16 MiB chunking across 64 MiB");
    }
    unsetenv("STRATA_MAX_ALLOC_MIB");
    unsetenv("STRATA_PIN_PARTS");

    std::printf("  Multi-Part Pinned Arena & Alignment: OK\n");
}

void test_thp_settle() {
    std::printf("[TEST] THP Compaction Settle Check...\n");

    double peak = 0.0;
    bool settled = strata::platform::settle_page_migration(0.2, 0.05, 100000.0, peak);
    TEST_CHECK(settled, "settle_page_migration returned false");
    std::printf("  settle_page_migration: settled=%s, peak=%.1f/s\n", settled ? "true" : "false", peak);

    // Test settle_before_ready with STRATA_SETTLE_S=0 (skip wait)
    setenv("STRATA_SETTLE_S", "0", 1);
    strata::platform::settle_before_ready();
    unsetenv("STRATA_SETTLE_S");

    std::printf("  THP Compaction Settle Check: OK\n");
}

} // namespace

int main() {
    std::printf("==================================================\n");
    std::printf("Running Level Zero Infrastructure & Memory Tests\n");
    std::printf("==================================================\n");

    test_copy_engine_queue();
    test_mark_calibration();
    test_device_status();
    test_chunked_device_memset();
    test_pinned_arena_alignment_and_pieces();
    test_thp_settle();

    if (g_failures == 0) {
        std::printf("==================================================\n");
        std::printf("All Level Zero & Memory Subsystem Tests PASSED!\n");
        std::printf("==================================================\n");
        return 0;
    } else {
        std::printf("==================================================\n");
        std::printf("%d test(s) FAILED!\n", g_failures);
        std::printf("==================================================\n");
        return 1;
    }
}
