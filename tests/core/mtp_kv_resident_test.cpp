// tests/core/mtp_kv_resident_test.cpp - Regression test and memory-budget evaluation for the MTP drafter
// K/V ring vs resident mode selection under `--kv-resident`.
//
// Verifies:
//   1. `mtp_kv_ring_cells(window, max_cells, max_t)` returns -1 when `window <= 0` or `window >= max_cells`
//      (so `--kv-resident` keeps the dense drafter K/V in `kv_mode = 0` instead of un-resolved Clock-LRU
//      paging `kv_mode = 1`), and returns `window + 4 * max_t + 64` when `0 < window < max_cells` (`kv_mode = 2`).
//   2. Live `qsa_state_bytes` + `qsa_state_init` behavior on device (`ring == 0` buggy Clock-LRU vs
//      `ring == -1` fixed resident vs `ring > 0` windowed ring) under `--kv-resident`.
//   3. Exact VRAM / pinned host memory bytes saved and equivalent MoE expert slots on a 12 GB Arc B580.

#include "strata/core/device.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/mtp_kv_ring.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/kv_stream.hpp"

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#else
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using strata::core::ModelGeometry;
using strata::core::QsaState;
using strata::core::kCompiledGeometry;
using strata::core::mtp_kv_ring_cells;
using strata::core::qsa_kv_host_bytes;
using strata::core::qsa_kv_hybrid;
using strata::core::qsa_kv_int8;
using strata::core::qsa_kv_q4;
using strata::core::qsa_kv_resident;
using strata::core::qsa_kv_resident_min;
using strata::core::qsa_set_kv_hybrid;
using strata::core::qsa_set_kv_int8;
using strata::core::qsa_set_kv_q4;
using strata::core::qsa_set_kv_resident;
using strata::core::qsa_shapes;
using strata::core::qsa_state_bytes;
using strata::core::qsa_state_init;
using strata::core::qsa_state_zero;

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-76s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

double mib(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }
double kib(uint64_t bytes) { return static_cast<double>(bytes) / 1024.0; }

struct KvGlobalGuard {
    int64_t resident = qsa_kv_resident();
    bool int8 = qsa_kv_int8();
    bool q4 = qsa_kv_q4();
    bool hybrid = qsa_kv_hybrid();
    ~KvGlobalGuard() {
        qsa_set_kv_resident(resident);
        qsa_set_kv_int8(int8);
        qsa_set_kv_q4(q4);
        qsa_set_kv_hybrid(hybrid);
    }
};

struct InitProbe {
    bool ok = false;
    uint64_t state_bytes = 0;
    uint64_t host_bytes_delta = 0;
    int kv_mode = -1;
    int64_t n_pages = 0;
    int64_t n_slots = 0;
    int64_t idx_pooled_rows = 0;
    bool has_stream_map = false;
    bool has_host_copy = false;
    int32_t page0 = -999;
};

InitProbe run_state_probe(const ModelGeometry& g, int64_t max_cells, int64_t ring_cells,
                          const QsaState* share_rope = nullptr) {
    InitProbe r;
    const bool with_rope = (share_rope == nullptr);
    r.state_bytes = qsa_state_bytes(g, max_cells, with_rope, ring_cells);
    QsaState st{};
    const uint64_t host_before = qsa_kv_host_bytes();

#if defined(SYCL_LANGUAGE_VERSION)
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    void* arena = nullptr;
    try {
        arena = sycl::malloc_device(r.state_bytes, q);
    } catch (...) {
        arena = nullptr;
    }
    if (!arena) return r;

    const uint64_t used = qsa_state_init(g, max_cells, arena, st, share_rope, ring_cells);
    if (used == 0) {
        sycl::free(arena, q);
        return r;
    }
    qsa_state_zero(st, g, nullptr);
    q.wait_and_throw();

    r.ok = true;
    r.host_bytes_delta = qsa_kv_host_bytes() - host_before;
    r.kv_mode = st.kv_mode;
    r.n_pages = st.n_pages;
    r.n_slots = st.n_slots;
    r.idx_pooled_rows = st.idx_pooled_rows;
    r.has_stream_map = st.map.slot_block != nullptr;
    r.has_host_copy = st.host.present();

    // Inspect page_table[0]: identity (0) in mode 0, evicted (-1) in mode 1, 0 in mode 2.
    if (st.page_table) {
        q.memcpy(&r.page0, st.page_table, sizeof(int32_t)).wait();
    }

    // cos_tab/sin_tab/page_table live inside `arena`; only host allocations need separate frees.
    if (st.host_step) sycl::free(st.host_step, q);
    if (st.host_pos) sycl::free(st.host_pos, q);
    if (st.host.k_q) sycl::free(st.host.k_q, q);
    else if (st.host.k_pool) sycl::free(st.host.k_pool, q);
    else if (st.host.k_q4) sycl::free(st.host.k_q4, q);
    sycl::free(arena, q);
#else
    void* arena = nullptr;
    if (cudaMalloc(&arena, r.state_bytes) != cudaSuccess || !arena) {
        cudaGetLastError();
        return r;
    }
    const uint64_t used = qsa_state_init(g, max_cells, arena, st, share_rope, ring_cells);
    if (used == 0) {
        cudaGetLastError();
        cudaFree(arena);
        return r;
    }
    qsa_state_zero(st, g, nullptr);
    cudaDeviceSynchronize();

    r.ok = true;
    r.host_bytes_delta = qsa_kv_host_bytes() - host_before;
    r.kv_mode = st.kv_mode;
    r.n_pages = st.n_pages;
    r.n_slots = st.n_slots;
    r.idx_pooled_rows = st.idx_pooled_rows;
    r.has_stream_map = st.map.slot_block != nullptr;
    r.has_host_copy = st.host.present();

    if (st.page_table) {
        cudaMemcpy(&r.page0, st.page_table, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    if (st.host_step) cudaFreeHost(st.host_step);
    if (st.host_pos) cudaFreeHost(st.host_pos);
    if (st.host.k_q) cudaFreeHost(st.host.k_q);
    else if (st.host.k_pool) cudaFreeHost(st.host.k_pool);
    else if (st.host.k_q4) cudaFreeHost(st.host.k_q4);
    cudaFree(arena);
#endif
    return r;
}

void test_ring_helper() {
    std::printf("\n[1] MTP ring_cells selection helper (mtp_kv_ring_cells)\n");
    constexpr int64_t kCtx256K = 262144;
    constexpr int64_t kCtx32K = 32768;
    constexpr int kMaxT = 8;

    check(mtp_kv_ring_cells(0, kCtx256K, kMaxT) == -1, "window == 0 -> ring = -1 (resident, not 0)");
    check(mtp_kv_ring_cells(-1, kCtx256K, kMaxT) == -1, "window < 0 -> ring = -1 (resident)");
    check(mtp_kv_ring_cells(kCtx256K, kCtx256K, kMaxT) == -1, "window == max_cells -> ring = -1 (resident)");
    check(mtp_kv_ring_cells(kCtx256K + 1024, kCtx256K, kMaxT) == -1, "window > max_cells -> ring = -1 (resident)");
    check(mtp_kv_ring_cells(32768, kCtx32K, kMaxT) == -1, "window == 32768 at 32K context -> ring = -1 (resident)");
    check(mtp_kv_ring_cells(32768, kCtx256K, kMaxT) == 32768 + 4 * kMaxT + 64,
          "window == 32768 at 256K context -> ring = 32864 (mode 2 ring)");
    check(mtp_kv_ring_cells(16384, kCtx256K, 4) == 16384 + 4 * 4 + 64,
          "window == 16384, max_t == 4 at 256K -> ring = 16464 (mode 2 ring)");
}

void report_memory_budget(const ModelGeometry& g) {
    std::printf("\n[3] 12 GB Arc B580 Memory-Budget Evaluation (Qwen3.8-Flash-Next MTP Drafter, INT8 KV)\n");
    KvGlobalGuard guard;
    qsa_set_kv_hybrid(false);
    qsa_set_kv_q4(false);
    qsa_set_kv_int8(true);
    qsa_set_kv_resident(32768);

    const auto s = qsa_shapes(g);
    constexpr uint64_t kExpertBlob = strata::kernels::cpu::BLOB;  // 1,382,400 B (Q2_0)
    constexpr int64_t kContexts[] = {131072, 262144};
    constexpr int kMaxT = 8;
    constexpr int64_t kWindow = 32768;

    for (const int64_t ctx : kContexts) {
        const int64_t pages = (ctx + s.page_size - 1) / s.page_size;
        const int64_t res_slots = (std::max<int64_t>(qsa_kv_resident(), qsa_kv_resident_min()) + s.page_size - 1) /
                                  s.page_size;
        const int64_t ring_window = mtp_kv_ring_cells(kWindow, ctx, kMaxT);
        const int64_t ring_resident = mtp_kv_ring_cells(0, ctx, kMaxT);

        const uint64_t vram_buggy_lru = qsa_state_bytes(g, ctx, false, 0);
        const uint64_t vram_fixed_res = qsa_state_bytes(g, ctx, false, ring_resident);
        const uint64_t vram_ring_win = qsa_state_bytes(g, ctx, false, ring_window);

        const uint64_t host_copy_bytes =
            static_cast<uint64_t>(pages) * strata::kernels::kv_block_bytes(s, strata::kernels::kKvInt8) + 4 * 256;
        const uint64_t map_bytes = strata::kernels::kv_stream_map_bytes(res_slots) + 6 * 16;

        const uint64_t total_buggy_lru = vram_buggy_lru + host_copy_bytes;
        const uint64_t total_fixed_res = vram_fixed_res;  // 0 host copy in kv_mode = 0
        const uint64_t host_saved_vs_lru = host_copy_bytes;
        const uint64_t vram_saved_win_vs_res = vram_fixed_res - vram_ring_win;
        const uint64_t slots_host_saved = host_saved_vs_lru / kExpertBlob;
        const uint64_t slots_win_saved = vram_saved_win_vs_res / kExpertBlob;

        std::printf("  Context %lld (%lld pages, kv_resident=%lld, expert_blob=%llu B):\n",
                    (long long)ctx, (long long)pages, (long long)qsa_kv_resident(),
                    (unsigned long long)kExpertBlob);
        std::printf("    - Buggy ring=0 (kv_mode=1 Clock-LRU):  VRAM = %6.2f MiB (%llu B), Host = %6.2f MiB (%llu B), "
                    "LRU map = %6.2f KiB (%llu B), Total = %6.2f MiB\n",
                    mib(vram_buggy_lru), (unsigned long long)vram_buggy_lru,
                    mib(host_copy_bytes), (unsigned long long)host_copy_bytes,
                    kib(map_bytes), (unsigned long long)map_bytes,
                    mib(total_buggy_lru));
        std::printf("    - Fixed ring=-1 (kv_mode=0 resident):  VRAM = %6.2f MiB (%llu B), Host =   0.00 MiB (0 B), "
                    "LRU map =   0.00 KiB (0 B), Total = %6.2f MiB\n",
                    mib(vram_fixed_res), (unsigned long long)vram_fixed_res,
                    mib(total_fixed_res));
        std::printf("    - Window ring=%lld (kv_mode=2):   VRAM = %6.2f MiB (%llu B), Host = %6.2f MiB (%llu B), "
                    "LRU map =   0.00 KiB (0 B)\n",
                    (long long)ring_window, mib(vram_ring_win), (unsigned long long)vram_ring_win,
                    mib(host_copy_bytes), (unsigned long long)host_copy_bytes);
        std::printf("    => Pinned host + map eliminated (ring=-1 vs ring=0): %6.2f MiB (%llu B host + %llu B map, "
                    "~%llu Q2_0 MoE expert slots eq.)\n",
                    mib(host_saved_vs_lru + map_bytes), (unsigned long long)host_saved_vs_lru,
                    (unsigned long long)map_bytes, (unsigned long long)((host_saved_vs_lru + map_bytes) / kExpertBlob));
        std::printf("    => VRAM saved with --mtp-window %lld vs full resident (ring=%lld vs -1): %6.2f MiB (%llu B, "
                    "+%llu Q2_0 MoE expert slots on 12 GB B580)\n",
                    (long long)kWindow, (long long)ring_window,
                    mib(vram_saved_win_vs_res), (unsigned long long)vram_saved_win_vs_res,
                    (unsigned long long)slots_win_saved);
        (void)slots_host_saved;
    }
}

}  // namespace

int main() {
    std::printf("mtp_kv_resident_test\n");
    test_ring_helper();

    const ModelGeometry g = kCompiledGeometry;
    const int devs = strata::core::device_count();
    std::printf("\n[2] Live qsa_state_init mode selection (devices=%d)\n", devs);

    if (devs <= 0) {
        std::printf("  SKIP live device qsa_state_init checks: no GPU device detected\n");
        report_memory_budget(g);
        std::printf(g_fail == 0 ? "\nmtp_kv_resident_test: all ok (CPU-only)\n"
                                : "\nmtp_kv_resident_test: %d FAILED\n",
                    g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    KvGlobalGuard guard;
    qsa_set_kv_hybrid(false);
    qsa_set_kv_q4(false);
    qsa_set_kv_int8(true);

    constexpr int64_t kMaxCells = 262144;
    constexpr int kMaxT = 8;
    constexpr int64_t kWindow = 32768;
    constexpr int64_t kKvResident = 32768;

    // Dummy share_rope handle so the drafter probes size without the shared 64 MiB RoPE table (as MtpDrafter::load).
    QsaState dummy_rope{};
    qsa_set_kv_resident(kKvResident);

    // 2a. Regression: old buggy ring = 0 under --kv-resident 32768 enters kv_mode = 1 (un-resolved Clock-LRU)
    const InitProbe buggy = run_state_probe(g, kMaxCells, 0, &dummy_rope);
    check(buggy.ok, "ring=0, kv_resident=32768: qsa_state_init succeeds (reproduces bug)");
    check(buggy.kv_mode == 1, "ring=0, kv_resident=32768: assigns buggy kv_mode == 1 (Clock-LRU)");
    check(buggy.has_stream_map, "ring=0, kv_resident=32768: allocates Clock-LRU residency map");
    check(buggy.has_host_copy && buggy.host_bytes_delta > (260ull << 20),
          "ring=0, kv_resident=32768: allocates >260 MiB pinned host copy");
    check(buggy.page0 == -1, "ring=0, kv_resident=32768: page_table[0] == -1 (un-resolved)");

    // 2b. Fix: window = 0 -> mtp_kv_ring_cells = -1 keeps dense drafter in kv_mode = 0 (compact resident)
    const int64_t ring_fixed = mtp_kv_ring_cells(0, kMaxCells, kMaxT);
    const InitProbe fixed = run_state_probe(g, kMaxCells, ring_fixed, &dummy_rope);
    check(fixed.ok, "ring=-1 (window=0), kv_resident=32768: qsa_state_init succeeds");
    check(fixed.kv_mode == 0, "ring=-1 (window=0), kv_resident=32768: keeps kv_mode == 0 (resident)");
    check(fixed.n_slots == fixed.n_pages, "ring=-1 (window=0): n_slots == n_pages (all pages resident)");
    check(!fixed.has_stream_map, "ring=-1 (window=0): no Clock-LRU residency map allocated");
    check(!fixed.has_host_copy && fixed.host_bytes_delta == 0,
          "ring=-1 (window=0): 0 bytes pinned host copy allocated");
    check(fixed.page0 == 0, "ring=-1 (window=0): page_table[0] == 0 (identity resident)");

    // 2c. Windowed: window = 32768 < 262144 -> ring = 32864 -> kv_mode = 2 (compact ring)
    const int64_t ring_win = mtp_kv_ring_cells(kWindow, kMaxCells, kMaxT);
    const InitProbe win = run_state_probe(g, kMaxCells, ring_win, &dummy_rope);
    check(win.ok, "ring=32864 (window=32768), kv_resident=32768: qsa_state_init succeeds");
    check(win.kv_mode == 2, "ring=32864 (window=32768), kv_resident=32768: selects kv_mode == 2 (ring)");
    check(win.n_slots == (ring_win + 3) / 4, "ring=32864 (window=32768): n_slots matches ring pages");
    check(win.idx_pooled_rows == 2, "ring=32864 (window=32768): idx_pooled_rows == 2 (no indexer bloat)");
    check(!win.has_stream_map, "ring=32864 (window=32768): no Clock-LRU residency map");
    check(win.has_host_copy && win.host_bytes_delta == buggy.host_bytes_delta,
          "ring=32864 (window=32768): host resume copy allocated");

    // 2d. Non-streaming (--kv-resident 0): always kv_mode = 0, 0 host bytes
    qsa_set_kv_resident(0);
    const InitProbe no_stream_res = run_state_probe(g, kMaxCells, ring_fixed, &dummy_rope);
    check(no_stream_res.ok && no_stream_res.kv_mode == 0 && no_stream_res.host_bytes_delta == 0,
          "kv_resident=0, ring=-1: kv_mode == 0 and 0 host bytes");
    const InitProbe no_stream_win = run_state_probe(g, kMaxCells, ring_win, &dummy_rope);
    check(no_stream_win.ok && no_stream_win.kv_mode == 0 && no_stream_win.host_bytes_delta == 0,
          "kv_resident=0, ring=32864: kv_mode == 0 and 0 host bytes");

    report_memory_budget(g);

    std::printf(g_fail == 0 ? "\nmtp_kv_resident_test: all ok\n" : "\nmtp_kv_resident_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
