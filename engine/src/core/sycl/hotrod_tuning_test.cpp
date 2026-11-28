// hotrod_tuning_test.cpp - Unit & GPU parity tests for the opt-in HotRod SYCL levers:
//   - STRATA_FUSE_WAIT (wait_copy_i32_from_mapped)
//   - STRATA_PCIE_EARLY (side-queue fetch_blobs + rebase_ptrs barrier overlap)
//   - STRATA_PCIE_BALANCE (PcieModel cost fitting & pick convergence)
//   - STRATA_PCIE_MIN1
//   - STRATA_ROUTE_TAIL_SKIP / STRATA_TAIL_CPU=0 / STRATA_RANK_CUTOFF
//   - STRATA_TAIL_WEIGHT / STRATA_TAIL_WEIGHT_FROM
//   - STRATA_LAYER_LIVE / STRATA_LAYER_SHARE / STRATA_LAYER_STATS
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s (%s)\n", __FILE__, __LINE__, #cond, (msg));     \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

// Minimal in-memory mock ExpertSource for host-side dispatch testing (no model files required).
class MockExpertSource final : public strata::core::ExpertSource {
public:
    MockExpertSource(int n_layers, int n_expert, size_t blob_bytes)
        : n_layers_(n_layers), n_expert_(n_expert), blob_(blob_bytes, 0) {}
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
        return blob_.data();
    }
    bool pinned(int64_t, int64_t) const override { return true; }
    bool pcie_layer(int64_t) const override { return true; }
    const uint8_t* device_alias(int64_t, int64_t) const override { return blob_.data(); }

private:
    int n_layers_;
    int n_expert_;
    std::vector<uint8_t> blob_;
};

int test_gpu_exact() {
    sycl::queue cs(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}});
    sycl::queue pf_cs(cs.get_context(), cs.get_device(), sycl::property_list{sycl::property::queue::in_order{}});

    // 1. Test STRATA_FUSE_WAIT kernel: wait_copy_i32_from_mapped
    {
        constexpr int64_t N = 256;
        uint32_t* h_flag = sycl::malloc_host<uint32_t>(1, cs);
        int32_t* h_src = sycl::malloc_host<int32_t>(N, cs);
        int32_t* d_dst = sycl::malloc_device<int32_t>(N, cs);
        CHECK(h_flag && h_src && d_dst, "USM allocation failed");
        *h_flag = 0;
        std::memset(h_src, 0, N * sizeof(int32_t));
        cs.memset(d_dst, 0, N * sizeof(int32_t)).wait();

        strata::kernels::wait_copy_i32_from_mapped(h_flag, 7u, d_dst, h_src, N, &cs);
        std::thread releaser([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            for (int64_t i = 0; i < N; ++i) h_src[i] = (int32_t)(i * 17 + 5);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*)h_flag = 7u;
        });
        cs.wait();
        releaser.join();

        std::vector<int32_t> got(N, 0);
        cs.memcpy(got.data(), d_dst, N * sizeof(int32_t)).wait();
        for (int64_t i = 0; i < N; ++i) {
            CHECK(got[(size_t)i] == (int32_t)(i * 17 + 5), "wait_copy_i32_from_mapped payload mismatch");
        }
        sycl::free(h_flag, cs);
        sycl::free(h_src, cs);
        sycl::free(d_dst, cs);
    }

    // 2. Test STRATA_PCIE_EARLY side-queue overlap: fork cs -> pf_cs, fetch_blobs + rebase_ptrs, join pf_cs -> cs
    {
        constexpr int kBlobs = 3;
        constexpr int64_t kBlobBytes = 4096;  // multiple of 16
        uint32_t* h_flagB = sycl::malloc_host<uint32_t>(1, cs);
        uint8_t* h_arena = sycl::malloc_host<uint8_t>((size_t)kBlobs * kBlobBytes, cs);
        uint8_t* d_stage = sycl::malloc_device<uint8_t>((size_t)kBlobs * kBlobBytes, cs);
        unsigned long long* d_ptrs = sycl::malloc_device<unsigned long long>(kBlobs, cs);
        int32_t* d_count = sycl::malloc_device<int32_t>(1, cs);
        int32_t* d_vram_work = sycl::malloc_device<int32_t>(1024, cs);
        CHECK(h_flagB && h_arena && d_stage && d_ptrs && d_count && d_vram_work, "USM allocation failed");

        for (int b = 0; b < kBlobs; ++b) {
            for (int64_t j = 0; j < kBlobBytes; ++j)
                h_arena[(size_t)b * kBlobBytes + (size_t)j] = (uint8_t)((b * 37 + j * 13 + 7) & 0xFF);
        }
        unsigned long long h_ptrs[kBlobs];
        for (int b = 0; b < kBlobs; ++b) h_ptrs[b] = (unsigned long long)(h_arena + (size_t)b * kBlobBytes);
        const int32_t h_cnt = kBlobs;
        *h_flagB = 0;
        cs.memcpy(d_ptrs, h_ptrs, sizeof(h_ptrs));
        cs.memcpy(d_count, &h_cnt, sizeof(h_cnt));
        cs.memset(d_stage, 0, (size_t)kBlobs * kBlobBytes);
        cs.wait();

        // Fork cs -> pf_cs
        sycl::event ev_fork = cs.ext_oneapi_submit_barrier();
        pf_cs.ext_oneapi_submit_barrier({ev_fork});
        strata::kernels::wait_flag_ge(h_flagB, 3u, &pf_cs);
        strata::kernels::fetch_blobs(d_ptrs, d_count, d_stage, kBlobBytes, kBlobs, &pf_cs);
        strata::kernels::rebase_ptrs(d_ptrs, d_count, d_stage, kBlobBytes, &pf_cs);
        sycl::event ev_join = pf_cs.ext_oneapi_submit_barrier();

        // Concurrent work on cs (simulating VRAM experts)
        cs.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> idx) { d_vram_work[idx] = (int32_t)(idx[0] * 3 + 1); });

        // Join pf_cs -> cs
        cs.ext_oneapi_submit_barrier({ev_join});

        std::thread releaser([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*)h_flagB = 3u;
        });
        cs.wait();
        releaser.join();

        std::vector<uint8_t> got_stage((size_t)kBlobs * kBlobBytes, 0);
        unsigned long long got_ptrs[kBlobs] = {};
        cs.memcpy(got_stage.data(), d_stage, got_stage.size());
        cs.memcpy(got_ptrs, d_ptrs, sizeof(got_ptrs));
        cs.wait();

        CHECK(std::memcmp(got_stage.data(), h_arena, got_stage.size()) == 0, "STRATA_PCIE_EARLY staged bytes mismatch");
        for (int b = 0; b < kBlobs; ++b) {
            const unsigned long long expected = (unsigned long long)(d_stage + (size_t)b * kBlobBytes);
            CHECK(got_ptrs[b] == expected, "STRATA_PCIE_EARLY rebased pointer mismatch");
        }
        sycl::free(h_flagB, cs);
        sycl::free(h_arena, cs);
        sycl::free(d_stage, cs);
        sycl::free(d_ptrs, cs);
        sycl::free(d_count, cs);
        sycl::free(d_vram_work, cs);
    }

    // 3. Test PcieModel convergence (STRATA_PCIE_BALANCE)
    {
        strata::core::PcieModel m;
        m.on = true;
        // Synthetic system: CPU = 0.2 + 1.0 * n_c + 0.1 * n_p, GPU = 0.4 + 0.5 * n_p.
        // For nmiss = 10:
        //   k=6: cpu = 0.2 + 4.0 + 0.6 = 4.8, gpu = 0.4 + 3.0 = 3.4 -> max 4.8
        //   k=7: cpu = 0.2 + 3.0 + 0.7 = 3.9, gpu = 0.4 + 3.5 = 3.9 -> max 3.9 (OPTIMAL!)
        //   k=8: cpu = 0.2 + 2.0 + 0.8 = 3.0, gpu = 0.4 + 4.0 = 4.4 -> max 4.4
        for (int step = 0; step < 80; ++step) {
            const int n_c = 1 + (step % 9);
            const int n_p = (step * 3) % 9;
            const double cpu = 0.2 + 1.0 * n_c + 0.1 * n_p;
            const double gpu = 0.4 + 0.5 * n_p;
            m.sample(n_c, n_p, cpu, gpu);
            if ((step & 7) == 7) m.solve();
        }
        m.solve();
        CHECK(m.ready(), "PcieModel should be ready after 80 samples");
        // Avoid explore tick on pick by setting explore = 1
        m.explore = 1;
        const int picked = m.pick(10, 2);
        CHECK(picked == 7, "PcieModel did not converge to optimal PCIe split k=7");
    }

    std::printf("hotrod_tuning_test [gpu_exact]: PASS\n");
    return 0;
}

struct DispatchFixture {
    static constexpr int kLayers = 2;
    static constexpr int kExperts = 512;
    static constexpr int kK = 10;
    static constexpr int kTok = 1;
    static constexpr int kH = strata::kernels::cpu::H;
    static constexpr int kFF = strata::kernels::cpu::FF;

    static size_t init_native_layout() {
        strata::kernels::cpu::NativeFmt f;
        std::string err;
        if (!strata::kernels::cpu::native_fmt(8, 8, kH, kFF, f, err)) {
            std::fprintf(stderr, "native_fmt failed: %s\n", err.c_str());
            std::exit(2);
        }
        const auto dir = std::filesystem::temp_directory_path() / "strata_hotrod_test_layout";
        std::filesystem::create_directories(dir);
        {
            std::ofstream out(dir / "native_experts.txt");
            out << "# strata native experts v1\n";
            for (int l = 0; l < kLayers; ++l)
                out << l << " 8 8 " << ((uint64_t)l * kExperts * f.bytes) << " " << f.bytes << "\n";
        }
        if (!strata::kernels::cpu::expert_layout_load(dir.string(), kLayers, kExperts, err)) {
            std::fprintf(stderr, "expert_layout_load failed: %s\n", err.c_str());
            std::exit(2);
        }
        return (size_t)f.bytes;
    }

    size_t blob_bytes;
    MockExpertSource src;
    strata::kernels::cpu::ExpertPool pool;
    strata::core::GpuPlanSink sink;
    strata::core::ExpertDispatch d;
    std::vector<int32_t> host_res;
    std::vector<int32_t> counts, start, dst, tok, start2;
    std::vector<unsigned long long> ptr, ptr2;
    std::vector<float> x, out;

    DispatchFixture()
        : blob_bytes(init_native_layout()),
          src(kLayers, kExperts, blob_bytes),
          pool(2),
          host_res((size_t)kLayers * kExperts, -1),
          counts(4, 0),
          start(65, 0),
          dst(64, 0),
          tok(64, 0),
          start2(65, 0),
          ptr(64, 0),
          ptr2(64, 0),
          x((size_t)kTok * kH, 0.0f),
          out((size_t)kTok * kK * kH, 1.0f) {
        sink.counts = counts.data();
        sink.start = start.data();
        sink.dst = dst.data();
        sink.tok = tok.data();
        sink.ptr = ptr.data();
        sink.ptr2 = ptr2.data();
        sink.start2 = start2.data();
        sink.staging = 0x100000ull;
        sink.staging_cap = 32;
        sink.cap = 64;
        sink.pcie_mode = 2;

        d.pool = &pool;
        d.src = &src;
        d.n_expert = kExperts;
        d.host_res = host_res.data();
        d.cache_base = reinterpret_cast<const uint8_t*>(0x200000ull);
        d.cache_blob = (int64_t)blob_bytes;
        d.plan = &sink;
        d.pcie_num = 51;  // --pcie-frac 0.2
        d.act_multi.resize(8);
        d.nact_multi.resize(8 * strata::kernels::cpu::kNativeActBytes);
        d.jobs_multi.resize(64);
        d.job_of.assign(kExperts, -1);
    }

    void run_layer(int layer, const int32_t ids[10], const float weights[10]) {
        d.layers = layer;
        std::fill(out.begin(), out.end(), 123.0f);  // sentinel: skipped/hit rows must become 0.0f
        strata::core::expert_dispatch_set_weights(weights);
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids, kTok, kK, out.data());
        strata::core::expert_dispatch_set_weights(nullptr);
    }
};

int test_dispatch_default() {
    DispatchFixture f;
    // Ranks 0..7 resident (experts 0..7), ranks 8..9 missed (experts 8..9) -> nmiss = 2
    for (int e = 0; e < 8; ++e) f.host_res[e] = e;
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[10] = {0.30f, 0.20f, 0.15f, 0.10f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f};
    f.run_layer(0, ids, w);
    CHECK(f.d.tail_experts == 0, "default must skip 0 experts");
    CHECK(f.d.last_n_pcie == 0, "default (2*51)>>8 must be 0 PCIe");
    CHECK(f.d.last_n_cpu == 2, "default must send both misses to CPU");
    CHECK(f.d.multi_misses == 2, "CPU must run 2 misses");
    std::printf("hotrod_tuning_test [dispatch_default]: PASS\n");
    return 0;
}

int test_pcie_min1() {
    DispatchFixture f;
    for (int e = 0; e < 8; ++e) f.host_res[e] = e;
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[10] = {0.30f, 0.20f, 0.15f, 0.10f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f};
    f.run_layer(0, ids, w);
    CHECK(f.d.tail_experts == 0, "pcie_min1 must skip 0 experts");
    CHECK(f.d.last_n_pcie == 1, "STRATA_PCIE_MIN1=1 must promote 1 of 2 misses to PCIe");
    CHECK(f.d.last_n_cpu == 1, "remaining 1 miss must go to CPU");
    std::printf("hotrod_tuning_test [pcie_min1]: PASS\n");
    return 0;
}

int test_tail_rank() {
    DispatchFixture f;
    // Experts 0,1 resident; 2 missed (rank 2 < 3 -> must compute); 3..9 missed (rank >= 3 -> must skip), except 5 resident (must NOT skip!)
    f.host_res[0] = 0;
    f.host_res[1] = 1;
    f.host_res[5] = 2;
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[10] = {0.30f, 0.20f, 0.15f, 0.10f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f};
    f.run_layer(0, ids, w);
    CHECK(f.d.tail_experts == 6, "STRATA_ROUTE_TAIL_SKIP=3 must skip 6 missed experts at ranks 3,4,6,7,8,9");
    CHECK(f.d.last_n_cpu == 1, "rank 2 miss must still run on CPU");
    CHECK(f.counts[0] == 3, "resident experts 0,1,5 must remain in VRAM plan");
    CHECK(f.out[3 * DispatchFixture::kH] == 0.0f, "skipped row must be zeroed");
    std::printf("hotrod_tuning_test [tail_rank]: PASS\n");
    return 0;
}

int test_tail_cpu_off() {
    DispatchFixture f;
    // STRATA_TAIL_CPU=0 + STRATA_ROUTE_RESIDENT_RANKS=4-9 -> floor 4
    f.host_res[0] = 0;
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[10] = {0.30f, 0.20f, 0.15f, 0.10f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f};
    f.run_layer(0, ids, w);
    CHECK(f.d.tail_experts == 6, "STRATA_TAIL_CPU=0 with ranks 4-9 must skip ranks 4..9 (6 experts)");
    CHECK(f.d.last_n_cpu + f.d.last_n_pcie == 3, "ranks 1,2,3 misses must still be served");
    std::printf("hotrod_tuning_test [tail_cpu_off]: PASS\n");
    return 0;
}

int test_rank_cutoff() {
    DispatchFixture f;
    // STRATA_RANK_CUTOFF=6: skips ALL ranks 6..9 even when resident!
    for (int e = 0; e < 10; ++e) f.host_res[e] = e;
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[10] = {0.30f, 0.20f, 0.15f, 0.10f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f};
    f.run_layer(0, ids, w);
    CHECK(f.d.tail_experts == 4, "STRATA_RANK_CUTOFF=6 must skip ranks 6..9 even when resident");
    CHECK(f.counts[0] == 6, "only ranks 0..5 should enter VRAM plan");
    std::printf("hotrod_tuning_test [rank_cutoff]: PASS\n");
    return 0;
}

int test_tail_weight() {
    DispatchFixture f;
    // STRATA_TAIL_WEIGHT=0.05 STRATA_TAIL_WEIGHT_FROM=3
    // All missed: ranks 0..2 (< 3) kept; ranks 3..6 (w >= 0.05) kept; ranks 7..9 (w = 0.03,0.02,0.01 < 0.05) skipped!
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[10] = {0.30f, 0.20f, 0.15f, 0.10f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f};
    f.run_layer(0, ids, w);
    CHECK(f.d.tail_experts == 3, "STRATA_TAIL_WEIGHT=0.05 must skip exactly ranks 7,8,9 (weights < 0.05)");
    CHECK(f.d.last_n_cpu + f.d.last_n_pcie == 7, "ranks 0..6 must be served");
    std::printf("hotrod_tuning_test [tail_weight]: PASS\n");
    return 0;
}

int test_layer_live() {
    DispatchFixture f;
    // STRATA_LAYER_LIVE=1 STRATA_LAYER_SHARE=0.5 STRATA_TAIL_WEIGHT_FROM=3
    // Layer 0 has small tail weights (0.01..0.035), Layer 1 has larger tail weights (0.08..0.11).
    // After 2500+ tail picks and 64 ticks, Layer 0 should skip much more than Layer 1!
    const int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w_l0[10] = {0.40f, 0.30f, 0.15f, 0.035f, 0.030f, 0.025f, 0.020f, 0.015f, 0.015f, 0.010f};
    const float w_l1[10] = {0.15f, 0.14f, 0.13f, 0.110f, 0.100f, 0.095f, 0.090f, 0.085f, 0.080f, 0.075f};
    for (int win = 0; win < 192; ++win) {
        f.run_layer(0, ids, w_l0);
        f.run_layer(1, ids, w_l1);
        strata::core::expert_dispatch_window_tick(2);
    }
    const int64_t before_l0 = f.d.tail_experts;
    f.run_layer(0, ids, w_l0);
    const int64_t skip_l0 = f.d.tail_experts - before_l0;
    const int64_t before_l1 = f.d.tail_experts;
    f.run_layer(1, ids, w_l1);
    const int64_t skip_l1 = f.d.tail_experts - before_l1;
    CHECK(skip_l0 > skip_l1, "STRATA_LAYER_LIVE=1 should skip more in low-weight layer 0 than high-weight layer 1");
    CHECK(skip_l0 + skip_l1 >= 5 && skip_l0 + skip_l1 <= 9, "total skipped tail picks out of 14 should be near 50% (~7)");
    std::printf("hotrod_tuning_test [layer_live]: PASS (skip_l0=%lld/7, skip_l1=%lld/7)\n",
                (long long)skip_l0, (long long)skip_l1);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc >= 2) ? argv[1] : "--mode=gpu_exact";
    if (mode == "--mode=gpu_exact") return test_gpu_exact();
    if (mode == "--mode=dispatch_default") return test_dispatch_default();
    if (mode == "--mode=pcie_min1") return test_pcie_min1();
    if (mode == "--mode=tail_rank") return test_tail_rank();
    if (mode == "--mode=tail_cpu_off") return test_tail_cpu_off();
    if (mode == "--mode=rank_cutoff") return test_rank_cutoff();
    if (mode == "--mode=tail_weight") return test_tail_weight();
    if (mode == "--mode=layer_live") return test_layer_live();
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
