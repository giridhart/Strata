// tests/sycl/prefill_dma_direct_test.cpp - Regression test for SYCL dma_direct USM range checks
// and Prefill::run_impl plan/walk consistency (!stream_all path).
//
// Verifies:
// 1. Non-2 MiB-aligned sycl::malloc_host slices (edge buckets whose 2 MiB-aligned outer bounds fall
//    outside the USM allocation) are accepted by sycl_dma_direct_check and can DMA via queue::memcpy.
// 2. Interior 2 MiB buckets stay cached without edge-bucket false negatives poisoning them.
// 3. When ExpertSource::pinned() is true over pageable memory (dma_direct() == false), Prefill::run
//    stages the blob cleanly through Stager instead of calling Stager::wait(-1).

#include <sycl/sycl.hpp>

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/sycl_dma_direct.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-80s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

class MockSource : public strata::core::ExpertSource {
public:
    const uint8_t* usm_blob = nullptr;
    const uint8_t* pageable_blob = nullptr;
    bool pin_usm = true;
    bool pin_pageable = true;

    const uint8_t* blob(int64_t /*layer*/, int64_t expert) override {
        if ((expert & 1) == 0 && usm_blob != nullptr) return usm_blob;
        return pageable_blob;
    }

    bool pinned(int64_t /*layer*/, int64_t expert) const override {
        if ((expert & 1) == 0 && usm_blob != nullptr) return pin_usm;
        return pin_pageable;
    }
};

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-pf-dma-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void write_synthetic_pack(const fs::path& dir, int64_t n_layers, int64_t n_vocab) {
    fs::create_directories(dir);
    std::vector<std::string> rows;
    auto add_row = [&](const std::string& name, int kind, uint64_t bytes, int64_t ne0, int64_t ne1,
                       int code_bits = 0, int group_elems = 0, uint64_t codes = 0, uint64_t scales = 0) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "%s 0 %d 0 %llu 0 %llu %lld %lld %d 0 %d 0 0 %llu %llu 0 0 0",
                      name.c_str(), kind, (unsigned long long)bytes, (unsigned long long)bytes,
                      (long long)ne0, (long long)ne1, code_bits, group_elems,
                      (unsigned long long)codes, (unsigned long long)scales);
        rows.push_back(buf);
    };

    const uint64_t emb_codes = (uint64_t)2560 * (uint64_t)n_vocab;
    const uint64_t emb_scales = ((uint64_t)2560 * (uint64_t)n_vocab / 32) * 4;
    add_row("token_embd.weight", 0, emb_codes + emb_scales, 2560, n_vocab, 8, 32, emb_codes, emb_scales);
    add_row("output.weight", 0, emb_codes + emb_scales, 2560, n_vocab, 8, 32, emb_codes, emb_scales);

    for (int64_t l = 0; l < n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add_row(p + "hc_attn_norm.weight", 2, 10240, 2560, 1);
        add_row(p + "hc_attn_down.weight", 4, 1638400, 2560, 320);
        add_row(p + "hc_attn_up.weight", 4, 2560, 320, 4);
        add_row(p + "hc_attn_inject.weight", 4, 20480, 4, 2560);
        add_row(p + "hc_ffn_norm.weight", 2, 10240, 2560, 1);
        add_row(p + "hc_ffn_down.weight", 4, 1638400, 2560, 320);
        add_row(p + "hc_ffn_up.weight", 4, 2560, 320, 4);
        add_row(p + "hc_ffn_inject.weight", 4, 20480, 4, 2560);
        add_row(p + "ffn_gate_inp.weight", 4, 2621440, 2560, 512);
        add_row(p + "ffn_gate_inp_shexp.weight", 4, 5120, 2560, 1);
        add_row(p + "ffn_gate_shexp.weight", 0, 256, 2560, 640);
        add_row(p + "ffn_up_shexp.weight", 0, 256, 2560, 640);
        add_row(p + "ffn_down_shexp.weight", 0, 256, 640, 2560);

        if (l % 4 != 3) {
            add_row(p + "attn_qkv.weight", 0, 256, 2560, 10240);
            add_row(p + "attn_gate.weight", 0, 256, 2560, 48);
            add_row(p + "ssm_out.weight", 0, 256, 6144, 2560);
            add_row(p + "ssm_alpha.weight", 4, 245760, 2560, 48);
            add_row(p + "ssm_beta.weight", 4, 245760, 2560, 48);
            add_row(p + "ssm_conv1d.weight", 2, 163840, 4, 10240);
            add_row(p + "ssm_norm.weight", 2, 512, 128, 1);
            add_row(p + "ssm_dt.bias", 2, 192, 48, 1);
            add_row(p + "ssm_a", 2, 192, 48, 1);
        } else {
            add_row(p + "attn_q.weight", 0, 256, 2560, 6144);
            add_row(p + "attn_k.weight", 0, 256, 2560, 512);
            add_row(p + "attn_v.weight", 0, 256, 2560, 512);
            add_row(p + "attn_output.weight", 0, 256, 6144, 2560);
            add_row(p + "attn_q_norm.weight", 2, 1024, 256, 1);
            add_row(p + "attn_k_norm.weight", 2, 1024, 256, 1);
            add_row(p + "indexer.q_proj.weight", 4, 2621440, 2560, 512);
            add_row(p + "indexer.k_proj.weight", 4, 655360, 2560, 128);
            add_row(p + "indexer.q_norm.weight", 2, 512, 128, 1);
            add_row(p + "indexer.k_norm.weight", 2, 512, 128, 1);
        }
    }

    std::ofstream idx_file(dir / "index.txt");
    idx_file << "# align 256 pool 67108864 tensors " << rows.size() << "\n";
    for (const auto& r : rows) idx_file << r << "\n";

    std::ofstream bin_file(dir / "dense.bin", std::ios::binary);
    std::vector<char> zeros(1024 * 1024, 0);
    for (int i = 0; i < 4; ++i) bin_file.write(zeros.data(), zeros.size());
}

void test_dma_direct_predicate(sycl::queue& q) {
    std::printf("[TEST] sycl_dma_direct_check USM edge & pageable checks...\n");
    const auto ctx = q.get_context();

    // Allocate 6 MiB + 8 KiB USM host buffer so we can guarantee both an unaligned 2 MiB head
    // bucket (k << 21 before allocation start) and/or an unaligned 2 MiB tail bucket, plus an
    // interior 2 MiB bucket and a cross-2 MiB boundary span.
    constexpr size_t kTwoMib = 2u << 20;
    constexpr size_t kBlobBytes = 1382400;
    constexpr size_t kBufBytes = 6 * kTwoMib + 8192;
    uint8_t* usm = static_cast<uint8_t*>(sycl::malloc_host(kBufBytes, q));
    check(usm != nullptr, "sycl::malloc_host 6+ MiB USM buffer allocated");
    if (!usm) return;

    const uintptr_t base = reinterpret_cast<uintptr_t>(usm);
    // Pick an edge offset near the start, an interior bucket, an upper edge, and a cross-2 MiB span.
    const uint8_t* head_edge = usm;
    const uint8_t* tail_edge = usm + kBufBytes - kBlobBytes;
    const uintptr_t first_aligned = (base + kTwoMib - 1) & ~(kTwoMib - 1);
    const uint8_t* cross_boundary =
        reinterpret_cast<const uint8_t*>(first_aligned + kTwoMib - kBlobBytes / 2);

    std::vector<uint8_t> pageable_buf(kBlobBytes, 0x3c);
    MockSource src;
    src.usm_blob = head_edge;
    src.pageable_blob = pageable_buf.data();

    check(strata::prefill::sycl_dma_direct_check(&src, 0, 0, head_edge, kBlobBytes, ctx, false),
          "USM head-edge slice accepted without STRATA_PF_DMA_PAGEABLE");
    check(strata::prefill::sycl_dma_direct_check(&src, 0, 0, tail_edge, kBlobBytes, ctx, false),
          "USM tail-edge slice accepted without STRATA_PF_DMA_PAGEABLE");
    check(strata::prefill::sycl_dma_direct_check(&src, 0, 0, cross_boundary, kBlobBytes, ctx, false),
          "USM slice spanning 2 MiB boundary accepted");
    check(strata::prefill::sycl_dma_direct_check(&src, 0, 0, reinterpret_cast<const void*>(first_aligned),
                                                 kBlobBytes, ctx, false),
          "USM interior aligned slice accepted (cached bucket path)");

    // Verify real DMA from the unaligned USM slice onto the device works byte-for-byte.
    for (size_t i = 0; i < kBlobBytes; ++i) usm[i] = static_cast<uint8_t>((i * 17) & 0xff);
    uint8_t* d_dst = sycl::malloc_device<uint8_t>(kBlobBytes, q);
    std::vector<uint8_t> back(kBlobBytes, 0);
    q.memcpy(d_dst, head_edge, kBlobBytes).wait();
    q.memcpy(back.data(), d_dst, kBlobBytes).wait();
    check(std::memcmp(head_edge, back.data(), kBlobBytes) == 0,
          "Device memcpy from unaligned USM slice round-trips cleanly");
    sycl::free(d_dst, q);

    // Pageable buffer with pinned() == true: must be rejected when pageable_ok == false.
    check(!strata::prefill::sycl_dma_direct_check(&src, 0, 1, pageable_buf.data(), kBlobBytes, ctx, false),
          "Pageable buffer with pinned()=true rejected when pageable_ok=false");
    check(strata::prefill::sycl_dma_direct_check(&src, 0, 1, pageable_buf.data(), kBlobBytes, ctx, true),
          "Pageable buffer accepted only when pageable_ok=true");
    check(!strata::prefill::sycl_dma_direct_check(&src, 0, 0, nullptr, kBlobBytes, ctx, true),
          "Null blob rejected even when pageable_ok=true");

    src.pin_usm = false;
    check(!strata::prefill::sycl_dma_direct_check(&src, 0, 0, head_edge, kBlobBytes, ctx, true),
          "Unpinned source rejected");

    sycl::free(usm, q);
}

void test_prefill_non_stream_all_consistency(sycl::queue& q) {
    std::printf("[TEST] Prefill::run_impl (!stream_all) plan vs walk consistency...\n");

    unsetenv("STRATA_PF_DMA_PAGEABLE");
    unsetenv("STRATA_PREFILL_LAYER_MAJOR");
    unsetenv("STRATA_PF_EARLY_COPY");
    unsetenv("STRATA_PF_PREDICT");

    const auto g = strata::core::kCompiledGeometry;
    constexpr int64_t n_vocab = 64;
    TempDir tmp;
    write_synthetic_pack(tmp.path, g.n_layers, n_vocab);

    constexpr uint64_t kPoolBytes = 64 * 1024 * 1024;
    void* dummy_pool = sycl::malloc_device(kPoolBytes, q);
    check(dummy_pool != nullptr, "Dummy device weight pool allocated");
    if (!dummy_pool) return;
    q.memset(dummy_pool, 0, kPoolBytes).wait();

    strata::core::WeightTable wt;
    std::string err;
    check(wt.load(tmp.path.string(), dummy_pool, kPoolBytes, err),
          "WeightTable loaded (" + (err.empty() ? "ok" : err) + ")");

    void* native_scratch = sycl::malloc_device(64 * 1024, q);
    q.memset(native_scratch, 0, 64 * 1024).wait();
    const char* native_suffixes[] = {
        "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"
    };
    for (int64_t l = 0; l < g.n_layers; ++l) {
        for (const char* suf : native_suffixes) {
            const auto* r = wt.find("blk." + std::to_string(l) + "." + suf);
            if (r) {
                auto* mut_r = const_cast<strata::core::WeightRef*>(r);
                mut_r->native_data = dummy_pool;
                mut_r->native_type = 8;  // GGML_TYPE_Q8_0
                mut_r->native_q8_1 = native_scratch;
            }
        }
    }

    constexpr int64_t max_cells = 128;
    constexpr int64_t k_experts = 10;
    const uint64_t s_bytes = strata::core::session_bytes(g, max_cells, k_experts);
    void* session_base = sycl::malloc_device(s_bytes, q);
    check(session_base != nullptr, "SessionState memory allocated");
    if (!session_base) {
        sycl::free(native_scratch, q);
        sycl::free(dummy_pool, q);
        return;
    }
    q.memset(session_base, 0, s_bytes).wait();

    strata::core::SessionState ss;
    check(strata::core::session_init(g, max_cells, k_experts, session_base, ss) > 0, "SessionState initialized");

    const uint64_t blob_bytes = strata::kernels::cpu::expert_layout().max_blob;
    strata::core::ExpertCache xcache;
    check(xcache.open(2, g.n_layers, g.n_expert, (int64_t)blob_bytes, err),
          "Tiny ExpertCache opened (" + (err.empty() ? "ok" : err) + ")");

    // Even experts sit inside a non-2 MiB-aligned USM host buffer (+4096 inside a 4 MiB USM region);
    // odd experts report pinned()=true over a pageable std::malloc buffer so dma_direct() returns false.
    // Previously this caused Prefill::run_impl to skip Stager in the plan (job_of=-1, ready=nullptr)
    // and then call Stager::wait(-1) inside stage_one().
    constexpr size_t kUsmBytes = (4u << 20) + 4096;
    uint8_t* usm_raw = static_cast<uint8_t*>(sycl::malloc_host(kUsmBytes, q));
    std::vector<uint8_t> pageable_blob(blob_bytes, 0);
    check(usm_raw != nullptr, "4 MiB USM host blob backing allocated");
    if (usm_raw) std::memset(usm_raw, 0, kUsmBytes);

    MockSource src;
    src.usm_blob = usm_raw ? usm_raw + kUsmBytes - blob_bytes : nullptr;  // guaranteed unaligned tail 2 MiB bucket
    src.pageable_blob = pageable_blob.data();
    src.pin_usm = true;
    src.pin_pageable = true;

    std::vector<int32_t> host_res((size_t)(g.n_layers * g.n_expert), strata::core::kNotResident);
    for (int64_t l = 0; l < g.n_layers; ++l) host_res[(size_t)(l * g.n_expert)] = 0;

    strata::core::layer_set_fused_gr(true);
    strata::kernels::native_gdn_set_enabled(true);
    strata::kernels::native_router_set_enabled(true);
    strata::kernels::native_qsa_set_enabled(true);
    strata::kernels::native_qsa_indexer_set_enabled(true);
    strata::kernels::native_rope_set_enabled(true);

    strata::prefill::Prefill pf;
    constexpr int64_t kChunk = 16;  // < stream_all_min() (1024) -> enters !stream_all path in Prefill::run_impl
    const bool init_ok = pf.init(wt, g, ss, &src, &xcache, host_res.data(), kChunk, &q, err);
    check(init_ok, "Prefill::init (chunk=16, !stream_all) succeeded" + (init_ok ? "" : ": " + err));
    if (init_ok) {
        const std::vector<int64_t> tokens = {1, 2, 3, 4};
        const bool run_ok = pf.run(tokens.data(), (int64_t)tokens.size(), 0, err);
        check(run_ok, "Prefill::run completed without segfault" + (run_ok ? "" : ": " + err));
        std::printf("  stats: streamed=%lld dma=%lld resident=%lld\n",
                    (long long)pf.stats().experts_streamed,
                    (long long)pf.stats().experts_dma,
                    (long long)pf.stats().experts_resident);
        check(pf.stats().experts_streamed > 0, "Non-resident experts were streamed during Prefill::run");
        check(pf.stats().experts_dma > 0, "Unaligned USM pinned experts used direct DMA");
    }

    if (usm_raw) sycl::free(usm_raw, q);
    xcache.close();
    sycl::free(session_base, q);
    sycl::free(native_scratch, q);
    sycl::free(dummy_pool, q);
}

}  // namespace

int main() {
    std::printf("==================================================\n");
    std::printf("SYCL Prefill dma_direct & Stager Regression Test\n");
    std::printf("==================================================\n");

    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    std::printf("  Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    test_dma_direct_predicate(q);
    test_prefill_non_stream_all_consistency(q);

    std::printf("==================================================\n");
    std::printf(g_fail ? "%d check(s) FAILED\n" : "All checks PASSED\n", g_fail);
    std::printf("==================================================\n");
    return g_fail ? 1 : 0;
}
