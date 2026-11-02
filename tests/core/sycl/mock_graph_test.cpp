// tests/core/sycl/mock_graph_test.cpp - End-to-end execution graph test with fake weights on SYCL/Xe2
//
// Validates full-scale (48-layer Qwen 3.8 Flash-Next) graph capture, finalization, and replay
// using weight aliasing into a single compact USM device pool (< 100 MB total VRAM).
//
// Tests monolithic graph, chunked graph (STRATA_VERIFY_GRAPH_CHUNK), and direct execution modes
// on constrained hardware without requiring full model weight files.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/device.hpp>

#include "strata/artifact/model_family.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::ModelGeometry;
using strata::core::SessionState;
using strata::core::Verifier;
using strata::core::VerifyHits;
using strata::core::WeightRef;
using strata::core::WeightTable;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-mock-graph-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

// Generates an index.txt and dense.bin for full Qwen 3.8 Flash-Next (48 layers)
// with all tensors aliased into a single 64 MB arena region.
void write_synthetic_pack(const fs::path& dir, int64_t n_layers, int64_t n_vocab) {
    fs::create_directories(dir);

    std::vector<std::string> rows;

    auto add_row = [&](const std::string& name, int kind, uint64_t bytes, int64_t ne0, int64_t ne1,
                       int code_bits = 0, int group_elems = 0, uint64_t codes = 0, uint64_t scales = 0) {
        char buf[512];
        // Format: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind
        std::snprintf(buf, sizeof(buf),
                      "%s 0 %d 0 %llu 0 %llu %lld %lld %d 0 %d 0 0 %llu %llu 0 0 0",
                      name.c_str(), kind, (unsigned long long)bytes, (unsigned long long)bytes,
                      (long long)ne0, (long long)ne1, code_bits, group_elems,
                      (unsigned long long)codes, (unsigned long long)scales);
        rows.push_back(buf);
    };

    // Model root tensors
    add_row("output_hc_norm.weight", 2, 10240, 2560, 1);
    add_row("output_hc_down.weight", 4, 1638400, 2560, 320);
    add_row("output_hc_up.weight", 4, 2560, 320, 4);

    // token_embd.weight & output.weight: quantized Q8_0 so embedding_gather_dev & lm_head succeed
    const uint64_t emb_codes = (uint64_t)2560 * (uint64_t)n_vocab;
    const uint64_t emb_scales = ((uint64_t)2560 * (uint64_t)n_vocab / 32) * 4;
    add_row("token_embd.weight", 0, emb_codes + emb_scales, 2560, n_vocab, 8, 32, emb_codes, emb_scales);
    add_row("output.weight", 0, emb_codes + emb_scales, 2560, n_vocab, 8, 32, emb_codes, emb_scales);

    // All 48 layers
    for (int64_t l = 0; l < n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";

        // Hyper-connections (attn & ffn)
        add_row(p + "hc_attn_norm.weight", 2, 10240, 2560, 1);
        add_row(p + "hc_attn_down.weight", 4, 1638400, 2560, 320);
        add_row(p + "hc_attn_up.weight", 4, 2560, 320, 4);
        add_row(p + "hc_attn_inject.weight", 4, 20480, 4, 2560);
        add_row(p + "hc_ffn_norm.weight", 2, 10240, 2560, 1);
        add_row(p + "hc_ffn_down.weight", 4, 1638400, 2560, 320);
        add_row(p + "hc_ffn_up.weight", 4, 2560, 320, 4);
        add_row(p + "hc_ffn_inject.weight", 4, 20480, 4, 2560);

        // Router & Shared Expert
        add_row(p + "ffn_gate_inp.weight", 4, 2621440, 2560, 512);
        add_row(p + "ffn_gate_inp_shexp.weight", 4, 5120, 2560, 1);
        add_row(p + "ffn_gate_shexp.weight", 0, 256, 2560, 640);
        add_row(p + "ffn_up_shexp.weight", 0, 256, 2560, 640);
        add_row(p + "ffn_down_shexp.weight", 0, 256, 640, 2560);

        if (l % 4 != 3) {
            // GDN Recurrence layer
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
            // QSA Decode Attention layer
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

    // Write index.txt
    std::ofstream idx_file(dir / "index.txt");
    idx_file << "# align 256 pool 67108864 tensors " << rows.size() << "\n";
    for (const auto& r : rows) {
        idx_file << r << "\n";
    }
    idx_file.close();

    // Write 16 MB dense.bin of zeros (sufficient for any single tensor read from offset 0)
    std::ofstream bin_file(dir / "dense.bin", std::ios::binary);
    std::vector<char> zeros(1024 * 1024, 0);
    for (int i = 0; i < 16; ++i) {
        bin_file.write(zeros.data(), zeros.size());
    }
    bin_file.close();
}

void dummy_pool_multi(void* /*user*/, const float* /*x_f*/, const int32_t* /*ids*/,
                      int64_t /*n_tok*/, int64_t /*k*/, float* /*out*/, int64_t /*layer*/) {
    // No-op CPU pool callback for mock graph execution
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::printf("====================================================================================\n");
    std::printf(" Strata SYCL Mock End-to-End Execution Graph Test (Qwen 3.8 Flash-Next / Xe2)\n");
    std::printf("====================================================================================\n");

    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    const auto dev = q.get_device();
    std::printf("  Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("  Driver: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());

    // 1. Configure engine execution flags
    strata::core::layer_set_fast_attn(true);
    strata::core::layer_set_publish_kernel(true);
    strata::core::layer_set_fused_gdn(true);
    strata::core::layer_set_fast_select(true);
    strata::core::layer_set_fused_gr(true);
    strata::core::layer_set_native_bf16(true);
    strata::core::layer_set_native_flash_attn_short(false);
    strata::kernels::native_gdn_set_enabled(true);
    strata::kernels::native_router_set_enabled(true);
    strata::kernels::native_qsa_set_enabled(true);
    strata::kernels::native_qsa_indexer_set_enabled(true);
    strata::kernels::native_rope_set_enabled(true);

    // 2. Set environment flags for self-contained device graph execution
    setenv("STRATA_VERIFY_DEVICE_PLAN", "1", 1);
    setenv("STRATA_VERIFY_NO_HOST", "1", 1);
    setenv("STRATA_VERIFY_LEAN", "1", 1);
    setenv("STRATA_NAN_GUARD", "0", 1);

    // 3. Model Geometry (48 layers, full Qwen 3.8 Flash-Next)
    const ModelGeometry g = strata::core::kCompiledGeometry;
    const int64_t n_vocab = 4096; // Compact vocabulary projection for testing
    check(g.n_layers == 48, "Geometry verification: 48 layers (36 GDN + 12 QSA)");
    check(g.n_expert == 512, "Geometry verification: 512 MoE experts");

    // 4. Generate synthetic pack
    TempDir tmp;
    write_synthetic_pack(tmp.path, g.n_layers, n_vocab);
    check(fs::exists(tmp.path / "index.txt"), "Synthetic pack: index.txt created");
    check(fs::exists(tmp.path / "dense.bin"), "Synthetic pack: dense.bin created");

    // 5. Allocate 64 MB USM Device Pool
    constexpr uint64_t kPoolBytes = 64 * 1024 * 1024;
    void* dummy_pool = sycl::malloc_device(kPoolBytes, q);
    check(dummy_pool != nullptr, "USM device memory: 64 MB dummy weight buffer allocated");
    q.memset(dummy_pool, 0, kPoolBytes).wait();

    // 6. Load WeightTable
    WeightTable wt;
    std::string err;
    const bool loaded = wt.load(tmp.path.string(), dummy_pool, kPoolBytes, err);
    check(loaded, "WeightTable load from synthetic pack: " + (loaded ? "success" : err));

    // 7. Bind native projections to dummy_pool (GGML_TYPE_Q8_0)
    void* native_scratch = sycl::malloc_device(64 * 1024, q);
    q.memset(native_scratch, 0, 64 * 1024).wait();
    int native_bound = 0;
    const char* native_suffixes[] = {
        "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"
    };

    for (int64_t l = 0; l < g.n_layers; ++l) {
        for (const char* suf : native_suffixes) {
            std::string full_name = "blk." + std::to_string(l) + "." + suf;
            const WeightRef* r = wt.find(full_name);
            if (r) {
                WeightRef* mut_r = const_cast<WeightRef*>(r);
                mut_r->native_data = dummy_pool;
                mut_r->native_type = 8; // GGML_TYPE_Q8_0
                mut_r->native_q8_1 = native_scratch;
                ++native_bound;
            }
        }
    }
    check(native_bound > 0, "Native weight descriptors attached (" + std::to_string(native_bound) + " projections)");

    // 8. Allocate SessionState (compact 256 cells context)
    const int64_t max_cells = 256;
    const int64_t k_experts = 10;
    const uint64_t s_bytes = strata::core::session_bytes(g, max_cells, k_experts);
    void* session_base = sycl::malloc_device(s_bytes, q);
    check(session_base != nullptr, "Session memory: " + std::to_string(s_bytes / 1048576) + " MiB allocated");
    q.memset(session_base, 0, s_bytes).wait();

    SessionState ss;
    const uint64_t used = strata::core::session_init(g, max_cells, k_experts, session_base, ss);
    check(used > 0, "SessionState initialized (" + std::to_string(used / 1048576) + " MiB carved)");

    // 9. Setup VerifyHits (Resident experts routed on device)
    int32_t* d_res = sycl::malloc_device<int32_t>(g.n_layers * g.n_expert, q);
    q.memset(d_res, 0, g.n_layers * g.n_expert * sizeof(int32_t)).wait();
    uint64_t slot_off[1] = {0};

    VerifyHits hits;
    hits.d_res = d_res;
    hits.cache_base = static_cast<const uint8_t*>(dummy_pool);
    hits.slot_off = slot_off;
    hits.n_slots = 1;
    hits.blob = 4096;

    // 10. Test Execution Across Graph Configurations
    struct TestCase {
        const char* name;
        const char* graph_env;
        const char* chunk_env;
    };

    TestCase cases[] = {
        {"Monolithic Whole Graph", "whole", "1"},
        {"Chunked Graph (4 layers/chunk)", "split", "4"},
        {"Direct Execution (Baseline)", "0", "1"}
    };

    for (const auto& tc : cases) {
        std::printf("\n--- Test Mode: %s (STRATA_VERIFY_GRAPH=%s, CHUNK=%s) ---\n",
                    tc.name, tc.graph_env, tc.chunk_env);
        setenv("STRATA_VERIFY_GRAPH", tc.graph_env, 1);
        setenv("STRATA_VERIFY_GRAPH_CHUNK", tc.chunk_env, 1);

        Verifier ver;
        ver.set_head_sampling(false);

        err.clear();
        const bool init_ok = ver.init(wt, g, ss, hits, nullptr, /*max_t=*/2, err);
        check(init_ok, std::string("Verifier::init: ") + (init_ok ? "ok" : err));
        if (!init_ok) continue;

        const int T = 2;
        int32_t tokens[2] = {101, 102};
        int32_t out[2] = {-1, -1};

        // First run triggers capture & instantiation
        const auto t_cap0 = std::chrono::steady_clock::now();
        const bool run1_ok = ver.run(T, tokens, 0, dummy_pool_multi, nullptr, out, err);
        const double cap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cap0).count();
        check(run1_ok, std::string("Capture + First Window: ") + (run1_ok ? "ok" : err) + " in " + std::to_string(cap_ms) + " ms");

        // Warmup & Replay iterations
        constexpr int kWarmSteps = 5;
        const auto t_rep0 = std::chrono::steady_clock::now();
        bool replay_ok = true;
        for (int step = 0; step < kWarmSteps; ++step) {
            tokens[0] = 100 + step;
            tokens[1] = 101 + step;
            if (!ver.run(T, tokens, (step + 1) * 2, dummy_pool_multi, nullptr, out, err)) {
                replay_ok = false;
                break;
            }
        }
        const double rep_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rep0).count();
        check(replay_ok, std::string("Replay: ") + std::to_string(kWarmSteps) + " iterations in " +
              std::to_string(rep_ms) + " ms (" + std::to_string(rep_ms / kWarmSteps) + " ms/window)");
    }

    // Cleanup device resources
    sycl::free(d_res, q);
    sycl::free(session_base, q);
    sycl::free(native_scratch, q);
    sycl::free(dummy_pool, q);

    std::printf("\n====================================================================================\n");
    std::printf(" Result: %s\n", g_fail == 0 ? "ALL PASS" : "FAILURES DETECTED");
    std::printf("====================================================================================\n");

    return g_fail == 0 ? 0 : 1;
}
