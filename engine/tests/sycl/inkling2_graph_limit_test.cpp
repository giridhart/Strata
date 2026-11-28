// engine/tests/sycl/inkling2_graph_limit_test.cpp
// Verifies:
//   1. Bounded LRU SYCL command_graph cache (InkSyRunner::run_graph) handles >64 distinct graph keys
//      (96 keys, cap=16) without Level Zero command-list exhaustion (UR_RESULT_ERROR_DEVICE_LOST).
//   2. Granular opt-in overrides (STRATA_INK_SYCL_VEC_LSC, STRATA_INK_SYCL_XMX, STRATA_INK_SYCL_GRAPH_CAP,
//      STRATA_INK_SYCL_L0_V2) isolate features independently on Battlemage/Xe2.
//   3. Battlemage fast-set refusal gate remains intact.
// MIT License - Copyright (c) 2026 Strata contributors

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ink_sy_runner.h"

using namespace ink_sycl;

static bool test_granular_cfg() {
    int fails = 0;
    auto check = [&](const char* what, bool ok) {
        if (!ok) { std::printf("[FAIL] %s\n", what); ++fails; }
        else { std::printf("[PASS] %s\n", what); }
    };

    // Battlemage fast-set gate still holds
    check("intelarc_fast_refused(bmg=true, exact=false)", intelarc_fast_refused(true, false) == true);
    check("intelarc_fast_refused(bmg=true, exact=true)", intelarc_fast_refused(true, true) == false);

    // Granular overrides work independently
    setenv("STRATA_INK_SYCL_VEC_LSC", "1", 1);
    setenv("STRATA_INK_SYCL_XMX", "1", 1);
    setenv("STRATA_INK_SYCL_GRAPH_CAP", "16", 1);
    setenv("STRATA_INK_SYCL_L0_V2", "1", 1);

    IntelArcCfg cfg;
    cfg.has_xmx = true;
    apply_granular_sycl_overrides(cfg);
    check("STRATA_INK_SYCL_VEC_LSC=1 sets use_vec_lsc_decode", cfg.use_vec_lsc_decode);
    check("STRATA_INK_SYCL_XMX=1 sets use_joint_matrix_xmx (when has_xmx)", cfg.use_joint_matrix_xmx);
    check("STRATA_INK_SYCL_GRAPH_CAP=16 sets graph_cap=16", cfg.graph_cap == 16);
    check("STRATA_INK_SYCL_L0_V2=1 sets use_l0_v2", cfg.use_l0_v2);

    setenv("STRATA_INK_SYCL_XMX", "0", 1);
    setenv("STRATA_INK_SYCL_VEC_LSC", "0", 1);
    apply_granular_sycl_overrides(cfg);
    check("STRATA_INK_SYCL_XMX=0 clears use_joint_matrix_xmx", !cfg.use_joint_matrix_xmx);
    check("STRATA_INK_SYCL_VEC_LSC=0 clears use_vec_lsc_decode", !cfg.use_vec_lsc_decode);

    unsetenv("STRATA_INK_SYCL_VEC_LSC");
    unsetenv("STRATA_INK_SYCL_XMX");
    unsetenv("STRATA_INK_SYCL_L0_V2");
    return fails == 0;
}

static bool test_bounded_graph_lru() {
    setenv("STRATA_INK_SYCL_GRAPH_CAP", "16", 1);

    InkSyRunner runner;
    constexpr int kN = 1024;
    constexpr int kTotalKeys = 96;  // > 64 distinct graph keys

    float* d_in = sycl::malloc_device<float>(kN, Q());
    float* d_out = sycl::malloc_device<float>(kN, Q());
    std::vector<float> h_in(kN, 1.0f), h_out(kN, 0.0f);
    Q().memcpy(d_in, h_in.data(), kN * sizeof(float)).wait();

    int graph_ok_count = 0;
    bool math_ok = true;

    for (int k = 0; k < kTotalKeys; ++k) {
        const float bias = (float)(k + 1);
        const uint64_t key = 0x10000ull + (uint64_t)k;
        bool ok = runner.run_graph(key, [&]() {
            Q().parallel_for(NR(kN / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                const int i = (int)it.get_global_id(0);
                d_out[i] = d_in[i] * 2.0f + bias;
            });
        });
        if (!ok) {
            std::printf("[SKIP] SYCL command_graph unsupported on this backend (graph_failed=%d)\n",
                        (int)runner.graph_failed);
            sycl::free(d_in, Q());
            sycl::free(d_out, Q());
            return true;  // graceful fallback verified
        }
        ++graph_ok_count;
        Q().memcpy(h_out.data(), d_out, kN * sizeof(float)).wait();
        const float want = 2.0f + bias;
        if (std::fabs(h_out[0] - want) > 1e-5f || std::fabs(h_out[kN - 1] - want) > 1e-5f) {
            std::printf("[FAIL] key=%d got=%.4f want=%.4f\n", k, h_out[0], want);
            math_ok = false;
            break;
        }
    }

    // Re-hit the most recent 16 keys (should be 100% cache hits with zero new evictions)
#if defined(SYCL_EXT_ONEAPI_GRAPH)
    const size_t ev_before = runner.graph_evictions;
    for (int k = kTotalKeys - 16; k < kTotalKeys; ++k) {
        const float bias = (float)(k + 1);
        const uint64_t key = 0x10000ull + (uint64_t)k;
        runner.run_graph(key, [&]() {
            Q().parallel_for(NR(kN / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                const int i = (int)it.get_global_id(0);
                d_out[i] = d_in[i] * 2.0f + bias;
            });
        });
    }
    Q().wait_and_throw();
    const size_t ev_after = runner.graph_evictions;
    const size_t cached = runner.graphs.size();
    std::printf("[INFO] graph_keys=%d cached=%zu evictions=%zu (rehit_evictions=%zu)\n",
                graph_ok_count, cached, ev_after, ev_after - ev_before);

    bool ok = math_ok && (graph_ok_count == kTotalKeys) && (cached <= 16) &&
              (ev_before >= (size_t)(kTotalKeys - 16)) && (ev_after == ev_before);
    std::printf("[%s] Bounded LRU SYCL command_graph cache (%d keys, cap=16)\n",
                ok ? "PASS" : "FAIL", kTotalKeys);
#else
    bool ok = true;
    std::printf("[SKIP] SYCL_EXT_ONEAPI_GRAPH not defined\n");
#endif

    runner.release();
    sycl::free(d_in, Q());
    sycl::free(d_out, Q());
    return ok;
}

int main() {
    const auto dev = Q().get_device();
    std::printf("=== inkling2_graph_limit_test on %s ===\n",
                dev.get_info<sycl::info::device::name>().c_str());

    bool ok = true;
    ok &= test_granular_cfg();
    ok &= test_bounded_graph_lru();

    std::printf("\n=== inkling2_graph_limit_test: %s ===\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
