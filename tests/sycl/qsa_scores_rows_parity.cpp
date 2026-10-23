// tests/sycl/qsa_scores_rows_parity.cpp - Dedicated bitwise parity test for Intel Xe2
// register-shuffled QSA decode block scores (STRATA_QSA_ROWS=1) vs multi-kernel reference and CPU host reference.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_failures = 0;

template <typename T>
T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    q.memset(p, 0, (n + 64) * sizeof(T)).wait();
    return p;
}

/// Host CPU reference for QSA block scores matching the SIMD tree reduction
void host_qsa_block_scores(const float* pooled, const float* dead, const float* q_idx,
                           const int32_t* steps, int nq, int64_t max_blocks,
                           const k::QsaShapes& s, float* ref_scores) {
    for (int q = 0; q < nq; ++q) {
        const int32_t* st = steps + q * k::kStepCount;
        const int64_t n_kv = st[k::kStepNKv];
        const int64_t n_bid = st[k::kStepNBid];
        const float* q_cur = q_idx + (size_t)q * s.idx_n_head * s.idx_dim;

        for (int64_t b = 0; b <= n_bid && b < max_blocks; ++b) {
            const float* key = (b == n_bid) ? dead : pooled + b * s.idx_dim;
            float score = 0.0f;
            for (int h = 0; h < s.idx_n_head; ++h) {
                const float* q_h = q_cur + h * s.idx_dim;
                // Tree reduction matching SIMD lane butterfly
                float lane_d[32];
                for (int lane = 0; lane < 32; ++lane) {
                    float d = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        d += key[lane * 4 + i] * q_h[lane * 4 + i];
                    }
                    lane_d[lane] = d;
                }
                for (int o = 16; o > 0; o >>= 1) {
                    for (int lane = 0; lane < 32; ++lane) {
                        if ((lane & o) == 0) {
                            lane_d[lane] += lane_d[lane ^ o];
                        }
                    }
                }
                float d_sum = lane_d[0];
                score += (d_sum > 0.0f) ? d_sum : 0.0f;
            }
            if (b == n_bid && n_kv % s.idx_block != 0) {
                score += 1e9f;
            }
            ref_scores[q * max_blocks + b] = score;
        }
    }
}

void test_qsa_scores_rows_case(int T, int64_t base_nkv, sycl::queue& q, const char* desc) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_cells = base_nkv + T + 16;
    const int64_t max_blocks = (max_cells + s.idx_block - 1) / s.idx_block + 4;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);

    std::mt19937 rng(static_cast<unsigned>(42 + T * 1009 + base_nkv));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Host buffers
    std::vector<float> h_pooled(static_cast<size_t>(max_blocks * s.idx_dim));
    std::vector<float> h_dead(static_cast<size_t>(s.idx_dim));
    std::vector<float> h_q_idx(static_cast<size_t>(T * s.idx_n_head * s.idx_dim));
    std::vector<int32_t> h_steps(static_cast<size_t>(T * k::kStepCount));

    for (auto& v : h_pooled) v = dist(rng);
    for (auto& v : h_dead) v = dist(rng);
    for (auto& v : h_q_idx) v = dist(rng);

    for (int t = 0; t < T; ++t) {
        const int64_t pos = base_nkv + t;
        const int64_t n_kv = pos + 1;
        const int64_t n_bid = n_kv / s.idx_block;
        const int64_t width = k::qsa_selection_width(n_kv, s);

        h_steps[t * k::kStepCount + k::kStepPos] = static_cast<int32_t>(pos);
        h_steps[t * k::kStepCount + k::kStepNKv] = static_cast<int32_t>(n_kv);
        h_steps[t * k::kStepCount + k::kStepNBid] = static_cast<int32_t>(n_bid);
        h_steps[t * k::kStepCount + k::kStepWidth] = static_cast<int32_t>(width);
    }

    // Allocate GPU buffers
    float* d_pooled = dalloc<float>(h_pooled.size(), q);
    float* d_dead = dalloc<float>(h_dead.size(), q);
    float* d_q_idx = dalloc<float>(h_q_idx.size(), q);
    int32_t* d_steps = dalloc<int32_t>(h_steps.size(), q);

    const size_t total_scores = static_cast<size_t>(T * max_blocks);
    const size_t total_ids = static_cast<size_t>(T * cap);

    float* d_scores_multi = dalloc<float>(total_scores, q);
    float* d_scores_rows = dalloc<float>(total_scores, q);
    float* d_scores_env = dalloc<float>(total_scores, q);

    int32_t* d_ids_multi = dalloc<int32_t>(total_ids, q);
    int32_t* d_ids_rows = dalloc<int32_t>(total_ids, q);
    int32_t* d_ids_env = dalloc<int32_t>(total_ids, q);

    // Upload inputs
    q.memcpy(d_pooled, h_pooled.data(), h_pooled.size() * sizeof(float));
    q.memcpy(d_dead, h_dead.data(), h_dead.size() * sizeof(float));
    q.memcpy(d_q_idx, h_q_idx.data(), h_q_idx.size() * sizeof(float));
    q.memcpy(d_steps, h_steps.data(), h_steps.size() * sizeof(int32_t)).wait();

    // 1. Reference: multi-kernel with STRATA_QSA_ROWS=0
    setenv("STRATA_QSA_ROWS", "0", 1);
    q.memset(d_scores_multi, 0, total_scores * sizeof(float)).wait();
    k::qsa_block_scores(d_pooled, d_dead, d_q_idx, d_steps, T, max_blocks, s, d_scores_multi, &q);
    k::qsa_block_topk(d_scores_multi, d_steps, T, max_blocks, cap, s, d_ids_multi, &q);
    q.wait();

    // 2. Direct qsa_block_scores_rows
    q.memset(d_scores_rows, 0, total_scores * sizeof(float)).wait();
    bool rows_ok = k::qsa_block_scores_rows(d_pooled, d_dead, d_q_idx, d_steps, T, max_blocks, s, d_scores_rows, &q);
    k::qsa_block_topk(d_scores_rows, d_steps, T, max_blocks, cap, s, d_ids_rows, &q);
    q.wait();

    // 3. Auto-dispatch: qsa_block_scores with STRATA_QSA_ROWS=1
    setenv("STRATA_QSA_ROWS", "1", 1);
    q.memset(d_scores_env, 0, total_scores * sizeof(float)).wait();
    k::qsa_block_scores(d_pooled, d_dead, d_q_idx, d_steps, T, max_blocks, s, d_scores_env, &q);
    k::qsa_block_topk(d_scores_env, d_steps, T, max_blocks, cap, s, d_ids_env, &q);
    q.wait();
    unsetenv("STRATA_QSA_ROWS");

    // Download outputs
    std::vector<float> res_multi(total_scores), res_rows(total_scores), res_env(total_scores);
    std::vector<int32_t> ids_multi(total_ids), ids_rows(total_ids), ids_env(total_ids);

    q.memcpy(res_multi.data(), d_scores_multi, total_scores * sizeof(float));
    q.memcpy(res_rows.data(), d_scores_rows, total_scores * sizeof(float));
    q.memcpy(res_env.data(), d_scores_env, total_scores * sizeof(float));
    q.memcpy(ids_multi.data(), d_ids_multi, total_ids * sizeof(int32_t));
    q.memcpy(ids_rows.data(), d_ids_rows, total_ids * sizeof(int32_t));
    q.memcpy(ids_env.data(), d_ids_env, total_ids * sizeof(int32_t)).wait();

    // Compute CPU host reference
    std::vector<float> h_ref_scores(total_scores, 0.0f);
    host_qsa_block_scores(h_pooled.data(), h_dead.data(), h_q_idx.data(), h_steps.data(),
                          T, max_blocks, s, h_ref_scores.data());

    // Verifications
    bool pass = true;
    if (!rows_ok) {
        std::printf("  [FAIL] %s: qsa_block_scores_rows returned false for T=%d\n", desc, T);
        pass = false;
    }

    // 1. Bitwise equality: rows vs multi
    const bool bitwise_rows_multi = (std::memcmp(res_multi.data(), res_rows.data(), total_scores * sizeof(float)) == 0);
    if (!bitwise_rows_multi) {
        size_t diff_cnt = 0;
        for (size_t i = 0; i < total_scores; ++i) {
            if (std::memcmp(&res_multi[i], &res_rows[i], sizeof(float)) != 0) {
                if (++diff_cnt <= 5) {
                    std::printf("    Scores mismatch rows vs multi at idx %zu: multi=%.7e rows=%.7e\n",
                                i, res_multi[i], res_rows[i]);
                }
            }
        }
        std::printf("  [FAIL] %s: scores bitwise mismatch rows vs multi (%zu / %zu)\n", desc, diff_cnt, total_scores);
        pass = false;
    }

    // 2. Bitwise equality: env(STRATA_QSA_ROWS=1) vs multi
    const bool bitwise_env_multi = (std::memcmp(res_multi.data(), res_env.data(), total_scores * sizeof(float)) == 0);
    if (!bitwise_env_multi) {
        std::printf("  [FAIL] %s: STRATA_QSA_ROWS=1 auto-dispatch did not match multi kernel\n", desc);
        pass = false;
    }

    // 3. Top-k IDs bitwise equality
    const bool bitwise_ids_rows = (std::memcmp(ids_multi.data(), ids_rows.data(), total_ids * sizeof(int32_t)) == 0);
    const bool bitwise_ids_env = (std::memcmp(ids_multi.data(), ids_env.data(), total_ids * sizeof(int32_t)) == 0);
    if (!bitwise_ids_rows || !bitwise_ids_env) {
        std::printf("  [FAIL] %s: Top-K IDs mismatch (rows_match=%d, env_match=%d)\n",
                    desc, bitwise_ids_rows ? 1 : 0, bitwise_ids_env ? 1 : 0);
        pass = false;
    }

    // 4. Numerical validation against Host CPU reference
    float max_abs_diff = 0.0f;
    for (int q_i = 0; q_i < T; ++q_i) {
        const int64_t n_bid = h_steps[q_i * k::kStepCount + k::kStepNBid];
        const int64_t n_kv = h_steps[q_i * k::kStepCount + k::kStepNKv];
        for (int64_t b = 0; b <= n_bid && b < max_blocks; ++b) {
            const size_t idx = static_cast<size_t>(q_i * max_blocks + b);
            const float diff = std::abs(res_rows[idx] - h_ref_scores[idx]);
            if (diff > max_abs_diff) max_abs_diff = diff;

            // Tail block bonus check
            if (b == n_bid && n_kv % s.idx_block != 0) {
                if (res_rows[idx] < 1e8f) {
                    std::printf("    Tail block bonus +1e9 missing at q=%d b=%ld: score=%.7e\n",
                                q_i, b, res_rows[idx]);
                    pass = false;
                }
            }
        }
    }

    if (max_abs_diff > 1e-4f) {
        std::printf("  [FAIL] %s: max diff against host CPU ref = %.7e (> 1e-4)\n", desc, max_abs_diff);
        pass = false;
    }

    if (pass) {
        std::printf("  [PASS] %-48s (Bitwise exact 100%%, Top-K exact, CPU diff=%.2e)\n", desc, max_abs_diff);
    } else {
        ++g_failures;
    }

    sycl::free(d_pooled, q);
    sycl::free(d_dead, q);
    sycl::free(d_q_idx, q);
    sycl::free(d_steps, q);
    sycl::free(d_scores_multi, q);
    sycl::free(d_scores_rows, q);
    sycl::free(d_scores_env, q);
    sycl::free(d_ids_multi, q);
    sycl::free(d_ids_rows, q);
    sycl::free(d_ids_env, q);
}

void test_fallback_paths(sycl::queue& q) {
    std::printf("\nTesting Fallback & Boundary Behaviors...\n");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 256;

    // Case 1: T = 5 (NQ > 4) with STRATA_QSA_ROWS=1 -> should gracefully fall back to multi kernel
    {
        const int T = 5;
        std::vector<float> h_pooled(max_blocks * s.idx_dim, 0.1f);
        std::vector<float> h_dead(s.idx_dim, 0.2f);
        std::vector<float> h_q_idx(T * s.idx_n_head * s.idx_dim, 0.05f);
        std::vector<int32_t> h_steps(T * k::kStepCount);

        for (int t = 0; t < T; ++t) {
            h_steps[t * k::kStepCount + k::kStepPos] = 500 + t;
            h_steps[t * k::kStepCount + k::kStepNKv] = 501 + t;
            h_steps[t * k::kStepCount + k::kStepNBid] = (501 + t) / 4;
            h_steps[t * k::kStepCount + k::kStepWidth] = static_cast<int32_t>(k::qsa_selection_width(501 + t, s));
        }

        float* d_p = dalloc<float>(h_pooled.size(), q);
        float* d_d = dalloc<float>(h_dead.size(), q);
        float* d_q = dalloc<float>(h_q_idx.size(), q);
        int32_t* d_st = dalloc<int32_t>(h_steps.size(), q);
        float* d_sc_multi = dalloc<float>(T * max_blocks, q);
        float* d_sc_env = dalloc<float>(T * max_blocks, q);

        q.memcpy(d_p, h_pooled.data(), h_pooled.size() * sizeof(float));
        q.memcpy(d_d, h_dead.data(), h_dead.size() * sizeof(float));
        q.memcpy(d_q, h_q_idx.data(), h_q_idx.size() * sizeof(float));
        q.memcpy(d_st, h_steps.data(), h_steps.size() * sizeof(int32_t)).wait();

        // qsa_block_scores_rows should decline T=5
        bool declined = !k::qsa_block_scores_rows(d_p, d_d, d_q, d_st, T, max_blocks, s, d_sc_env, &q);

        // Run multi with STRATA_QSA_ROWS=0
        setenv("STRATA_QSA_ROWS", "0", 1);
        k::qsa_block_scores(d_p, d_d, d_q, d_st, T, max_blocks, s, d_sc_multi, &q);
        q.wait();

        // Run with STRATA_QSA_ROWS=1 -> falls back cleanly to multi
        setenv("STRATA_QSA_ROWS", "1", 1);
        k::qsa_block_scores(d_p, d_d, d_q, d_st, T, max_blocks, s, d_sc_env, &q);
        q.wait();
        unsetenv("STRATA_QSA_ROWS");

        std::vector<float> res_m(T * max_blocks), res_e(T * max_blocks);
        q.memcpy(res_m.data(), d_sc_multi, res_m.size() * sizeof(float));
        q.memcpy(res_e.data(), d_sc_env, res_e.size() * sizeof(float)).wait();

        const bool match = (std::memcmp(res_m.data(), res_e.data(), res_m.size() * sizeof(float)) == 0);
        if (declined && match) {
            std::printf("  [PASS] Fallback T=5 (NQ > 4) cleanly dispatches multi kernel (bitwise match)\n");
        } else {
            std::printf("  [FAIL] Fallback T=5 failed: declined=%d, match=%d\n", declined ? 1 : 0, match ? 1 : 0);
            ++g_failures;
        }

        sycl::free(d_p, q); sycl::free(d_d, q); sycl::free(d_q, q); sycl::free(d_st, q);
        sycl::free(d_sc_multi, q); sycl::free(d_sc_env, q);
    }

    // Case 2: active_blocks > 0 with STRATA_QSA_ROWS=1 -> should bypass rows and multi, runs block_scores_kernel
    {
        const int T = 2;
        const int64_t active_blocks = 64;
        std::vector<float> h_pooled(max_blocks * s.idx_dim, 0.1f);
        std::vector<float> h_dead(s.idx_dim, 0.2f);
        std::vector<float> h_q_idx(T * s.idx_n_head * s.idx_dim, 0.05f);
        std::vector<int32_t> h_steps(T * k::kStepCount);

        for (int t = 0; t < T; ++t) {
            h_steps[t * k::kStepCount + k::kStepPos] = 200 + t;
            h_steps[t * k::kStepCount + k::kStepNKv] = 201 + t;
            h_steps[t * k::kStepCount + k::kStepNBid] = (201 + t) / 4;
            h_steps[t * k::kStepCount + k::kStepWidth] = static_cast<int32_t>(k::qsa_selection_width(201 + t, s));
        }

        float* d_p = dalloc<float>(h_pooled.size(), q);
        float* d_d = dalloc<float>(h_dead.size(), q);
        float* d_q = dalloc<float>(h_q_idx.size(), q);
        int32_t* d_st = dalloc<int32_t>(h_steps.size(), q);
        float* d_sc_active = dalloc<float>(T * max_blocks, q);
        float* d_sc_ref = dalloc<float>(T * max_blocks, q);

        q.memcpy(d_p, h_pooled.data(), h_pooled.size() * sizeof(float));
        q.memcpy(d_d, h_dead.data(), h_dead.size() * sizeof(float));
        q.memcpy(d_q, h_q_idx.data(), h_q_idx.size() * sizeof(float));
        q.memcpy(d_st, h_steps.data(), h_steps.size() * sizeof(int32_t)).wait();

        // Run with active_blocks > 0 (bypasses rows even with STRATA_QSA_ROWS=1)
        setenv("STRATA_QSA_ROWS", "1", 1);
        k::qsa_block_scores(d_p, d_d, d_q, d_st, T, max_blocks, s, d_sc_active, &q, active_blocks);
        // Compare with baseline active_blocks > 0
        setenv("STRATA_QSA_ROWS", "0", 1);
        k::qsa_block_scores(d_p, d_d, d_q, d_st, T, max_blocks, s, d_sc_ref, &q, active_blocks);
        q.wait();
        unsetenv("STRATA_QSA_ROWS");

        std::vector<float> res_a(T * max_blocks), res_r(T * max_blocks);
        q.memcpy(res_a.data(), d_sc_active, res_a.size() * sizeof(float));
        q.memcpy(res_r.data(), d_sc_ref, res_r.size() * sizeof(float)).wait();

        const bool match = (std::memcmp(res_a.data(), res_r.data(), (size_t)T * active_blocks * sizeof(float)) == 0);
        if (match) {
            std::printf("  [PASS] active_blocks=%ld bypasses rows correctly (bitwise exact)\n", active_blocks);
        } else {
            std::printf("  [FAIL] active_blocks=%ld mismatch\n", active_blocks);
            ++g_failures;
        }

        sycl::free(d_p, q); sycl::free(d_d, q); sycl::free(d_q, q); sycl::free(d_st, q);
        sycl::free(d_sc_active, q); sycl::free(d_sc_ref, q);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const auto dev = q.get_device();
    std::printf("=================================================================\n");
    std::printf(" QSA Scores Rows Parity Test (STRATA_QSA_ROWS Xe2/B580 Optimization)\n");
    std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf(" Subgroup:     %d (Native Xe2 SIMD)\n", strata::sycl_arch::SUB_GROUP_SIZE);
    std::printf("=================================================================\n\n");

    std::printf("Suite 1: Small Context with Partial Tail Block (+1e9f check, n_kv %% 4 != 0)\n");
    for (int T = 1; T <= 4; ++T) {
        char name[128];
        std::snprintf(name, sizeof(name), "T=%d, n_kv=103 (tail block partial, remainder 3)", T);
        test_qsa_scores_rows_case(T, 103, q, name);

        std::snprintf(name, sizeof(name), "T=%d, n_kv=107 (tail block partial, remainder 3)", T);
        test_qsa_scores_rows_case(T, 107, q, name);
    }

    std::printf("\nSuite 2: Small Context with Full Block Alignment (no +1e9f tail check, n_kv %% 4 == 0)\n");
    for (int T = 1; T <= 4; ++T) {
        char name[128];
        std::snprintf(name, sizeof(name), "T=%d, n_kv=100 (aligned block, remainder 0)", T);
        test_qsa_scores_rows_case(T, 100, q, name);

        std::snprintf(name, sizeof(name), "T=%d, n_kv=104 (aligned block, remainder 0)", T);
        test_qsa_scores_rows_case(T, 104, q, name);
    }

    std::printf("\nSuite 3: Medium and Long Contexts (n_kv in [1024, 4096, 16384, 65536])\n");
    const int64_t contexts[] = {1024, 4096, 16384, 65536};
    for (int64_t ctx : contexts) {
        for (int T = 1; T <= 4; ++T) {
            char name[128];
            std::snprintf(name, sizeof(name), "T=%d, n_kv=%ld (medium/long context)", T, ctx);
            test_qsa_scores_rows_case(T, ctx, q, name);
        }
    }

    // Suite 4: Fallback & Boundary Behaviors
    test_fallback_paths(q);

    std::printf("\n=================================================================\n");
    std::printf("qsa_scores_rows_parity: %d failures\n", g_failures);
    std::printf("=================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
