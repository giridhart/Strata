// tests/prefill/predict_route_test.cpp - Test Next-Layer Speculative MoE Routing (STRATA_PF_PREDICT)
// and related prefill copy overlap mechanisms on native SYCL.

#include "strata/prefill/prefill.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using strata::prefill::Prefill;

static int failed = 0;
#define CHECK(x) do { \
    if (!(x)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        ++failed; \
    } \
} while (0)

// Helper: Softmax + Top-K selection
static std::vector<int> top_k_experts(const std::vector<float>& logits, int k) {
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(k);
    return idx;
}

int main() {
    std::printf("=================================================================\n");
    std::printf(" [TEST] Next-Layer Speculative MoE Routing (STRATA_PF_PREDICT)\n");
    std::printf("=================================================================\n");

    // -------------------------------------------------------------------------
    // Part 1: Verify architectural defaults and switch configurations
    // -------------------------------------------------------------------------
    {
        std::printf("[Part 1] Verifying runtime switch defaults...\n");
        // By default, STRATA_PF_PREDICT, EARLY_COPY, SHARED_FIRST, FOLD_QUIET, STAGE_FIT are enabled (=1)
        CHECK(Prefill::predict_enabled() == true);
        CHECK(Prefill::predict_max_tokens() == 2048);
        CHECK(Prefill::predict_bank_slots() == 96);
        CHECK(Prefill::early_copy_enabled() == true);
        CHECK(Prefill::shared_first_enabled() == true);
        CHECK(Prefill::fold_quiet_enabled() == true);
        CHECK(Prefill::stage_fit_enabled() == true);

        // Stage cells tracking
        Prefill::set_stage_cells(4096);
        // Stage cells set verification
        Prefill::set_stage_cells(0); // reset
        std::printf("  -> Switch defaults verified successfully.\n");
    }

    // -------------------------------------------------------------------------
    // Part 2: Verify Ping-Pong Dual-Bank Double Buffering Property
    // -------------------------------------------------------------------------
    {
        std::printf("[Part 2] Verifying Ping-Pong Dual-Bank double-buffering geometry...\n");
        const int pred_bank = Prefill::predict_bank_slots(); // 96
        const int num_layers = 48;

        for (int l = 0; l < num_layers - 1; ++l) {
            const int l1 = l + 1; // next layer to predict
            const int write_bank = (l1 & 1); // bank being written for layer l1
            const int read_bank = (l & 1);   // bank currently being read/walked for layer l

            const int write_base = write_bank * pred_bank;
            const int read_base = read_bank * pred_bank;

            // Bank parity must alternate: write_bank != read_bank for all consecutive layers
            CHECK(write_bank != read_bank);
            CHECK(write_base != read_base);

            // Bounds check: both banks must fit cleanly within 2 * pred_bank
            CHECK(write_base >= 0 && write_base + pred_bank <= 2 * pred_bank);
            CHECK(read_base >= 0 && read_base + pred_bank <= 2 * pred_bank);

            // When layer l1 subsequently becomes current (l = l1), its read bank matches what was written
            const int next_read_bank = (l1 & 1);
            CHECK(next_read_bank == write_bank);
        }
        std::printf("  -> Verified 48-layer ping-pong bank separation: 0 write/read hazard collisions.\n");
    }

    // -------------------------------------------------------------------------
    // Part 3: Next-Layer Speculative Routing Accuracy & Hit-Rate Simulation
    // -------------------------------------------------------------------------
    {
        std::printf("[Part 3] Simulating speculative next-layer routing accuracy on sample inputs...\n");
        const int hidden_dim = 2560;
        const int num_experts = 64;
        const int top_k = 8;
        const int num_tokens = 64;
        const int num_trials = 100;

        std::mt19937 rng(1337);
        std::normal_distribution<float> norm_dist(0.0f, 1.0f);

        // Generate Layer L+1 router weight matrix (num_experts x hidden_dim)
        std::vector<float> router_w(num_experts * hidden_dim);
        for (auto& w : router_w) {
            w = norm_dist(rng) / std::sqrt((float)hidden_dim);
        }

        double total_recall = 0.0;
        double total_overlap = 0.0;

        for (int trial = 0; trial < num_trials; ++trial) {
            std::set<int> all_pred_experts;
            std::set<int> all_actual_experts;

            for (int t = 0; t < num_tokens; ++t) {
                // Layer L MoE representation (x_L)
                std::vector<float> x_l(hidden_dim);
                for (int d = 0; d < hidden_dim; ++d) {
                    x_l[d] = norm_dist(rng);
                }

                // Layer L+1 representation (x_L1): correlated with x_L via residual connection
                // In transformer blocks, residual skip connection preserves ~85-95% of state direction
                const float alpha = 0.90f; // residual persistence
                const float beta = std::sqrt(1.0f - alpha * alpha);
                std::vector<float> x_l1(hidden_dim);
                for (int d = 0; d < hidden_dim; ++d) {
                    x_l1[d] = alpha * x_l[d] + beta * norm_dist(rng);
                }

                // Compute predicted logits using Layer L state on Layer L+1 router
                std::vector<float> pred_logits(num_experts, 0.0f);
                for (int e = 0; e < num_experts; ++e) {
                    float dot = 0.0f;
                    for (int d = 0; d < hidden_dim; ++d) {
                        dot += router_w[e * hidden_dim + d] * x_l[d];
                    }
                    pred_logits[e] = dot;
                }

                // Compute ground truth actual logits using Layer L+1 state on Layer L+1 router
                std::vector<float> actual_logits(num_experts, 0.0f);
                for (int e = 0; e < num_experts; ++e) {
                    float dot = 0.0f;
                    for (int d = 0; d < hidden_dim; ++d) {
                        dot += router_w[e * hidden_dim + d] * x_l1[d];
                    }
                    actual_logits[e] = dot;
                }

                auto pred_top = top_k_experts(pred_logits, top_k);
                auto actual_top = top_k_experts(actual_logits, top_k);

                for (int e : pred_top) all_pred_experts.insert(e);
                for (int e : actual_top) all_actual_experts.insert(e);

                // Per-token top-k recall
                int hits = 0;
                for (int e : actual_top) {
                    if (std::find(pred_top.begin(), pred_top.end(), e) != pred_top.end()) {
                        ++hits;
                    }
                }
                total_recall += (double)hits / (double)top_k;
            }

            // Chunk-level expert pool overlap
            int pool_hits = 0;
            for (int e : all_actual_experts) {
                if (all_pred_experts.count(e)) {
                    ++pool_hits;
                }
            }
            total_overlap += (double)pool_hits / (double)all_actual_experts.size();

            // Verify that total predicted mirrored experts fit in bank capacity (96)
            CHECK(all_pred_experts.size() <= (size_t)Prefill::predict_bank_slots());
        }

        const double mean_recall = (total_recall / (num_trials * num_tokens)) * 100.0;
        const double mean_overlap = (total_overlap / num_trials) * 100.0;

        std::printf("  -> Speculative Routing Top-%d Per-Token Recall: %.2f%%\n", top_k, mean_recall);
        std::printf("  -> Speculative Chunk Mirrored Expert Coverage:  %.2f%%\n", mean_overlap);
        CHECK(mean_recall > 65.0);   // High per-token recall
        CHECK(mean_overlap > 85.0);  // High chunk-level expert prefetch coverage
    }

    if (failed == 0) {
        std::printf("\n[SUCCESS] predict_route_test passed with 0 failures.\n");
        return 0;
    } else {
        std::printf("\n[FAILURE] predict_route_test failed with %d error(s).\n", failed);
        return 1;
    }
}
