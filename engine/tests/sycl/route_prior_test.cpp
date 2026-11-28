// tests/sycl/route_prior_test.cpp - Test for STRATA_ROUTE_PRIOR defaults (B580 12GB = 1.5f)
// and SYCL route_prior_top10 kernel correctness against CPU oracle.

#include "strata/core/defaults_026.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/kernels/route_prior.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {
using namespace strata::kernels;

constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

void cpu_route_prior_ref(const float* logits, const int32_t* res, float lambda,
                         int32_t* ids, float* weights, int n_tok) {
    for (int t = 0; t < n_tok; ++t) {
        const float* lg = logits + (size_t)t * 512;
        int32_t* id = ids + (size_t)t * 10;
        float* w = weights + (size_t)t * 10;

        float mx = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < 512; ++i) {
            mx = std::max(mx, lg[i]);
        }

        struct Cand {
            int e;
            float score;
            float prob;
        };
        std::vector<Cand> cands(512);
        for (int e = 0; e < 512; ++e) {
            cands[e].e = e;
            cands[e].score = lg[e] + (res[e] >= 0 ? lambda : 0.0f);
            cands[e].prob = std::exp(lg[e] - mx);
        }

        std::vector<Cand> working = cands;
        float sum_p = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            int best_idx = 0;
            for (size_t j = 1; j < working.size(); ++j) {
                if (working[j].score > working[best_idx].score ||
                    (working[j].score == working[best_idx].score && working[j].e < working[best_idx].e)) {
                    best_idx = (int)j;
                }
            }
            id[rank] = working[best_idx].e;
            w[rank] = working[best_idx].prob;
            sum_p += working[best_idx].prob;
            working[best_idx].score = -std::numeric_limits<float>::infinity();
        }

        for (int rank = 0; rank < 10; ++rank) {
            w[rank] /= std::max(sum_p, 1e-30f);
        }
    }
}

}  // namespace

int main() {
    int bad = 0;

    // Part 1: Verify architectural defaults for Intel Arc B580 and other SYCL GPUs
    {
        // route_prior_default() is 0.0f on all architectures by default (off by default per upstream #4402)
        float b580_default = strata::route_prior_default(false, false, 12.0 * GiB, false);
        if (b580_default != 0.0f) {
            std::printf("FAIL: Arc B580 12GB route_prior_default expected 0.0f, got %.2f\n", b580_default);
            ++bad;
        }

        // Opt-in resolution (STRATA_ROUTE_PRIOR=1.5) provides kRoutePriorSmallCard (1.5f)
        float b580_opt_in = strata::route_prior_resolve("1.5", false, false, 12.0 * GiB, false);
        if (b580_opt_in != strata::kRoutePriorSmallCard) {
            std::printf("FAIL: Arc B580 12GB route_prior_resolve expected %.2f, got %.2f\n",
                        strata::kRoutePriorSmallCard, b580_opt_in);
            ++bad;
        }

        // Arc A770: Discrete GPU with 16.0 GiB global memory -> should default to 0.0f
        float a770_default = strata::route_prior_default(false, false, 16.0 * GiB, false);
        if (a770_default != 0.0f) {
            std::printf("FAIL: Arc A770 16GB route_prior_default expected 0.0f, got %.2f\n", a770_default);
            ++bad;
        }

        // Lunar Lake: Integrated Xe2 GPU (shared memory) -> should default to 0.0f
        float lnl_default = strata::route_prior_default(false, true, 8.0 * GiB, false);
        if (lnl_default != 0.0f) {
            std::printf("FAIL: Lunar Lake integrated route_prior_default expected 0.0f, got %.2f\n", lnl_default);
            ++bad;
        }

        // Under STRATA_DEFAULTS=026 master flag -> must be 0.0f
        float v026_default = strata::route_prior_default(false, false, 12.0 * GiB, true);
        if (v026_default != 0.0f) {
            std::printf("FAIL: v026 route_prior_default expected 0.0f, got %.2f\n", v026_default);
            ++bad;
        }
    }

    // Part 2: Verify SYCL route_prior_top10 kernel on device
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running route_prior_top10 test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int n_tok = 16;
    const int n_expert = 512;
    const int k = 10;
    const float lambda = 1.5f;

    std::mt19937 rng(42);
    std::normal_distribution<float> logit_dist(0.0f, 2.5f);

    std::vector<float> h_logits(n_tok * n_expert);
    for (auto& x : h_logits) x = logit_dist(rng);

    // Mock resident experts: ~128 experts cached in VRAM
    std::vector<int32_t> h_res(n_expert, -1);
    for (int e = 0; e < n_expert; ++e) {
        if (rng() % 4 == 0) {
            h_res[e] = (int32_t)(rng() % 128);
        }
    }

    std::vector<int32_t> ref_ids(n_tok * k);
    std::vector<float> ref_weights(n_tok * k);
    cpu_route_prior_ref(h_logits.data(), h_res.data(), lambda, ref_ids.data(), ref_weights.data(), n_tok);

    float* d_logits = sycl::malloc_device<float>(h_logits.size(), q);
    int32_t* d_res = sycl::malloc_device<int32_t>(h_res.size(), q);
    int32_t* d_ids = sycl::malloc_device<int32_t>(n_tok * k, q);
    float* d_weights = sycl::malloc_device<float>(n_tok * k, q);

    if (!d_logits || !d_res || !d_ids || !d_weights) {
        std::printf("Allocation failed\n");
        return 1;
    }

    q.memcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float));
    q.memcpy(d_res, h_res.data(), h_res.size() * sizeof(int32_t));
    q.wait();

    route_prior_top10(d_logits, d_res, lambda, d_ids, d_weights, n_tok, &q);
    q.wait();

    std::vector<int32_t> dev_ids(n_tok * k);
    std::vector<float> dev_weights(n_tok * k);
    q.memcpy(dev_ids.data(), d_ids, dev_ids.size() * sizeof(int32_t));
    q.memcpy(dev_weights.data(), d_weights, dev_weights.size() * sizeof(float));
    q.wait();

    sycl::free(d_logits, q);
    sycl::free(d_res, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);

    int id_diffs = 0;
    int weight_diffs = 0;
    for (int i = 0; i < n_tok * k; ++i) {
        if (dev_ids[i] != ref_ids[i]) {
            ++id_diffs;
        }
        if (std::fabs(dev_weights[i] - ref_weights[i]) > 1e-4f) {
            ++weight_diffs;
        }
    }

    std::printf("  route_prior_top10 (lambda=1.5): %d id diffs, %d weight diffs across %d entries\n",
                id_diffs, weight_diffs, n_tok * k);
    if (id_diffs > 0 || weight_diffs > 0) {
        ++bad;
    }

    std::printf("route_prior_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
