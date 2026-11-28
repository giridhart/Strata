// Parity test for the SYCL implementation of the native router.
#include <sycl/sycl.hpp>
#include "strata/kernels/native_router.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr double RENORM_CLAMP = 6.103515625e-05;

void reference(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e)
        mx = std::max(mx, (double)logits[e]);
    double sum = 0;
    std::vector<double> p((size_t)n_expert);
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t)e] = std::exp((double)logits[e] - mx);
        sum += p[(size_t)e];
    }
    for (int e = 0; e < n_expert; ++e)
        p[(size_t)e] /= sum;

    std::vector<int> idx((size_t)n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return p[(size_t)a] > p[(size_t)b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[(size_t)i];
        w[i] = (float)p[(size_t)idx[(size_t)i]];
        s += p[(size_t)idx[(size_t)i]];
    }
    s = std::max(s, RENORM_CLAMP);
    for (int i = 0; i < k; ++i)
        w[i] = (float)((double)w[i] / s);
}

int test_native_router(sycl::queue& q, int n_tokens) {
    constexpr int n_expert = 512, k = 10;
    std::vector<float> logits((size_t)n_tokens * n_expert);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = dist(rng);
    }
    // Add ties in first token
    logits[10] = 5.0f;
    logits[20] = 5.0f;
    logits[30] = 5.0f;

    std::vector<int> ref_ids((size_t)n_tokens * k);
    std::vector<float> ref_w((size_t)n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t)t * n_expert], n_expert, k, &ref_ids[(size_t)t * k], &ref_w[(size_t)t * k]);
    }

    float* d_l = sycl::malloc_device<float>(logits.size(), q);
    int32_t* d_ids = sycl::malloc_device<int32_t>((size_t)n_tokens * k, q);
    float* d_w = sycl::malloc_device<float>((size_t)n_tokens * k, q);

    q.memcpy(d_l, logits.data(), logits.size() * sizeof(float)).wait();

    if (n_tokens == 1) {
        strata::kernels::native_router_top10(d_l, d_ids, d_w, &q);
    } else {
        strata::kernels::native_router_top10_multi(d_l, d_ids, d_w, n_tokens, &q);
    }
    q.wait();

    std::vector<int32_t> act_ids((size_t)n_tokens * k);
    std::vector<float> act_w((size_t)n_tokens * k);
    q.memcpy(act_ids.data(), d_ids, act_ids.size() * sizeof(int32_t)).wait();
    q.memcpy(act_w.data(), d_w, act_w.size() * sizeof(float)).wait();

    sycl::free(d_l, q);
    sycl::free(d_ids, q);
    sycl::free(d_w, q);

    int id_bad = 0, w_bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < act_ids.size(); ++i) {
        if (act_ids[i] != ref_ids[i]) ++id_bad;
        double rel = std::fabs((double)act_w[i] - (double)ref_w[i]) /
                     (std::fabs((double)ref_w[i]) > 1e-30 ? std::fabs((double)ref_w[i]) : 1e-30);
        worst = std::max(worst, rel);
        if (rel > 1e-3) ++w_bad;
    }

    std::printf("  n_tokens=%d: ids %s (%d bad) weights worst rel %.3e (%d over tol)\n", n_tokens,
                id_bad ? "*** WRONG ***" : "exact", id_bad, worst, w_bad);
    return id_bad + w_bad;
}
int test_native_route_resident(sycl::queue& q) {
    constexpr int n_tokens = 4;
    constexpr int n_expert = 512, k = 10;
    std::vector<float> logits((size_t)n_tokens * n_expert, 0.0f);
    std::vector<int32_t> ids((size_t)n_tokens * k, 0);
    std::vector<float> weights((size_t)n_tokens * k, 0.1f);
    std::vector<int32_t> res(n_expert, 1);  // Most resident

    // For token 0: ids[6] is non-resident (res = -1), logit = 5.0f
    // Resident candidate expert 100 has logit = 4.8f (diff 0.2f <= margin 0.5f) -> should swap!
    res[50] = -1;  // non-resident
    ids[6] = 50;
    logits[50] = 5.0f;
    logits[100] = 4.8f;
    res[100] = 1;  // resident

    float* d_l = sycl::malloc_device<float>(logits.size(), q);
    int32_t* d_ids = sycl::malloc_device<int32_t>(ids.size(), q);
    float* d_w = sycl::malloc_device<float>(weights.size(), q);
    int32_t* d_res = sycl::malloc_device<int32_t>(res.size(), q);
    unsigned long long* d_stats = sycl::malloc_device<unsigned long long>(4, q);

    q.memcpy(d_l, logits.data(), logits.size() * sizeof(float));
    q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t));
    q.memcpy(d_w, weights.data(), weights.size() * sizeof(float));
    q.memcpy(d_res, res.data(), res.size() * sizeof(int32_t));
    q.fill(d_stats, 0ULL, 4).wait();

    strata::kernels::native_route_resident(d_l, d_ids, d_w, d_res, n_tokens, 0.5f, 6, 9, d_stats, &q);
    q.wait();

    std::vector<int32_t> out_ids(ids.size());
    std::vector<unsigned long long> stats(4);
    q.memcpy(out_ids.data(), d_ids, out_ids.size() * sizeof(int32_t));
    q.memcpy(stats.data(), d_stats, 4 * sizeof(unsigned long long)).wait();

    sycl::free(d_l, q);
    sycl::free(d_ids, q);
    sycl::free(d_w, q);
    sycl::free(d_res, q);
    sycl::free(d_stats, q);

    int bad = 0;
    if (out_ids[6] != 100) {
        std::fprintf(stderr, "route_resident: expected id[6] == 100, got %d\n", out_ids[6]);
        ++bad;
    }
    if (stats[1] != 1) {  // 1 swap
        std::fprintf(stderr, "route_resident: expected 1 swap, got %llu\n", stats[1]);
        ++bad;
    }
    std::printf("  native_route_resident: %s (swaps=%llu, tail=%llu)\n",
                bad ? "*** WRONG ***" : "exact", stats[1], stats[0]);
    return bad;
}

int test_native_router_topk_256_8(sycl::queue& q, int n_tokens) {
    constexpr int n_expert = 256, k = 8;
    std::vector<float> logits((size_t)n_tokens * n_expert);
    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = dist(rng);
    }

    std::vector<int> ref_ids((size_t)n_tokens * k);
    std::vector<float> ref_w((size_t)n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t)t * n_expert], n_expert, k, &ref_ids[(size_t)t * k], &ref_w[(size_t)t * k]);
    }

    float* d_l = sycl::malloc_device<float>(logits.size(), q);
    int32_t* d_ids = sycl::malloc_device<int32_t>((size_t)n_tokens * k, q);
    float* d_w = sycl::malloc_device<float>((size_t)n_tokens * k, q);

    q.memcpy(d_l, logits.data(), logits.size() * sizeof(float)).wait();

    strata::kernels::native_router_topk(d_l, d_ids, d_w, n_expert, k, n_tokens, &q);
    q.wait();

    std::vector<int32_t> act_ids((size_t)n_tokens * k);
    std::vector<float> act_w((size_t)n_tokens * k);
    q.memcpy(act_ids.data(), d_ids, act_ids.size() * sizeof(int32_t)).wait();
    q.memcpy(act_w.data(), d_w, act_w.size() * sizeof(float)).wait();

    sycl::free(d_l, q);
    sycl::free(d_ids, q);
    sycl::free(d_w, q);

    int id_bad = 0, w_bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < act_ids.size(); ++i) {
        if (act_ids[i] != ref_ids[i]) ++id_bad;
        double rel = std::fabs((double)act_w[i] - (double)ref_w[i]) /
                     (std::fabs((double)ref_w[i]) > 1e-30 ? std::fabs((double)ref_w[i]) : 1e-30);
        worst = std::max(worst, rel);
        if (rel > 1e-3) ++w_bad;
    }

    std::printf("  topk (256, 8) n_tokens=%d: ids %s (%d bad) weights worst rel %.3e (%d over tol)\n", n_tokens,
                id_bad ? "*** WRONG ***" : "exact", id_bad, worst, w_bad);
    return id_bad + w_bad;
}

void bench_native_router(sycl::queue& q) {
    std::printf("\n--- Native Router Microbenchmarks ---\n");
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    constexpr int reps = 100;

    for (int n_tokens : {1, 4, 16, 64}) {
        // 512 / 10
        std::vector<float> l_512((size_t)n_tokens * 512);
        for (auto& x : l_512) x = dist(rng);
        float* d_l512 = sycl::malloc_device<float>(l_512.size(), q);
        int32_t* d_ids512 = sycl::malloc_device<int32_t>((size_t)n_tokens * 10, q);
        float* d_w512 = sycl::malloc_device<float>((size_t)n_tokens * 10, q);
        q.memcpy(d_l512, l_512.data(), l_512.size() * sizeof(float)).wait();

        // Warmup
        for (int i = 0; i < 5; ++i) {
            if (n_tokens == 1) strata::kernels::native_router_top10(d_l512, d_ids512, d_w512, &q);
            else strata::kernels::native_router_top10_multi(d_l512, d_ids512, d_w512, n_tokens, &q);
        }
        q.wait();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            if (n_tokens == 1) strata::kernels::native_router_top10(d_l512, d_ids512, d_w512, &q);
            else strata::kernels::native_router_top10_multi(d_l512, d_ids512, d_w512, n_tokens, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        double us_512 = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

        // 256 / 8
        std::vector<float> l_256((size_t)n_tokens * 256);
        for (auto& x : l_256) x = dist(rng);
        float* d_l256 = sycl::malloc_device<float>(l_256.size(), q);
        int32_t* d_ids256 = sycl::malloc_device<int32_t>((size_t)n_tokens * 8, q);
        float* d_w256 = sycl::malloc_device<float>((size_t)n_tokens * 8, q);
        q.memcpy(d_l256, l_256.data(), l_256.size() * sizeof(float)).wait();

        for (int i = 0; i < 5; ++i) {
            strata::kernels::native_router_topk(d_l256, d_ids256, d_w256, 256, 8, n_tokens, &q);
        }
        q.wait();

        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            strata::kernels::native_router_topk(d_l256, d_ids256, d_w256, 256, 8, n_tokens, &q);
        }
        q.wait();
        t1 = std::chrono::steady_clock::now();
        double us_256 = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

        std::printf("BENCH Router T=%-2d | 512/10: %6.2f us (%7.1f tok/s) | 256/8: %6.2f us (%7.1f tok/s) | speedup: %.2fx\n",
                    n_tokens, us_512, (n_tokens / us_512) * 1e6, us_256, (n_tokens / us_256) * 1e6, us_512 / us_256);

        sycl::free(d_l512, q);
        sycl::free(d_ids512, q);
        sycl::free(d_w512, q);
        sycl::free(d_l256, q);
        sycl::free(d_ids256, q);
        sycl::free(d_w256, q);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    sycl::queue q{sycl::property::queue::in_order()};
    int fails = 0;
    fails += test_native_router(q, 1);
    fails += test_native_router(q, 4);
    fails += test_native_router(q, 16);
    fails += test_native_router(q, 128);
    fails += test_native_route_resident(q);

    // Qwen3.6-35B-A3B: 256 experts, top 8
    fails += test_native_router_topk_256_8(q, 1);
    fails += test_native_router_topk_256_8(q, 4);
    fails += test_native_router_topk_256_8(q, 16);

    if (fails != 0) {
        std::fprintf(stderr, "native_router_test FAILED with %d failures\n", fails);
        return 1;
    }
    std::puts("native_router SYCL parity OK (0 failures)");

    if (bench) {
        bench_native_router(q);
    }
    return 0;
}
