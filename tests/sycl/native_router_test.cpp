// Parity test for the SYCL implementation of the native router.
#include <sycl/sycl.hpp>
#include "strata/kernels/native_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
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
}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    int fails = 0;
    fails += test_native_router(q, 1);
    fails += test_native_router(q, 4);
    fails += test_native_router(q, 16);
    fails += test_native_router(q, 128);

    if (fails != 0) {
        std::fprintf(stderr, "native_router_test FAILED with %d failures\n", fails);
        return 1;
    }
    std::puts("native_router SYCL parity OK (0 failures)");
    return 0;
}
