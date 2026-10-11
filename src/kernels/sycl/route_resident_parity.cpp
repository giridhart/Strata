// route_resident_parity - the warp-parallel STRATA_ROUTE_RESIDENT swap against a scalar reference of the same
// semantics, bit for bit (Pure SYCL 2020 edition).
//
// route_resident_k is one subgroup (32 lanes) per token: lane j scans experts j, j + 32, ... and the subgroup
// reduces the best logit via shift_group_left, where the form it replaced walked all 512 experts on a single thread.
// The two have to land on the same expert and leave the same bits in ids and weights.
#include "strata/kernels/native_router.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kExperts = 512;
constexpr int kTop = 10;
constexpr int kMaxTok = 8;
constexpr int kWeightUlp = 8;

// The spec. One thread per token, serial scan of all 512 experts.
// Runs on device so that exp, and therefore the softmax bits, are evaluated by the same runtime math library.
void route_resident_ref_k(const float* __restrict__ logits, int32_t* __restrict__ ids,
                          float* __restrict__ weights, const int32_t* __restrict__ res, int n_tok,
                          float margin, int lo, int hi, unsigned long long* __restrict__ stats,
                          const sycl::nd_item<1>& item) {
    const int t = (int)item.get_global_id(0);
    if (t >= n_tok) return;
    const float* l = logits + (size_t)t * kExperts;
    int32_t* id = ids + (size_t)t * kTop;
    int32_t my[kTop];
    for (int r = 0; r < kTop; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    for (int r = 0; r < kTop; ++r) before += (res[my[r]] < 0);
    for (int r = lo; r <= hi && r < kTop; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;  // resident already: nothing to swap
        ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < kTop; ++q) used |= (my[q] == f);
            if (used) continue;
            if (l[f] > bv) {
                bv = l[f];
                bi = f;
            }
        }
        if (bi >= 0 && l[e] - bv <= margin) {
            my[r] = bi;
            ++swaps;
        }
    }
    int after = 0;
    for (int r = 0; r < kTop; ++r) after += (res[my[r]] < 0);
    if (swaps) {
        float m = -INFINITY;
        for (int r = 0; r < kTop; ++r) m = sycl::fmax(m, l[my[r]]);
        float ex[kTop], sum = 0.0f;
        for (int r = 0; r < kTop; ++r) {
            ex[r] = sycl::exp(l[my[r]] - m);
            sum += ex[r];
        }
        for (int r = 0; r < kTop; ++r) weights[(size_t)t * kTop + r] = ex[r] / sum;
    }
    for (int r = 0; r < kTop; ++r) id[r] = my[r];
    if (stats) {
        auto a_tail = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>(stats[0]);
        auto a_swaps = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(stats[1]);
        auto a_before = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>(stats[2]);
        auto a_after = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(stats[3]);
        a_tail.fetch_add((unsigned long long)tail);
        a_swaps.fetch_add((unsigned long long)swaps);
        a_before.fetch_add((unsigned long long)before);
        a_after.fetch_add((unsigned long long)after);
    }
}

// The router's top-10 of a row, ties to the smallest index - what the kernel is handed as `ids`.
void top10(const float* l, int32_t* out) {
    for (int r = 0; r < kTop; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            bool used = false;
            for (int q = 0; q < r; ++q) used |= (out[q] == f);
            if (used) continue;
            if (l[f] > bv) {
                bv = l[f];
                bi = f;
            }
        }
        out[r] = bi;
    }
}

}  // namespace

int main() {
    sycl::queue q(sycl::property::queue::in_order{});

    const float margins[] = {-0.5f, 0.0f, 0.05f, 0.25f, 0.5f, 1.0f, 1.5f};
    const int n_margins = (int)(sizeof(margins) / sizeof(margins[0]));
    const int cases = 240;
    std::mt19937 rng(20261010u);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::uniform_int_distribution<int> pick(0, 1);
    std::uniform_int_distribution<int> coin(0, 99);

    std::vector<float> logits((size_t)kMaxTok * kExperts), weights((size_t)kMaxTok * kTop);
    std::vector<int32_t> ids((size_t)kMaxTok * kTop), res(kExperts);
    std::vector<float> sentinel(weights.size());
    const uint32_t pattern = 0xA5A5A5A5u;
    for (size_t i = 0; i < sentinel.size(); ++i) std::memcpy(&sentinel[i], &pattern, 4);

    float* d_logits = sycl::malloc_device<float>(logits.size(), q);
    float* d_w_ref = sycl::malloc_device<float>(weights.size(), q);
    float* d_w_par = sycl::malloc_device<float>(weights.size(), q);
    int32_t* d_ids_ref = sycl::malloc_device<int32_t>(ids.size(), q);
    int32_t* d_ids_par = sycl::malloc_device<int32_t>(ids.size(), q);
    int32_t* d_res = sycl::malloc_device<int32_t>(res.size(), q);
    unsigned long long* d_stats = sycl::malloc_device<unsigned long long>(4, q);

    int bad = 0, swapped_cases = 0, stats_bad = 0, dumped = 0, worst_ulp = 0;
    unsigned long long swaps_total = 0;

    for (int c = 0; c < cases; ++c) {
        const int n_tok = 1 + (c % kMaxTok);
        const float margin = margins[c % n_margins];
        const int lo = pick(rng) ? 4 + (c % 4) : 0;
        const int hi = pick(rng) ? kTop - 1 : lo + (c % (kTop - lo));
        const bool tie_heavy = pick(rng) != 0;
        const bool ids_from_router = pick(rng) != 0;
        const int resident_pct = 40 + coin(rng) % 45;

        for (int e = 0; e < kExperts; ++e) res[e] = (coin(rng) < resident_pct) ? (int32_t)(e % 7) : -1;
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = ud(rng);
            if (tie_heavy) v = std::round(v * 2.0f) * 0.5f;
            logits[i] = v;
        }
        for (int t = 0; t < n_tok; ++t) {
            int32_t* row = ids.data() + (size_t)t * kTop;
            if (ids_from_router) {
                top10(logits.data() + (size_t)t * kExperts, row);
            } else {
                for (int r = 0; r < kTop; ++r) row[r] = (int32_t)(rng() % kExperts);
            }
        }

        q.memcpy(d_logits, logits.data(), logits.size() * sizeof(float)).wait();
        q.memcpy(d_res, res.data(), res.size() * sizeof(int32_t)).wait();

        // Reference kernel
        q.memcpy(d_ids_ref, ids.data(), ids.size() * sizeof(int32_t)).wait();
        q.memcpy(d_w_ref, sentinel.data(), sentinel.size() * sizeof(float)).wait();
        q.memset(d_stats, 0, 4 * sizeof(unsigned long long)).wait();
        const size_t ref_grid = ((size_t)n_tok + 31) / 32 * 32;
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(ref_grid), sycl::range<1>(32)),
                       [=](sycl::nd_item<1> item) {
                           route_resident_ref_k(d_logits, d_ids_ref, d_w_ref, d_res, n_tok, margin, lo, hi, d_stats, item);
                       }).wait();
        unsigned long long st_ref[4] = {0, 0, 0, 0};
        q.memcpy(st_ref, d_stats, sizeof(st_ref)).wait();

        // Parallel subgroup kernel
        q.memcpy(d_ids_par, ids.data(), ids.size() * sizeof(int32_t)).wait();
        q.memcpy(d_w_par, sentinel.data(), sentinel.size() * sizeof(float)).wait();
        q.memset(d_stats, 0, 4 * sizeof(unsigned long long)).wait();
        strata::kernels::native_route_resident(d_logits, d_ids_par, d_w_par, d_res, n_tok, margin, lo, hi,
                                               d_stats, static_cast<void*>(&q));
        q.wait();
        unsigned long long st_par[4] = {0, 0, 0, 0};
        q.memcpy(st_par, d_stats, sizeof(st_par)).wait();

        std::vector<int32_t> i_ref(ids.size()), i_par(ids.size());
        std::vector<float> w_ref(weights.size()), w_par(weights.size());
        q.memcpy(i_ref.data(), d_ids_ref, i_ref.size() * sizeof(int32_t)).wait();
        q.memcpy(i_par.data(), d_ids_par, i_par.size() * sizeof(int32_t)).wait();
        q.memcpy(w_ref.data(), d_w_ref, w_ref.size() * sizeof(float)).wait();
        q.memcpy(w_par.data(), d_w_par, w_par.size() * sizeof(float)).wait();

        const bool ids_same = (i_ref == i_par);
        int w_ulp = 0;
        for (size_t i = 0; i < w_ref.size(); ++i) {
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &w_ref[i], 4);
            std::memcpy(&b, &w_par[i], 4);
            if (a != b) {
                const uint32_t d = a > b ? a - b : b - a;
                if ((int)d > w_ulp) w_ulp = (int)d;
            }
        }
        if (w_ulp > worst_ulp) worst_ulp = w_ulp;
        const bool w_same = (w_ulp <= kWeightUlp);
        const bool st_same = (std::memcmp(st_ref, st_par, sizeof(st_ref)) == 0);

        if (!ids_same || !w_same || !st_same) {
            ++bad;
            if (!st_same) ++stats_bad;
            std::printf("  case %3d  n_tok %d  margin %.2f  ranks [%d,%d]  ties %d  ids %s  weights %s (%d ulp)  stats %s\n",
                        c, n_tok, margin, lo, hi, (int)tie_heavy, ids_same ? "same" : "*** DIFFER ***",
                        w_same ? "same" : "*** DIFFER ***", w_ulp, st_same ? "same" : "*** DIFFER ***");
            if (std::getenv("STRATA_PARITY_DUMP") && dumped < 2) {
                ++dumped;
                for (int t = 0; t < n_tok; ++t) {
                    std::printf("    token %d  ids in ", t);
                    for (int r = 0; r < kTop; ++r) std::printf(" %d", ids[(size_t)t * kTop + r]);
                    std::printf("\n      ids ref ");
                    for (int r = 0; r < kTop; ++r) std::printf(" %d", i_ref[(size_t)t * kTop + r]);
                    std::printf("\n      ids par ");
                    for (int r = 0; r < kTop; ++r) std::printf(" %d", i_par[(size_t)t * kTop + r]);
                    std::printf("\n");
                }
            }
        }
        if (st_par[1]) {
            ++swapped_cases;
            swaps_total += st_par[1];
        }
    }

    std::printf("route_resident_parity: %d cases, %d with swaps (%llu swaps), %d mismatches, weights within %d ulp\n",
                cases, swapped_cases, swaps_total, bad, worst_ulp);
    if (stats_bad) std::printf("  (%d of them in the counters)\n", stats_bad);
    if (swapped_cases == 0) {
        std::printf("route_resident_parity: no case swapped anything - the test proved nothing\n");
        ++bad;
    }
    std::printf(bad ? "route_resident_parity: %d FAILURES\n" : "route_resident_parity OK\n", bad);

    sycl::free(d_logits, q);
    sycl::free(d_w_ref, q);
    sycl::free(d_w_par, q);
    sycl::free(d_ids_ref, q);
    sycl::free(d_ids_par, q);
    sycl::free(d_res, q);
    sycl::free(d_stats, q);
    return bad ? 1 : 0;
}
