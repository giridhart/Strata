// tests/sycl/qsa_select_long_parity.cpp - STRATA_SELECT_LONG's decode select (PR #3820):
// qsa_block_scores_long (block_scores_dx_kernel) and qsa_block_topk_long (multi-workgroup top-k)
// against qsa_block_scores and qsa_block_topk, bit for bit (SYCL, synthetic data, no model).

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <sycl/sycl.hpp>

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

template <typename T>
T* dalloc(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) {
        std::fprintf(stderr, "qsa_select_long_parity: malloc_device failed for %zu items\n", n);
        std::exit(2);
    }
    q.fill(p, T{}, n).wait();
    return p;
}

template <typename T>
T* upload(sycl::queue& q, const std::vector<T>& h) {
    T* d = sycl::malloc_device<T>(h.size(), q);
    if (!d) {
        std::fprintf(stderr, "qsa_select_long_parity: upload malloc_device failed\n");
        std::exit(2);
    }
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

// steps of a decode window: query i at position pos0 + i
std::vector<int32_t> make_steps(int nq, int64_t pos0, const k::QsaShapes& s) {
    std::vector<int32_t> st((size_t)nq * k::kStepCount, 0);
    for (int i = 0; i < nq; ++i) {
        int32_t* p = st.data() + (size_t)i * k::kStepCount;
        const int64_t pos = pos0 + i;
        p[k::kStepPos] = (int32_t)pos;
        p[k::kStepNKv] = (int32_t)(pos + 1);
        p[k::kStepNBid] = (int32_t)((pos + 1) / s.idx_block);
        p[k::kStepWidth] = (int32_t)k::qsa_selection_width(pos + 1, s);
    }
    return st;
}

long long g_cases = 0, g_failed = 0;

void check_scores(sycl::queue& q, const k::QsaShapes& s, int nq, int64_t n_bid0, std::mt19937& rng) {
    const int64_t D = s.idx_dim, H = s.idx_n_head;
    const int64_t max_blocks = n_bid0 + nq + 8;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> pooled((size_t)(max_blocks * D)), dead((size_t)D), q_vec((size_t)(nq * H * D));
    for (auto& x : pooled) x = nd(rng);
    for (auto& x : dead) x = nd(rng);
    for (auto& x : q_vec) x = nd(rng);

    const std::vector<int32_t> steps = make_steps(nq, n_bid0 * s.idx_block - 1 + (int64_t)(rng() % 4u) - (nq - 1), s);
    float *dp = upload(q, pooled), *dd = upload(q, dead), *dq = upload(q, q_vec);
    int32_t* dst = upload(q, steps);
    float* a = dalloc<float>(q, (size_t)(nq * max_blocks));
    float* b = dalloc<float>(q, (size_t)(nq * max_blocks));
    q.fill(a, -1.0f, (size_t)(nq * max_blocks)).wait();
    q.fill(b, -1.0f, (size_t)(nq * max_blocks)).wait();

    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, a, &q);
    k::qsa_block_scores_long(dp, dd, dq, dst, nq, max_blocks, s, b, &q, -1);
    q.wait();

    std::vector<float> ha((size_t)(nq * max_blocks)), hb(ha.size());
    q.memcpy(ha.data(), a, ha.size() * sizeof(float)).wait();
    q.memcpy(hb.data(), b, hb.size() * sizeof(float)).wait();

    bool ok = true;
    for (int i = 0; i < nq && ok; ++i) {
        const int64_t nb = steps[(size_t)i * k::kStepCount + k::kStepNBid];
        for (int64_t j = 0; j <= nb; ++j) {
            if (std::memcmp(&ha[(size_t)(i * max_blocks + j)], &hb[(size_t)(i * max_blocks + j)], 4) != 0) {
                std::printf("FAIL scores nq %d n_bid0 %lld: query %d block %lld: %.9g vs %.9g\n", nq, (long long)n_bid0, i,
                            (long long)j, ha[(size_t)(i * max_blocks + j)], hb[(size_t)(i * max_blocks + j)]);
                ok = false;
                break;
            }
        }
    }
    ++g_cases;
    if (!ok) ++g_failed;

    sycl::free(dp, q);
    sycl::free(dd, q);
    sycl::free(dq, q);
    sycl::free(dst, q);
    sycl::free(a, q);
    sycl::free(b, q);
}

void check_topk(sycl::queue& q, const k::QsaShapes& s, int nq, int64_t ctx, int mode, int64_t max_blocks,
                std::mt19937& rng, bool& multi_ran) {
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const std::vector<int32_t> steps = make_steps(nq, ctx - nq, s);
    std::vector<float> sc((size_t)(nq * max_blocks), -999.0f);
    std::uniform_real_distribution<float> ur(0.0f, 200.0f);
    for (int i = 0; i < nq; ++i) {
        const int32_t nb = steps[(size_t)i * k::kStepCount + k::kStepNBid];
        const bool tail = steps[(size_t)i * k::kStepCount + k::kStepNKv] % s.idx_block != 0;
        for (int64_t j = 0; j <= nb; ++j) {
            float v = mode == 1 ? 3.0f : mode == 2 ? (float)(rng() % 7u) : ur(rng);
            sc[(size_t)(i * max_blocks + j)] = v;
        }
        if (tail && mode != 1) sc[(size_t)(i * max_blocks + nb)] += 1e9f;
    }

    float* ds = upload(q, sc);
    int32_t* dst = upload(q, steps);
    int32_t* ia = dalloc<int32_t>(q, (size_t)(nq * cap));
    int32_t* ib = dalloc<int32_t>(q, (size_t)(nq * cap));
    const int64_t scratch_ints = k::qsa_topk_multi_scratch_ints(nq);
    int32_t* scratch = dalloc<int32_t>(q, (size_t)scratch_ints);

    q.fill(scratch, -1, (size_t)scratch_ints).wait();
    q.fill(ia, -1, (size_t)(nq * cap)).wait();
    q.fill(ib, -1, (size_t)(nq * cap)).wait();

    k::qsa_block_topk(ds, dst, nq, max_blocks, cap, s, ia, &q);
    k::qsa_block_topk_long(ds, dst, nq, max_blocks, cap, s, ib, &q, -1, scratch);
    q.wait();

    std::vector<int32_t> ha((size_t)(nq * cap)), hb(ha.size()), hs((size_t)scratch_ints);
    q.memcpy(ha.data(), ia, ha.size() * sizeof(int32_t)).wait();
    q.memcpy(hb.data(), ib, hb.size() * sizeof(int32_t)).wait();
    q.memcpy(hs.data(), scratch, hs.size() * sizeof(int32_t)).wait();

    const bool touched = std::any_of(hs.begin(), hs.end(), [](int32_t v) { return v != -1; });
    if (touched) multi_ran = true;

    bool ok = true;
    for (int i = 0; i < nq && ok; ++i) {
        const int64_t w = steps[(size_t)i * k::kStepCount + k::kStepWidth];
        if (!std::equal(ha.begin() + i * cap, ha.begin() + i * cap + w, hb.begin() + i * cap)) {
            size_t first = 0;
            while (ha[(size_t)(i * cap) + first] == hb[(size_t)(i * cap) + first]) ++first;
            std::printf("FAIL topk nq %d ctx %lld mode %d max_blocks %lld: query %d entry %zu: %d vs %d\n", nq,
                        (long long)ctx, mode, (long long)max_blocks, i, first, ha[(size_t)(i * cap) + first],
                        hb[(size_t)(i * cap) + first]);
            ok = false;
        }
    }
    ++g_cases;
    if (!ok) ++g_failed;

    sycl::free(ds, q);
    sycl::free(dst, q);
    sycl::free(ia, q);
    sycl::free(ib, q);
    sycl::free(scratch, q);
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("Running qsa_select_long_parity on: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    if (k::qsa_topk_multi_scratch_ints(1) <= 0) {
        std::puts("SKIP: this build has no multi-CTA select");
        return 77;
    }

    setenv("STRATA_SCORES_DX", "1", 1);
    setenv("STRATA_TOPK_MULTI", "1", 1);

    const k::QsaShapes s = k::qsa_real_shapes();
    std::mt19937 rng(20261026u);

    for (int nq : {1, 2, 3, 5, 8}) {
        for (int64_t nb : {(int64_t)8, (int64_t)31, (int64_t)32, (int64_t)33, (int64_t)1000, (int64_t)70001, (int64_t)131072}) {
            check_scores(q, s, nq, nb, rng);
        }
    }

    bool multi_ran = false;
    const int64_t max_ctx = 524288, max_blocks = max_ctx / s.idx_block + 2;
    for (int nq : {1, 2, 5, 8, 16}) {
        for (int64_t ctx : {(int64_t)1000, (int64_t)2051, (int64_t)2055, (int64_t)8192, (int64_t)70000, (int64_t)135164,
                            (int64_t)135168, (int64_t)300003, max_ctx}) {
            for (int mode = 0; mode < 3; ++mode) {
                check_topk(q, s, nq, ctx, mode, max_blocks, rng, multi_ran);
            }
        }
    }

    for (int nq : {1, 4}) {
        check_topk(q, s, nq, 40000, 0, 40000 / s.idx_block + 2, rng, multi_ran);
    }

    if (!multi_ran) {
        std::printf("FAIL: the multi-CTA top-k never wrote its scratch: the new kernels did not run\n");
        ++g_failed;
    }

    std::printf("qsa_select_long_parity: %lld cases, %lld differ -> %s\n", g_cases, g_failed, g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 1;
}
