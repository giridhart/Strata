// src/kernels/sycl/qsa_topk_parity.cpp - pure SYCL 2020 QSA top-k engine dispatch (qsa_block_topk) against reference (qsa_block_topk_ref).
// Synthetic, no model.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

bool run_case(sycl::queue& q, int64_t ctx, int64_t nq, int reps, int64_t capacity, bool counted) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t)(nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t)pos;
        st[k::kStepNKv] = (int32_t)(pos + 1);
        st[k::kStepNBid] = (int32_t)((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t)k::qsa_selection_width(pos + 1, s);
    }
    std::mt19937 rng((uint32_t)(ctx * 31 + capacity));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> scores((size_t)(nq * max_blocks)), tied(scores.size()), equal(scores.size(), 0.25f);
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        float* row = scores.data() + i * max_blocks;
        for (int64_t b = 0; b < max_blocks; ++b) row[b] = nd(rng);
        if (st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;
        for (int64_t b = 0; b < max_blocks; ++b)
            tied[(size_t)(i * max_blocks + b)] = row[b] > 1e8f ? row[b] : std::floor(row[b] * 8.0f) / 8.0f;
    }

    int32_t* d_steps = sycl::malloc_device<int32_t>(steps.size(), q);
    float* d_scores = sycl::malloc_device<float>(scores.size(), q);
    int32_t* ids_ref = sycl::malloc_device<int32_t>((size_t)(nq * cap), q);
    int32_t* ids_new = sycl::malloc_device<int32_t>((size_t)(nq * cap), q);
    q.memcpy(d_steps, steps.data(), steps.size() * sizeof(int32_t)).wait();

    const int64_t active = counted ? steps[(size_t)((nq - 1) * k::kStepCount + k::kStepNBid)] + 1 : -1;
    std::vector<int32_t> a((size_t)(nq * cap)), b(a.size());
    auto same = [&](const std::vector<float>& sc) -> int64_t {
        q.memcpy(d_scores, sc.data(), sc.size() * sizeof(float)).wait();
        q.memset(ids_ref, 0xff, a.size() * sizeof(int32_t));
        q.memset(ids_new, 0xff, a.size() * sizeof(int32_t)).wait();
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, &q);
        k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, &q, active);
        q.wait();
        q.memcpy(a.data(), ids_ref, a.size() * sizeof(int32_t));
        q.memcpy(b.data(), ids_new, b.size() * sizeof(int32_t)).wait();
        int64_t n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t)(i * k::kStepCount + k::kStepWidth)];
            n += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
        }
        return n;
    };
    const int64_t same_tied = same(tied), same_equal = same(equal), same_cont = same(scores);

    auto timed = [&](auto f) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            f();
            q.wait();
        }
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<float, std::milli>(t1 - t0).count() / (float)reps;
    };
    const float t_ref = timed([&] { k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, &q); });
    const float t_new = timed([&] { k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, &q, active); });

    sycl::free(d_steps, q);
    sycl::free(d_scores, q);
    sycl::free(ids_ref, q);
    sycl::free(ids_new, q);

    const bool ok = same_cont == nq && same_tied == nq && same_equal == nq;
    std::printf("%s context %lld under a capacity of %lld (%lld blocks), %lld queries %s: reference %.3f ms, "
                "dispatched %.3f ms (%.1fx); identical ids: continuous %lld/%lld, 64 levels %lld/%lld, equal scores "
                "%lld/%lld\n", ok ? "PASS" : "FAIL", (long long)ctx, (long long)capacity, (long long)max_blocks,
                (long long)nq, counted ? "with their block count" : "without a block count", t_ref, t_new,
                t_new > 0.0f ? t_ref / t_new : 1.0f,
                (long long)same_cont, (long long)nq, (long long)same_tied,
                (long long)nq, (long long)same_equal, (long long)nq);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        const int64_t cases[][2] = {{9000, 9000},    {135160, 135160}, {135176, 135176}, {2100, 262144},
                                    {9000, 262144},  {32768, 262144},  {262144, 262144}, {300001, 524288},
                                    {524288, 524288}};
        bool ok = true;
        for (const auto& c : cases) {
            ok = run_case(q, c[0], 8, 3, c[1], false) && ok;
            ok = run_case(q, c[0], 64, 3, c[1], true) && ok;
        }
        std::printf("%s\n", ok ? "qsa_topk_parity: PASS" : "qsa_topk_parity: FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_topk_parity --selftest | CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] "
                             "[COUNT=1]\n");
        return 2;
    }
    const int64_t ctx = std::atoll(argv[1]);
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : ctx;
    if (ctx < nq || nq <= 0 || reps <= 0 || capacity < ctx) {
        std::fprintf(stderr, "qsa_topk_parity: CONTEXT >= QUERIES > 0, REPS > 0, CAPACITY >= CONTEXT\n");
        return 2;
    }
    return run_case(q, ctx, nq, reps, capacity, argc <= 5 || std::atoi(argv[5]) != 0) ? 0 : 1;
}
