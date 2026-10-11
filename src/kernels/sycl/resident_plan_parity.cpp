// src/kernels/sycl/resident_plan_parity.cpp - pure SYCL 2020 verify window device-built expert plan against host replay.
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

template <typename T>
T* dalloc(size_t n, sycl::queue& q, int fill = 0) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    q.memset(p, fill, (n + 64) * sizeof(T)).wait();
    return p;
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("resident_plan_parity: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    const int K = 10, NE = 512;
    const long long capx = (long long)k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t)(ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(783);
    const long long blob = 1337;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));  // never dereferenced

    int32_t* d_ids = dalloc<int32_t>(128, q);
    int32_t* d_res = dalloc<int32_t>(NE, q);
    unsigned long long* d_off = dalloc<unsigned long long>(1024, q);
    int32_t* d_plan = dalloc<int32_t>(plan_words, q);
    uint32_t* d_skip = dalloc<uint32_t>(1, q);
    uint32_t* d_err = dalloc<uint32_t>(1, q);

    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[e] = (e * 37 + 11) % 1000;  // distinct slots, not the identity
    std::vector<unsigned long long> off(1024);
    for (int s = 0; s < 1024; ++s) off[s] = (unsigned long long)s * 4096 + (s % 7) * 64;

    q.memcpy(d_res, res.data(), NE * sizeof(int32_t)).wait();
    q.memcpy(d_off, off.data(), 1024 * sizeof(unsigned long long)).wait();

    int cases = 0, bad = 0;
    for (int n = 1; n <= 80; ++n) {
        for (int variant = 0; variant < 4; ++variant) {
            const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = (int32_t)((rng() % pool) * (512 / pool));
            const bool use_off = (n + variant) % 2 == 1;
            q.memcpy(d_ids, ids.data(), n * sizeof(int32_t)).wait();

            // host replay
            std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
            std::vector<unsigned long long> ptr;
            std::vector<int> first_seen;  // group -> expert
            for (int i = 0; i < n; ++i) {
                int g = -1;
                for (size_t q_idx = 0; q_idx < first_seen.size(); ++q_idx)
                    if (first_seen[q_idx] == ids[i]) g = (int)q_idx;
                if (g < 0) {
                    first_seen.push_back(ids[i]);
                    g = (int)first_seen.size() - 1;
                }
            }
            const int groups = (int)first_seen.size();
            counts[0] = groups;
            start.assign(groups + 1, 0);
            int pos = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = pos;
                ptr.push_back((unsigned long long)(uintptr_t)cache_base +
                              (use_off ? off[res[first_seen[g]]]
                                       : (unsigned long long)res[first_seen[g]] * (unsigned long long)blob));
                for (int i = 0; i < n; ++i)
                    if (ids[i] == first_seen[g]) {
                        dst[pos] = i;
                        tok[pos] = i / K;
                        ++pos;
                    }
            }
            start[groups] = n;

            for (int mode = 0; mode < 2; ++mode) {  // 0: general kernel with a skip word, 1: all resident
                q.memset(d_plan, 0xAB, plan_words * sizeof(int32_t)).wait();
                q.memset(d_skip, 0, sizeof(uint32_t)).wait();
                q.memset(d_err, 0, sizeof(uint32_t)).wait();
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx,
                                 mode == 0 ? d_skip : nullptr, 7u, &q, d_err);
                q.wait();

                std::vector<int32_t> plan(plan_words);
                q.memcpy(plan.data(), d_plan, plan_words * sizeof(int32_t)).wait();
                uint32_t skip = 0;
                q.memcpy(&skip, d_skip, sizeof(uint32_t)).wait();
                bool ok = mode == 1 || skip == 7u;
                ok = ok && std::memcmp(plan.data(), counts.data(), 3 * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t)(groups + 1) * sizeof(int32_t)) == 0;
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t)n * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t)n * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t)groups * sizeof(unsigned long long)) == 0;
                ok = ok && plan[(size_t)(ptr_off + 4 * capx)] == n;
                ++cases;
                if (!ok) {
                    std::printf("    *** n=%d pool=%d %s %s: the plan differs from the host replay ***\n", n, pool,
                                use_off ? "slot_off" : "blob", mode ? "all_resident" : "general");
                    ++bad;
                }
            }
        }
    }

    // a routed expert that is not resident: the general kernel leaves the plan alone and zeroes *skip
    {
        std::vector<int32_t> res2 = res;
        res2[5] = -1;
        q.memcpy(d_res, res2.data(), NE * sizeof(int32_t)).wait();
        const int32_t ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        q.memcpy(d_ids, ids, sizeof(ids)).wait();
        q.memset(d_plan, 0xAB, plan_words * sizeof(int32_t)).wait();
        const uint32_t seven = 7;
        q.memcpy(d_skip, &seven, sizeof(uint32_t)).wait();
        k::resident_plan(d_ids, 10, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, &q, d_err);
        q.wait();
        std::vector<int32_t> plan(plan_words);
        q.memcpy(plan.data(), d_plan, plan_words * sizeof(int32_t)).wait();
        uint32_t skip = 99;
        q.memcpy(&skip, d_skip, sizeof(uint32_t)).wait();
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t)w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched) {
            std::printf("    *** a non-resident expert: skip=%u untouched=%d (want 0 / 1) ***\n", skip, (int)untouched);
            ++bad;
        }
    }

    std::printf("resident_plan: %d cases, %d failures\n", cases, bad);

    sycl::free(d_ids, q);
    sycl::free(d_res, q);
    sycl::free(d_off, q);
    sycl::free(d_plan, q);
    sycl::free(d_skip, q);
    sycl::free(d_err, q);

    if (bad) return 1;
    std::printf("resident_plan_parity OK\n");
    return 0;
}
