// src/kernels/sycl/resident_plan_parity.cpp - pure SYCL 2020 verify window device-built expert plan against host replay.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

int main(int argc, char** argv) {
    bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) run_bench = true;
    }

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

    // STRATA_MIRROR_STAGE: a third of the experts host-mirrored; the plan lists the VRAM groups first (routing order,
    // counts[3] of them), then the mirrored ones (routing order), these also as a list of their own (ptr2 / start2 /
    // counts[2]) pointing at stage + j * stride, their mirror addresses in src[j]
    {
        std::vector<int32_t> res3(NE);
        std::vector<unsigned long long> mtab(NE, 0ull);
        for (int e = 0; e < NE; ++e) {
            const bool mirrored = (e * 7 + 3) % 3 == 0;
            res3[e] = mirrored ? -1 : (e * 37 + 11) % 1000;
            if (mirrored) mtab[e] = 0x7f0000000000ull + (unsigned long long)e * 1536 * 1024;  // never dereferenced
        }
        unsigned long long* d_tab = dalloc<unsigned long long>(NE, q);
        unsigned long long* d_src = dalloc<unsigned long long>(capx, q);
        q.memcpy(d_res, res3.data(), NE * sizeof(int32_t)).wait();
        q.memcpy(d_tab, mtab.data(), NE * sizeof(unsigned long long)).wait();
        k::resident_plan_set_mirror(d_res, d_tab, nullptr, NE);
        uint8_t* const stage = reinterpret_cast<uint8_t*>(uintptr_t(0x50000000));  // never dereferenced
        const long long stride = 1536 * 1024;
        for (int n = 1; n <= 80; ++n)
            for (int variant = 0; variant < 4; ++variant) {
                const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
                std::vector<int32_t> ids(n);
                for (int i = 0; i < n; ++i) ids[i] = (int32_t)((rng() % pool) * (512 / pool));
                q.memcpy(d_ids, ids.data(), n * sizeof(int32_t)).wait();
                std::vector<int> vexp, mexp;  // distinct experts in routing order, VRAM / mirrored
                for (int i = 0; i < n; ++i) {
                    std::vector<int>& L = res3[ids[i]] >= 0 ? vexp : mexp;
                    if (std::find(L.begin(), L.end(), ids[i]) == L.end()) L.push_back(ids[i]);
                }
                const int gv = (int)vexp.size(), gm = (int)mexp.size(), groups = gv + gm;
                std::vector<int32_t> start(groups + 1), dst(n), tok(n), start2(gm + 1);
                std::vector<unsigned long long> ptr(groups), ptr2(gm), src(gm);
                int pos = 0;
                for (int g = 0; g < groups; ++g) {
                    const int e = g < gv ? vexp[g] : mexp[g - gv];
                    start[g] = pos;
                    ptr[g] = g < gv ? (unsigned long long)(uintptr_t)cache_base + (unsigned long long)res3[e] * (unsigned long long)blob
                                    : (unsigned long long)(uintptr_t)stage + (unsigned long long)(g - gv) * (unsigned long long)stride;
                    if (g >= gv) {
                        ptr2[g - gv] = ptr[g];
                        start2[g - gv] = pos;
                        src[g - gv] = mtab[e];
                    }
                    for (int i = 0; i < n; ++i)
                        if (ids[i] == e) {
                            dst[pos] = i;
                            tok[pos] = i / K;
                            ++pos;
                        }
                }
                start[groups] = n;
                start2[gm] = n;
                const int32_t counts[4] = {groups, n, gm, gv};
                q.memset(d_plan, 0xAB, plan_words * sizeof(int32_t)).wait();
                k::resident_plan_set_stage(stage, stride, d_src);
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, nullptr, 0u, &q, d_err);
                k::resident_plan_set_stage(nullptr, 0, nullptr);
                q.wait();
                std::vector<int32_t> plan(plan_words);
                std::vector<unsigned long long> got_src(gm + 1);
                q.memcpy(plan.data(), d_plan, plan_words * sizeof(int32_t)).wait();
                if (gm > 0) q.memcpy(got_src.data(), d_src, gm * sizeof(unsigned long long)).wait();
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                const unsigned long long* pptr = (const unsigned long long*)(plan.data() + ptr_off);
                bool ok = std::memcmp(plan.data(), counts, 4 * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t)(groups + 1) * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t)n * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t)n * sizeof(int32_t)) == 0;
                ok = ok && std::memcmp(pptr, ptr.data(), (size_t)groups * sizeof(unsigned long long)) == 0;
                ok = ok && (gm == 0 || std::memcmp(pptr + capx, ptr2.data(), (size_t)gm * sizeof(unsigned long long)) == 0);
                ok = ok && std::memcmp(plan.data() + ptr_off + 4 * capx, start2.data(), (size_t)(gm + 1) * sizeof(int32_t)) == 0;
                ok = ok && (gm == 0 || std::memcmp(got_src.data(), src.data(), (size_t)gm * sizeof(unsigned long long)) == 0);
                ++cases;
                if (!ok) {
                    std::printf("    *** stage n=%d pool=%d (%d VRAM + %d mirrored groups): the plan differs from the host replay ***\n",
                                n, pool, gv, gm);
                    ++bad;
                }
            }

        // STRATA_MIRROR_RING: 400 calls in a row through a ring of 256 slots: within a call no two mirrored groups share a
        // ring slot; a group copies (src = its mirror address) exactly when its expert was not in the ring before the call,
        // and then the ring says it is there now; a group that does not copy reads the slot the ring already held it in
        {
            std::string rerr;
            const long long rstride = 4096;
            const int R = 256;
            if (!k::mirror_ring_create(R, rstride, rerr)) {
                std::printf("    *** ring: %s ***\n", rerr.c_str());
                ++bad;
            }
            std::vector<int> where(NE, -1);  // the host's view: expert -> ring slot
            unsigned long long base = 0;
            int ring_bad = 0, hits = 0, inserts = 0;
            const unsigned long long st_lo = (unsigned long long)(uintptr_t)stage, st_hi = st_lo + (unsigned long long)capx * rstride;
            for (int call = 0; call < 400 && ring_bad == 0; ++call) {
                const int n = 1 + (int)(rng() % 60);
                std::vector<int32_t> ids(n);
                for (int i = 0; i < n; ++i) ids[i] = (int32_t)(rng() % (call < 200 ? 96 : 512));
                q.memcpy(d_ids, ids.data(), n * sizeof(int32_t)).wait();
                k::resident_plan_set_stage(stage, rstride, d_src, true);
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, nullptr, 0u, &q, d_err);
                k::resident_plan_set_stage(nullptr, 0, nullptr);
                q.wait();
                std::vector<int32_t> plan(plan_words);
                q.memcpy(plan.data(), d_plan, plan_words * sizeof(int32_t)).wait();
                const int gm = plan[2], gv = plan[3];
                std::vector<unsigned long long> got_src(gm + 1);
                if (gm > 0) q.memcpy(got_src.data(), d_src, gm * sizeof(unsigned long long)).wait();
                const unsigned long long* pptr = (const unsigned long long*)(plan.data() + ptr_off);
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                const int32_t* pstart = plan.data() + 4;
                if (base == 0) {  // the ring is new: the first call that inserts takes slots from 0, the lowest is its base
                    unsigned long long lo = ~0ull;
                    for (int j = 0; j < gm; ++j)
                        if (got_src[j] != 0 && (pptr[capx + j] < st_lo || pptr[capx + j] >= st_hi)) lo = std::min(lo, pptr[capx + j]);
                    if (lo != ~0ull) base = lo;
                }
                std::vector<int> nw = where, used;
                for (int j = 0; j < gm && ring_bad == 0; ++j) {
                    const int e = ids[pdst[pstart[gv + j]]];  // the group's expert: its first entry's id
                    const unsigned long long a = pptr[capx + j];
                    if (res3[e] >= 0 || pptr[gv + j] != a) {
                        ++ring_bad;
                        break;
                    }
                    int slot = -1;
                    if (got_src[j] == 0) {  // a hit: the ring held it there before this call
                        ++hits;
                        slot = where[e];
                        if (slot < 0 || a != base + (unsigned long long)slot * rstride) {
                            ++ring_bad;
                            break;
                        }
                    } else {
                        if (got_src[j] != mtab[e] || where[e] >= 0) {
                            ++ring_bad;
                            break;
                        }  // copies its mirror; was not in the ring
                        if (a >= st_lo && a < st_hi) continue;  // no ring slot left: staging
                        if (a < base || (a - base) % rstride != 0 || (a - base) / rstride >= (unsigned long long)R) {
                            ++ring_bad;
                            break;
                        }
                        slot = (int)((a - base) / rstride);
                        for (int x = 0; x < NE; ++x)
                            if (nw[x] == slot) nw[x] = -1;  // evicted
                        nw[e] = slot;
                        ++inserts;
                    }
                    if (std::find(used.begin(), used.end(), slot) != used.end()) {
                        ++ring_bad;
                        break;
                    }
                    used.push_back(slot);
                }
                where = nw;
            }
            ++cases;
            if (ring_bad || hits == 0 || inserts == 0) {
                std::printf("    *** ring: %d inconsistencies (%d hits, %d inserts) ***\n", ring_bad, hits, inserts);
                ++bad;
            } else {
                std::printf("ring: 400 plans, %d hits, %d inserts, consistent\n", hits, inserts);
            }
        }
        k::resident_plan_set_mirror(nullptr, nullptr, nullptr, 0);
        sycl::free(d_tab, q);
        sycl::free(d_src, q);
    }

    if (run_bench) {
        std::printf("\n--- Mirror Staging & Ring Microbenchmarks (Xe2) ---\n");
        // 1. mirror_stage Bandwidth: test with real host-pinned memory and device VRAM
        const size_t bench_blob = 1536 * 1024;  // 1.5 MiB expert blob
        for (int n_staged : {1, 2, 4, 8}) {
            uint8_t* d_mhost = sycl::malloc_host<uint8_t>(n_staged * bench_blob, q);
            uint8_t* d_mvram = sycl::malloc_device<uint8_t>(n_staged * bench_blob, q);
            int32_t* d_bplan = dalloc<int32_t>(plan_words, q);
            unsigned long long* d_bsrc = dalloc<unsigned long long>(n_staged, q);

            std::vector<int32_t> bplan(plan_words, 0);
            bplan[2] = n_staged;  // counts[2] = gm
            std::vector<unsigned long long> bsrc(n_staged);
            std::vector<unsigned long long> bdst(n_staged);
            for (int j = 0; j < n_staged; ++j) {
                bsrc[j] = (unsigned long long)(d_mhost + j * bench_blob);
                bdst[j] = (unsigned long long)(d_mvram + j * bench_blob);
            }
            unsigned long long* pptr = (unsigned long long*)(bplan.data() + ptr_off);
            for (int j = 0; j < n_staged; ++j) pptr[capx + j] = bdst[j];

            q.memcpy(d_bplan, bplan.data(), plan_words * sizeof(int32_t)).wait();
            q.memcpy(d_bsrc, bsrc.data(), n_staged * sizeof(unsigned long long)).wait();
            q.memset(d_mhost, 0x42, n_staged * bench_blob).wait();

            // Warmup
            for (int w = 0; w < 10; ++w) {
                k::mirror_stage(d_bplan, capx, bench_blob, d_bsrc, &q);
                q.wait();
            }

            const int iters = 200;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                k::mirror_stage(d_bplan, capx, bench_blob, d_bsrc, &q);
            }
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            double total_bytes = (double)n_staged * (double)bench_blob;
            double gbps = (total_bytes / (ms * 1e-3)) / 1e9;
            std::printf("mirror_stage (%d blobs x %.2f MiB = %.2f MiB total): %.3f ms -> %.2f GB/s\n",
                        n_staged, (double)bench_blob / (1024 * 1024), total_bytes / (1024 * 1024), ms, gbps);

            sycl::free(d_mhost, q);
            sycl::free(d_mvram, q);
            sycl::free(d_bplan, q);
            sycl::free(d_bsrc, q);
        }

        // 2. resident_plan latency with ring hits vs cold staging
        {
            const int n = 40;  // 4 tokens x 10 experts
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = i % 16;  // 16 distinct experts
            q.memcpy(d_ids, ids.data(), n * sizeof(int32_t)).wait();

            std::vector<int32_t> res_ring(NE, -1);
            std::vector<unsigned long long> mtab_ring(NE);
            for (int e = 0; e < NE; ++e) mtab_ring[e] = 0x7f0000000000ull + e * 4096;
            unsigned long long* d_tab = dalloc<unsigned long long>(NE, q);
            unsigned long long* d_src = dalloc<unsigned long long>(capx, q);
            q.memcpy(d_res, res_ring.data(), NE * sizeof(int32_t)).wait();
            q.memcpy(d_tab, mtab_ring.data(), NE * sizeof(unsigned long long)).wait();
            k::resident_plan_set_mirror(d_res, d_tab, nullptr, NE);

            uint8_t* const stage = reinterpret_cast<uint8_t*>(uintptr_t(0x50000000));
            // First call warms / inserts into ring
            k::resident_plan_set_stage(stage, 4096, d_src, true);
            k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, nullptr, 0u, &q, d_err);
            q.wait();

            // Next calls are ring hits
            const int iters = 500;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, nullptr, 0u, &q, d_err);
            }
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            double us_hit = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            std::printf("resident_plan with VRAM ring hit (n=%d entries, 16 distinct experts): %.2f us\n", n, us_hit);

            k::resident_plan_set_stage(nullptr, 0, nullptr);
            k::resident_plan_set_mirror(nullptr, nullptr, nullptr, 0);
            sycl::free(d_tab, q);
            sycl::free(d_src, q);
        }
        std::printf("---------------------------------------------------\n\n");
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
