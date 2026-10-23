// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include <sycl/sycl.hpp>
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace strata::kernels {
namespace {

[[maybe_unused]] void check([[maybe_unused]] const char* what) {
    // DPCT error checking removed.
}

constexpr int RT = 1024;  // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int)(s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvHybrid) {  // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int)s.head_dim, scales = rows * (int)(s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int)kv_q4_bytes_per_head((int)s.head_dim);
        r.src[0] = (const uint8_t*)host.k_q;
        r.dst[0] = (uint8_t*)slots.k_q;
        r.len[0] = codes;
        r.src[1] = (const uint8_t*)host.k_scale;
        r.dst[1] = (uint8_t*)slots.k_scale;
        r.len[1] = scales;
        r.src[2] = (const uint8_t*)host.v_q4;
        r.dst[2] = (uint8_t*)slots.v_q4;
        r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int)kv_q4_bytes_per_head((int)s.head_dim);
        r.src[0] = (const uint8_t*)host.k_q4;
        r.dst[0] = (uint8_t*)slots.k_q4;
        r.len[0] = bytes;
        r.src[1] = (const uint8_t*)host.v_q4;
        r.dst[1] = (uint8_t*)slots.v_q4;
        r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int)s.head_dim, scales = rows * (int)(s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*)host.k_q;
        r.dst[0] = (uint8_t*)slots.k_q;
        r.len[0] = codes;
        r.src[1] = (const uint8_t*)host.v_q;
        r.dst[1] = (uint8_t*)slots.v_q;
        r.len[1] = codes;
        r.src[2] = (const uint8_t*)host.k_scale;
        r.dst[2] = (uint8_t*)slots.k_scale;
        r.len[2] = scales;
        r.src[3] = (const uint8_t*)host.v_scale;
        r.dst[3] = (uint8_t*)slots.v_scale;
        r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int)s.head_dim * 2;
        r.src[0] = (const uint8_t*)host.k_pool;
        r.dst[0] = (uint8_t*)slots.k_pool;
        r.len[0] = bytes;
        r.src[1] = (const uint8_t*)host.v_pool;
        r.dst[1] = (uint8_t*)slots.v_pool;
        r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
int block_scan(sycl::nd_item<1> item, int v, int* warp_sums, int& total) {
    const int lane = item.get_local_id(0) & 31;
    const int w = item.get_local_id(0) >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = sycl::shift_group_right(item.get_sub_group(), x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    item.barrier(sycl::access::fence_space::local_space);
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = sycl::shift_group_right(item.get_sub_group(), t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    item.barrier(sycl::access::fence_space::local_space);
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    item.barrier(sycl::access::fence_space::local_space);
    return excl;
}

void resolve_kernel(sycl::nd_item<1> item, KvStreamMap m, const int32_t* __restrict__ ids,
                    const int32_t* __restrict__ steps, int n_q, int cap, int page_size, int& s_nmiss, int& s_lookups,
                    int& s_cut, int* warp_sums) {
    const int epoch = m.ctl[0] + 1;
    if (item.get_local_id(0) == 0) {
        s_nmiss = 0;
        s_lookups = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long)q * cap;
        for (int i = item.get_local_id(0); i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;  // ids are ascending: one lookup per block
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1) {
                int32_t expected = -1;
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    pt_ref(m.page_table[b]);
                if (pt_ref.compare_exchange_strong(expected, -2)) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        nmiss_ref(s_nmiss);
                    m.miss_block[nmiss_ref.fetch_add(1)] = b;
                }
            }
        }
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        lookups_ref(s_lookups);
    lookups_ref.fetch_add(lookups);
    item.barrier(sycl::access::fence_space::local_space);

    const int need = s_nmiss, n = (int)m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int)(((long long)hand + item.get_local_id(0)) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(item, cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (item.get_local_id(0) == 0) s_cut = RT;
        item.barrier(sycl::access::fence_space::local_space);
        if (cand && rank == want - 1) s_cut = item.get_local_id(0) + 1;  // the hand stops just past the last slot taken
        item.barrier(sycl::access::fence_space::local_space);
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;  // taken: a sweep that wraps around must not take it twice
        } else if (int(item.get_local_id(0)) < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int)(((long long)hand + cut) % n);
        item.barrier(sycl::access::fence_space::local_space);
    }
    const int placed = got < need ? got : need;
    for (int k = item.get_local_id(0); k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) {
            m.page_table[b] = -1;
            continue;
        }  // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (item.get_local_id(0) == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long)placed;
        c[1] += (unsigned long long)s_lookups;
        c[2] += 1ull;
    }
}

void copy_kernel(sycl::nd_item<1> item, KvStreamMap m, Runs r) {
    const int need = m.ctl[2];
    for (int k = item.get_group(0); k < need; k += item.get_group_range(0)) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const sycl::uint4* src = reinterpret_cast<const sycl::uint4*>(r.src[a] + b * r.len[a]);
            sycl::uint4* dst = reinterpret_cast<sycl::uint4*>(r.dst[a] + sl * r.len[a]);
            for (int i = item.get_local_id(0); i < r.len[a] / 16; i += item.get_local_range(0))
                dst[i] = src[i];
        }
    }
}

void reset_kernel(sycl::nd_item<1> item, KvStreamMap m) {
    const long long i0 = item.get_global_id(0);
    const long long st = item.get_global_range(0);
    for (long long i = i0; i < m.n_blocks; i += st)
        m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

void ring_kernel(sycl::nd_item<1> item, int32_t* table, long long n_blocks, long long n_slots) {
    for (long long i = item.get_global_id(0); i < n_blocks; i += item.get_global_range(0))
        table[i] = (int32_t)(i % n_slots);
}

}  // namespace

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t)(s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int)s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t)s.head_dim * 2 + rows * (uint64_t)(s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t)s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->parallel_for(sycl::nd_range<1>(128 * 256, 256), [=](sycl::nd_item<1> item) { reset_kernel(item, m); });
    if (stream == nullptr) q_ptr->wait();
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d)\n", (long long)m.n_slots, RT);
        std::exit(1);
    }

    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_queue();

    sycl::event ev1 = q_ptr->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> s_nmiss_acc(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> s_lookups_acc(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> s_cut_acc(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> warp_sums_acc(sycl::range<1>(32), cgh);

        cgh.parallel_for(sycl::nd_range<1>(RT, RT), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            resolve_kernel(item, m, ids, steps, (int)n_q, (int)cap, (int)s.page_size, s_nmiss_acc[0], s_lookups_acc[0],
                           s_cut_acc[0], warp_sums_acc.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });

    q_ptr->submit([&](sycl::handler& cgh) {
        cgh.depends_on(ev1);
        auto runs = runs_of(slots, host, fmt, s);
        cgh.parallel_for(sycl::nd_range<1>(96 * 128, 128), [=](sycl::nd_item<1> item) { copy_kernel(item, m, runs); });
    });
    if (stream == nullptr) q_ptr->wait();
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->parallel_for(sycl::nd_range<1>(64 * 256, 256),
                        [=](sycl::nd_item<1> item) { ring_kernel(item, page_table, n_blocks, n_slots); });
    if (stream == nullptr) q_ptr->wait();
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_queue();
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);
        for (int a = 0; a < r.n; ++a) {
            q_ptr->memcpy(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t)(run * r.len[a]));
        }
        b += run;
    }
    if (stream == nullptr) q_ptr->wait();
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_queue();
    for (int a = 0; a < r.n; ++a) {
        q_ptr->memcpy(r.dst[a], r.src[a], (size_t)(n_blocks * r.len[a]));
    }
    if (stream == nullptr) q_ptr->wait();
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr) return r;

    sycl::queue* q_ptr = &default_queue();  // Synchronous memcpy requires in-order default_queue

    q_ptr->memcpy(c, m.ctl, sizeof(c)).wait();

    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
