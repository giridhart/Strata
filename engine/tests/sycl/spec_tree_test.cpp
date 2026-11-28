#include <sycl/sycl.hpp>
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/spec/spec_tree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check_cond(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

uint16_t f32_to_f16_bits(float v) {
    sycl::half h(v);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float f16_bits_to_f32(uint16_t bits) {
    sycl::half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<float>(h);
}

// ---- 1. CPU Unit Tests for SpecTree, walk_accepted_tree, penalty_rows_tree ----

void test_spec_tree_cpu() {
    using namespace strata::spec;
    using namespace strata::kernels;

    // 1a. indexer_tree_safe invariants: <= 2048 always safe, > 2048 only within same /4 tail block.
    check_cond(indexer_tree_safe(0, 0), "indexer_tree_safe(0, 0)");
    check_cond(indexer_tree_safe(2040, 7), "indexer_tree_safe(2040, 7)");
    check_cond(indexer_tree_safe(2047, 3), "indexer_tree_safe(2047, 3): 2048..2051 /4 == 512");
    check_cond(!indexer_tree_safe(2047, 4), "indexer_tree_safe crosses 2051->2052 /4 block boundary");
    check_cond(indexer_tree_safe(4095, 3), "indexer_tree_safe(4095, 3): (4096..4099)/4 == 1024");
    check_cond(!indexer_tree_safe(4095, 4), "indexer_tree_safe(4095, 4): (4096)/4 != (4100)/4");
    check_cond(indexer_tree_safe(4097, 1), "indexer_tree_safe(4097, 1): same /4 block");
    check_cond(!indexer_tree_safe(4097, 2), "indexer_tree_safe(4097, 2): crosses /4 block");

    // 1b. Mode 2 wide tree construction: trunk + sfx branch + mtp top2_0 + mtp top2_1
    {
        int32_t win[kVerifyMaxT] = {10, 20, 30, 40};
        float dprob[kVerifyMaxT] = {0.75f, 0.70f, 0.65f};
        int32_t sbuf[kVerifyMaxT] = {20, 99, 100, 101};
        SpecTreeInput in;
        in.mode = 2;
        in.pos0 = 100;
        in.T_base = 4;
        in.max_room = 8;
        in.from_sfx = false;
        in.draft_on = true;
        in.mtp_n = 3;
        in.drafts = win + 1;
        in.dprob = dprob;
        in.mtp_top2_0 = 200;
        in.mtp_top2_p0 = 0.15f;
        in.mtp_top2_1 = 201;
        in.mtp_top2_p1 = 0.12f;
        in.sfx_k = 4;
        in.sfx_match = 4;
        in.sbuf = sbuf;

        int T_out = 0;
        VerifyTreeMap map{};
        const bool ok = build_spec_tree(in, win, T_out, map);
        check_cond(ok, "mode 2 build_spec_tree succeeds");
        check_cond(T_out >= 6 && T_out <= 8, "mode 2 T_out in [6, 8]");
        check_cond(map.pos0 == 100 && map.n_tok == T_out, "mode 2 map metadata");
        check_cond(map.parent[0] == -1 && map.depth[0] == 0, "trunk root invariants");
        for (int t = 1; t < 4; ++t) {
            check_cond(map.parent[t] == t - 1 && map.depth[t] == t, "trunk parent/depth preserved");
        }
        // Check ancestor table and conv history indices
        for (int t = 0; t < T_out; ++t) {
            check_cond(map.anc_slot[t][map.depth[t]] == t, "anc_slot leaf self");
            int cur = t;
            for (int k = map.depth[t]; k >= 0; --k) {
                check_cond(map.anc_slot[t][k] == cur, "anc_slot chain matches parent walk");
                cur = map.parent[cur];
            }
            for (int j = 0; j < 4; ++j) {
                const int rel = map.depth[t] - (3 - j);
                const int expect = rel < 0 ? rel : map.anc_slot[t][rel];
                check_cond(map.anc_conv[t][j] == expect, "anc_conv matches causal window");
            }
        }
        // Verify branch_mask marks every node that spawned a non-trunk edge
        uint32_t expect_mask = 0;
        for (int t = 1; t < T_out; ++t) {
            if (map.parent[t] != t - 1) expect_mask |= (1u << (unsigned)map.parent[t]);
        }
        check_cond(map.branch_mask == expect_mask && map.branch_mask != 0, "branch_mask exact");

        // Edge case 1: full trunk match -> zero compaction, no branch hit
        int32_t outv[kVerifyMaxT] = {20, 30, 40, 50, 0, 0, 0, 0};
        int path[kVerifyMaxT] = {};
        bool needs_compact = true, branch_hit = true;
        int k_acc = walk_accepted_tree(map, win, outv, path, needs_compact, branch_hit);
        check_cond(k_acc == 4 && !needs_compact && !branch_hit, "trunk match zero-copy fast path");
        for (int i = 0; i < 4; ++i) check_cond(path[i] == i, "trunk path[i] == i");

        // Edge case 2: root divergence (miss trunk and all root children) -> k_acc == 1
        int32_t outv_div[kVerifyMaxT] = {999, 30, 40, 50, 0, 0, 0, 0};
        k_acc = walk_accepted_tree(map, win, outv_div, path, needs_compact, branch_hit);
        check_cond(k_acc == 1 && !needs_compact && !branch_hit && path[0] == 0, "root divergence accepts 1");

        // Edge case 3: sfx diverged at depth 1 (parent 1 -> slot 4 = 99 -> slot 5 = 100 -> slot 6 = 101)
        // Find slot holding 99 with parent 1
        int s99 = -1, s100 = -1;
        for (int t = 4; t < T_out; ++t) {
            if (map.parent[t] == 1 && win[t] == 99) s99 = t;
            if (s99 >= 0 && map.parent[t] == s99 && win[t] == 100) s100 = t;
        }
        check_cond(s99 >= 4 && s100 > s99, "sfx branch chain materialized");
        int32_t outv_br[kVerifyMaxT] = {};
        outv_br[0] = 20;   // accept trunk slot 1
        outv_br[1] = 99;   // diverge to s99!
        outv_br[s99] = 100;
        outv_br[s100] = 777;
        k_acc = walk_accepted_tree(map, win, outv_br, path, needs_compact, branch_hit);
        check_cond(k_acc == 4 && needs_compact && branch_hit, "branch walk triggers compaction");
        check_cond(path[0] == 0 && path[1] == 1 && path[2] == s99 && path[3] == s100, "branch path exact");

        // Verify penalty_rows_tree matches linear decode along each branch
        const int h = 6;
        std::vector<int32_t> tail = {1, 2, 3, 4, 5};
        std::vector<int32_t> tree_hist((size_t)T_out * h, -99);
        penalty_rows_tree(tail.data(), (int64_t)tail.size(), win, map, h, tree_hist.data());
        for (int t = 0; t < T_out; ++t) {
            std::vector<int32_t> lin_win;
            for (int d = 0; d <= map.depth[t]; ++d) lin_win.push_back(win[map.anc_slot[t][d]]);
            std::vector<int32_t> lin_hist(lin_win.size() * h, -99);
            penalty_rows(tail.data(), (int64_t)tail.size(), lin_win.data(), (int)lin_win.size(), h, lin_hist.data());
            const int32_t* got_row = tree_hist.data() + (size_t)t * h;
            const int32_t* exp_row = lin_hist.data() + (size_t)map.depth[t] * h;
            for (int j = 0; j < h; ++j) check_cond(got_row[j] == exp_row[j], "penalty_rows_tree parity");
        }
    }

    // 1c. Mode 1 cap (<= 6) and Mode 3 tile-4 discipline (<= max(T_base, 4))
    {
        int32_t win[kVerifyMaxT] = {1, 2, 3, 4};
        float dprob[kVerifyMaxT] = {0.95f, 0.92f, 0.90f};
        SpecTreeInput in;
        in.mode = 3;
        in.pos0 = 50;
        in.T_base = 4;
        in.max_room = 8;
        in.draft_on = true;
        in.mtp_n = 3;
        in.drafts = win + 1;
        in.dprob = dprob;
        in.mtp_top2_0 = 55;
        in.mtp_top2_p0 = 0.04f;  // weak runner-up vs strong 0.95*0.92*0.90 tail -> no replacement, stays linear
        int T_out = 0;
        VerifyTreeMap map{};
        check_cond(!build_spec_tree(in, win, T_out, map) && T_out == 4, "mode 3 skips weak replacement at T=4");

        // Now strong top2 at root + weak tail -> tail replaced, T_out == 4 (fits tile-4!)
        dprob[0] = 0.6f; dprob[1] = 0.4f; dprob[2] = 0.3f;
        in.mtp_top2_p0 = 0.25f;
        check_cond(build_spec_tree(in, win, T_out, map) && T_out == 4, "mode 3 replaces slot 3 within tile-4");
        check_cond(map.parent[3] == 0 && win[3] == 55, "mode 3 slot 3 holds root branch");

        // Mode 1 bounds check
        int32_t win1[kVerifyMaxT] = {1, 2, 3, 4};
        int32_t sbuf1[kVerifyMaxT] = {9, 10, 11, 12};
        SpecTreeInput in1 = in;
        in1.mode = 1;
        in1.mtp_top2_p0 = 0.25f;
        in1.mtp_top2_1 = 66;
        in1.mtp_top2_p1 = 0.20f;
        in1.sfx_k = 4;
        in1.sfx_match = 4;
        in1.sbuf = sbuf1;
        check_cond(build_spec_tree(in1, win1, T_out, map) && T_out <= 6, "mode 1 bounded by 6");
    }

    // 1d. Sibling isolation in walk_accepted_tree: outv[0] matching slot whose parent != 0 is ignored
    {
        VerifyTreeMap map{};
        map.pos0 = 10;
        map.n_tok = 4;
        map.parent[0] = -1; map.depth[0] = 0;
        map.parent[1] = 0;  map.depth[1] = 1;
        map.parent[2] = 1;  map.depth[2] = 2;
        map.parent[3] = 1;  map.depth[3] = 2;
        int32_t win[4] = {10, 20, 30, 99};
        int32_t outv[4] = {99, 0, 0, 0};  // 99 is child of 1, NOT child of 0!
        int path[4] = {};
        bool nc = false, bh = false;
        const int k = walk_accepted_tree(map, win, outv, path, nc, bh);
        check_cond(k == 1 && !nc && !bh, "walk_accepted_tree ignores non-child match");
    }

    std::printf("test_spec_tree_cpu: %s\n", g_failures == 0 ? "PASS" : "FAIL");
}

// ---- 2. SYCL GPU Kernel Tests ----

void test_row_top2_mapped(sycl::queue& q) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    int errs = 0;

    for (int nv : {2, 2048, 15193, 32000}) {
        for (bool use_dvocab : {false, true}) {
            std::vector<float> h_logits(nv);
            for (float& v : h_logits) v = dist(rng);
            // Force a deterministic clear top1 and top2, plus test a tie case when nv == 2048
            if (nv == 2048 && !use_dvocab) {
                h_logits[500] = 12.0f;
                h_logits[1234] = 11.25f;
                h_logits[777] = 11.25f;  // ascending tie should pick 777
            }

            std::vector<int32_t> h_dvocab(nv);
            for (int i = 0; i < nv; ++i) h_dvocab[i] = 100000 + i * 3;

            const size_t arg_scratch_bytes = (size_t)strata::kernels::argmax_rows_scratch_bytes(1);
            float* d_logits = sycl::malloc_device<float>(nv, q);
            void* d_arg_scratch = sycl::malloc_device(arg_scratch_bytes, q);
            int32_t* d_top1_id = sycl::malloc_device<int32_t>(1, q);
            float* d_top1_prob = sycl::malloc_device<float>(1, q);
            int32_t* d_dvocab = use_dvocab ? sycl::malloc_device<int32_t>(nv, q) : nullptr;
            int32_t* h_out2 = sycl::malloc_host<int32_t>(2, q);
            float* h_prob2 = sycl::malloc_host<float>(2, q);
            h_out2[0] = -1;
            h_prob2[0] = -1.0f;

            q.memset(d_arg_scratch, 0, arg_scratch_bytes);
            q.memcpy(d_logits, h_logits.data(), (size_t)nv * sizeof(float));
            if (d_dvocab) q.memcpy(d_dvocab, h_dvocab.data(), (size_t)nv * sizeof(int32_t));
            q.wait();

            strata::kernels::argmax_rows(d_logits, 1, nv, d_arg_scratch, d_top1_id, &q);
            strata::kernels::row_top_prob(d_logits, 1, nv, d_top1_id, d_top1_prob, &q);
            strata::kernels::row_top2_mapped(d_logits, nv, d_top1_id, d_top1_prob, d_dvocab, h_out2, h_prob2, &q);
            q.wait();

            int32_t got_top1 = -1;
            float got_prob1 = 0.0f;
            q.memcpy(&got_top1, d_top1_id, sizeof(int32_t));
            q.memcpy(&got_prob1, d_top1_prob, sizeof(float));
            q.wait();

            // CPU FP64 reference
            int ref_top1 = 0;
            for (int i = 1; i < nv; ++i) {
                if (h_logits[i] > h_logits[ref_top1]) ref_top1 = i;
            }
            int ref_top2 = -1;
            for (int i = 0; i < nv; ++i) {
                if (i == ref_top1) continue;
                if (ref_top2 < 0 || h_logits[i] > h_logits[ref_top2]) ref_top2 = i;
            }
            double sum_exp = 0.0;
            const double m = h_logits[ref_top1];
            for (int i = 0; i < nv; ++i) sum_exp += std::exp((double)h_logits[i] - m);
            const double ref_p1 = 1.0 / sum_exp;
            const double ref_p2 = ref_p1 * std::exp((double)h_logits[ref_top2] - m);
            const int32_t expect_id2 = use_dvocab ? h_dvocab[ref_top2] : ref_top2;

            if (got_top1 != ref_top1 || h_out2[0] != expect_id2 ||
                std::fabs((double)h_prob2[0] - ref_p2) > 2e-5) {
                std::fprintf(stderr,
                             "row_top2_mapped mismatch (nv=%d dvocab=%d): got id2=%d (exp=%d) prob2=%.6f (exp=%.6f)\n",
                             nv, (int)use_dvocab, h_out2[0], expect_id2, h_prob2[0], ref_p2);
                ++errs;
            }

            sycl::free(d_logits, q);
            sycl::free(d_arg_scratch, q);
            sycl::free(d_top1_id, q);
            sycl::free(d_top1_prob, q);
            if (d_dvocab) sycl::free(d_dvocab, q);
            sycl::free(h_out2, q);
            sycl::free(h_prob2, q);
        }
    }

    check_cond(errs == 0, "row_top2_mapped GPU vs CPU reference");
    std::printf("test_row_top2_mapped: %s\n", errs == 0 ? "PASS" : "FAIL");
}

void test_verify_compact_row(sycl::queue& q) {
    constexpr int n_layers = 12;
    constexpr int MT = 8;
    constexpr int row_floats = 5120;
    constexpr int64_t layer_stride = (int64_t)MT * row_floats;
    const size_t total = (size_t)n_layers * layer_stride;

    std::vector<float> h_init(total);
    for (int l = 0; l < n_layers; ++l) {
        for (int r = 0; r < MT; ++r) {
            for (int c = 0; c < row_floats; ++c) {
                h_init[(size_t)l * layer_stride + (size_t)r * row_floats + c] =
                    (float)(l * 100 + r) + 0.001f * (float)(c % 251);
            }
        }
    }

    float* d_buf = sycl::malloc_device<float>(total, q);
    q.memcpy(d_buf, h_init.data(), total * sizeof(float)).wait();

    // Compact path {0, 4, 6}: move 4 -> 1, then 6 -> 2
    strata::kernels::verify_compact_row(d_buf, layer_stride, n_layers, 4, 1, row_floats, &q);
    strata::kernels::verify_compact_row(d_buf, layer_stride, n_layers, 6, 2, row_floats, &q);
    q.wait();

    std::vector<float> h_got(total);
    q.memcpy(h_got.data(), d_buf, total * sizeof(float)).wait();

    int errs = 0;
    for (int l = 0; l < n_layers; ++l) {
        for (int r = 0; r < MT; ++r) {
            const int src_row = (r == 1) ? 4 : (r == 2) ? 6 : r;
            for (int c = 0; c < row_floats; ++c) {
                const float exp_v = h_init[(size_t)l * layer_stride + (size_t)src_row * row_floats + c];
                const float got_v = h_got[(size_t)l * layer_stride + (size_t)r * row_floats + c];
                if (exp_v != got_v) ++errs;
            }
        }
    }

    sycl::free(d_buf, q);
    check_cond(errs == 0, "verify_compact_row across 12 layers");
    std::printf("test_verify_compact_row: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
}

void test_kv_copy_cell(sycl::queue& q) {
    using namespace strata::kernels;
    const QsaShapes s = qsa_real_shapes();
    constexpr int64_t n_blocks = 8;
    constexpr int64_t n_slots = 4;  // test paged/host split too
    constexpr int64_t src_pos = 15; // block 3, offset 3
    constexpr int64_t dst_pos = 13; // block 3, offset 1 (same block)
    constexpr int64_t src2 = 22;    // block 5, offset 2 (cross-block)
    constexpr int64_t dst2 = 14;    // block 3, offset 2

    int errs = 0;

    // A) Resident FP16 pool with identity page_table
    {
        const size_t cells = (size_t)n_blocks * s.page_size;
        const size_t pool_elems = cells * s.n_head_kv * s.head_dim;
        std::vector<uint16_t> h_k(pool_elems), h_v(pool_elems);
        for (size_t i = 0; i < pool_elems; ++i) {
            h_k[i] = (uint16_t)((i * 7 + 13) & 0xffff);
            h_v[i] = (uint16_t)((i * 11 + 29) & 0xffff);
        }
        std::vector<int32_t> pt(n_blocks);
        std::iota(pt.begin(), pt.end(), 0);

        uint16_t* d_k = sycl::malloc_device<uint16_t>(pool_elems, q);
        uint16_t* d_v = sycl::malloc_device<uint16_t>(pool_elems, q);
        int32_t* d_pt = sycl::malloc_device<int32_t>(n_blocks, q);
        q.memcpy(d_k, h_k.data(), pool_elems * sizeof(uint16_t));
        q.memcpy(d_v, h_v.data(), pool_elems * sizeof(uint16_t));
        q.memcpy(d_pt, pt.data(), n_blocks * sizeof(int32_t));
        q.wait();

        QsaAttnPools pools{};
        pools.k_pool = d_k;
        pools.v_pool = d_v;
        pools.page_table = d_pt;
        KvHostPools empty_host{};
        kv_copy_cell(pools, empty_host, d_pt, kKvF16, src_pos, dst_pos, s, &q);
        kv_copy_cell(pools, empty_host, d_pt, kKvF16, src2, dst2, s, &q);
        q.wait();

        std::vector<uint16_t> got_k(pool_elems), got_v(pool_elems);
        q.memcpy(got_k.data(), d_k, pool_elems * sizeof(uint16_t));
        q.memcpy(got_v.data(), d_v, pool_elems * sizeof(uint16_t));
        q.wait();

        auto cell_idx = [&](int64_t pos, int h) -> size_t {
            const int64_t blk = pos / s.page_size, off = pos % s.page_size;
            return (size_t)((blk * s.n_head_kv + h) * s.page_size + off) * s.head_dim;
        };
        for (int64_t pos = 0; pos < (int64_t)cells; ++pos) {
            const int64_t expect_pos = (pos == dst_pos) ? src_pos : (pos == dst2) ? src2 : pos;
            for (int h = 0; h < s.n_head_kv; ++h) {
                const size_t gi = cell_idx(pos, h);
                const size_t ei = cell_idx(expect_pos, h);
                for (int d = 0; d < s.head_dim; ++d) {
                    if (got_k[gi + d] != h_k[ei + d] || got_v[gi + d] != h_v[ei + d]) ++errs;
                }
            }
        }
        sycl::free(d_k, q);
        sycl::free(d_v, q);
        sycl::free(d_pt, q);
    }

    // B) Paged INT8 pool + pinned host mirror (source non-resident ssl=-1, dst resident dsl>=0)
    {
        const size_t host_q_elems = (size_t)n_blocks * s.n_head_kv * s.page_size * s.head_dim;
        const size_t host_s_elems = (size_t)n_blocks * s.n_head_kv * s.page_size * (s.head_dim / 64);
        const size_t dev_q_elems = (size_t)n_slots * s.n_head_kv * s.page_size * s.head_dim;
        const size_t dev_s_elems = (size_t)n_slots * s.n_head_kv * s.page_size * (s.head_dim / 64);

        int8_t* h_kq = sycl::malloc_host<int8_t>(host_q_elems, q);
        int8_t* h_vq = sycl::malloc_host<int8_t>(host_q_elems, q);
        uint16_t* h_ks = sycl::malloc_host<uint16_t>(host_s_elems, q);
        uint16_t* h_vs = sycl::malloc_host<uint16_t>(host_s_elems, q);
        int8_t* d_kq = sycl::malloc_device<int8_t>(dev_q_elems, q);
        int8_t* d_vq = sycl::malloc_device<int8_t>(dev_q_elems, q);
        uint16_t* d_ks = sycl::malloc_device<uint16_t>(dev_s_elems, q);
        uint16_t* d_vs = sycl::malloc_device<uint16_t>(dev_s_elems, q);
        int32_t* d_pt = sycl::malloc_device<int32_t>(n_blocks, q);

        for (size_t i = 0; i < host_q_elems; ++i) {
            h_kq[i] = (int8_t)((int)(i * 13 + 3) % 127 - 60);
            h_vq[i] = (int8_t)((int)(i * 17 + 5) % 127 - 60);
        }
        for (size_t i = 0; i < host_s_elems; ++i) {
            h_ks[i] = (uint16_t)(0x3c00u + (i % 17));
            h_vs[i] = (uint16_t)(0x3800u + (i % 19));
        }
        std::vector<int8_t> init_dev_q(dev_q_elems, 0);
        std::vector<uint16_t> init_dev_s(dev_s_elems, 0);
        // Block 3 resident at slot 2; block 5 evicted (-1) so read comes from host!
        std::vector<int32_t> pt(n_blocks, -1);
        pt[3] = 2;
        q.memcpy(d_kq, init_dev_q.data(), dev_q_elems);
        q.memcpy(d_vq, init_dev_q.data(), dev_q_elems);
        q.memcpy(d_ks, init_dev_s.data(), dev_s_elems * sizeof(uint16_t));
        q.memcpy(d_vs, init_dev_s.data(), dev_s_elems * sizeof(uint16_t));
        q.memcpy(d_pt, pt.data(), n_blocks * sizeof(int32_t)).wait();

        QsaAttnPools pools{};
        pools.k_q = d_kq; pools.v_q = d_vq;
        pools.k_scale = d_ks; pools.v_scale = d_vs;
        pools.page_table = d_pt;
        KvHostPools host{};
        host.k_q = h_kq; host.v_q = h_vq;
        host.k_scale = h_ks; host.v_scale = h_vs;

        // Snapshot expected source bytes from block 5 offset 2 (pos 22) -> copy to block 3 offset 2 (pos 14, slot 2)
        kv_copy_cell(pools, host, d_pt, kKvInt8, src2, dst2, s, &q);
        q.wait();

        std::vector<int8_t> got_dkq(dev_q_elems), got_dvq(dev_q_elems);
        std::vector<uint16_t> got_dks(dev_s_elems), got_dvs(dev_s_elems);
        q.memcpy(got_dkq.data(), d_kq, dev_q_elems);
        q.memcpy(got_dvq.data(), d_vq, dev_q_elems);
        q.memcpy(got_dks.data(), d_ks, dev_s_elems * sizeof(uint16_t));
        q.memcpy(got_dvs.data(), d_vs, dev_s_elems * sizeof(uint16_t));
        q.wait();

        const int64_t sb = src2 / s.page_size, soff = src2 % s.page_size;
        const int64_t db = dst2 / s.page_size, doff = dst2 % s.page_size;
        const int64_t dsl = pt[db];
        for (int h = 0; h < s.n_head_kv; ++h) {
            const size_t sq = (size_t)((sb * s.n_head_kv + h) * s.page_size + soff) * s.head_dim;
            const size_t hq = (size_t)((db * s.n_head_kv + h) * s.page_size + doff) * s.head_dim;
            const size_t dq = (size_t)((dsl * s.n_head_kv + h) * s.page_size + doff) * s.head_dim;
            for (int d = 0; d < s.head_dim; ++d) {
                if (h_kq[hq + d] != h_kq[sq + d] || got_dkq[dq + d] != h_kq[sq + d] ||
                    h_vq[hq + d] != h_vq[sq + d] || got_dvq[dq + d] != h_vq[sq + d])
                    ++errs;
            }
            const int n_sc = (int)(s.head_dim / 64);
            const size_t ss = (size_t)((sb * s.n_head_kv + h) * s.page_size + soff) * n_sc;
            const size_t hs = (size_t)((db * s.n_head_kv + h) * s.page_size + doff) * n_sc;
            const size_t ds = (size_t)((dsl * s.n_head_kv + h) * s.page_size + doff) * n_sc;
            for (int sc = 0; sc < n_sc; ++sc) {
                if (h_ks[hs + sc] != h_ks[ss + sc] || got_dks[ds + sc] != h_ks[ss + sc] ||
                    h_vs[hs + sc] != h_vs[ss + sc] || got_dvs[ds + sc] != h_vs[ss + sc])
                    ++errs;
            }
        }

        sycl::free(h_kq, q); sycl::free(h_vq, q); sycl::free(h_ks, q); sycl::free(h_vs, q);
        sycl::free(d_kq, q); sycl::free(d_vq, q); sycl::free(d_ks, q); sycl::free(d_vs, q);
        sycl::free(d_pt, q);
    }

    // C) Resident Q4_0 pool
    {
        constexpr size_t q4_cell_bytes = (256 / 32) * 18;  // 144 B
        const size_t pool_bytes = (size_t)n_blocks * s.n_head_kv * s.page_size * q4_cell_bytes;
        std::vector<uint8_t> h_kq4(pool_bytes), h_vq4(pool_bytes);
        for (size_t i = 0; i < pool_bytes; ++i) {
            h_kq4[i] = (uint8_t)((i * 31 + 7) & 0xff);
            h_vq4[i] = (uint8_t)((i * 43 + 19) & 0xff);
        }
        std::vector<int32_t> pt(n_blocks);
        std::iota(pt.begin(), pt.end(), 0);

        uint8_t* d_kq4 = sycl::malloc_device<uint8_t>(pool_bytes, q);
        uint8_t* d_vq4 = sycl::malloc_device<uint8_t>(pool_bytes, q);
        int32_t* d_pt = sycl::malloc_device<int32_t>(n_blocks, q);
        q.memcpy(d_kq4, h_kq4.data(), pool_bytes);
        q.memcpy(d_vq4, h_vq4.data(), pool_bytes);
        q.memcpy(d_pt, pt.data(), n_blocks * sizeof(int32_t)).wait();

        QsaAttnPools pools{};
        pools.k_q4 = d_kq4; pools.v_q4 = d_vq4; pools.page_table = d_pt;
        KvHostPools empty_host{};
        kv_copy_cell(pools, empty_host, d_pt, kKvQ4, src_pos, dst_pos, s, &q);
        q.wait();

        std::vector<uint8_t> got_k(pool_bytes), got_v(pool_bytes);
        q.memcpy(got_k.data(), d_kq4, pool_bytes);
        q.memcpy(got_v.data(), d_vq4, pool_bytes);
        q.wait();

        auto q4_off = [&](int64_t pos, int h) -> size_t {
            const int64_t b = pos / s.page_size, o = pos % s.page_size;
            return (size_t)((b * s.n_head_kv + h) * s.page_size + o) * q4_cell_bytes;
        };
        for (int64_t pos = 0; pos < n_blocks * s.page_size; ++pos) {
            const int64_t expect_pos = (pos == dst_pos) ? src_pos : pos;
            for (int h = 0; h < s.n_head_kv; ++h) {
                const size_t g0 = q4_off(pos, h), e0 = q4_off(expect_pos, h);
                for (size_t b = 0; b < q4_cell_bytes; ++b) {
                    if (got_k[g0 + b] != h_kq4[e0 + b] || got_v[g0 + b] != h_vq4[e0 + b]) ++errs;
                }
            }
        }
        sycl::free(d_kq4, q); sycl::free(d_vq4, q); sycl::free(d_pt, q);
    }

    check_cond(errs == 0, "kv_copy_cell (F16 + paged INT8/host + Q4)");
    std::printf("test_kv_copy_cell: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
}

void test_gdn_tree_parity(sycl::queue& q) {
    using namespace strata::kernels;
    constexpr int S = 128, h_k = 4, h_v = 8;
    constexpr int qk = S * h_k, C = 2 * qk + S * h_v, Vdim = S * h_v;
    constexpr float eps = 1e-6f;
    // Build a 6-node multi-branch tree:
    // trunk: 0 -> 1 -> 2
    // alt at root: 0 -> 3 -> 5
    // alt at 1:    1 -> 4
    VerifyTreeMap map{};
    map.pos0 = 32;
    map.n_tok = 6;
    const int parents[6] = {-1, 0, 1, 0, 1, 3};
    const int depths[6]  = {0,  1, 2, 1, 2, 2};
    for (int t = 0; t < 6; ++t) {
        map.parent[t] = parents[t];
        map.depth[t] = depths[t];
        if (t > 0 && parents[t] != t - 1) map.branch_mask |= (1u << (unsigned)parents[t]);
        int cur = t;
        for (int d = depths[t]; d >= 0; --d) {
            map.anc_slot[t][d] = cur;
            cur = parents[cur];
        }
        for (int j = 0; j < 4; ++j) {
            const int rel = depths[t] - (3 - j);
            map.anc_conv[t][j] = rel < 0 ? rel : map.anc_slot[t][rel];
        }
    }

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<float> h_hist0(C * 3), h_conv_w(C * 4), h_qkv(6 * C);
    std::vector<float> h_state0((size_t)S * h_v * S), h_gate(6 * h_v), h_beta(6 * h_v);
    std::vector<float> h_z(6 * Vdim), h_gamma(S);
    for (auto& v : h_hist0) v = dist(rng);
    for (auto& v : h_conv_w) v = dist(rng);
    for (auto& v : h_qkv) v = dist(rng);
    for (auto& v : h_state0) v = dist(rng);
    for (auto& v : h_gate) v = -0.2f + dist(rng);
    for (auto& v : h_beta) v = 0.5f + dist(rng);
    for (auto& v : h_z) v = dist(rng);
    for (auto& v : h_gamma) v = 0.8f + dist(rng);

    const size_t snap_floats = (size_t)kVerifyNarrowT * h_v * S * S;
    float* d_hist = sycl::malloc_device<float>(h_hist0.size(), q);
    float* d_conv_w = sycl::malloc_device<float>(h_conv_w.size(), q);
    float* d_qkv = sycl::malloc_device<float>(h_qkv.size(), q);
    float* d_h = sycl::malloc_device<float>(h_qkv.size(), q);
    float* d_state = sycl::malloc_device<float>(h_state0.size(), q);
    float* d_snap = sycl::malloc_device<float>(snap_floats, q);
    float* d_gate = sycl::malloc_device<float>(h_gate.size(), q);
    float* d_beta = sycl::malloc_device<float>(h_beta.size(), q);
    float* d_z = sycl::malloc_device<float>(h_z.size(), q);
    float* d_gamma = sycl::malloc_device<float>(h_gamma.size(), q);
    float* d_y_tree = sycl::malloc_device<float>(6 * Vdim, q);
    VerifyTreeMap* d_tree = sycl::malloc_device<VerifyTreeMap>(1, q);

    q.memcpy(d_hist, h_hist0.data(), h_hist0.size() * sizeof(float));
    q.memcpy(d_conv_w, h_conv_w.data(), h_conv_w.size() * sizeof(float));
    q.memcpy(d_qkv, h_qkv.data(), h_qkv.size() * sizeof(float));
    q.memcpy(d_state, h_state0.data(), h_state0.size() * sizeof(float));
    q.memset(d_snap, 0, snap_floats * sizeof(float));
    q.memcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float));
    q.memcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float));
    q.memcpy(d_z, h_z.data(), h_z.size() * sizeof(float));
    q.memcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float));
    q.memcpy(d_tree, &map, sizeof(VerifyTreeMap));
    q.wait();

    gdn_conv_l2_tree(d_hist, d_qkv, d_conv_w, d_h, C, 2 * h_k, eps, 6, d_tree, &q);
    gdn_step_norm_tree(d_state, d_snap, d_h, C, d_gate, d_beta, d_z, d_gamma, eps, d_y_tree, h_k, h_v, 6, d_tree, &q);
    q.wait();

    std::vector<float> h_y_tree(6 * Vdim);
    q.memcpy(h_y_tree.data(), d_y_tree, h_y_tree.size() * sizeof(float)).wait();

    // Verify each branch path independently using sequential fused_gdn_conv_l2 + fused_gdn_step_norm
    const std::vector<std::vector<int>> paths = {{0, 1, 2}, {0, 3, 5}, {0, 1, 4}};
    float* d_y_ref = sycl::malloc_device<float>(Vdim, q);
    float* d_h_ref = sycl::malloc_device<float>(C, q);
    float* d_qkv1 = sycl::malloc_device<float>(C, q);
    float* d_g1 = sycl::malloc_device<float>(h_v, q);
    float* d_b1 = sycl::malloc_device<float>(h_v, q);
    float* d_z1 = sycl::malloc_device<float>(Vdim, q);

    float max_diff = 0.0f;
    for (const auto& path : paths) {
        q.memcpy(d_hist, h_hist0.data(), h_hist0.size() * sizeof(float));
        q.memcpy(d_state, h_state0.data(), h_state0.size() * sizeof(float)).wait();
        for (int slot : path) {
            q.memcpy(d_qkv1, h_qkv.data() + (size_t)slot * C, C * sizeof(float));
            q.memcpy(d_g1, h_gate.data() + (size_t)slot * h_v, h_v * sizeof(float));
            q.memcpy(d_b1, h_beta.data() + (size_t)slot * h_v, h_v * sizeof(float));
            q.memcpy(d_z1, h_z.data() + (size_t)slot * Vdim, Vdim * sizeof(float)).wait();

            fused_gdn_conv_l2(d_hist, d_qkv1, d_conv_w, d_h_ref, C, 2 * h_k, eps, &q);
            fused_gdn_step_norm(d_state, d_h_ref, d_h_ref + qk, d_h_ref + 2 * qk, d_g1, d_b1, d_z1, d_gamma, eps,
                                d_y_ref, h_k, h_v, &q);
            q.wait();

            std::vector<float> ref_y(Vdim);
            q.memcpy(ref_y.data(), d_y_ref, Vdim * sizeof(float)).wait();
            for (int i = 0; i < Vdim; ++i) {
                const float diff = std::fabs(h_y_tree[(size_t)slot * Vdim + i] - ref_y[i]);
                max_diff = std::max(max_diff, diff);
            }
        }
    }

    sycl::free(d_hist, q); sycl::free(d_conv_w, q); sycl::free(d_qkv, q); sycl::free(d_h, q);
    sycl::free(d_state, q); sycl::free(d_snap, q); sycl::free(d_gate, q); sycl::free(d_beta, q);
    sycl::free(d_z, q); sycl::free(d_gamma, q); sycl::free(d_y_tree, q); sycl::free(d_tree, q);
    sycl::free(d_y_ref, q); sycl::free(d_h_ref, q); sycl::free(d_qkv1, q);
    sycl::free(d_g1, q); sycl::free(d_b1, q); sycl::free(d_z1, q);

    check_cond(max_diff < 1e-5f, "gdn_step_norm_tree matches step-by-step single-branch decode");
    std::printf("test_gdn_tree_parity: %s (max_diff=%.2e)\n", max_diff < 1e-5f ? "PASS" : "FAIL", max_diff);
}

void test_qsa_causal_tree_mask(sycl::queue& q) {
    using namespace strata::kernels;
    const QsaShapes s = qsa_real_shapes();
    constexpr int64_t pos0 = 12;  // 3 past blocks (cells 0..11)
    constexpr int T = 6;          // tree: trunk 0->1->2, branch A 0->3->5, branch B 1->4
    constexpr int64_t n_blocks = 8;
    const int64_t cap = qsa_selection_width(pos0 + T, s);

    VerifyTreeMap map{};
    map.pos0 = pos0;
    map.n_tok = T;
    const int parents[T] = {-1, 0, 1, 0, 1, 3};
    const int depths[T]  = {0,  1, 2, 1, 2, 2};
    for (int t = 0; t < T; ++t) {
        map.parent[t] = parents[t];
        map.depth[t] = depths[t];
        int cur = t;
        for (int d = depths[t]; d >= 0; --d) {
            map.anc_slot[t][d] = cur;
            cur = parents[cur];
        }
    }

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    const size_t pool_elems = (size_t)n_blocks * s.n_head_kv * s.page_size * s.head_dim;
    const size_t q_elems = (size_t)T * s.n_head * s.head_dim;
    const size_t kv_step_elems = (size_t)s.n_head_kv * s.head_dim;

    std::vector<uint16_t> h_k16(pool_elems, 0), h_v16(pool_elems, 0);
    std::vector<int32_t> pt(n_blocks);
    std::iota(pt.begin(), pt.end(), 0);
    // Past KV cells 0..11
    for (int64_t c = 0; c < pos0; ++c) {
        const int64_t blk = c / s.page_size, off = c % s.page_size;
        for (int h = 0; h < s.n_head_kv; ++h) {
            const size_t base = (size_t)((blk * s.n_head_kv + h) * s.page_size + off) * s.head_dim;
            for (int d = 0; d < s.head_dim; ++d) {
                h_k16[base + d] = f32_to_f16_bits(dist(rng));
                h_v16[base + d] = f32_to_f16_bits(dist(rng));
            }
        }
    }

    // Tree K/V appended at physical slots pos0 + 0 .. pos0 + 5, plus distinct queries per tree node.
    // Deliberately make sibling slot 1 K strongly aligned with slot 3 Q (branch A) with distinct V,
    // so ANY causal leak across siblings causes an immediate huge error!
    std::vector<float> h_q(q_elems), h_kt(T * kv_step_elems), h_vt(T * kv_step_elems);
    for (size_t i = 0; i < q_elems; ++i) h_q[i] = dist(rng);
    for (size_t i = 0; i < h_kt.size(); ++i) h_kt[i] = dist(rng);
    for (size_t i = 0; i < h_vt.size(); ++i) h_vt[i] = dist(rng);
    for (int h = 0; h < s.n_head_kv; ++h) {
        for (int d = 0; d < s.head_dim; ++d) {
            // Branch A query (slot 3) strongly aligns with trunk slot 1's key, whose V is huge (+8.0)
            h_kt[1 * kv_step_elems + h * s.head_dim + d] = 1.5f;
            h_vt[1 * kv_step_elems + h * s.head_dim + d] = 8.0f;
            h_vt[3 * kv_step_elems + h * s.head_dim + d] = -3.0f;
        }
    }
    for (int h = 0; h < s.n_head; ++h) {
        for (int d = 0; d < s.head_dim; ++d) {
            h_q[(size_t)3 * s.n_head * s.head_dim + h * s.head_dim + d] = 1.5f;
            h_q[(size_t)5 * s.n_head * s.head_dim + h * s.head_dim + d] = 1.5f;
        }
    }

    uint16_t* d_k16 = sycl::malloc_device<uint16_t>(pool_elems, q);
    uint16_t* d_v16 = sycl::malloc_device<uint16_t>(pool_elems, q);
    int32_t* d_pt = sycl::malloc_device<int32_t>(n_blocks, q);
    float* d_q = sycl::malloc_device<float>(q_elems, q);
    float* d_kt = sycl::malloc_device<float>(T * kv_step_elems, q);
    float* d_vt = sycl::malloc_device<float>(T * kv_step_elems, q);
    int32_t* d_steps = sycl::malloc_device<int32_t>(T * kStepCount, q);
    int32_t* d_steps_app = sycl::malloc_device<int32_t>(T * kStepCount, q);
    int32_t* d_sel = sycl::malloc_device<int32_t>(T * cap, q);
    VerifyTreeMap* d_tree = sycl::malloc_device<VerifyTreeMap>(1, q);
    const size_t scratch_n = (size_t)T * qsa_decode_attn_scratch_floats(cap, s);
    float* d_scratch = sycl::malloc_device<float>(scratch_n, q);
    float* d_attn = sycl::malloc_device<float>(q_elems, q);

    std::vector<int32_t> h_steps(T * kStepCount), h_steps_app(T * kStepCount), h_sel(T * cap, -1);
    for (int t = 0; t < T; ++t) {
        qsa_step_fill(h_steps.data() + t * kStepCount, pos0 + depths[t], s);
        qsa_step_fill(h_steps_app.data() + t * kStepCount, pos0 + t, s);
        // Simulate raw top-k logical selection 0 .. pos0 + depths[t] (as qsa_block_topk returns in dense regime)
        const int w = h_steps[t * kStepCount + kStepWidth];
        for (int j = 0; j < w; ++j) h_sel[(size_t)t * cap + j] = j;
    }

    q.memcpy(d_k16, h_k16.data(), pool_elems * sizeof(uint16_t));
    q.memcpy(d_v16, h_v16.data(), pool_elems * sizeof(uint16_t));
    q.memcpy(d_pt, pt.data(), n_blocks * sizeof(int32_t));
    q.memcpy(d_q, h_q.data(), q_elems * sizeof(float));
    q.memcpy(d_kt, h_kt.data(), T * kv_step_elems * sizeof(float));
    q.memcpy(d_vt, h_vt.data(), T * kv_step_elems * sizeof(float));
    q.memcpy(d_steps, h_steps.data(), h_steps.size() * sizeof(int32_t));
    q.memcpy(d_steps_app, h_steps_app.data(), h_steps_app.size() * sizeof(int32_t));
    q.memcpy(d_sel, h_sel.data(), h_sel.size() * sizeof(int32_t));
    q.memcpy(d_tree, &map, sizeof(VerifyTreeMap));
    q.wait();

    QsaAttnPools pools{};
    pools.k_pool = d_k16;
    pools.v_pool = d_v16;
    pools.page_table = d_pt;

    // 1. Append tree K/V at physical cells pos0 + 0 .. pos0 + 5
    for (int t = 0; t < T; ++t) {
        kv_append_step(d_k16, d_v16, d_pt, d_steps_app + t * kStepCount, d_kt + t * kv_step_elems,
                       d_vt + t * kv_step_elems, s, &q);
    }
    // 2. Remap selected IDs >= pos0 using tree->anc_slot[q][id - pos0]
    qsa_remap_tree_sel(d_sel, d_steps, T, cap, d_tree, &q);
    // 3. Batch attention over remapped causal cells
    q.memset(d_scratch, 0, scratch_n * sizeof(float));
    q.memset(d_attn, 0, q_elems * sizeof(float));
    qsa_decode_attn_batch(d_q, pools, d_sel, d_steps, cap, s, d_scratch, d_attn, T, &q);
    q.wait();

    std::vector<float> got_attn(q_elems);
    q.memcpy(got_attn.data(), d_attn, q_elems * sizeof(float)).wait();

    // Check remapped IDs explicitly + verify against isolated single-branch qsa_decode_attn_step!
    std::vector<int32_t> got_sel(T * cap);
    q.memcpy(got_sel.data(), d_sel, got_sel.size() * sizeof(int32_t)).wait();

    int remap_errs = 0;
    for (int t = 0; t < T; ++t) {
        const int w = h_steps[t * kStepCount + kStepWidth];
        if (w != (int)(pos0 + depths[t] + 1)) ++remap_errs;
        for (int d = 0; d <= depths[t]; ++d) {
            if (got_sel[(size_t)t * cap + pos0 + d] != pos0 + map.anc_slot[t][d]) ++remap_errs;
        }
    }
    check_cond(remap_errs == 0, "qsa_remap_tree_sel ancestor remapping exact");

    // Run isolated single-query attention per tree slot against a dedicated clean KV cache containing ONLY that slot's path
    uint16_t* d_iso_k = sycl::malloc_device<uint16_t>(pool_elems, q);
    uint16_t* d_iso_v = sycl::malloc_device<uint16_t>(pool_elems, q);
    float* d_iso_attn = sycl::malloc_device<float>((size_t)s.n_head * s.head_dim, q);
    QsaAttnPools iso_pools{};
    iso_pools.k_pool = d_iso_k;
    iso_pools.v_pool = d_iso_v;
    iso_pools.page_table = d_pt;

    float max_diff = 0.0f;
    for (int t = 0; t < T; ++t) {
        q.memcpy(d_iso_k, h_k16.data(), pool_elems * sizeof(uint16_t));
        q.memcpy(d_iso_v, h_v16.data(), pool_elems * sizeof(uint16_t)).wait();
        for (int d = 0; d <= depths[t]; ++d) {
            const int src_slot = map.anc_slot[t][d];
            // Put src_slot into logical cell pos0 + d using h_steps[d]?! Just use step with pos0+d!
            int32_t st_lin[kStepCount];
            qsa_step_fill(st_lin, pos0 + d, s);
            q.memcpy(d_steps_app, st_lin, sizeof(st_lin)).wait();
            kv_append_step(d_iso_k, d_iso_v, d_pt, d_steps_app, d_kt + src_slot * kv_step_elems,
                           d_vt + src_slot * kv_step_elems, s, &q);
        }
        // Single-query selection: linear [0 .. pos0 + depths[t]]
        const int w = (int)(pos0 + depths[t] + 1);
        std::vector<int32_t> iso_sel(cap, -1);
        for (int j = 0; j < w; ++j) iso_sel[j] = j;
        int32_t st_q[kStepCount];
        qsa_step_fill(st_q, pos0 + depths[t], s);
        q.memcpy(d_steps_app, st_q, sizeof(st_q));
        q.memcpy(d_sel, iso_sel.data(), cap * sizeof(int32_t));
        q.memset(d_scratch, 0, qsa_decode_attn_scratch_floats(cap, s) * sizeof(float));
        q.memset(d_iso_attn, 0, (size_t)s.n_head * s.head_dim * sizeof(float));
        q.wait();

        qsa_decode_attn_step(d_q + (size_t)t * s.n_head * s.head_dim, iso_pools, d_sel, d_steps_app, cap, s,
                             d_scratch, d_iso_attn, &q);
        q.wait();

        std::vector<float> ref_a((size_t)s.n_head * s.head_dim);
        q.memcpy(ref_a.data(), d_iso_attn, ref_a.size() * sizeof(float)).wait();
        for (size_t i = 0; i < ref_a.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(got_attn[(size_t)t * s.n_head * s.head_dim + i] - ref_a[i]));
        }
    }

    sycl::free(d_iso_k, q); sycl::free(d_iso_v, q); sycl::free(d_iso_attn, q);
    sycl::free(d_k16, q); sycl::free(d_v16, q); sycl::free(d_pt, q);
    sycl::free(d_q, q); sycl::free(d_kt, q); sycl::free(d_vt, q);
    sycl::free(d_steps, q); sycl::free(d_steps_app, q); sycl::free(d_sel, q);
    sycl::free(d_tree, q); sycl::free(d_scratch, q); sycl::free(d_attn, q);

    check_cond(max_diff < 1e-5f, "qsa_decode_attn_batch + qsa_remap_tree_sel causal tree isolation");
    std::printf("test_qsa_causal_tree_mask: %s (max_diff=%.2e)\n",
                (remap_errs == 0 && max_diff < 1e-5f) ? "PASS" : "FAIL", max_diff);
}

// ---- 3. Perf Benchmark & Multi-Branch Acceptance Evaluation (--bench) ----

void run_benchmarks(sycl::queue& q) {
    using namespace strata::kernels;
    using namespace strata::spec;
    std::printf("\n=== SpecTree & Top2 SYCL Kernel Microbenchmarks ===\n");
    constexpr int warmup = 10, reps = 100;

    // 1) row_top2_mapped
    for (int nv : {2048, 151936, 248320}) {
        float* d_logits = sycl::malloc_device<float>(nv, q);
        int32_t* d_t1 = sycl::malloc_device<int32_t>(1, q);
        float* d_p1 = sycl::malloc_device<float>(1, q);
        int32_t* d_dv = (nv == 2048) ? sycl::malloc_device<int32_t>(nv, q) : nullptr;
        int32_t* h_id2 = sycl::malloc_host<int32_t>(2, q);
        float* h_p2 = sycl::malloc_host<float>(2, q);
        q.fill(d_logits, 0.1f, nv);
        q.fill(d_t1, 10, 1);
        q.fill(d_p1, 0.6f, 1).wait();

        for (int i = 0; i < warmup; ++i) row_top2_mapped(d_logits, nv, d_t1, d_p1, d_dv, h_id2, h_p2, &q);
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) row_top2_mapped(d_logits, nv, d_t1, d_p1, d_dv, h_id2, h_p2, &q);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("BENCH row_top2_mapped (nv=%6d, dvocab=%d): %7.2f us/call\n", nv, d_dv != nullptr ? 1 : 0, us);

        sycl::free(d_logits, q); sycl::free(d_t1, q); sycl::free(d_p1, q);
        if (d_dv) sycl::free(d_dv, q);
        sycl::free(h_id2, q); sycl::free(h_p2, q);
    }

    // 2) verify_compact_row (36 GDN layers, C=8192)
    {
        constexpr int nL = 36, MT = 8, rf = 8192;
        float* d_buf = sycl::malloc_device<float>((size_t)nL * MT * rf, q);
        q.fill(d_buf, 0.0f, (size_t)nL * MT * rf).wait();
        for (int i = 0; i < warmup; ++i) verify_compact_row(d_buf, (int64_t)MT * rf, nL, 4, 1, rf, &q);
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) verify_compact_row(d_buf, (int64_t)MT * rf, nL, 4, 1, rf, &q);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("BENCH verify_compact_row (L=%d, row_floats=%d): %7.2f us/call\n", nL, rf, us);
        sycl::free(d_buf, q);
    }

    // 3) kv_copy_cell (F16, INT8, Q4 resident)
    {
        const QsaShapes s = qsa_real_shapes();
        constexpr int64_t n_blocks = 8;
        const size_t cells = (size_t)n_blocks * s.page_size;
        std::vector<int32_t> pt(n_blocks);
        std::iota(pt.begin(), pt.end(), 0);
        int32_t* d_pt = sycl::malloc_device<int32_t>(n_blocks, q);
        q.memcpy(d_pt, pt.data(), n_blocks * sizeof(int32_t)).wait();
        KvHostPools empty_host{};

        // F16
        uint16_t* d_kf16 = sycl::malloc_device<uint16_t>(cells * s.n_head_kv * s.head_dim, q);
        uint16_t* d_vf16 = sycl::malloc_device<uint16_t>(cells * s.n_head_kv * s.head_dim, q);
        QsaAttnPools p_f16{}; p_f16.k_pool = d_kf16; p_f16.v_pool = d_vf16; p_f16.page_table = d_pt;
        for (int i = 0; i < warmup; ++i) kv_copy_cell(p_f16, empty_host, d_pt, kKvF16, 15, 13, s, &q);
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) kv_copy_cell(p_f16, empty_host, d_pt, kKvF16, 15, 13, s, &q);
        q.wait();
        double us_f16 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        sycl::free(d_kf16, q); sycl::free(d_vf16, q);

        // INT8
        int8_t* d_kq8 = sycl::malloc_device<int8_t>(cells * s.n_head_kv * s.head_dim, q);
        int8_t* d_vq8 = sycl::malloc_device<int8_t>(cells * s.n_head_kv * s.head_dim, q);
        uint16_t* d_ks8 = sycl::malloc_device<uint16_t>(cells * s.n_head_kv * (s.head_dim / 64), q);
        uint16_t* d_vs8 = sycl::malloc_device<uint16_t>(cells * s.n_head_kv * (s.head_dim / 64), q);
        QsaAttnPools p_i8{}; p_i8.k_q = d_kq8; p_i8.v_q = d_vq8; p_i8.k_scale = d_ks8; p_i8.v_scale = d_vs8; p_i8.page_table = d_pt;
        for (int i = 0; i < warmup; ++i) kv_copy_cell(p_i8, empty_host, d_pt, kKvInt8, 15, 13, s, &q);
        q.wait();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) kv_copy_cell(p_i8, empty_host, d_pt, kKvInt8, 15, 13, s, &q);
        q.wait();
        double us_i8 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        sycl::free(d_kq8, q); sycl::free(d_vq8, q); sycl::free(d_ks8, q); sycl::free(d_vs8, q);

        std::printf("BENCH kv_copy_cell (F16): %7.2f us/call | (INT8): %7.2f us/call\n", us_f16, us_i8);
        sycl::free(d_pt, q);
    }

    // 4) qsa_remap_tree_sel (T=6, width=2052)
    {
        constexpr int T = 6;
        constexpr int64_t cap = 2052;
        int32_t* d_sel = sycl::malloc_device<int32_t>((size_t)T * cap, q);
        int32_t* d_steps = sycl::malloc_device<int32_t>(T * kStepCount, q);
        VerifyTreeMap* d_tree = sycl::malloc_device<VerifyTreeMap>(1, q);
        std::vector<int32_t> h_steps(T * kStepCount, 0);
        std::vector<int32_t> h_sel((size_t)T * cap, 0);
        for (int t = 0; t < T; ++t) {
            h_steps[t * kStepCount + kStepWidth] = (int32_t)cap;
            for (int j = 0; j < cap; ++j) h_sel[(size_t)t * cap + j] = j;
        }
        VerifyTreeMap hm{};
        hm.pos0 = 2046;
        hm.n_tok = T;
        hm.branch_mask = 0x3u;
        for (int t = 0; t < T; ++t) {
            hm.parent[t] = (t == 0) ? -1 : (t < 4 ? t - 1 : (t == 4 ? 0 : 1));
            for (int d = 0; d < kVerifyMaxT; ++d) hm.anc_slot[t][d] = std::min(d, t);
        }
        q.memcpy(d_sel, h_sel.data(), h_sel.size() * sizeof(int32_t));
        q.memcpy(d_steps, h_steps.data(), h_steps.size() * sizeof(int32_t));
        q.memcpy(d_tree, &hm, sizeof(VerifyTreeMap));
        q.wait();

        for (int i = 0; i < warmup; ++i) qsa_remap_tree_sel(d_sel, d_steps, T, cap, d_tree, &q);
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) qsa_remap_tree_sel(d_sel, d_steps, T, cap, d_tree, &q);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("BENCH qsa_remap_tree_sel (T=%d, width=%lld): %7.2f us/call\n", T, (long long)cap, us);
        sycl::free(d_sel, q); sycl::free(d_steps, q); sycl::free(d_tree, q);
    }

    // 5) gdn_step_norm_tree vs gdn_step_norm_multi (T=4,6,8; h_k=16, h_v=32)
    {
        constexpr int S = 128, h_k = 16, h_v = 32, C = 2 * S * h_k + S * h_v, Vdim = S * h_v;
        float* d_state = sycl::malloc_device<float>((size_t)S * h_v * S, q);
        float* d_snap = sycl::malloc_device<float>((size_t)kVerifyNarrowT * h_v * S * S, q);
        float* d_h = sycl::malloc_device<float>((size_t)kVerifyNarrowT * C, q);
        float* d_g = sycl::malloc_device<float>((size_t)kVerifyNarrowT * h_v, q);
        float* d_b = sycl::malloc_device<float>((size_t)kVerifyNarrowT * h_v, q);
        float* d_z = sycl::malloc_device<float>((size_t)kVerifyNarrowT * Vdim, q);
        float* d_gm = sycl::malloc_device<float>(S, q);
        float* d_y = sycl::malloc_device<float>((size_t)kVerifyNarrowT * Vdim, q);
        VerifyTreeMap* d_tree = sycl::malloc_device<VerifyTreeMap>(1, q);
        q.fill(d_state, 0.01f, (size_t)S * h_v * S);
        q.fill(d_h, 0.02f, (size_t)kVerifyNarrowT * C);
        q.fill(d_g, -0.3f, (size_t)kVerifyNarrowT * h_v);
        q.fill(d_b, 0.5f, (size_t)kVerifyNarrowT * h_v);
        q.fill(d_z, 0.1f, (size_t)kVerifyNarrowT * Vdim);
        q.fill(d_gm, 1.0f, S).wait();

        for (int T : {4, 6, 8}) {
            VerifyTreeMap hm{};
            hm.n_tok = T;
            hm.branch_mask = 0x3u;
            for (int t = 0; t < T; ++t) {
                hm.parent[t] = (t == 0) ? -1 : (t < T - 2 ? t - 1 : (t == T - 2 ? 0 : 1));
            }
            q.memcpy(d_tree, &hm, sizeof(VerifyTreeMap)).wait();

            for (int i = 0; i < warmup; ++i)
                gdn_step_norm_multi(d_state, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, h_k, h_v, T, nullptr, &q, 0);
            q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i)
                gdn_step_norm_multi(d_state, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, h_k, h_v, T, nullptr, &q, 0);
            q.wait();
            double us_lin = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;

            for (int i = 0; i < warmup; ++i)
                gdn_step_norm_tree(d_state, d_snap, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, h_k, h_v, T, d_tree, &q);
            q.wait();
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i)
                gdn_step_norm_tree(d_state, d_snap, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, h_k, h_v, T, d_tree, &q);
            q.wait();
            double us_tree = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;

            std::printf("BENCH GDN step_norm (T=%d, hv=%d) | linear: %7.2f us | tree: %7.2f us\n", T, h_v, us_lin, us_tree);
        }

        sycl::free(d_state, q); sycl::free(d_snap, q); sycl::free(d_h, q);
        sycl::free(d_g, q); sycl::free(d_b, q); sycl::free(d_z, q);
        sycl::free(d_gm, q); sycl::free(d_y, q); sycl::free(d_tree, q);
    }

    // 6) Simulated multi-branch acceptance evaluation (trunk vs spec-tree 1/2/3)
    std::printf("\n=== Simulated Multi-Branch Acceptance Speedup (4000 waves/setting) ===\n");
    std::mt19937 sim_rng(2026);
    constexpr int waves = 4000;
    constexpr int T_base = 4;  // 1 root + 3 drafts
    for (double p1 : {0.55, 0.70, 0.80}) {
        for (double p2_rec : {0.15, 0.25, 0.35}) {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            double tok_sum[4] = {};
            int compact_cnt[4] = {};
            for (int w = 0; w < waves; ++w) {
                // Ground truth targets for trunk depth 0, 1, 2
                int32_t target[3];
                int32_t draft1[3] = {10, 20, 30};
                int32_t runner_up[2] = {11, 21};
                int32_t sfx_alt[3] = {12, 22, 32};
                // Simulate whether each trunk step accepts top-1, falls to top-2 runner-up, or misses all
                for (int d = 0; d < 3; ++d) {
                    const double r = u(sim_rng);
                    if (r < p1) target[d] = draft1[d];
                    else if (d < 2 && r < p1 + (1.0 - p1) * p2_rec) target[d] = runner_up[d];
                    else if (d == 1 && r < p1 + (1.0 - p1) * (p2_rec + 0.15)) target[d] = sfx_alt[d];
                    else target[d] = 999;
                }
                for (int mode : {0, 1, 2, 3}) {
                    int32_t win[kVerifyMaxT] = {1, 10, 20, 30};
                    float dprob[kVerifyMaxT] = {(float)p1, (float)p1, (float)p1};
                    int T_out = T_base;
                    VerifyTreeMap map{};
                    bool is_tree = false;
                    if (mode > 0) {
                        SpecTreeInput in;
                        in.mode = mode;
                        in.pos0 = 128;
                        in.T_base = T_base;
                        in.max_room = 8;
                        in.draft_on = true;
                        in.mtp_n = 3;
                        in.drafts = draft1;
                        in.dprob = dprob;
                        in.mtp_top2_0 = runner_up[0];
                        in.mtp_top2_p0 = (float)((1.0 - p1) * p2_rec + 0.05);
                        in.mtp_top2_1 = runner_up[1];
                        in.mtp_top2_p1 = (float)((1.0 - p1) * p2_rec + 0.05);
                        in.sfx_k = 3;
                        in.sfx_match = 4;
                        in.sbuf = sfx_alt;
                        is_tree = build_spec_tree(in, win, T_out, map);
                    }
                    int32_t outv[kVerifyMaxT] = {};
                    if (!is_tree) {
                        for (int d = 0; d < 3; ++d) outv[d] = target[d];
                        int a = 0;
                        while (a < T_base - 1 && win[a + 1] == outv[a]) ++a;
                        tok_sum[mode] += (a + 1);
                    } else {
                        for (int t = 0; t < T_out; ++t) {
                            const int dep = map.depth[t];
                            if (dep < 3) {
                                // Only preserve target if slot's ancestor prefix actually matched target[0..dep-1],
                                // else simulate post-divergence target (miss further drafts for conservative bound)
                                bool prefix_ok = true;
                                for (int k = 1; k <= dep; ++k) {
                                    if (win[map.anc_slot[t][k]] != target[k - 1]) { prefix_ok = false; break; }
                                }
                                outv[t] = prefix_ok ? target[dep] : 888;
                            }
                        }
                        int path[kVerifyMaxT] = {};
                        bool nc = false, bh = false;
                        const int k_acc = walk_accepted_tree(map, win, outv, path, nc, bh);
                        tok_sum[mode] += k_acc;
                        if (nc) ++compact_cnt[mode];
                    }
                }
            }
            const double b = tok_sum[0] / waves;
            std::printf("SIM p1=%.2f p2_rec=%.2f | trunk=%.3f | tree1=%.3f (%+.1f%%) | tree2=%.3f (%+.1f%%) | tree3=%.3f (%+.1f%%)\n",
                        p1, p2_rec, b,
                        tok_sum[1] / waves, 100.0 * (tok_sum[1] / waves - b) / b,
                        tok_sum[2] / waves, 100.0 * (tok_sum[2] / waves - b) / b,
                        tok_sum[3] / waves, 100.0 * (tok_sum[3] / waves - b) / b);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";

    test_spec_tree_cpu();

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running sycl_spec_tree on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    test_row_top2_mapped(q);
    test_verify_compact_row(q);
    test_kv_copy_cell(q);
    test_gdn_tree_parity(q);
    test_qsa_causal_tree_mask(q);

    std::printf("\nsycl_spec_tree: %s (failures: %d)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);

    if (bench && g_failures == 0) {
        run_benchmarks(q);
    }
    return g_failures == 0 ? 0 : 1;
}
