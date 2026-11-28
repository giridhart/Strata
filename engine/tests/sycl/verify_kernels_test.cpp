#include <sycl/sycl.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running verify_kernels_test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int total_failures = 0;

    // 1. broadcast_streams test
    {
        constexpr int64_t n_embd = 2560;
        constexpr int hc = 4;
        constexpr int n_tok = 3;
        std::vector<float> h_x(n_tok * n_embd);
        for (auto& v : h_x)
            v = dist(rng);

        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        float* d_R = sycl::malloc_device<float>(n_tok * hc * n_embd, q);
        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float)).wait();

        strata::kernels::broadcast_streams(d_x, d_R, n_embd, hc, n_tok, &q);
        q.wait();

        std::vector<float> h_R(n_tok * hc * n_embd);
        q.memcpy(h_R.data(), d_R, h_R.size() * sizeof(float)).wait();

        int errs = 0;
        for (int t = 0; t < n_tok; ++t) {
            for (int c = 0; c < hc; ++c) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    float expected = h_x[t * n_embd + i];
                    float got = h_R[t * hc * n_embd + c * n_embd + i];
                    if (expected != got) errs++;
                }
            }
        }
        std::printf("broadcast_streams: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_x, q);
        sycl::free(d_R, q);
    }

    // 2. add_streams_broadcast test
    {
        constexpr int64_t n_embd = 2560;
        constexpr int hc = 4;
        constexpr int n_tok = 2;
        std::vector<float> h_h(n_tok * hc * n_embd);
        std::vector<float> h_e(n_tok * n_embd);
        for (auto& v : h_h)
            v = dist(rng);
        for (auto& v : h_e)
            v = dist(rng);

        float* d_h = sycl::malloc_device<float>(h_h.size(), q);
        float* d_e = sycl::malloc_device<float>(h_e.size(), q);
        float* d_R = sycl::malloc_device<float>(h_h.size(), q);
        q.memcpy(d_h, h_h.data(), h_h.size() * sizeof(float)).wait();
        q.memcpy(d_e, h_e.data(), h_e.size() * sizeof(float)).wait();

        strata::kernels::add_streams_broadcast(d_h, d_e, d_R, n_embd, hc, n_tok, &q);
        q.wait();

        std::vector<float> h_R(h_h.size());
        q.memcpy(h_R.data(), d_R, h_R.size() * sizeof(float)).wait();

        int errs = 0;
        for (int t = 0; t < n_tok; ++t) {
            for (int c = 0; c < hc; ++c) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    float expected = h_h[t * hc * n_embd + c * n_embd + i] + h_e[t * n_embd + i];
                    float got = h_R[t * hc * n_embd + c * n_embd + i];
                    if (expected != got) errs++;
                }
            }
        }
        std::printf("add_streams_broadcast: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_h, q);
        sycl::free(d_e, q);
        sycl::free(d_R, q);
    }

    // 3. copy_indexed test
    {
        constexpr int64_t stride = 512;
        constexpr int64_t n = 256;
        constexpr int rows = 8;
        std::vector<float> h_src(rows * stride);
        for (auto& v : h_src)
            v = dist(rng);
        int32_t index_val = 3;

        float* d_src = sycl::malloc_device<float>(h_src.size(), q);
        float* d_dst = sycl::malloc_device<float>(n, q);
        int32_t* d_idx = sycl::malloc_device<int32_t>(1, q);

        q.memcpy(d_src, h_src.data(), h_src.size() * sizeof(float)).wait();
        q.memcpy(d_idx, &index_val, sizeof(int32_t)).wait();

        strata::kernels::copy_indexed(d_dst, d_src, stride, d_idx, n, &q);
        q.wait();

        std::vector<float> h_dst(n);
        q.memcpy(h_dst.data(), d_dst, n * sizeof(float)).wait();

        int errs = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (h_dst[i] != h_src[index_val * stride + i]) errs++;
        }
        std::printf("copy_indexed: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_src, q);
        sycl::free(d_dst, q);
        sycl::free(d_idx, q);
    }

    // 4. ident_hits test
    {
        constexpr int n = 64;
        std::vector<int32_t> h_ids(n);
        for (int i = 0; i < n; ++i)
            h_ids[i] = i * 2;

        int32_t* d_ids = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_slot = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_dst = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_count = sycl::malloc_device<int32_t>(1, q);

        q.memcpy(d_ids, h_ids.data(), n * sizeof(int32_t)).wait();

        strata::kernels::ident_hits(d_ids, n, d_slot, d_dst, d_count, &q);
        q.wait();

        std::vector<int32_t> h_slot(n), h_dst(n);
        int32_t count = 0;
        q.memcpy(h_slot.data(), d_slot, n * sizeof(int32_t)).wait();
        q.memcpy(h_dst.data(), d_dst, n * sizeof(int32_t)).wait();
        q.memcpy(&count, d_count, sizeof(int32_t)).wait();

        int errs = (count == n) ? 0 : 1;
        for (int i = 0; i < n; ++i) {
            if (h_slot[i] != h_ids[i]) errs++;
            if (h_dst[i] != i) errs++;
        }
        std::printf("ident_hits: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_ids, q);
        sycl::free(d_slot, q);
        sycl::free(d_dst, q);
        sycl::free(d_count, q);
    }

    // 5. dense_steps test
    {
        constexpr int n = 8;
        std::vector<int32_t> h_cells = {0, 3, 7, 15, 20, 25, 31, 40};
        int32_t* d_cells = sycl::malloc_device<int32_t>(n, q);
        int32_t* d_steps = sycl::malloc_device<int32_t>(n * 4, q);
        q.memcpy(d_cells, h_cells.data(), n * sizeof(int32_t)).wait();

        strata::kernels::dense_steps(d_cells, n, d_steps, &q);
        q.wait();

        std::vector<int32_t> h_steps(n * 4);
        q.memcpy(h_steps.data(), d_steps, n * 4 * sizeof(int32_t)).wait();

        int errs = 0;
        for (int i = 0; i < n; ++i) {
            int c = h_cells[i];
            if (h_steps[i * 4 + 0] != c) errs++;
            if (h_steps[i * 4 + 1] != c + 1) errs++;
            if (h_steps[i * 4 + 2] != (c + 1) / 4) errs++;
            if (h_steps[i * 4 + 3] != c + 1) errs++;
        }
        std::printf("dense_steps: %s (%d mismatches)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        sycl::free(d_cells, q);
        sycl::free(d_steps, q);
    }

    // 6. gdn_conv_l2_multi vs fused_gdn_conv_l2 parity (1 token)
    {
        constexpr int channels = 2048;
        constexpr int qk_heads = 4;
        constexpr float eps = 1e-5f;

        std::vector<float> h_hist(channels * 3);
        std::vector<float> h_qkv(channels);
        std::vector<float> h_w(channels * 4);
        for (auto& v : h_hist)
            v = dist(rng);
        for (auto& v : h_qkv)
            v = dist(rng);
        for (auto& v : h_w)
            v = dist(rng);

        float* d_hist1 = sycl::malloc_device<float>(h_hist.size(), q);
        float* d_hist2 = sycl::malloc_device<float>(h_hist.size(), q);
        float* d_qkv = sycl::malloc_device<float>(channels, q);
        float* d_w = sycl::malloc_device<float>(channels * 4, q);
        float* d_h_single = sycl::malloc_device<float>(channels, q);
        float* d_h_multi = sycl::malloc_device<float>(channels, q);

        q.memcpy(d_hist1, h_hist.data(), h_hist.size() * sizeof(float)).wait();
        q.memcpy(d_hist2, h_hist.data(), h_hist.size() * sizeof(float)).wait();
        q.memcpy(d_qkv, h_qkv.data(), channels * sizeof(float)).wait();
        q.memcpy(d_w, h_w.data(), channels * 4 * sizeof(float)).wait();

        // Run single-token oracle
        strata::kernels::fused_gdn_conv_l2(d_hist1, d_qkv, d_w, d_h_single, channels, qk_heads, eps, &q);

        // Run multi-token verify kernel (1 token)
        strata::kernels::gdn_conv_l2_multi(d_hist2, d_qkv, d_w, d_h_multi, channels, qk_heads, eps, 1, &q, 0);
        q.wait();

        std::vector<float> res_single(channels), res_multi(channels);
        q.memcpy(res_single.data(), d_h_single, channels * sizeof(float)).wait();
        q.memcpy(res_multi.data(), d_h_multi, channels * sizeof(float)).wait();

        int errs = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < channels; ++i) {
            float diff = std::fabs(res_single[i] - res_multi[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-6f) errs++;
        }
        std::printf("gdn_conv_l2_multi parity: %s (max_diff: %e, mismatches: %d)\n", errs == 0 ? "PASS" : "FAIL",
                    max_diff, errs);
        total_failures += errs;

        sycl::free(d_hist1, q);
        sycl::free(d_hist2, q);
        sycl::free(d_qkv, q);
        sycl::free(d_w, q);
        sycl::free(d_h_single, q);
        sycl::free(d_h_multi, q);
    }

    // 7. gdn_ab_multi vs fused_gdn_ab parity (1 token)
    {
        constexpr int n_embd = 2048;
        constexpr int h_v = 32;

        std::vector<float> h_x(n_embd);
        std::vector<uint16_t> h_wa(h_v * n_embd), h_wb(h_v * n_embd);
        std::vector<float> h_dt(h_v), h_ssm_a(h_v);
        for (auto& v : h_x)
            v = dist(rng);
        for (auto& v : h_wa)
            v = (uint16_t)(rng() & 0xFFFF);
        for (auto& v : h_wb)
            v = (uint16_t)(rng() & 0xFFFF);
        for (auto& v : h_dt)
            v = dist(rng);
        for (auto& v : h_ssm_a)
            v = -std::fabs(dist(rng));

        float* d_x = sycl::malloc_device<float>(n_embd, q);
        uint16_t* d_wa = sycl::malloc_device<uint16_t>(h_wa.size(), q);
        uint16_t* d_wb = sycl::malloc_device<uint16_t>(h_wb.size(), q);
        float* d_dt = sycl::malloc_device<float>(h_v, q);
        float* d_ssm_a = sycl::malloc_device<float>(h_v, q);

        float* d_gate_s = sycl::malloc_device<float>(h_v, q);
        float* d_beta_s = sycl::malloc_device<float>(h_v, q);
        float* d_gate_m = sycl::malloc_device<float>(h_v, q);
        float* d_beta_m = sycl::malloc_device<float>(h_v, q);

        q.memcpy(d_x, h_x.data(), n_embd * sizeof(float)).wait();
        q.memcpy(d_wa, h_wa.data(), h_wa.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_wb, h_wb.data(), h_wb.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_dt, h_dt.data(), h_v * sizeof(float)).wait();
        q.memcpy(d_ssm_a, h_ssm_a.data(), h_v * sizeof(float)).wait();

        // Run single-token oracle
        strata::kernels::fused_gdn_ab(d_x, d_wa, d_wb, d_dt, d_ssm_a, d_gate_s, d_beta_s, n_embd, h_v, &q);

        // Run multi-token verify kernel (1 token)
        strata::kernels::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_ssm_a, d_gate_m, d_beta_m, n_embd, h_v, 1, &q);
        q.wait();

        std::vector<float> gate_s(h_v), gate_m(h_v), beta_s(h_v), beta_m(h_v);
        q.memcpy(gate_s.data(), d_gate_s, h_v * sizeof(float)).wait();
        q.memcpy(gate_m.data(), d_gate_m, h_v * sizeof(float)).wait();
        q.memcpy(beta_s.data(), d_beta_s, h_v * sizeof(float)).wait();
        q.memcpy(beta_m.data(), d_beta_m, h_v * sizeof(float)).wait();

        int errs = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < h_v; ++i) {
            float dg = std::fabs(gate_s[i] - gate_m[i]);
            float db = std::fabs(beta_s[i] - beta_m[i]);
            max_diff = std::max(max_diff, std::max(dg, db));
            if (dg > 1e-6f || db > 1e-6f) errs++;
        }
        std::printf("gdn_ab_multi parity: %s (max_diff: %e, mismatches: %d)\n", errs == 0 ? "PASS" : "FAIL", max_diff,
                    errs);
        total_failures += errs;

        sycl::free(d_x, q);
        sycl::free(d_wa, q);
        sycl::free(d_wb, q);
        sycl::free(d_dt, q);
        sycl::free(d_ssm_a, q);
        sycl::free(d_gate_s, q);
        sycl::free(d_beta_s, q);
        sycl::free(d_gate_m, q);
        sycl::free(d_beta_m, q);
    }

    // 8. gdn_commit_layers parity test
    {
        constexpr int S_dim = 128, h_k = 1, h_v = 4, L = 5, MT = strata::kernels::kVerifyMaxT;
        constexpr int qk = S_dim * h_k, C = 2 * qk + S_dim * h_v;
        constexpr int64_t conv_off = (int64_t)S_dim * h_v * S_dim, stride = conv_off + (int64_t)C * 3;
        std::vector<float> st0((size_t)(stride * L)), hb((size_t)L * MT * C), qkv((size_t)L * MT * C),
            gt((size_t)L * MT * h_v), bt((size_t)L * MT * h_v), zdummy((size_t)MT * S_dim * h_v), gm(S_dim, 1.0f);
        for (auto& x : st0) x = dist(rng) * 0.1f;
        for (auto& x : hb) x = dist(rng) * 0.1f;
        for (auto& x : qkv) x = dist(rng);
        for (auto& x : gt) x = -(0.5f + (float)(rng() % 100) / 50.0f);
        for (auto& x : bt) x = (float)(rng() % 100) / 100.0f;
        float *d_a = sycl::malloc_device<float>(st0.size(), q);
        float *d_b = sycl::malloc_device<float>(st0.size(), q);
        float *d_h = sycl::malloc_device<float>(hb.size(), q);
        float *d_qkv = sycl::malloc_device<float>(qkv.size(), q);
        float *d_g = sycl::malloc_device<float>(gt.size(), q);
        float *d_bt = sycl::malloc_device<float>(bt.size(), q);
        float *d_z = sycl::malloc_device<float>(zdummy.size(), q);
        float *d_y = sycl::malloc_device<float>(zdummy.size(), q);
        float *d_gm = sycl::malloc_device<float>(gm.size(), q);
        int32_t *d_nk = sycl::malloc_device<int32_t>(1, q);

        q.memcpy(d_h, hb.data(), hb.size() * sizeof(float));
        q.memcpy(d_qkv, qkv.data(), qkv.size() * sizeof(float));
        q.memcpy(d_g, gt.data(), gt.size() * sizeof(float));
        q.memcpy(d_bt, bt.data(), bt.size() * sizeof(float));
        q.memcpy(d_gm, gm.data(), gm.size() * sizeof(float));
        q.memset(d_z, 0, zdummy.size() * sizeof(float)).wait();

        int lb_bad = 0;
        for (int nk = 1; nk <= MT; ++nk) {
            q.memcpy(d_nk, &nk, sizeof(int32_t));
            q.memcpy(d_a, st0.data(), st0.size() * sizeof(float));
            q.memcpy(d_b, st0.data(), st0.size() * sizeof(float)).wait();

            for (int l = 0; l < L; ++l) {
                float* stl = d_a + (size_t)l * stride;
                strata::kernels::gdn_conv_commit(stl + conv_off, d_qkv + (size_t)l * MT * C, C, d_nk, &q);
                strata::kernels::gdn_step_norm_multi(stl, d_h + (size_t)l * MT * C, C, d_g + (size_t)l * MT * h_v,
                                                     d_bt + (size_t)l * MT * h_v, d_z, d_gm, 1e-6f, d_y, h_k, h_v, MT,
                                                     d_nk, &q, MT);
            }
            bool launched = strata::kernels::gdn_commit_layers(d_b, stride, conv_off, d_qkv, (int64_t)MT * C, d_h,
                                                               (int64_t)MT * C, d_g, d_bt, (int64_t)MT * h_v, C, h_k,
                                                               h_v, MT, L, d_nk, &q);
            if (!launched) {
                std::printf("gdn_commit_layers failed to launch\n");
                lb_bad++;
                break;
            }
            q.wait();

            std::vector<float> a(st0.size()), b(st0.size());
            q.memcpy(a.data(), d_a, a.size() * sizeof(float));
            q.memcpy(b.data(), d_b, b.size() * sizeof(float)).wait();

            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0 ||
                std::memcmp(a.data(), st0.data(), a.size() * sizeof(float)) == 0) {
                std::printf("    *** gdn_commit_layers n_keep=%d differs from per-layer commit ***\n", nk);
                lb_bad++;
            }
        }
        std::printf("gdn_commit_layers parity: %s (%d layers, MT=%d)\n", lb_bad == 0 ? "PASS" : "FAIL", L, MT);
        total_failures += lb_bad;

        sycl::free(d_a, q); sycl::free(d_b, q); sycl::free(d_h, q); sycl::free(d_qkv, q);
        sycl::free(d_g, q); sycl::free(d_bt, q); sycl::free(d_z, q); sycl::free(d_y, q);
        sycl::free(d_gm, q); sycl::free(d_nk, q);
    }

    // 9. native_qsa_indexer_append_steps_layers test
    {
        constexpr int NL = 3;
        constexpr int IDXD = 128, R = 4, ROT = 64;
        constexpr int64_t NB = 8;
        constexpr int32_t BASE = 100;
        constexpr int64_t MC = 1024;
        constexpr size_t prows = (size_t)(MC / R + 1) * IDXD;
        constexpr size_t trows = (size_t)(R - 1) * IDXD;
        constexpr float EPS = 1e-6f;

        strata::kernels::QsaShapes shapes{};
        shapes.idx_dim = IDXD;
        shapes.idx_block = R;
        shapes.n_rot = ROT;
        strata::kernels::RopeScaling scaling{};

        std::vector<float> dw_kn(IDXD);
        for (auto& x : dw_kn) x = 1.0f + 0.1f * dist(rng);
        float* d_gamma = sycl::malloc_device<float>(IDXD, q);
        q.memcpy(d_gamma, dw_kn.data(), IDXD * sizeof(float)).wait();

        std::vector<float> rawL((size_t)NL * NB * IDXD);
        for (auto& x : rawL) x = dist(rng);
        float* d_rawL = sycl::malloc_device<float>(rawL.size(), q);
        q.memcpy(d_rawL, rawL.data(), rawL.size() * sizeof(float)).wait();

        std::vector<int32_t> hp((size_t)NB);
        for (int64_t t = 0; t < NB; ++t) hp[(size_t)t] = (int32_t)(2 + t);
        int32_t* d_posL = sycl::malloc_device<int32_t>(NB, q);
        q.memcpy(d_posL, hp.data(), NB * sizeof(int32_t)).wait();

        float *pr[NL], *dr[NL], *tr[NL], *pl[NL], *dl[NL], *tl[NL], *tsrc[NL];
        int32_t *br[NL], *bl[NL];
        for (int i = 0; i < NL; ++i) {
            pr[i] = sycl::malloc_device<float>(prows, q);
            dr[i] = sycl::malloc_device<float>(IDXD, q);
            tr[i] = sycl::malloc_device<float>(trows, q);
            pl[i] = sycl::malloc_device<float>(prows, q);
            dl[i] = sycl::malloc_device<float>(IDXD, q);
            tl[i] = sycl::malloc_device<float>(trows, q);
            tsrc[i] = sycl::malloc_device<float>(trows, q);
            br[i] = sycl::malloc_device<int32_t>(1, q);
            bl[i] = sycl::malloc_device<int32_t>(1, q);

            std::vector<float> ts(trows);
            for (auto& x : ts) x = dist(rng);
            q.memcpy(tsrc[i], ts.data(), trows * sizeof(float));

            q.memset(pr[i], 0, prows * sizeof(float));
            q.memset(dr[i], 0, IDXD * sizeof(float));
            q.memset(tr[i], 0, trows * sizeof(float));
            q.memset(pl[i], 0, prows * sizeof(float));
            q.memset(dl[i], 0, IDXD * sizeof(float));
            q.memset(tl[i], 0, trows * sizeof(float));
            q.memset(br[i], 0, sizeof(int32_t));
            q.memset(bl[i], 0, sizeof(int32_t));
        }
        q.wait();

        strata::kernels::QsaIndexerLayers il;
        for (int i = 0; i < NL; ++i) {
            q.memcpy(tr[i], tsrc[i], trows * sizeof(float));
            strata::kernels::QsaIndexerBuffers b{tr[i], dr[i], pr[i], br[i]};
            strata::kernels::native_qsa_indexer_append_steps(d_rawL + (size_t)i * NB * IDXD, d_posL, 1, (int)NB, BASE,
                                                             d_gamma, EPS, b, shapes, MC, scaling, &q);
            il.gamma[i] = d_gamma;
            il.tail_src[i] = tsrc[i];
            il.tail[i] = tl[i];
            il.dead[i] = dl[i];
            il.pooled[i] = pl[i];
            il.block_pos[i] = bl[i];
        }
        strata::kernels::native_qsa_indexer_append_steps_layers(d_rawL, (int64_t)NB * IDXD, d_posL, 1, (int)NB, BASE,
                                                                EPS, il, NL, shapes, MC, scaling, &q);
        q.wait();

        int errs = 0;
        for (int i = 0; i < NL; ++i) {
            std::vector<float> h_pr(prows), h_pl(prows);
            std::vector<float> h_dr(IDXD), h_dl(IDXD);
            std::vector<float> h_tr(trows), h_tl(trows);
            int32_t h_br = 0, h_bl = 0;
            q.memcpy(h_pr.data(), pr[i], prows * sizeof(float));
            q.memcpy(h_pl.data(), pl[i], prows * sizeof(float));
            q.memcpy(h_dr.data(), dr[i], IDXD * sizeof(float));
            q.memcpy(h_dl.data(), dl[i], IDXD * sizeof(float));
            q.memcpy(h_tr.data(), tr[i], trows * sizeof(float));
            q.memcpy(h_tl.data(), tl[i], trows * sizeof(float));
            q.memcpy(&h_br, br[i], sizeof(int32_t));
            q.memcpy(&h_bl, bl[i], sizeof(int32_t)).wait();

            if (std::memcmp(h_pr.data(), h_pl.data(), prows * sizeof(float)) != 0 ||
                std::memcmp(h_dr.data(), h_dl.data(), IDXD * sizeof(float)) != 0 ||
                std::memcmp(h_tr.data(), h_tl.data(), trows * sizeof(float)) != 0 ||
                h_br != h_bl) {
                errs++;
            }
        }
        std::printf("native_qsa_indexer_append_steps_layers parity: %s (mismatches: %d)\n", errs == 0 ? "PASS" : "FAIL", errs);
        total_failures += errs;

        for (int i = 0; i < NL; ++i) {
            sycl::free(pr[i], q); sycl::free(dr[i], q); sycl::free(tr[i], q);
            sycl::free(pl[i], q); sycl::free(dl[i], q); sycl::free(tl[i], q);
            sycl::free(tsrc[i], q); sycl::free(br[i], q); sycl::free(bl[i], q);
        }
        sycl::free(d_gamma, q);
        sycl::free(d_rawL, q);
        sycl::free(d_posL, q);
    }

    std::printf("\nverify_kernels_test: %s (total failures: %d)\n", total_failures == 0 ? "PASSED" : "FAILED",
                total_failures);

    if (bench && total_failures == 0) {
        std::printf("\n--- Batch Commit Microbenchmarks (STRATA_COMMIT_BATCH) ---\n");
        constexpr int reps = 50;
        constexpr int S_dim = 128, h_k = 1, h_v = 4, L = 48, MT = strata::kernels::kVerifyMaxT;
        constexpr int qk = S_dim * h_k, C = 2 * qk + S_dim * h_v;
        constexpr int64_t conv_off = (int64_t)S_dim * h_v * S_dim, stride = conv_off + (int64_t)C * 3;

        float* d_st = sycl::malloc_device<float>((size_t)stride * L, q);
        float* d_qkv = sycl::malloc_device<float>((size_t)L * MT * C, q);
        float* d_h = sycl::malloc_device<float>((size_t)L * MT * C, q);
        float* d_g = sycl::malloc_device<float>((size_t)L * MT * h_v, q);
        float* d_bt = sycl::malloc_device<float>((size_t)L * MT * h_v, q);
        float* d_z = sycl::malloc_device<float>((size_t)MT * S_dim * h_v, q);
        float* d_gm = sycl::malloc_device<float>(S_dim, q);
        float* d_y = sycl::malloc_device<float>((size_t)MT * S_dim * h_v, q);
        int32_t* d_nk = sycl::malloc_device<int32_t>(1, q);

        q.fill(d_st, 0.01f, (size_t)stride * L);
        q.fill(d_qkv, 0.05f, (size_t)L * MT * C);
        q.fill(d_h, 0.05f, (size_t)L * MT * C);
        q.fill(d_g, -0.5f, (size_t)L * MT * h_v);
        q.fill(d_bt, 0.8f, (size_t)L * MT * h_v);
        q.fill(d_z, 0.2f, (size_t)MT * S_dim * h_v);
        q.fill(d_gm, 1.0f, S_dim);
        q.wait();

        for (int nk : {1, 2, 4, 8}) {
            q.fill(d_nk, nk, 1).wait();

            // 1. Sequential per-layer commit
            for (int i = 0; i < 3; ++i) {
                for (int l = 0; l < L; ++l) {
                    float* stl = d_st + (size_t)l * stride;
                    strata::kernels::gdn_conv_commit(stl + conv_off, d_qkv + (size_t)l * MT * C, C, d_nk, &q);
                    strata::kernels::gdn_step_norm_multi(stl, d_h + (size_t)l * MT * C, C, d_g + (size_t)l * MT * h_v,
                                                         d_bt + (size_t)l * MT * h_v, d_z, d_gm, 1e-6f, d_y, h_k, h_v, MT,
                                                         d_nk, &q, MT);
                }
            }
            q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                for (int l = 0; l < L; ++l) {
                    float* stl = d_st + (size_t)l * stride;
                    strata::kernels::gdn_conv_commit(stl + conv_off, d_qkv + (size_t)l * MT * C, C, d_nk, &q);
                    strata::kernels::gdn_step_norm_multi(stl, d_h + (size_t)l * MT * C, C, d_g + (size_t)l * MT * h_v,
                                                         d_bt + (size_t)l * MT * h_v, d_z, d_gm, 1e-6f, d_y, h_k, h_v, MT,
                                                         d_nk, &q, MT);
                }
            }
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            double us_seq = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

            // 2. Batched gdn_commit_layers
            for (int i = 0; i < 3; ++i) {
                strata::kernels::gdn_commit_layers(d_st, stride, conv_off, d_qkv, (int64_t)MT * C, d_h,
                                                   (int64_t)MT * C, d_g, d_bt, (int64_t)MT * h_v, C, h_k,
                                                   h_v, MT, L, d_nk, &q);
            }
            q.wait();
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                strata::kernels::gdn_commit_layers(d_st, stride, conv_off, d_qkv, (int64_t)MT * C, d_h,
                                                   (int64_t)MT * C, d_g, d_bt, (int64_t)MT * h_v, C, h_k,
                                                   h_v, MT, L, d_nk, &q);
            }
            q.wait();
            t1 = std::chrono::steady_clock::now();
            double us_batched = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

            std::printf("BENCH GDN Commit L=%2d steps=%d | sequential: %7.2f us | batched: %7.2f us | speedup: %.2fx\n",
                        L, nk, us_seq, us_batched, us_seq / us_batched);
        }

        sycl::free(d_st, q);
        sycl::free(d_qkv, q);
        sycl::free(d_h, q);
        sycl::free(d_g, q);
        sycl::free(d_bt, q);
        sycl::free(d_z, q);
        sycl::free(d_gm, q);
        sycl::free(d_y, q);
        sycl::free(d_nk, q);
    }

    return total_failures == 0 ? 0 : 1;
}
