// tests/sycl/sycl_queue_sync_test.cpp - Stress test for SYCL multi-queue synchronization,
// event chaining, and asynchronous memory transfers to prevent cross-queue data races
// (such as the YaRN scaling indexer append race observed on Intel Arc B580).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_failures = 0;

void test_event_chained_producer_consumer(sycl::queue& q1, sycl::queue& q2) {
    std::printf("Testing cross-queue event-chained producer-consumer... ");
    constexpr size_t N = 16384;
    std::vector<float> h_in(N);
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : h_in) v = dist(rng);

    float* d_buf1 = sycl::malloc_device<float>(N, q1);
    float* d_buf2 = sycl::malloc_device<float>(N, q2);

    int bad_iterations = 0;
    for (int iter = 0; iter < 20; ++iter) {
        for (auto& v : h_in) v = dist(rng);

        // Queue 1: upload input to d_buf1 and scale by 2.0
        sycl::event e_copy = q1.memcpy(d_buf1, h_in.data(), N * sizeof(float));
        sycl::event e_scale = q1.submit([&](sycl::handler& cgh) {
            cgh.depends_on(e_copy);
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                d_buf1[idx] *= 2.0f;
            });
        });

        // Queue 2: depends on e_scale from Queue 1, adds 1.5 to each element into d_buf2
        sycl::event e_add = q2.submit([&](sycl::handler& cgh) {
            cgh.depends_on(e_scale);
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                d_buf2[idx] = d_buf1[idx] + 1.5f;
            });
        });

        std::vector<float> h_out(N);
        q2.memcpy(h_out.data(), d_buf2, N * sizeof(float), e_add).wait();

        for (size_t i = 0; i < N; ++i) {
            float expected = h_in[i] * 2.0f + 1.5f;
            if (std::abs(h_out[i] - expected) > 1e-5f) {
                bad_iterations++;
                break;
            }
        }
    }

    if (bad_iterations == 0) {
        std::printf("PASS (20 iterations)\n");
    } else {
        std::printf("FAIL (%d bad iterations)\n", bad_iterations);
        g_failures++;
    }

    sycl::free(d_buf1, q1);
    sycl::free(d_buf2, q2);
}

void test_cross_queue_indexer_append_yarn(sycl::queue& q_main, sycl::queue& q_async) {
    std::printf("Testing multi-queue YaRN indexer append consistency... ");
    const k::QsaShapes S = k::qsa_real_shapes();
    const int64_t IDXD = S.idx_dim;
    const int64_t R = S.idx_block;
    const int64_t MC = 256;
    const int64_t NB = 16;
    const float EPS = 1e-6f;
    const int64_t BASE = 0;

    k::RopeScaling sc_yarn;
    sc_yarn.type = k::RopeScalingType::YaRN;
    sc_yarn.factor = 2.0f;
    sc_yarn.ext_factor = 1.0f;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> w_kn((size_t)IDXD, 1.0f);
    std::vector<float> raws((size_t)NB * IDXD);
    for (auto& v : raws) v = dist(rng);

    float* d_wkn = sycl::malloc_device<float>((size_t)IDXD, q_main);
    q_main.memcpy(d_wkn, w_kn.data(), (size_t)IDXD * sizeof(float)).wait();

    // 1. Single queue baseline (run entirely on q_main)
    const size_t prows = (size_t)(MC / R + 1) * IDXD;
    const size_t trows = (size_t)(R - 1) * IDXD;

    float* d_tail_base = sycl::malloc_device<float>(trows, q_main);
    float* d_dead_base = sycl::malloc_device<float>((size_t)IDXD, q_main);
    float* d_pooled_base = sycl::malloc_device<float>(prows, q_main);
    int32_t* d_bpos_base = sycl::malloc_device<int32_t>(1, q_main);
    float* d_draw_base = sycl::malloc_device<float>((size_t)IDXD, q_main);
    int32_t* d_pos_base = sycl::malloc_device<int32_t>(1, q_main);

    q_main.memset(d_tail_base, 0, trows * sizeof(float)).wait();
    q_main.memset(d_dead_base, 0, (size_t)IDXD * sizeof(float)).wait();
    q_main.memset(d_pooled_base, 0, prows * sizeof(float)).wait();
    q_main.memset(d_bpos_base, 0, sizeof(int32_t)).wait();

    k::QsaIndexerBuffers bufs_base{d_tail_base, d_dead_base, d_pooled_base, d_bpos_base};
    for (int64_t t = 0; t < NB; ++t) {
        q_main.memcpy(d_draw_base, &raws[(size_t)t * IDXD], (size_t)IDXD * sizeof(float)).wait();
        int32_t tp = (int32_t)t;
        q_main.memcpy(d_pos_base, &tp, sizeof(int32_t)).wait();
        k::native_qsa_indexer_append(d_draw_base, d_pos_base, BASE, d_wkn, EPS, bufs_base, S, MC, sc_yarn, &q_main);
        q_main.wait();
    }

    std::vector<float> h_pooled_base(prows), h_dead_base((size_t)IDXD), h_tail_base(trows);
    int32_t h_bpos_base = 0;
    q_main.memcpy(h_pooled_base.data(), d_pooled_base, prows * sizeof(float)).wait();
    q_main.memcpy(h_dead_base.data(), d_dead_base, (size_t)IDXD * sizeof(float)).wait();
    q_main.memcpy(h_tail_base.data(), d_tail_base, trows * sizeof(float)).wait();
    q_main.memcpy(&h_bpos_base, d_bpos_base, sizeof(int32_t)).wait();

    // 2. Multi-queue execution: data staging and transfers on q_async, kernel execution on q_main
    float* d_tail_multi = sycl::malloc_device<float>(trows, q_main);
    float* d_dead_multi = sycl::malloc_device<float>((size_t)IDXD, q_main);
    float* d_pooled_multi = sycl::malloc_device<float>(prows, q_main);
    int32_t* d_bpos_multi = sycl::malloc_device<int32_t>(1, q_main);
    float* d_draw_multi = sycl::malloc_device<float>((size_t)IDXD, q_main);
    int32_t* d_pos_multi = sycl::malloc_device<int32_t>(1, q_main);

    q_main.memset(d_tail_multi, 0, trows * sizeof(float)).wait();
    q_main.memset(d_dead_multi, 0, (size_t)IDXD * sizeof(float)).wait();
    q_main.memset(d_pooled_multi, 0, prows * sizeof(float)).wait();
    q_main.memset(d_bpos_multi, 0, sizeof(int32_t)).wait();

    k::QsaIndexerBuffers bufs_multi{d_tail_multi, d_dead_multi, d_pooled_multi, d_bpos_multi};
    for (int64_t t = 0; t < NB; ++t) {
        // Enqueue staging copy on q_async
        sycl::event e_copy1 = q_async.memcpy(d_draw_multi, &raws[(size_t)t * IDXD], (size_t)IDXD * sizeof(float));
        int32_t tp = (int32_t)t;
        sycl::event e_copy2 = q_async.memcpy(d_pos_multi, &tp, sizeof(int32_t));

        // q_async waits before q_main runs the kernel, preventing cross-queue race
        q_async.wait();
        k::native_qsa_indexer_append(d_draw_multi, d_pos_multi, BASE, d_wkn, EPS, bufs_multi, S, MC, sc_yarn, &q_main);
        q_main.wait();
    }

    std::vector<float> h_pooled_multi(prows), h_dead_multi((size_t)IDXD), h_tail_multi(trows);
    int32_t h_bpos_multi = 0;
    q_main.memcpy(h_pooled_multi.data(), d_pooled_multi, prows * sizeof(float)).wait();
    q_main.memcpy(h_dead_multi.data(), d_dead_multi, (size_t)IDXD * sizeof(float)).wait();
    q_main.memcpy(h_tail_multi.data(), d_tail_multi, trows * sizeof(float)).wait();
    q_main.memcpy(&h_bpos_multi, d_bpos_multi, sizeof(int32_t)).wait();

    int diff_words = 0;
    for (size_t i = 0; i < prows; ++i) {
        if (std::memcmp(&h_pooled_base[i], &h_pooled_multi[i], 4) != 0) diff_words++;
    }
    for (size_t i = 0; i < (size_t)IDXD; ++i) {
        if (std::memcmp(&h_dead_base[i], &h_dead_multi[i], 4) != 0) diff_words++;
    }
    for (size_t i = 0; i < trows; ++i) {
        if (std::memcmp(&h_tail_base[i], &h_tail_multi[i], 4) != 0) diff_words++;
    }
    if (h_bpos_base != h_bpos_multi) diff_words++;

    if (diff_words == 0) {
        std::printf("PASS (bit-identical across all state words)\n");
    } else {
        std::printf("FAIL (%d state words differ)\n", diff_words);
        g_failures++;
    }

    sycl::free(d_wkn, q_main);
    sycl::free(d_tail_base, q_main); sycl::free(d_dead_base, q_main);
    sycl::free(d_pooled_base, q_main); sycl::free(d_bpos_base, q_main);
    sycl::free(d_draw_base, q_main); sycl::free(d_pos_base, q_main);

    sycl::free(d_tail_multi, q_main); sycl::free(d_dead_multi, q_main);
    sycl::free(d_pooled_multi, q_main); sycl::free(d_bpos_multi, q_main);
    sycl::free(d_draw_multi, q_main); sycl::free(d_pos_multi, q_main);
}

} // namespace

int main() {
    sycl::queue q_main{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue q_async{q_main.get_context(), q_main.get_device(), sycl::property::queue::in_order()};

    std::printf("Running SYCL Queue Synchronization test on: %s\n",
                q_main.get_device().get_info<sycl::info::device::name>().c_str());

    test_event_chained_producer_consumer(q_main, q_async);
    test_cross_queue_indexer_append_yarn(q_main, q_async);

    if (g_failures == 0) {
        std::printf("sycl_queue_sync: 0 failures -> PASS\n");
        return 0;
    } else {
        std::printf("sycl_queue_sync: %d failures -> FAIL\n", g_failures);
        return 1;
    }
}
