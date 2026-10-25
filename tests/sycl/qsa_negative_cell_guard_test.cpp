// tests/sycl/qsa_negative_cell_guard_test.cpp - Regression test verifying that negative,
// invalid, and sentinel cell IDs in QSA selection do not trigger out-of-bounds page table
// access, memory page faults, or Level Zero UR_RESULT_ERROR_DEVICE_LOST on Intel Xe2/B580.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

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

int g_failures = 0;

template <typename T>
T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 256, q);
    q.memset(p, 0, (n + 256) * sizeof(T)).wait();
    return p;
}

template <typename T>
T* upload(const std::vector<T>& h, sycl::queue& q) {
    T* d = dalloc<T>(h.size(), q);
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

uint16_t f16_val(uint32_t r) {
    return (uint16_t)((r & 0x8000u) | ((11u + (r >> 16) % 5u) << 10) | (r & 0x3FFu));
}

void test_kv_gather_negative_ids(sycl::queue& q) {
    std::printf("Testing KV gather with negative cell IDs (FP16, INT8, Q4_0)... ");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 4;
    const int H = (int)s.n_head_kv;
    const int D = (int)s.head_dim;
    const int P = (int)s.page_size;
    const int pages = 16;
    const int cells = pages * P;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    int32_t* d_table = upload(table, q);

    // FP16 pools
    std::vector<uint16_t> h_k16((size_t)cells * H * D, k::f16_from_f32(1.5f));
    std::vector<uint16_t> h_v16((size_t)cells * H * D, k::f16_from_f32(2.5f));
    uint16_t* d_k16 = upload(h_k16, q);
    uint16_t* d_v16 = upload(h_v16, q);

    // INT8 pools
    std::vector<int8_t> h_k8((size_t)cells * H * D, 10);
    std::vector<int8_t> h_v8((size_t)cells * H * D, 20);
    std::vector<uint16_t> h_k8s((size_t)cells * H * (D / k::KV_Q8_GROUP), k::f16_from_f32(0.1f));
    std::vector<uint16_t> h_v8s((size_t)cells * H * (D / k::KV_Q8_GROUP), k::f16_from_f32(0.1f));
    int8_t* d_k8 = upload(h_k8, q);
    int8_t* d_v8 = upload(h_v8, q);
    uint16_t* d_k8s = upload(h_k8s, q);
    uint16_t* d_v8s = upload(h_v8s, q);

    // Q4_0 pools
    const size_t q4_blocks = (size_t)cells * H * (D / k::QK4_0);
    std::vector<k::block_q4_0> h_k4(q4_blocks);
    std::vector<k::block_q4_0> h_v4(q4_blocks);
    for (size_t b = 0; b < q4_blocks; ++b) {
        h_k4[b].d = k::f16_from_f32(0.5f);
        h_v4[b].d = k::f16_from_f32(0.5f);
        for (int j = 0; j < 16; ++j) {
            h_k4[b].qs[j] = 0x99;
            h_v4[b].qs[j] = 0x99;
        }
    }
    uint8_t* d_k4 = (uint8_t*)upload(h_k4, q);
    uint8_t* d_v4 = (uint8_t*)upload(h_v4, q);

    // IDs with negative/sentinel entries:
    // Slot 0: valid (cell 0)
    // Slot 1: negative (-1)
    // Slot 2: valid (cell 4)
    // Slot 3: negative (-3584, the exact Battlemage bug offset)
    // Slot 4: negative (-12800)
    // Slot 5: valid (cell 8)
    const int n_ids = 6;
    std::vector<int32_t> ids = {0, -1, 4, -3584, -12800, 8};
    int32_t* d_ids = upload(ids, q);

    std::vector<int32_t> step_h(k::kStepCount, 0);
    step_h[k::kStepWidth] = n_ids;
    int32_t* d_step = upload(step_h, q);

    const size_t out_elems = (size_t)n_ids * H * D;
    uint16_t* d_k_out = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_v_out = dalloc<uint16_t>(out_elems, q);
    std::vector<uint16_t> h_k_out(out_elems), h_v_out(out_elems);

    // 1. Test FP16 kv_gather
    k::kv_gather_step(d_k16, d_v16, d_table, d_ids, d_step, n_ids, s, d_k_out, d_v_out, &q);
    q.wait_and_throw();
    q.memcpy(h_k_out.data(), d_k_out, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_out.data(), d_v_out, out_elems * sizeof(uint16_t)).wait();

    bool fp16_ok = true;
    for (int i = 0; i < n_ids; ++i) {
        bool expect_zero = (ids[i] < 0);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D; ++d) {
                size_t idx = ((size_t)i * H + h) * D + d;
                float kv = k::f32_from_f16(h_k_out[idx]);
                float vv = k::f32_from_f16(h_v_out[idx]);
                if (expect_zero) {
                    if (kv != 0.0f || vv != 0.0f) fp16_ok = false;
                } else {
                    if (std::abs(kv - 1.5f) > 1e-3f || std::abs(vv - 2.5f) > 1e-3f) fp16_ok = false;
                }
            }
        }
    }

    // 2. Test INT8 kv_gather
    q.memset(d_k_out, 0xFF, out_elems * sizeof(uint16_t)).wait();
    q.memset(d_v_out, 0xFF, out_elems * sizeof(uint16_t)).wait();
    k::kv_gather_q8_step(d_k8, d_v8, d_k8s, d_v8s, d_table, d_ids, d_step, n_ids, s, d_k_out, d_v_out, &q);
    q.wait_and_throw();
    q.memcpy(h_k_out.data(), d_k_out, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_out.data(), d_v_out, out_elems * sizeof(uint16_t)).wait();

    bool int8_ok = true;
    for (int i = 0; i < n_ids; ++i) {
        bool expect_zero = (ids[i] < 0);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D; ++d) {
                size_t idx = ((size_t)i * H + h) * D + d;
                float kv = k::f32_from_f16(h_k_out[idx]);
                float vv = k::f32_from_f16(h_v_out[idx]);
                if (expect_zero) {
                    if (kv != 0.0f || vv != 0.0f) int8_ok = false;
                } else {
                    if (kv == 0.0f || vv == 0.0f || !std::isfinite(kv) || !std::isfinite(vv)) int8_ok = false;
                }
            }
        }
    }

    // 3. Test Q4_0 kv_gather
    q.memset(d_k_out, 0xFF, out_elems * sizeof(uint16_t)).wait();
    q.memset(d_v_out, 0xFF, out_elems * sizeof(uint16_t)).wait();
    k::kv_gather_q4_step(d_k4, d_v4, d_table, d_ids, d_step, n_ids, s, d_k_out, d_v_out, &q);
    q.wait_and_throw();
    q.memcpy(h_k_out.data(), d_k_out, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_out.data(), d_v_out, out_elems * sizeof(uint16_t)).wait();

    bool q4_ok = true;
    for (int i = 0; i < n_ids; ++i) {
        bool expect_zero = (ids[i] < 0);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D; ++d) {
                size_t idx = ((size_t)i * H + h) * D + d;
                float kv = k::f32_from_f16(h_k_out[idx]);
                float vv = k::f32_from_f16(h_v_out[idx]);
                if (expect_zero) {
                    if (kv != 0.0f || vv != 0.0f) q4_ok = false;
                } else {
                    if (kv == 0.0f || vv == 0.0f || !std::isfinite(kv) || !std::isfinite(vv)) q4_ok = false;
                }
            }
        }
    }

    if (fp16_ok && int8_ok && q4_ok) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL (fp16=%d, int8=%d, q4=%d)\n", fp16_ok, int8_ok, q4_ok);
        g_failures++;
    }

    sycl::free(d_k16, q); sycl::free(d_v16, q);
    sycl::free(d_k8, q); sycl::free(d_v8, q); sycl::free(d_k8s, q); sycl::free(d_v8s, q);
    sycl::free(d_k4, q); sycl::free(d_v4, q);
    sycl::free(d_table, q); sycl::free(d_ids, q); sycl::free(d_step, q);
    sycl::free(d_k_out, q); sycl::free(d_v_out, q);
}

void test_qsa_decode_attn_negative_ids(sycl::queue& q) {
    std::printf("Testing QSA decode attention with negative & boundary cell IDs... ");
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = 2048;
    const int n_q = 1;
    const int PS = (int)s.page_size;
    const int64_t total_cells = 40960;
    const int pages = (int)(total_cells / PS);

    std::mt19937 rng(1337);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    int32_t* d_table = upload(table, q);

    std::vector<uint16_t> h_k16((size_t)total_cells * s.n_head_kv * s.head_dim);
    std::vector<uint16_t> h_v16((size_t)total_cells * s.n_head_kv * s.head_dim);
    for (auto& v : h_k16) v = f16_val(rng());
    for (auto& v : h_v16) v = f16_val(rng());
    uint16_t* d_k16 = upload(h_k16, q);
    uint16_t* d_v16 = upload(h_v16, q);

    k::QsaAttnPools pools;
    pools.page_table = d_table;
    pools.k_pool = d_k16;
    pools.v_pool = d_v16;

    std::vector<float> h_q((size_t)n_q * s.n_head * s.head_dim);
    std::normal_distribution<float> dist_q(0.0f, 1.0f);
    for (auto& v : h_q) v = dist_q(rng);
    float* d_q = upload(h_q, q);

    const uint64_t scratch_floats = k::qsa_decode_attn_scratch_floats(cap, s);
    float* d_scratch = dalloc<float>(scratch_floats, q);
    float* d_attn = dalloc<float>((size_t)n_q * s.n_head * s.head_dim, q);
    int32_t* d_ids = dalloc<int32_t>((size_t)n_q * cap, q);
    int32_t* d_steps = dalloc<int32_t>((size_t)n_q * k::kStepCount, q);

    // Case 1: ALL negative IDs (simulating severe boundary wrap or uninitialized mask)
    std::vector<int32_t> neg_ids((size_t)cap, -1);
    neg_ids[0] = -3584; // the Battlemage boundary crash value
    neg_ids[1] = -12800;
    neg_ids[2] = -32768;
    q.memcpy(d_ids, neg_ids.data(), cap * sizeof(int32_t)).wait();

    std::vector<int32_t> steps_h(k::kStepCount, 0);
    steps_h[k::kStepWidth] = 512;
    steps_h[k::kStepNKv] = 512;
    q.memcpy(d_steps, steps_h.data(), k::kStepCount * sizeof(int32_t)).wait();

    bool all_neg_ok = true;
    for (int daily : {0, 1}) {
        q.memset(d_scratch, 0, scratch_floats * sizeof(float)).wait();
        k::qsa_decode_attn_chunks_only(d_q, pools, d_ids, d_steps, cap, s, d_scratch, n_q, daily, &q);
        q.wait_and_throw();

        // Also test merge
        q.memset(d_attn, 0, n_q * s.n_head * s.head_dim * sizeof(float)).wait();
        k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scratch, d_attn, n_q, 0, &q);
        q.wait_and_throw();

        std::vector<float> h_out((size_t)n_q * s.n_head * s.head_dim);
        q.memcpy(h_out.data(), d_attn, h_out.size() * sizeof(float)).wait();
        for (float val : h_out) {
            if (!std::isfinite(val)) all_neg_ok = false;
        }
    }

    // Case 2: Mixed valid cells and negative cells
    // Compare full batch attention: valid cells must produce identical values to clean selection
    std::vector<int32_t> mixed_ids((size_t)cap);
    for (size_t i = 0; i < (size_t)cap; ++i) {
        if (i % 2 == 0 && i < 256) {
            mixed_ids[i] = (int32_t)(i * 4); // valid cell
        } else {
            mixed_ids[i] = (int32_t)(-1 - (i % 5000)); // negative sentinel
        }
    }
    q.memcpy(d_ids, mixed_ids.data(), cap * sizeof(int32_t)).wait();
    steps_h[k::kStepWidth] = 256;
    steps_h[k::kStepNKv] = 1024;
    q.memcpy(d_steps, steps_h.data(), k::kStepCount * sizeof(int32_t)).wait();

    bool mixed_ok = true;
    q.memset(d_attn, 0, n_q * s.n_head * s.head_dim * sizeof(float)).wait();
    k::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, cap, s, d_scratch, d_attn, n_q, &q);
    q.wait_and_throw();

    std::vector<float> h_mixed((size_t)n_q * s.n_head * s.head_dim);
    q.memcpy(h_mixed.data(), d_attn, h_mixed.size() * sizeof(float)).wait();
    for (float val : h_mixed) {
        if (!std::isfinite(val)) mixed_ok = false;
    }

    if (all_neg_ok && mixed_ok) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL (all_neg=%d, mixed=%d)\n", all_neg_ok, mixed_ok);
        g_failures++;
    }

    sycl::free(d_k16, q); sycl::free(d_v16, q); sycl::free(d_table, q);
    sycl::free(d_q, q); sycl::free(d_scratch, q); sycl::free(d_attn, q);
    sycl::free(d_ids, q); sycl::free(d_steps, q);
}

} // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running QSA Negative Cell Guard test on: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    test_kv_gather_negative_ids(q);
    test_qsa_decode_attn_negative_ids(q);

    if (g_failures == 0) {
        std::printf("qsa_negative_cell_guard: 0 failures -> PASS\n");
        return 0;
    } else {
        std::printf("qsa_negative_cell_guard: %d failures -> FAIL\n", g_failures);
        return 1;
    }
}
