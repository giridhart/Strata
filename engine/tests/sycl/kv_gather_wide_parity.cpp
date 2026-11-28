// tests/sycl/kv_gather_wide_parity.cpp - Parity verification for 8-wide vs 4-wide INT8 KV cache gather on Intel Xe2.
#include <sycl/sycl.hpp>
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

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
    T* p = (T*)sycl::malloc_device(n * sizeof(T) + 64, q);
    q.memset(p, 0, n * sizeof(T) + 64).wait();
    return p;
}

void test_kv_gather_case(int T, int page_size, sycl::queue& q) {
    std::printf("Testing KV gather wide parity [T=%4d, page_size=%2d]... ", T, page_size);

    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = page_size;
    const int H = (int)s.n_head_kv;
    const int D = (int)s.head_dim;
    const int P = (int)s.page_size;
    const int G = D / k::KV_Q8_GROUP;
    const int pages = (T + P - 1) / P;
    const int cells = pages * P;

    std::mt19937 rng(42 + T * 17 + page_size * 31);

    // Random non-identity page table permutation
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);

    // Populate INT8 codes and FP16 scales with challenging values:
    // positive, negative, zero, full clamp limits (+/-127), tiny/large scales
    std::vector<int8_t> h_kq((size_t)cells * H * D), h_vq((size_t)cells * H * D);
    std::vector<uint16_t> h_ks((size_t)cells * H * G), h_vs((size_t)cells * H * G);

    std::uniform_int_distribution<int> dist_code(-127, 127);
    for (auto& c : h_kq) c = (int8_t)dist_code(rng);
    for (auto& c : h_vq) c = (int8_t)dist_code(rng);

    std::uniform_real_distribution<float> dist_scale(0.0005f, 3.5f);
    for (auto& sc : h_ks) sc = k::f16_from_f32(dist_scale(rng));
    for (auto& sc : h_vs) sc = k::f16_from_f32(dist_scale(rng));

    // Selection IDs: test selecting T cells in randomized non-contiguous order
    const int n_ids = T;
    std::vector<int32_t> ids(n_ids);
    for (int i = 0; i < n_ids; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);

    int8_t* d_kq = dalloc<int8_t>(h_kq.size(), q);
    int8_t* d_vq = dalloc<int8_t>(h_vq.size(), q);
    uint16_t* d_ks = dalloc<uint16_t>(h_ks.size(), q);
    uint16_t* d_vs = dalloc<uint16_t>(h_vs.size(), q);
    int32_t* d_table = dalloc<int32_t>(table.size(), q);
    int32_t* d_ids = dalloc<int32_t>(ids.size(), q);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount, q);

    const size_t out_elems = (size_t)n_ids * H * D;
    uint16_t* d_k_scratch_std = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_v_scratch_std = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_k_scratch_wide = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_v_scratch_wide = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_k_scratch_env = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_v_scratch_env = dalloc<uint16_t>(out_elems, q);

    q.memcpy(d_kq, h_kq.data(), h_kq.size()).wait();
    q.memcpy(d_vq, h_vq.data(), h_vq.size()).wait();
    q.memcpy(d_ks, h_ks.data(), h_ks.size() * sizeof(uint16_t)).wait();
    q.memcpy(d_vs, h_vs.data(), h_vs.size() * sizeof(uint16_t)).wait();
    q.memcpy(d_table, table.data(), table.size() * sizeof(int32_t)).wait();
    q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t)).wait();

    int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
    q.memcpy(d_step, hstep, sizeof(hstep)).wait();

    // 1. Launch standard 4-wide gather
    k::kv_gather_q8_standard_step(d_kq, d_vq, d_ks, d_vs, d_table, d_ids, d_step, n_ids, s,
                                  d_k_scratch_std, d_v_scratch_std, &q);

    // 2. Launch wide 8-wide gather
    k::kv_gather_q8_wide_step(d_kq, d_vq, d_ks, d_vs, d_table, d_ids, d_step, n_ids, s,
                              d_k_scratch_wide, d_v_scratch_wide, &q);

    // 3. Launch via kv_gather_q8_step (with default environment -> selects wide)
    k::kv_gather_q8_step(d_kq, d_vq, d_ks, d_vs, d_table, d_ids, d_step, n_ids, s,
                         d_k_scratch_env, d_v_scratch_env, &q);
    q.wait();

    std::vector<uint16_t> h_k_std(out_elems), h_v_std(out_elems);
    std::vector<uint16_t> h_k_wide(out_elems), h_v_wide(out_elems);
    std::vector<uint16_t> h_k_env(out_elems), h_v_env(out_elems);

    q.memcpy(h_k_std.data(), d_k_scratch_std, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_std.data(), d_v_scratch_std, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_k_wide.data(), d_k_scratch_wide, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_wide.data(), d_v_scratch_wide, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_k_env.data(), d_k_scratch_env, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_env.data(), d_v_scratch_env, out_elems * sizeof(uint16_t)).wait();

    // Verification 1: 100% bitwise exact between wide and standard, and default API
    size_t wide_std_diff = 0;
    for (size_t i = 0; i < out_elems; ++i) {
        if (h_k_wide[i] != h_k_std[i]) ++wide_std_diff;
        if (h_v_wide[i] != h_v_std[i]) ++wide_std_diff;
        if (h_k_wide[i] != h_k_env[i]) ++wide_std_diff;
        if (h_v_wide[i] != h_v_env[i]) ++wide_std_diff;
    }

    // Verification 2: 100% bitwise exact against host reference
    size_t host_ref_diff = 0;
    for (int j = 0; j < n_ids; ++j) {
        const int cell = ids[j];
        const int page = table[cell / P];
        for (int h = 0; h < H; ++h) {
            const long long row = ((long long)page * H + h) * P + (cell % P);
            for (int d = 0; d < D; ++d) {
                const size_t o = ((size_t)j * H + h) * D + d;
                const float ks = k::f32_from_f16(h_ks[row * G + d / k::KV_Q8_GROUP]);
                const float vs = k::f32_from_f16(h_vs[row * G + d / k::KV_Q8_GROUP]);
                const int8_t kc = h_kq[row * D + d];
                const int8_t vc = h_vq[row * D + d];
                const uint16_t want_k = k::f16_from_f32((float)kc * ks);
                const uint16_t want_v = k::f16_from_f32((float)vc * vs);

                if (h_k_wide[o] != want_k) ++host_ref_diff;
                if (h_v_wide[o] != want_v) ++host_ref_diff;
            }
        }
    }

    // Verification 3: fallback with STRATA_Q8_GATHER_WIDE=0
    setenv("STRATA_Q8_GATHER_WIDE", "0", 1);
    uint16_t* d_k_scratch_fb = dalloc<uint16_t>(out_elems, q);
    uint16_t* d_v_scratch_fb = dalloc<uint16_t>(out_elems, q);
    k::kv_gather_q8_step(d_kq, d_vq, d_ks, d_vs, d_table, d_ids, d_step, n_ids, s,
                         d_k_scratch_fb, d_v_scratch_fb, &q);
    q.wait();
    std::vector<uint16_t> h_k_fb(out_elems), h_v_fb(out_elems);
    q.memcpy(h_k_fb.data(), d_k_scratch_fb, out_elems * sizeof(uint16_t)).wait();
    q.memcpy(h_v_fb.data(), d_v_scratch_fb, out_elems * sizeof(uint16_t)).wait();
    unsetenv("STRATA_Q8_GATHER_WIDE");

    size_t fb_diff = 0;
    for (size_t i = 0; i < out_elems; ++i) {
        if (h_k_fb[i] != h_k_std[i]) ++fb_diff;
        if (h_v_fb[i] != h_v_std[i]) ++fb_diff;
    }

    sycl::free(d_kq, q); sycl::free(d_vq, q);
    sycl::free(d_ks, q); sycl::free(d_vs, q);
    sycl::free(d_table, q); sycl::free(d_ids, q); sycl::free(d_step, q);
    sycl::free(d_k_scratch_std, q); sycl::free(d_v_scratch_std, q);
    sycl::free(d_k_scratch_wide, q); sycl::free(d_v_scratch_wide, q);
    sycl::free(d_k_scratch_env, q); sycl::free(d_v_scratch_env, q);
    sycl::free(d_k_scratch_fb, q); sycl::free(d_v_scratch_fb, q);

    if (wide_std_diff == 0 && host_ref_diff == 0 && fb_diff == 0) {
        std::printf("PASS (exact bits, %zu values verified, 0 diffs)\n", out_elems * 2);
    } else {
        std::printf("FAIL (wide_std_diff=%zu, host_ref_diff=%zu, fb_diff=%zu)\n",
                    wide_std_diff, host_ref_diff, fb_diff);
        ++g_failures;
    }
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("=== Intel Xe2 8-Wide INT8 KV Cache Gather Parity Verification ===\n");
    std::printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const std::vector<int> cell_counts = {64, 256, 1024, 4096};
    const std::vector<int> page_sizes = {16, 64};

    for (int page_size : page_sizes) {
        for (int T : cell_counts) {
            test_kv_gather_case(T, page_size, q);
        }
    }

    std::printf("\n=================================================================\n");
    if (g_failures == 0) {
        std::printf("All 8/8 KV Gather wide parity suites PASSED! (0 failures)\n");
        return 0;
    } else {
        std::printf("FAILED: %d test suites failed!\n", g_failures);
        return 1;
    }
}
