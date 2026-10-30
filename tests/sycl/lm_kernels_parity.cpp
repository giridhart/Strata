// tests/sycl/lm_kernels_parity.cpp - the two kernels of the layer-major prompt path (STRATA_PREFILL_LAYER_MAJOR=1)
// against a host reference, on synthetic data of the model's width (N = 2560): no model, a few MiB.
//   moe_shared_init   acc[t, :] = shared[t, :] * sigmoid(sg[t])      (within 1e-5 relative)
//   moe_scatter_add   acc[u_tok[u], :] += sum_{j in [u_off[u], u_off[u+1])} u_w[j] * Dm[u_rows[j], :], in that j order,
//                     one block per token and no atomics: the same bits as the host's fmaf chain, and the same bits on a
//                     second launch (deterministic)
// Exit 0 pass, 1 fail, 77 no SYCL device (ctest SKIP_RETURN_CODE).
#include "strata/prefill/kernels.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int N = 2560;

int fails = 0;

void check_shared_init(sycl::queue& q, int64_t T, std::mt19937& g) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> shared((size_t)T * N), sg((size_t)T), ref((size_t)T * N), got((size_t)T * N);
    for (auto& x : shared) x = nd(g);
    for (auto& x : sg) x = nd(g) * 3.0f;
    for (int64_t t = 0; t < T; ++t)
        for (int d = 0; d < N; ++d)
            ref[(size_t)t * N + d] = shared[(size_t)t * N + d] * (1.0f / (1.0f + std::exp(-sg[(size_t)t])));

    float* d_sh = sycl::malloc_device<float>(shared.size(), q);
    float* d_sg = sycl::malloc_device<float>(sg.size(), q);
    float* d_acc = sycl::malloc_device<float>(shared.size(), q);

    q.memcpy(d_sh, shared.data(), shared.size() * sizeof(float));
    q.memcpy(d_sg, sg.data(), sg.size() * sizeof(float));
    q.memset(d_acc, 0xff, shared.size() * sizeof(float));
    q.wait();

    strata::prefill::moe_shared_init(d_sh, d_sg, d_acc, T, &q);
    q.wait();

    q.memcpy(got.data(), d_acc, got.size() * sizeof(float)).wait();

    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double scale = std::fabs(ref[i]) + 1e-6;
        worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]) / scale);
    }
    const bool ok = worst < 1e-5;
    std::printf("moe_shared_init T=%lld: worst relative error %.2e %s\n", (long long)T, worst, ok ? "ok" : "FAIL");
    fails += ok ? 0 : 1;

    sycl::free(d_sh, q);
    sycl::free(d_sg, q);
    sycl::free(d_acc, q);
}

void check_scatter(sycl::queue& q, int64_t tokens, int64_t rows, int max_per_tok, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, (int)tokens - 1);
    std::vector<int> per((size_t)tokens, 0);
    std::vector<int32_t> row_tok;
    while ((int64_t)row_tok.size() < rows) {
        const int t = pick(g);
        if (per[(size_t)t] >= max_per_tok) continue;
        ++per[(size_t)t];
        row_tok.push_back(t);
    }
    std::vector<int32_t> u_tok, u_off, u_rows(rows);
    std::vector<int> slot((size_t)tokens, -1);
    for (int32_t r = 0; r < (int32_t)rows; ++r) {
        const int t = row_tok[(size_t)r];
        if (slot[(size_t)t] < 0) {
            slot[(size_t)t] = (int)u_tok.size();
            u_tok.push_back(t);
            u_off.push_back(0);
        }
        ++u_off[(size_t)slot[(size_t)t]];
    }
    const int64_t nu = (int64_t)u_tok.size();
    std::vector<int32_t> fillp((size_t)nu);
    int32_t acc_n = 0;
    for (int64_t i = 0; i < nu; ++i) {
        const int32_t c = u_off[(size_t)i];
        u_off[(size_t)i] = acc_n;
        fillp[(size_t)i] = acc_n;
        acc_n += c;
    }
    u_off.push_back((int32_t)rows);
    std::vector<float> u_w(rows);
    std::vector<float> w_row((size_t)rows);
    for (auto& x : w_row) x = std::fabs(nd(g)) * 0.2f;
    for (int32_t r = 0; r < (int32_t)rows; ++r) {
        const int32_t p = fillp[(size_t)slot[(size_t)row_tok[(size_t)r]]]++;
        u_rows[(size_t)p] = r;
        u_w[(size_t)p] = w_row[(size_t)r];
    }
    std::vector<float> Dm((size_t)rows * N), acc0((size_t)tokens * N);
    for (auto& x : Dm) x = nd(g);
    for (auto& x : acc0) x = nd(g);

    std::vector<float> ref = acc0;
    for (int64_t u = 0; u < nu; ++u)
        for (int d = 0; d < N; ++d) {
            float s = 0.0f;
            for (int32_t j = u_off[(size_t)u]; j < u_off[(size_t)u + 1]; ++j)
                s = std::fma(u_w[(size_t)j], Dm[(size_t)u_rows[(size_t)j] * N + d], s);
            ref[(size_t)u_tok[(size_t)u] * N + d] += s;
        }

    float* d_Dm = sycl::malloc_device<float>(Dm.size(), q);
    float* d_acc = sycl::malloc_device<float>(acc0.size(), q);
    float* d_w = sycl::malloc_device<float>(u_w.size(), q);
    int32_t* d_ut = sycl::malloc_device<int32_t>(u_tok.size(), q);
    int32_t* d_uo = sycl::malloc_device<int32_t>(u_off.size(), q);
    int32_t* d_ur = sycl::malloc_device<int32_t>(u_rows.size(), q);

    q.memcpy(d_Dm, Dm.data(), Dm.size() * sizeof(float));
    q.memcpy(d_w, u_w.data(), u_w.size() * sizeof(float));
    q.memcpy(d_ut, u_tok.data(), u_tok.size() * sizeof(int32_t));
    q.memcpy(d_uo, u_off.data(), u_off.size() * sizeof(int32_t));
    q.memcpy(d_ur, u_rows.data(), u_rows.size() * sizeof(int32_t));
    q.wait();

    std::vector<float> got1(acc0.size()), got2(acc0.size());
    for (int run = 0; run < 2; ++run) {
        q.memcpy(d_acc, acc0.data(), acc0.size() * sizeof(float)).wait();
        strata::prefill::moe_scatter_add(d_Dm, d_ut, d_uo, d_ur, d_w, nu, d_acc, &q);
        q.wait();
        q.memcpy((run ? got2 : got1).data(), d_acc, acc0.size() * sizeof(float)).wait();
    }

    size_t diff = 0, diff_runs = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        diff += std::memcmp(&ref[i], &got1[i], 4) != 0;
        diff_runs += std::memcmp(&got1[i], &got2[i], 4) != 0;
    }
    const bool ok = diff == 0 && diff_runs == 0;
    std::printf(
        "moe_scatter_add tokens=%lld rows=%lld (<= %d per token, %lld tokens in the group): %zu values differ from "
        "the host chain, %zu differ between two launches %s\n",
        (long long)tokens, (long long)rows, max_per_tok, (long long)nu, diff, diff_runs, ok ? "ok" : "FAIL");
    fails += ok ? 0 : 1;

    sycl::free(d_Dm, q);
    sycl::free(d_acc, q);
    sycl::free(d_w, q);
    sycl::free(d_ut, q);
    sycl::free(d_uo, q);
    sycl::free(d_ur, q);
}

void check_combine(sycl::queue& q, int64_t tokens, int64_t rows, bool with_row_sd, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> pick_row(0, (int32_t)rows - 1);

    std::vector<int32_t> slot((size_t)tokens * 10);
    for (auto& s : slot) s = pick_row(g);

    std::vector<float> w((size_t)tokens * 10);
    for (auto& x : w) x = std::fabs(nd(g)) * 0.1f;

    std::vector<float> shared((size_t)tokens * N), sg((size_t)tokens);
    for (auto& x : shared) x = nd(g);
    for (auto& x : sg) x = nd(g) * 2.0f;

    std::vector<float> Dm((size_t)rows * N);
    for (auto& x : Dm) x = nd(g);

    std::vector<float> row_sd(with_row_sd ? rows : 0);
    if (with_row_sd) {
        for (auto& x : row_sd) x = std::fabs(nd(g)) + 0.1f;
    }

    // Host reference calculation
    std::vector<float> ref((size_t)tokens * N);
    for (int64_t t = 0; t < tokens; ++t) {
        const float g_val = 1.0f / (1.0f + std::exp(-sg[(size_t)t]));
        for (int d = 0; d < N; ++d) {
            float s = 0.0f;
            for (int k = 0; k < 10; ++k) {
                const int64_t r = slot[(size_t)t * 10 + k];
                const float v = with_row_sd ? Dm[(size_t)r * N + d] * row_sd[(size_t)r] : Dm[(size_t)r * N + d];
                s = std::fma(w[(size_t)t * 10 + k], v, s);
            }
            ref[(size_t)t * N + d] = s + shared[(size_t)t * N + d] * g_val;
        }
    }

    float* d_Dm = sycl::malloc_device<float>(Dm.size(), q);
    int32_t* d_slot = sycl::malloc_device<int32_t>(slot.size(), q);
    float* d_w = sycl::malloc_device<float>(w.size(), q);
    float* d_shared = sycl::malloc_device<float>(shared.size(), q);
    float* d_sg = sycl::malloc_device<float>(sg.size(), q);
    float* d_bo = sycl::malloc_device<float>(ref.size(), q);
    float* d_row_sd = with_row_sd ? sycl::malloc_device<float>(row_sd.size(), q) : nullptr;

    q.memcpy(d_Dm, Dm.data(), Dm.size() * sizeof(float));
    q.memcpy(d_slot, slot.data(), slot.size() * sizeof(int32_t));
    q.memcpy(d_w, w.data(), w.size() * sizeof(float));
    q.memcpy(d_shared, shared.data(), shared.size() * sizeof(float));
    q.memcpy(d_sg, sg.data(), sg.size() * sizeof(float));
    if (with_row_sd) {
        q.memcpy(d_row_sd, row_sd.data(), row_sd.size() * sizeof(float));
    }
    q.wait();

    // 1. Run default (vectorized 128-bit moe_combine4)
    unsetenv("STRATA_NO_COMB4");
    q.memset(d_bo, 0, ref.size() * sizeof(float)).wait();
    strata::prefill::moe_combine(d_Dm, d_slot, d_w, d_shared, d_sg, d_bo, tokens, &q, d_row_sd);
    q.wait();
    std::vector<float> got_vec(ref.size());
    q.memcpy(got_vec.data(), d_bo, got_vec.size() * sizeof(float)).wait();

    // 2. Run scalar fallback kernel
    setenv("STRATA_NO_COMB4", "1", 1);
    q.memset(d_bo, 0, ref.size() * sizeof(float)).wait();
    strata::prefill::moe_combine(d_Dm, d_slot, d_w, d_shared, d_sg, d_bo, tokens, &q, d_row_sd);
    q.wait();
    std::vector<float> got_scalar(ref.size());
    q.memcpy(got_scalar.data(), d_bo, got_scalar.size() * sizeof(float)).wait();
    unsetenv("STRATA_NO_COMB4");

    // Check accuracy against host reference and check bitwise parity between vector & scalar
    double max_err_vec = 0.0, max_err_scalar = 0.0, max_val = 0.0;
    size_t diff_vec_scalar = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff_v = std::fabs((double)got_vec[i] - ref[i]);
        const double diff_s = std::fabs((double)got_scalar[i] - ref[i]);
        max_err_vec = std::fmax(max_err_vec, diff_v);
        max_err_scalar = std::fmax(max_err_scalar, diff_s);
        max_val = std::fmax(max_val, std::fabs((double)ref[i]));
        if (std::memcmp(&got_vec[i], &got_scalar[i], sizeof(float)) != 0) {
            ++diff_vec_scalar;
        }
    }

    const double rel_vec = max_err_vec / (max_val > 0.0 ? max_val : 1.0);
    const double rel_scalar = max_err_scalar / (max_val > 0.0 ? max_val : 1.0);
    const bool ok = (rel_vec < 1e-4) && (rel_scalar < 1e-4) && (diff_vec_scalar == 0);
    std::printf("moe_combine tokens=%lld rows=%lld row_sd=%d: rel_vec=%.2e rel_scalar=%.2e diff_vec_scalar=%zu %s\n",
                (long long)tokens, (long long)rows, with_row_sd ? 1 : 0,
                rel_vec, rel_scalar, diff_vec_scalar, ok ? "ok" : "FAIL");
    fails += ok ? 0 : 1;

    sycl::free(d_Dm, q);
    sycl::free(d_slot, q);
    sycl::free(d_w, q);
    sycl::free(d_shared, q);
    sycl::free(d_sg, q);
    sycl::free(d_bo, q);
    if (d_row_sd) sycl::free(d_row_sd, q);
}

}  // namespace

int main() {
    sycl::device dev;
    try {
        dev = sycl::device(sycl::default_selector_v);
    } catch (...) {
        std::printf("SKIP: no SYCL device\n");
        return 77;
    }
    if (!dev.is_gpu()) {
        std::printf("SKIP: default SYCL device is not GPU\n");
        return 77;
    }
    sycl::queue q{dev, sycl::property::queue::in_order()};
    std::printf("Running lm_kernels_parity on: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    std::mt19937 g(7);
    check_shared_init(q, 1, g);
    check_shared_init(q, 512, g);
    check_shared_init(q, 2049, g);
    check_scatter(q, 512, 1, 8, 1);        // one row
    check_scatter(q, 512, 700, 8, 2);      // a group inside one sub-chunk
    check_scatter(q, 4096, 5000, 8, 3);    // several sub-chunks of a super-chunk
    check_scatter(q, 8192, 8192, 1, 4);    // every token once
    check_scatter(q, 300, 2400, 8, 5);     // every token at its limit
    check_combine(q, 1, 100, false, 10);   // single token
    check_combine(q, 16, 200, false, 11);  // small batch
    check_combine(q, 512, 1000, false, 12); // chunk
    check_combine(q, 512, 1000, true, 13);  // chunk with row_sd (NVFP4 scale)
    check_combine(q, 2049, 2000, false, 14); // super-chunk length
    std::printf(fails == 0 ? "PASS\n" : "FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
