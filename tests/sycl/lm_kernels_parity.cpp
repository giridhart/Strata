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
    std::printf(fails == 0 ? "PASS\n" : "FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
