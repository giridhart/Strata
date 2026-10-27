// tests/sycl/kva_sycl_test.cpp - Test KVA configuration logic and on-device
// GDN recurrence state-only parity against full GDN recurrence.
#include "strata/prefill/kva.hpp"
#include "strata/prefill/kva_floor.hpp"
#include "strata/prefill/kernels.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::prefill;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
static void clear_all() {
    for (const char* k : {"STRATA_KVA", "STRATA_KVA_SPLIT", "STRATA_KVA_SUFFIX", "STRATA_KVA_MIN_PROMPT", "STRATA_KVA_PARITY",
                          "STRATA_KVA_CAPTURE", "STRATA_KVA_CAPTURE_FP32", "STRATA_KVA_ORACLE", "STRATA_KVA_ORACLE_FP32",
                          "STRATA_KVA_PROJECTOR", "STRATA_KVA_MODE", "STRATA_KVA_ONLY", "STRATA_KVA_DRAFT_FLOOR", "STRATA_KVA_SKIP", "STRATA_KVA_ANY_MODEL"})
        unset(k);
}

void test_kva_config() {
    clear_all();
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(!c.on && !c.explicit_on);
        CHECK(c.split == 36 && c.min_prompt == 4096 && c.suffix_env == 0 && kKvaSplitDefault == 36);
        CHECK(c.mode == KvaConfig::Mode::Identity && c.only == KvaConfig::Only::All);
        KvaEngine eng;
        CHECK(!c.wants_cell_floor(eng, true));
        CHECK(c.decline(eng, 0, 30000, 0, -1, 48) != nullptr);
    }
    put("STRATA_KVA", "1");
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(c.on && c.explicit_on);
        CHECK(c.split == 36 && c.effective_suffix() == 2048);
        KvaEngine eng;
        CHECK(c.decline(eng, 0, 14600, 0, -1, 48) == nullptr);
    }
    put("STRATA_KVA_SPLIT", "32");
    put("STRATA_KVA_SUFFIX", "1024");
    put("STRATA_KVA_SKIP", "moe");
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(c.split == 32);
        CHECK(c.effective_suffix() == 1024);
        CHECK(c.skip == KvaConfig::Skip::Moe);
    }
    clear_all();
    std::printf("test_kva_config: OK\n");
}

void test_gdn_recurrence_parity(sycl::queue& q, int64_t T) {
    constexpr int64_t S = 128, HV = 48, C = 10240;
    const size_t state_size = (size_t)HV * S * S;
    std::mt19937 gen(42 + (uint32_t)T);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    std::vector<float> h_state_init(state_size);
    for (auto& v : h_state_init) v = nd(gen);

    std::vector<float> h_h((size_t)T * C);
    for (auto& v : h_h) v = nd(gen);

    std::vector<float> h_gate((size_t)T * HV);
    for (auto& v : h_gate) v = -std::fabs(nd(gen)); // gate < 0 so exp(gate) < 1

    std::vector<float> h_beta((size_t)T * HV);
    for (auto& v : h_beta) v = nd(gen) * 0.1f;

    std::vector<float> h_z((size_t)T * HV * S);
    for (auto& v : h_z) v = nd(gen);

    std::vector<float> h_gamma((size_t)S);
    for (auto& v : h_gamma) v = 1.0f + 0.1f * nd(gen);

    float* d_state_full = sycl::malloc_device<float>(state_size, q);
    float* d_state_approx = sycl::malloc_device<float>(state_size, q);
    float* d_h = sycl::malloc_device<float>(h_h.size(), q);
    float* d_gate = sycl::malloc_device<float>(h_gate.size(), q);
    float* d_beta = sycl::malloc_device<float>(h_beta.size(), q);
    float* d_z = sycl::malloc_device<float>(h_z.size(), q);
    float* d_gamma = sycl::malloc_device<float>(h_gamma.size(), q);
    float* d_y_full = sycl::malloc_device<float>((size_t)T * HV * S, q);
    float* d_y_approx = sycl::malloc_device<float>((size_t)T * HV * S, q);
    uint16_t* d_y16 = sycl::malloc_device<uint16_t>((size_t)T * HV * S, q);

    q.memcpy(d_state_full, h_state_init.data(), state_size * sizeof(float));
    q.memcpy(d_state_approx, h_state_init.data(), state_size * sizeof(float));
    q.memcpy(d_h, h_h.data(), h_h.size() * sizeof(float));
    q.memcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float));
    q.memcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float));
    q.memcpy(d_z, h_z.data(), h_z.size() * sizeof(float));
    q.memcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float));
    q.wait();

    // Run full GDN recurrence
    strata::prefill::gdn_recurrence(d_state_full, d_h, d_gate, d_beta, d_z, d_gamma, 1e-6f, d_y_full, d_y16, T, &q);
    q.wait();

    // Run state-only GDN recurrence (as KVA does)
    strata::prefill::gdn_recurrence_state_only(d_state_approx, d_h, d_gate, d_beta, d_y_approx, T, &q);
    q.wait();

    std::vector<float> got_full(state_size), got_approx(state_size);
    q.memcpy(got_full.data(), d_state_full, state_size * sizeof(float));
    q.memcpy(got_approx.data(), d_state_approx, state_size * sizeof(float));
    q.wait();

    sycl::free(d_state_full, q);
    sycl::free(d_state_approx, q);
    sycl::free(d_h, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_z, q);
    sycl::free(d_gamma, q);
    sycl::free(d_y_full, q);
    sycl::free(d_y_approx, q);
    sycl::free(d_y16, q);

    size_t diffs = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < state_size; ++i) {
        if (std::memcmp(&got_full[i], &got_approx[i], 4) != 0) {
            ++diffs;
            max_err = std::fmax(max_err, std::fabs((double)got_full[i] - got_approx[i]));
        }
    }
    const bool ok = (diffs == 0);
    std::printf("test_gdn_recurrence_parity T=%lld: %zu of %zu state entries differ (max_err=%.2e) %s\n",
                (long long)T, diffs, state_size, max_err, ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}

int main() {
    test_kva_config();

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
    std::printf("Running kva_sycl_test on: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    test_gdn_recurrence_parity(q, 16);
    test_gdn_recurrence_parity(q, 64);
    test_gdn_recurrence_parity(q, 128);

    std::printf(fails == 0 ? "PASS\n" : "FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
