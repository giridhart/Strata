// bench/sycl/gdn_chunk_bench.cpp - the chunked GDN prompt recurrence against the token recurrence,
// real GDN geometry (S=128, 16 q/k heads, 48 v heads, C=10240), pure SYCL 2020.
//
// One call of `gdn_recurrence` per side (the whole prompt chunk for one GDN layer: conv output -> out norm),
// A/B'd with STRATA_GDN_CHUNK=0 / =1 so the dispatch, the tail and the norm are timed exactly as the engine
// runs them. The two paths are not bit-exact (the chunked one reorders the sums); the bench reports the max
// relative difference it saw so a silent divergence cannot hide behind a speedup.
//
//   usage: gdn_chunk_bench [T ...]        (default 1024 4096 8192)
#include <sycl/sycl.hpp>
#include "strata/prefill/kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int S = 128, HV = 48, C = 10240;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct Rng {
    uint64_t s = 12345;
    float uni() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float) ((s >> 40) * (1.0 / 16777216.0));
    }
};

float timed(sycl::queue& q, const char* mode, const char* warp, float* st, float* st0, float* h, float* g, float* b,
            float* z, float* ga, float* y, uint16_t* y16, int64_t T, int iters) {
    set_env("STRATA_GDN_CHUNK", mode);
    set_env("STRATA_GDN_CHUNKED", mode);
    set_env("STRATA_GDN_REC_WARP", warp);

    q.memcpy(st, st0, (size_t) S * HV * S * sizeof(float));
    strata::prefill::gdn_recurrence(st, h, g, b, z, ga, 1e-5f, y, y16, T, &q);
    q.wait();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        q.memcpy(st, st0, (size_t) S * HV * S * sizeof(float));
        strata::prefill::gdn_recurrence(st, h, g, b, z, ga, 1e-5f, y, y16, T, &q);
    }
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int64_t> ts;
    for (int i = 1; i < argc; ++i) ts.push_back(std::atoll(argv[i]));
    if (ts.empty()) ts = {1024, 4096, 8192};
    int64_t tmax = 0;
    for (int64_t t : ts) tmax = t > tmax ? t : tmax;

    sycl::queue q{sycl::default_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    auto dev = q.get_device();
    std::string dev_name = dev.get_info<sycl::info::device::name>();
    std::printf("%s  GDN prompt recurrence: chunked vs token-by-token\n", dev_name.c_str());

    Rng rng;
    const size_t n_h = (size_t) tmax * C, n_y = (size_t) tmax * HV * S, n_g = (size_t) tmax * HV;
    float* d_h = sycl::malloc_device<float>(n_h, q);
    float* d_y = sycl::malloc_device<float>(n_y, q);
    float* d_z = sycl::malloc_device<float>(n_y, q);
    float* d_g = sycl::malloc_device<float>(n_g, q);
    float* d_b = sycl::malloc_device<float>(n_g, q);
    float* d_ga = sycl::malloc_device<float>(S, q);
    float* d_st = sycl::malloc_device<float>((size_t) S * HV * S, q);
    float* d_st0 = sycl::malloc_device<float>((size_t) S * HV * S, q);
    uint16_t* d_y16 = sycl::malloc_device<uint16_t>(n_y, q);

    if (!d_h || !d_y || !d_z || !d_g || !d_b || !d_ga || !d_st || !d_st0 || !d_y16) {
        std::fprintf(stderr, "Failed to allocate device memory\n");
        return 1;
    }

    std::vector<float> tmp((size_t) tmax * C);
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = rng.uni() - 0.5f;
    // q/k rows are L2-normalised rows (as gdn_conv leaves them); v rows stay O(1).  A degenerate near-zero
    // v would make the out-norm amplify rounding noise by 1/sqrt(eps) and the diff column meaningless.
    for (int64_t t = 0; t < tmax; ++t)
        for (int row = 0; row < 2 * 16; ++row) {
            float* r = tmp.data() + (size_t) t * C + row * S;
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += (double) r[i] * r[i];
            const float nrm = (float) (1.0 / std::sqrt(ss + 1e-5));
            for (int i = 0; i < S; ++i) r[i] *= nrm;
        }
    q.memcpy(d_h, tmp.data(), n_h * sizeof(float));
    tmp.resize(n_g);
    for (size_t i = 0; i < n_g; ++i) tmp[i] = -0.02f - 0.3f * rng.uni();
    q.memcpy(d_g, tmp.data(), n_g * sizeof(float));
    for (size_t i = 0; i < n_g; ++i) tmp[i] = 1.0f / (1.0f + std::exp(-(4.0f * rng.uni() - 2.0f)));
    q.memcpy(d_b, tmp.data(), n_g * sizeof(float));
    tmp.resize(n_y);
    for (size_t i = 0; i < n_y; ++i) tmp[i] = 2.0f * rng.uni() - 1.0f;
    q.memcpy(d_z, tmp.data(), n_y * sizeof(float));
    tmp.resize(S);
    for (int i = 0; i < S; ++i) tmp[i] = 0.5f + 2.0f * rng.uni();
    q.memcpy(d_ga, tmp.data(), S * sizeof(float));
    tmp.resize((size_t) S * HV * S);
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = 0.1f * (rng.uni() - 0.5f);
    q.memcpy(d_st0, tmp.data(), tmp.size() * sizeof(float));
    q.wait();

    std::printf("%8s %12s %12s %12s %12s %12s   %s\n", "T", "pipe(rec)", "warp(rec)", "fast(rec)", "chunk FMA",
                "chunk wmma", "max abs diff vs pipe");
    for (int64_t T : ts) {
        const int iters = T >= 4096 ? 5 : 20;
        const float rec = timed(q, "0", "0", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        std::vector<float> y0((size_t) T * HV * S);
        q.memcpy(y0.data(), d_y, y0.size() * sizeof(float)).wait();
        const float wrp = timed(q, "0", "1", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        const float fst = timed(q, "0", "2", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        const float fma = timed(q, "2", "0", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        const float ch = timed(q, "1", "0", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        std::vector<float> y1((size_t) T * HV * S);
        q.memcpy(y1.data(), d_y, y1.size() * sizeof(float)).wait();
        double absd = 0;
        for (size_t i = 0; i < y0.size(); ++i) {
            const double d = std::fabs((double) y1[i] - (double) y0[i]);
            if (d > absd) absd = d;
        }
        std::printf("%8lld %11.3fms %11.3fms %11.3fms %11.3fms %11.3fms   %.2e (wmma vs pipe)\n", (long long) T,
                    rec, wrp, fst, fma, ch, absd);
    }

    sycl::free(d_h, q);
    sycl::free(d_y, q);
    sycl::free(d_z, q);
    sycl::free(d_g, q);
    sycl::free(d_b, q);
    sycl::free(d_ga, q);
    sycl::free(d_st, q);
    sycl::free(d_st0, q);
    sycl::free(d_y16, q);

    return 0;
}
