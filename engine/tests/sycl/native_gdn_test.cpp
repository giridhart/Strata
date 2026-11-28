#include <sycl/sycl.hpp>
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void ref_step(std::vector<double>& state, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta, int S,
              int h_k, int h_v, std::vector<float>& o) {
    auto at = [&](int i, int j, int h) -> double& { return state[(size_t)(i * S + j) * h_v + h]; };
    std::vector<float> oo((size_t)h_v * S, 0.0f);
    const double scale = 1.0 / std::sqrt((double)S);
    for (int h = 0; h < h_v; ++h) {
        const int src = h % h_k;
        const double dec = std::exp((double)gate[(size_t)h]);
        const double b = (double)beta[(size_t)h];
        for (int j = 0; j < S; ++j) {
            double sk = 0.0;
            for (int i = 0; i < S; ++i)
                sk += at(i, j, h) * (double)k[(size_t)(src * S + i)];
            const double d = ((double)v[(size_t)(h * S + j)] - dec * sk) * b;
            for (int i = 0; i < S; ++i)
                at(i, j, h) = dec * at(i, j, h) + (double)k[(size_t)(src * S + i)] * d;
            double dot = 0.0;
            for (int i = 0; i < S; ++i)
                dot += at(i, j, h) * (double)q[(size_t)(src * S + i)];
            oo[(size_t)(h * S + j)] = (float)(dot * scale);
        }
    }
    o = oo;
}

}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    printf("Running native_gdn SYCL test on device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    constexpr int S = 128;
    struct Config {
        int h_k, h_v;
        const char* desc;
    };
    const Config configs[] = {
        {1, 1, "h_k=1, h_v=1"}, {2, 4, "h_k=2, h_v=4"}, {16, 48, "Qwen Flash-Next (h_k=16, h_v=48)"}};

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int total_bad = 0;

    for (const auto& cfg : configs) {
        const int h_k = cfg.h_k;
        const int h_v = cfg.h_v;

        std::vector<double> st_ref((size_t)S * S * h_v);
        std::vector<float> st_dev((size_t)S * h_v * S);
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                for (int h = 0; h < h_v; ++h) {
                    const float val = (float)(0.001 * (i + 2 * j + 3 * h) - 1.0);
                    st_ref[(size_t)(i * S + j) * h_v + h] = val;
                    st_dev[(size_t)(i * h_v + h) * S + j] = val;  // Strata layout: (i*h_v + h)*S + j
                }
            }
        }

        std::vector<float> q_vec((size_t)h_k * S), k_vec((size_t)h_k * S), v_vec((size_t)h_v * S);
        std::vector<float> gate_vec((size_t)h_v), beta_vec((size_t)h_v);
        for (auto& x : q_vec)
            x = gauss(rng) * 0.1f;
        for (auto& x : k_vec)
            x = gauss(rng) * 0.1f;
        for (auto& x : v_vec)
            x = gauss(rng);
        for (auto& x : gate_vec)
            x = -(2.0f + 4.0f * (float)(rng() % 100) / 100.0f);
        for (auto& x : beta_vec)
            x = (float)(rng() % 100) / 100.0f;

        std::vector<float> want_o;
        std::vector<double> st_ref_mod = st_ref;
        ref_step(st_ref_mod, q_vec, k_vec, v_vec, gate_vec, beta_vec, S, h_k, h_v, want_o);

        float* d_st = sycl::malloc_device<float>(st_dev.size(), q);
        float* d_q = sycl::malloc_device<float>(q_vec.size(), q);
        float* d_k = sycl::malloc_device<float>(k_vec.size(), q);
        float* d_v = sycl::malloc_device<float>(v_vec.size(), q);
        float* d_g = sycl::malloc_device<float>(gate_vec.size(), q);
        float* d_b = sycl::malloc_device<float>(beta_vec.size(), q);
        float* d_o = sycl::malloc_device<float>((size_t)h_v * S, q);

        q.memcpy(d_st, st_dev.data(), st_dev.size() * sizeof(float));
        q.memcpy(d_q, q_vec.data(), q_vec.size() * sizeof(float));
        q.memcpy(d_k, k_vec.data(), k_vec.size() * sizeof(float));
        q.memcpy(d_v, v_vec.data(), v_vec.size() * sizeof(float));
        q.memcpy(d_g, gate_vec.data(), gate_vec.size() * sizeof(float));
        q.memcpy(d_b, beta_vec.data(), beta_vec.size() * sizeof(float));
        q.wait();

        strata::kernels::GdnShapes sh{S, h_k, h_v};
        strata::kernels::native_gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, &q);

        std::vector<float> got_o((size_t)h_v * S);
        std::vector<float> got_st(st_dev.size());
        q.memcpy(got_o.data(), d_o, got_o.size() * sizeof(float));
        q.memcpy(got_st.data(), d_st, got_st.size() * sizeof(float));
        q.wait();

        // Check output
        int bad_o = 0;
        float max_rel_o = 0.0f;
        for (size_t idx = 0; idx < got_o.size(); ++idx) {
            float diff = std::fabs(got_o[idx] - want_o[idx]);
            float denom = std::max(std::fabs(want_o[idx]), 1e-6f);
            float rel = diff / denom;
            if (rel > max_rel_o) max_rel_o = rel;
            if (rel > 1e-3f && diff > 1e-4f) bad_o++;
        }

        // Check updated state
        int bad_st = 0;
        float max_rel_st = 0.0f;
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                for (int h = 0; h < h_v; ++h) {
                    const size_t dev_idx = (size_t)(i * h_v + h) * S + j;
                    const double w = st_ref_mod[(size_t)(i * S + j) * h_v + h];
                    const double g = got_st[dev_idx];
                    float diff = (float)std::fabs(g - w);
                    float denom = (float)std::max(std::fabs(w), 1e-6);
                    float rel = diff / denom;
                    if (rel > max_rel_st) max_rel_st = rel;
                    if (rel > 1e-3f && diff > 1e-4f) bad_st++;
                }
            }
        }

        printf("  %s: out max_rel=%.3e (bad=%d), state max_rel=%.3e (bad=%d)\n", cfg.desc, max_rel_o, bad_o, max_rel_st,
               bad_st);
        total_bad += bad_o + bad_st;

        sycl::free(d_st, q);
        sycl::free(d_q, q);
        sycl::free(d_k, q);
        sycl::free(d_v, q);
        sycl::free(d_g, q);
        sycl::free(d_b, q);
        sycl::free(d_o, q);
    }

    if (total_bad == 0) {
        printf("native_gdn SYCL parity OK (0 failures)\n");
        return 0;
    } else {
        printf("native_gdn SYCL parity FAILED (%d bad)\n", total_bad);
        return 1;
    }
}
