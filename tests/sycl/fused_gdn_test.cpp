#include <sycl/sycl.hpp>
#include "strata/kernels/fused_gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using strata::kernels::fused_gdn_ab;
using strata::kernels::fused_gdn_conv_l2;
using strata::kernels::fused_gdn_step_norm;
using strata::kernels::fused_gdn_step_norm_silu;

namespace {

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    printf("Running fused_gdn SYCL test on device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int total_bad = 0;

    // 1. Test fused_gdn_conv_l2
    {
        const int channels = 2560;  // 20 heads of 128
        const int qk_heads = 16;
        const float eps = 1e-5f;

        std::vector<float> h_hist(channels * 3);
        std::vector<float> h_qkv(channels);
        std::vector<float> h_w(channels * 4);
        std::vector<float> ref_h(channels);
        std::vector<float> ref_hist(channels * 3);

        for (auto& v : h_hist)
            v = dist(rng);
        for (auto& v : h_qkv)
            v = dist(rng);
        for (auto& v : h_w)
            v = dist(rng) * 0.1f;

        // CPU reference
        for (int c = 0; c < channels; ++c) {
            float v0 = h_hist[c * 3], v1 = h_hist[c * 3 + 1], v2 = h_hist[c * 3 + 2], x = h_qkv[c];
            float sum = v0 * h_w[c * 4] + v1 * h_w[c * 4 + 1] + v2 * h_w[c * 4 + 2] + x * h_w[c * 4 + 3];
            ref_hist[c * 3] = v1;
            ref_hist[c * 3 + 1] = v2;
            ref_hist[c * 3 + 2] = x;
            ref_h[c] = sum / (1.0f + std::exp(-sum));
        }

        // Apply L2 norm per head for qk_heads
        for (int h = 0; h < qk_heads; ++h) {
            float sum_sq = 0.0f;
            for (int i = 0; i < 128; ++i) {
                float val = ref_h[h * 128 + i];
                sum_sq += val * val;
            }
            float scale = 1.0f / std::sqrt(sum_sq + eps);
            for (int i = 0; i < 128; ++i) {
                ref_h[h * 128 + i] *= scale;
            }
        }

        float* d_hist = sycl::malloc_device<float>(h_hist.size(), q);
        float* d_qkv = sycl::malloc_device<float>(h_qkv.size(), q);
        float* d_w = sycl::malloc_device<float>(h_w.size(), q);
        float* d_h = sycl::malloc_device<float>(channels, q);

        q.memcpy(d_hist, h_hist.data(), h_hist.size() * sizeof(float));
        q.memcpy(d_qkv, h_qkv.data(), h_qkv.size() * sizeof(float));
        q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(float));
        q.wait();

        fused_gdn_conv_l2(d_hist, d_qkv, d_w, d_h, channels, qk_heads, eps, &q);

        std::vector<float> dev_h(channels), dev_hist(h_hist.size());
        q.memcpy(dev_h.data(), d_h, channels * sizeof(float));
        q.memcpy(dev_hist.data(), d_hist, h_hist.size() * sizeof(float));
        q.wait();

        int bad_conv = 0;
        for (int c = 0; c < channels; ++c) {
            float diff = std::fabs(dev_h[c] - ref_h[c]);
            float denom = std::max(std::fabs(ref_h[c]), 1e-4f);
            if (diff / denom > 1e-4f && diff > 1e-4f) bad_conv++;
            for (int tap = 0; tap < 3; ++tap) {
                if (std::fabs(dev_hist[c * 3 + tap] - ref_hist[c * 3 + tap]) > 1e-6f) bad_conv++;
            }
        }
        printf("  fused_gdn_conv_l2: bad = %d / %d\n", bad_conv, channels);
        total_bad += bad_conv;

        sycl::free(d_hist, q);
        sycl::free(d_qkv, q);
        sycl::free(d_w, q);
        sycl::free(d_h, q);
    }

    // 2. Test fused_gdn_ab
    {
        const int n_embd = 2560;
        const int h_v = 32;

        std::vector<float> h_x(n_embd);
        std::vector<uint16_t> h_wa(h_v * n_embd);
        std::vector<uint16_t> h_wb(h_v * n_embd);
        std::vector<float> h_dt(h_v);
        std::vector<float> h_ssm_a(h_v);
        std::vector<float> ref_gate(h_v);
        std::vector<float> ref_beta(h_v);

        for (auto& v : h_x)
            v = dist(rng) * 0.1f;
        for (auto& v : h_dt)
            v = dist(rng) * 0.1f;
        for (auto& v : h_ssm_a)
            v = dist(rng) * 0.1f;

        auto float_to_bf16 = [](float f) -> uint16_t {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return static_cast<uint16_t>(u >> 16);
        };
        auto bf16_to_float = [](uint16_t b) -> float {
            uint32_t u = static_cast<uint32_t>(b) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        };

        // BF16 bit patterns: encode float as bf16
        for (size_t i = 0; i < h_wa.size(); ++i) {
            float f = dist(rng) * 0.05f;
            h_wa[i] = float_to_bf16(f);
        }
        for (size_t i = 0; i < h_wb.size(); ++i) {
            float f = dist(rng) * 0.05f;
            h_wb[i] = float_to_bf16(f);
        }

        // CPU reference for ab
        for (int r = 0; r < h_v; ++r) {
            // gate
            double acc_a = 0.0;
            for (int j = 0; j < n_embd; ++j) {
                float w = bf16_to_float(h_wa[r * n_embd + j]);
                acc_a += (double)w * (double)h_x[j];
            }
            float va = (float)acc_a + h_dt[r];
            float sp = va > 20.0f ? va : std::log1p(std::exp(va));
            ref_gate[r] = sp * h_ssm_a[r];

            // beta
            double acc_b = 0.0;
            for (int j = 0; j < n_embd; ++j) {
                float w = bf16_to_float(h_wb[r * n_embd + j]);
                acc_b += (double)w * (double)h_x[j];
            }
            ref_beta[r] = 1.0f / (1.0f + std::exp(-(float)acc_b));
        }

        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        uint16_t* d_wa = sycl::malloc_device<uint16_t>(h_wa.size(), q);
        uint16_t* d_wb = sycl::malloc_device<uint16_t>(h_wb.size(), q);
        float* d_dt = sycl::malloc_device<float>(h_dt.size(), q);
        float* d_ssm_a = sycl::malloc_device<float>(h_ssm_a.size(), q);
        float* d_gate = sycl::malloc_device<float>(h_v, q);
        float* d_beta = sycl::malloc_device<float>(h_v, q);

        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float));
        q.memcpy(d_wa, h_wa.data(), h_wa.size() * sizeof(uint16_t));
        q.memcpy(d_wb, h_wb.data(), h_wb.size() * sizeof(uint16_t));
        q.memcpy(d_dt, h_dt.data(), h_dt.size() * sizeof(float));
        q.memcpy(d_ssm_a, h_ssm_a.data(), h_ssm_a.size() * sizeof(float));
        q.wait();

        fused_gdn_ab(d_x, d_wa, d_wb, d_dt, d_ssm_a, d_gate, d_beta, n_embd, h_v, &q);

        std::vector<float> dev_gate(h_v), dev_beta(h_v);
        q.memcpy(dev_gate.data(), d_gate, h_v * sizeof(float));
        q.memcpy(dev_beta.data(), d_beta, h_v * sizeof(float));
        q.wait();

        int bad_ab = 0;
        for (int r = 0; r < h_v; ++r) {
            float diff_g = std::fabs(dev_gate[r] - ref_gate[r]);
            float denom_g = std::max(std::fabs(ref_gate[r]), 1e-4f);
            if (diff_g / denom_g > 1e-3f && diff_g > 1e-3f) bad_ab++;

            float diff_b = std::fabs(dev_beta[r] - ref_beta[r]);
            float denom_b = std::max(std::fabs(ref_beta[r]), 1e-4f);
            if (diff_b / denom_b > 1e-3f && diff_b > 1e-3f) bad_ab++;
        }
        printf("  fused_gdn_ab: bad = %d / %d\n", bad_ab, h_v);
        total_bad += bad_ab;

        sycl::free(d_x, q);
        sycl::free(d_wa, q);
        sycl::free(d_wb, q);
        sycl::free(d_dt, q);
        sycl::free(d_ssm_a, q);
        sycl::free(d_gate, q);
        sycl::free(d_beta, q);
    }

    // 3. Test fused_gdn_step_norm
    {
        const int h_k = 16;
        const int h_v = 32;
        const int S = 128;
        const float eps = 1e-5f;

        std::vector<float> h_state(h_v * S * S);
        std::vector<float> h_q(h_k * S);
        std::vector<float> h_k_vec(h_k * S);
        std::vector<float> h_v_vec(h_v * S);
        std::vector<float> h_gate(h_v);
        std::vector<float> h_beta(h_v);
        std::vector<float> h_z(h_v * S);
        std::vector<float> h_gamma(S);
        std::vector<float> ref_y(h_v * S);
        std::vector<float> ref_state(h_v * S * S);

        for (auto& v : h_state)
            v = dist(rng) * 0.05f;
        for (auto& v : h_q)
            v = dist(rng);
        for (auto& v : h_k_vec)
            v = dist(rng);
        for (auto& v : h_v_vec)
            v = dist(rng);
        for (auto& v : h_gate)
            v = -0.1f * std::fabs(dist(rng));
        for (auto& v : h_beta)
            v = sigmoid_f(dist(rng));
        for (auto& v : h_z)
            v = dist(rng);
        for (auto& v : h_gamma)
            v = 1.0f + 0.1f * dist(rng);

        ref_state = h_state;

        // CPU reference for fused_gdn_step_norm
        for (int head = 0; head < h_v; ++head) {
            const int qh = head % h_k;
            const float g = std::exp(h_gate[head]);
            const float b = h_beta[head];

            // 1. kv_col for each col: sum_r state[r, col] * k[r]
            std::vector<float> kv_col(S, 0.0f);
            for (int col = 0; col < S; ++col) {
                float kv = 0.0f;
                for (int r = 0; r < S; ++r) {
                    float s_val = ref_state[(r * h_v + head) * S + col];
                    kv += s_val * h_k_vec[qh * S + r];
                }
                kv_col[col] = kv;
            }

            // 2. update state and compute o_col
            std::vector<float> oc(S, 0.0f);
            for (int col = 0; col < S; ++col) {
                float delta = (h_v_vec[head * S + col] - g * kv_col[col]) * b;
                float o = 0.0f;
                for (int r = 0; r < S; ++r) {
                    size_t s_idx = (r * h_v + head) * S + col;
                    float new_s = g * ref_state[s_idx] + h_k_vec[qh * S + r] * delta;
                    ref_state[s_idx] = new_s;
                    o += new_s * h_q[qh * S + r];
                }
                oc[col] = o * (1.0f / std::sqrt((float)S));
            }

            // 3. RMSNorm over head's 128 outputs + silu(z) gating
            float sum_sq = 0.0f;
            for (int col = 0; col < S; ++col) {
                sum_sq += oc[col] * oc[col];
            }
            float scale = 1.0f / std::sqrt(sum_sq / (float)S + eps);
            for (int col = 0; col < S; ++col) {
                float zz = h_z[head * S + col];
                float gate_z = 1.0f / (1.0f + std::exp(-zz));
                ref_y[head * S + col] = oc[col] * scale * h_gamma[col] * gate_z;
            }
        }

        float* d_state = sycl::malloc_device<float>(h_state.size(), q);
        float* d_q = sycl::malloc_device<float>(h_q.size(), q);
        float* d_k = sycl::malloc_device<float>(h_k_vec.size(), q);
        float* d_v = sycl::malloc_device<float>(h_v_vec.size(), q);
        float* d_gate = sycl::malloc_device<float>(h_gate.size(), q);
        float* d_beta = sycl::malloc_device<float>(h_beta.size(), q);
        float* d_z = sycl::malloc_device<float>(h_z.size(), q);
        float* d_gamma = sycl::malloc_device<float>(h_gamma.size(), q);
        float* d_y = sycl::malloc_device<float>(ref_y.size(), q);

        q.memcpy(d_state, h_state.data(), h_state.size() * sizeof(float));
        q.memcpy(d_q, h_q.data(), h_q.size() * sizeof(float));
        q.memcpy(d_k, h_k_vec.data(), h_k_vec.size() * sizeof(float));
        q.memcpy(d_v, h_v_vec.data(), h_v_vec.size() * sizeof(float));
        q.memcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float));
        q.memcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float));
        q.memcpy(d_z, h_z.data(), h_z.size() * sizeof(float));
        q.memcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float));
        q.wait();

        fused_gdn_step_norm(d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, eps, d_y, h_k, h_v, &q);

        std::vector<float> dev_y(ref_y.size()), dev_state(h_state.size());
        q.memcpy(dev_y.data(), d_y, ref_y.size() * sizeof(float));
        q.memcpy(dev_state.data(), d_state, h_state.size() * sizeof(float));
        q.wait();

        int bad_step = 0;
        for (size_t i = 0; i < ref_y.size(); ++i) {
            float diff = std::fabs(dev_y[i] - ref_y[i]);
            float denom = std::max(std::fabs(ref_y[i]), 1e-4f);
            if (diff / denom > 2e-3f && diff > 1e-3f) bad_step++;
        }
        for (size_t i = 0; i < ref_state.size(); ++i) {
            float diff = std::fabs(dev_state[i] - ref_state[i]);
            float denom = std::max(std::fabs(ref_state[i]), 1e-4f);
            if (diff / denom > 2e-3f && diff > 1e-3f) bad_step++;
        }
        printf("  fused_gdn_step_norm: bad = %d / %zu\n", bad_step, ref_y.size());
        total_bad += bad_step;

        sycl::free(d_state, q);
        sycl::free(d_q, q);
        sycl::free(d_k, q);
        sycl::free(d_v, q);
        sycl::free(d_gate, q);
        sycl::free(d_beta, q);
        sycl::free(d_z, q);
        sycl::free(d_gamma, q);
        sycl::free(d_y, q);
    }

    // 4. Test fused_gdn_step_norm_silu (Qwen3.6)
    {
        const int h_k = 16;
        const int h_v = 32;
        const int S = 128;
        const float eps = 1e-5f;

        std::vector<float> h_state(h_v * S * S);
        std::vector<float> h_q(h_k * S);
        std::vector<float> h_k_vec(h_k * S);
        std::vector<float> h_v_vec(h_v * S);
        std::vector<float> h_gate(h_v);
        std::vector<float> h_beta(h_v);
        std::vector<float> h_z(h_v * S);
        std::vector<float> h_gamma(S);
        std::vector<float> ref_y(h_v * S);
        std::vector<float> ref_state(h_v * S * S);

        for (auto& v : h_state)
            v = dist(rng) * 0.05f;
        for (auto& v : h_q)
            v = dist(rng);
        for (auto& v : h_k_vec)
            v = dist(rng);
        for (auto& v : h_v_vec)
            v = dist(rng);
        for (auto& v : h_gate)
            v = -0.1f * std::fabs(dist(rng));
        for (auto& v : h_beta)
            v = sigmoid_f(dist(rng));
        for (auto& v : h_z)
            v = dist(rng);
        for (auto& v : h_gamma)
            v = 1.0f + 0.1f * dist(rng);

        ref_state = h_state;

        // CPU reference for fused_gdn_step_norm_silu
        for (int head = 0; head < h_v; ++head) {
            const int qh = head % h_k;
            const float g = std::exp(h_gate[head]);
            const float b = h_beta[head];

            std::vector<float> kv_col(S, 0.0f);
            for (int col = 0; col < S; ++col) {
                float kv = 0.0f;
                for (int r = 0; r < S; ++r) {
                    float s_val = ref_state[(r * h_v + head) * S + col];
                    kv += s_val * h_k_vec[qh * S + r];
                }
                kv_col[col] = kv;
            }

            std::vector<float> oc(S, 0.0f);
            for (int col = 0; col < S; ++col) {
                float delta = (h_v_vec[head * S + col] - g * kv_col[col]) * b;
                float o = 0.0f;
                for (int r = 0; r < S; ++r) {
                    float s_val = ref_state[(r * h_v + head) * S + col];
                    float s_new = g * s_val + h_k_vec[qh * S + r] * delta;
                    o += s_new * h_q[qh * S + r];
                    ref_state[(r * h_v + head) * S + col] = s_new;
                }
                oc[col] = o / std::sqrt((float)S);
            }

            float sum_sq = 0.0f;
            for (int col = 0; col < S; ++col) {
                sum_sq += oc[col] * oc[col];
            }
            float scale = 1.0f / std::sqrt(sum_sq / (float)S + eps);
            for (int col = 0; col < S; ++col) {
                float zz = h_z[head * S + col];
                float silu_z = zz / (1.0f + std::exp(-zz));
                ref_y[head * S + col] = oc[col] * scale * h_gamma[col] * silu_z;
            }
        }

        float* d_state = sycl::malloc_device<float>(h_state.size(), q);
        float* d_q = sycl::malloc_device<float>(h_q.size(), q);
        float* d_k = sycl::malloc_device<float>(h_k_vec.size(), q);
        float* d_v = sycl::malloc_device<float>(h_v_vec.size(), q);
        float* d_gate = sycl::malloc_device<float>(h_gate.size(), q);
        float* d_beta = sycl::malloc_device<float>(h_beta.size(), q);
        float* d_z = sycl::malloc_device<float>(h_z.size(), q);
        float* d_gamma = sycl::malloc_device<float>(h_gamma.size(), q);
        float* d_y = sycl::malloc_device<float>(ref_y.size(), q);

        q.memcpy(d_state, h_state.data(), h_state.size() * sizeof(float));
        q.memcpy(d_q, h_q.data(), h_q.size() * sizeof(float));
        q.memcpy(d_k, h_k_vec.data(), h_k_vec.size() * sizeof(float));
        q.memcpy(d_v, h_v_vec.data(), h_v_vec.size() * sizeof(float));
        q.memcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float));
        q.memcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float));
        q.memcpy(d_z, h_z.data(), h_z.size() * sizeof(float));
        q.memcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float));
        q.wait();

        fused_gdn_step_norm_silu(d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, eps, d_y, h_k, h_v, &q);

        std::vector<float> dev_y(ref_y.size()), dev_state(h_state.size());
        q.memcpy(dev_y.data(), d_y, ref_y.size() * sizeof(float));
        q.memcpy(dev_state.data(), d_state, h_state.size() * sizeof(float));
        q.wait();

        int bad_step = 0;
        for (size_t i = 0; i < ref_y.size(); ++i) {
            float diff = std::fabs(dev_y[i] - ref_y[i]);
            float denom = std::max(std::fabs(ref_y[i]), 1e-4f);
            if (diff / denom > 2e-3f && diff > 1e-3f) bad_step++;
        }
        for (size_t i = 0; i < ref_state.size(); ++i) {
            float diff = std::fabs(dev_state[i] - ref_state[i]);
            float denom = std::max(std::fabs(ref_state[i]), 1e-4f);
            if (diff / denom > 2e-3f && diff > 1e-3f) bad_step++;
        }
        printf("  fused_gdn_step_norm_silu: bad = %d / %zu\n", bad_step, ref_y.size());
        total_bad += bad_step;

        sycl::free(d_state, q);
        sycl::free(d_q, q);
        sycl::free(d_k, q);
        sycl::free(d_v, q);
        sycl::free(d_gate, q);
        sycl::free(d_beta, q);
        sycl::free(d_z, q);
        sycl::free(d_gamma, q);
        sycl::free(d_y, q);
    }

    if (total_bad == 0) {
        printf("fused_gdn SYCL parity OK (0 failures)\n");
        return 0;
    } else {
        printf("fused_gdn SYCL parity FAILED (%d bad)\n", total_bad);
        return 1;
    }
}
