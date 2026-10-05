#include <sycl/sycl.hpp>
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    printf("Running native_gdn_preprocess SYCL test on device: %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int total_bad = 0;

    // 1. Test native_gdn_conv_silu
    {
        const int64_t channels = 2560;
        const int64_t d_conv = 4;
        std::vector<float> h_hist(channels * 3);
        std::vector<float> h_inp(channels);
        std::vector<float> h_w(channels * 4);
        std::vector<float> ref_raw(channels);
        std::vector<float> ref_silu(channels);
        std::vector<float> ref_hist(channels * 3);

        for (auto& v : h_hist)
            v = dist(rng);
        for (auto& v : h_inp)
            v = dist(rng);
        for (auto& v : h_w)
            v = dist(rng) * 0.1f;

        // CPU reference
        for (int c = 0; c < channels; ++c) {
            float vals[4] = {h_hist[c * 3], h_hist[c * 3 + 1], h_hist[c * 3 + 2], h_inp[c]};
            float sum = 0.0f;
            for (int tap = 0; tap < 4; ++tap)
                sum += vals[tap] * h_w[c * 4 + tap];
            ref_raw[c] = sum;
            ref_silu[c] = sum / (1.0f + std::exp(-sum));
            for (int tap = 0; tap < 3; ++tap)
                ref_hist[c * 3 + tap] = vals[tap + 1];
        }

        float* d_hist = sycl::malloc_device<float>(h_hist.size(), q);
        float* d_inp = sycl::malloc_device<float>(h_inp.size(), q);
        float* d_w = sycl::malloc_device<float>(h_w.size(), q);
        float* d_raw = sycl::malloc_device<float>(channels, q);
        float* d_silu = sycl::malloc_device<float>(channels, q);

        q.memcpy(d_hist, h_hist.data(), h_hist.size() * sizeof(float));
        q.memcpy(d_inp, h_inp.data(), h_inp.size() * sizeof(float));
        q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(float));
        q.wait();

        strata::kernels::native_gdn_conv_silu(d_hist, d_inp, d_w, d_raw, d_silu, channels, d_conv, &q);

        std::vector<float> dev_raw(channels), dev_silu(channels), dev_hist(h_hist.size());
        q.memcpy(dev_raw.data(), d_raw, channels * sizeof(float));
        q.memcpy(dev_silu.data(), d_silu, channels * sizeof(float));
        q.memcpy(dev_hist.data(), d_hist, h_hist.size() * sizeof(float));
        q.wait();

        int bad_conv = 0;
        for (int c = 0; c < channels; ++c) {
            if (std::fabs(dev_raw[c] - ref_raw[c]) > 1e-4f) bad_conv++;
            if (std::fabs(dev_silu[c] - ref_silu[c]) > 1e-4f) bad_conv++;
            for (int tap = 0; tap < 3; ++tap) {
                if (std::fabs(dev_hist[c * 3 + tap] - ref_hist[c * 3 + tap]) > 1e-6f) bad_conv++;
            }
        }
        printf("  conv_silu: bad = %d / %ld\n", bad_conv, (long)channels);
        total_bad += bad_conv;

        sycl::free(d_hist, q);
        sycl::free(d_inp, q);
        sycl::free(d_w, q);
        sycl::free(d_raw, q);
        sycl::free(d_silu, q);
    }

    // 2. Test native_gdn_l2_norm
    {
        const int64_t rows = 16;
        const int64_t cols = 128;
        const float eps = 1e-5f;
        std::vector<float> h_inp(rows * cols);
        std::vector<float> ref_inp(rows * cols);

        for (auto& v : h_inp)
            v = dist(rng);
        ref_inp = h_inp;

        // CPU reference
        for (int r = 0; r < rows; ++r) {
            float sumsq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float v = ref_inp[r * cols + c];
                sumsq += v * v;
            }
            float scale = 1.0f / std::sqrt(sumsq / cols + eps / cols);
            float scale_after = 1.0f / std::sqrt(128.0f);
            for (int c = 0; c < cols; ++c) {
                ref_inp[r * cols + c] = (scale * ref_inp[r * cols + c]) * scale_after;
            }
        }

        float* d_inp = sycl::malloc_device<float>(h_inp.size(), q);
        q.memcpy(d_inp, h_inp.data(), h_inp.size() * sizeof(float)).wait();

        strata::kernels::native_gdn_l2_norm(d_inp, rows, cols, eps, &q);

        std::vector<float> dev_inp(h_inp.size());
        q.memcpy(dev_inp.data(), d_inp, dev_inp.size() * sizeof(float)).wait();

        int bad_l2 = 0;
        for (size_t i = 0; i < dev_inp.size(); ++i) {
            float diff = std::fabs(dev_inp[i] - ref_inp[i]);
            float denom = std::max(std::fabs(ref_inp[i]), 1e-6f);
            if (diff / denom > 1e-3f && diff > 1e-4f) bad_l2++;
        }
        printf("  l2_norm: bad = %d / %zu\n", bad_l2, dev_inp.size());
        total_bad += bad_l2;

        sycl::free(d_inp, q);
    }

    // 3. Test native_gdn_beta_gate
    {
        const int64_t heads = 48;
        std::vector<float> h_beta(heads);
        std::vector<float> ref_beta(heads);
        for (auto& v : h_beta)
            v = dist(rng);
        for (int i = 0; i < heads; ++i)
            ref_beta[i] = sigmoid_f(h_beta[i]);

        float* d_beta = sycl::malloc_device<float>(heads, q);
        q.memcpy(d_beta, h_beta.data(), heads * sizeof(float)).wait();

        strata::kernels::native_gdn_beta_gate(d_beta, heads, &q);

        std::vector<float> dev_beta(heads);
        q.memcpy(dev_beta.data(), d_beta, heads * sizeof(float)).wait();

        int bad_beta = 0;
        for (int i = 0; i < heads; ++i) {
            if (std::fabs(dev_beta[i] - ref_beta[i]) > 1e-5f) bad_beta++;
        }
        printf("  beta_gate: bad = %d / %ld\n", bad_beta, (long)heads);
        total_bad += bad_beta;

        sycl::free(d_beta, q);
    }

    // 4. Test native_gdn_gate
    {
        const int64_t heads = 48;
        std::vector<float> h_alpha(heads), h_dt(heads), h_ssm_a(heads), ref_gate(heads);
        for (auto& v : h_alpha)
            v = dist(rng);
        for (auto& v : h_dt)
            v = dist(rng);
        for (auto& v : h_ssm_a)
            v = -std::exp(dist(rng));

        for (int i = 0; i < heads; ++i) {
            float val = h_alpha[i] + h_dt[i];
            float softplus = val > 20.0f ? val : std::log1p(std::exp(val));
            ref_gate[i] = softplus * h_ssm_a[i];
        }

        float* d_alpha = sycl::malloc_device<float>(heads, q);
        float* d_dt = sycl::malloc_device<float>(heads, q);
        float* d_ssm_a = sycl::malloc_device<float>(heads, q);
        float* d_gate = sycl::malloc_device<float>(heads, q);

        q.memcpy(d_alpha, h_alpha.data(), heads * sizeof(float));
        q.memcpy(d_dt, h_dt.data(), heads * sizeof(float));
        q.memcpy(d_ssm_a, h_ssm_a.data(), heads * sizeof(float));
        q.wait();

        strata::kernels::native_gdn_gate(d_alpha, d_dt, d_ssm_a, d_gate, heads, &q);

        std::vector<float> dev_gate(heads);
        q.memcpy(dev_gate.data(), d_gate, heads * sizeof(float)).wait();

        int bad_gate = 0;
        for (int i = 0; i < heads; ++i) {
            float diff = std::fabs(dev_gate[i] - ref_gate[i]);
            float denom = std::max(std::fabs(ref_gate[i]), 1e-6f);
            if (diff / denom > 1e-3f && diff > 1e-4f) bad_gate++;
        }
        printf("  gate: bad = %d / %ld\n", bad_gate, (long)heads);
        total_bad += bad_gate;

        sycl::free(d_alpha, q);
        sycl::free(d_dt, q);
        sycl::free(d_ssm_a, q);
        sycl::free(d_gate, q);
    }

    // 5. Test native_gdn_out_norm
    {
        const int64_t heads = 48;
        const int64_t cols = 128;
        const float eps = 1e-5f;
        std::vector<float> h_out(heads * cols), h_z(heads * cols), h_gamma(cols), ref_dst(heads * cols);
        for (auto& v : h_out)
            v = dist(rng);
        for (auto& v : h_z)
            v = dist(rng);
        for (auto& v : h_gamma)
            v = dist(rng) * 0.5f + 1.0f;

        for (int h = 0; h < heads; ++h) {
            float sumsq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float v = h_out[h * cols + c];
                sumsq += v * v;
            }
            float scale = 1.0f / std::sqrt(sumsq / cols + eps);
            for (int c = 0; c < cols; ++c) {
                float weighted = scale * h_out[h * cols + c] * h_gamma[c];
                ref_dst[h * cols + c] = weighted * sigmoid_f(h_z[h * cols + c]);
            }
        }

        float* d_out = sycl::malloc_device<float>(heads * cols, q);
        float* d_z = sycl::malloc_device<float>(heads * cols, q);
        float* d_gamma = sycl::malloc_device<float>(cols, q);
        float* d_dst = sycl::malloc_device<float>(heads * cols, q);

        q.memcpy(d_out, h_out.data(), heads * cols * sizeof(float));
        q.memcpy(d_z, h_z.data(), heads * cols * sizeof(float));
        q.memcpy(d_gamma, h_gamma.data(), cols * sizeof(float));
        q.wait();

        strata::kernels::native_gdn_out_norm(d_out, d_z, d_gamma, d_dst, heads, cols, eps, &q);

        std::vector<float> dev_dst(heads * cols);
        q.memcpy(dev_dst.data(), d_dst, heads * cols * sizeof(float)).wait();

        int bad_out_norm = 0;
        for (size_t i = 0; i < dev_dst.size(); ++i) {
            float diff = std::fabs(dev_dst[i] - ref_dst[i]);
            float denom = std::max(std::fabs(ref_dst[i]), 1e-6f);
            if (diff / denom > 1e-3f && diff > 1e-4f) bad_out_norm++;
        }
        printf("  out_norm: bad = %d / %zu\n", bad_out_norm, dev_dst.size());
        total_bad += bad_out_norm;

        sycl::free(d_out, q);
        sycl::free(d_z, q);
        sycl::free(d_gamma, q);
        sycl::free(d_dst, q);
    }

    if (total_bad == 0) {
        printf("native_gdn_preprocess SYCL parity OK (0 failures)\n");
        return 0;
    } else {
        printf("native_gdn_preprocess SYCL parity FAILED (%d bad)\n", total_bad);
        return 1;
    }
}
