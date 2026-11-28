// Parity test for the SYCL implementation of native PLE postops.
#include <sycl/sycl.hpp>
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;

void ref_rms_norm(const float* in, const float* gamma, float* out, int n_cols, int n_rows, float eps) {
    for (int r = 0; r < n_rows; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < n_cols; ++c) {
            float v = in[r * n_cols + c];
            sum_sq += v * v;
        }
        float scale = 1.0f / std::sqrt(sum_sq / float(n_cols) + eps);
        for (int c = 0; c < n_cols; ++c) {
            out[r * n_cols + c] = scale * in[r * n_cols + c] * gamma[r * n_cols + c];
        }
    }
}

int test_native_ple_postops(sycl::queue& q) {
    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> proj_key(D), hidden(D), value(N), history(D * HISTORY);
    std::vector<float> norm_key(D), norm_query(D), norm_conv(D);
    std::vector<uint16_t> conv1d_f16(D * 4);

    for (int i = 0; i < D; ++i) {
        proj_key[i] = dist(rng) * 0.1f;
        hidden[i] = dist(rng) * 0.1f;
        norm_key[i] = 1.0f + 0.01f * dist(rng);
        norm_query[i] = 1.0f + 0.01f * dist(rng);
        norm_conv[i] = 1.0f + 0.01f * dist(rng);
        for (int k = 0; k < 4; ++k) {
            sycl::half h = sycl::vec<float, 1>(dist(rng) * 0.2f).convert<sycl::half, sycl::rounding_mode::rte>()[0];
            conv1d_f16[i * 4 + k] = sycl::bit_cast<uint16_t>(h);
        }
    }
    for (int i = 0; i < N; ++i)
        value[i] = dist(rng) * 0.1f;
    for (size_t i = 0; i < history.size(); ++i)
        history[i] = dist(rng) * 0.05f;

    // Host reference calculation
    std::vector<float> ref_key(D), ref_query(D), ref_gate(H), ref_gated(D);
    std::vector<float> ref_norm(D), ref_conv(D), ref_result(D);

    ref_rms_norm(proj_key.data(), norm_key.data(), ref_key.data(), N, H, strata::kernels::NG_RMS_EPS);
    ref_rms_norm(hidden.data(), norm_query.data(), ref_query.data(), N, H, strata::kernels::NG_RMS_EPS);

    const float scale = 1.0f / std::sqrt(float(N));
    for (int h = 0; h < H; ++h) {
        float sum = 0.0f;
        for (int d = 0; d < N; ++d) {
            sum += ref_key[h * N + d] * ref_query[h * N + d];
        }
        float s = scale * sum;
        float mag = std::sqrt(std::max(std::abs(s), 1e-6f));
        float sign = float((s > 0.0f) - (s < 0.0f));
        ref_gate[h] = 1.0f / (1.0f + std::exp(-sign * mag));
    }

    for (int i = 0; i < D; ++i) {
        ref_gated[i] = value[i % N] * ref_gate[i / N];
    }

    ref_rms_norm(ref_gated.data(), norm_conv.data(), ref_norm.data(), N, H, strata::kernels::NG_RMS_EPS);

    for (int c = 0; c < D; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            float x = (k == 3 ? ref_norm[c] : history[c * HISTORY + 3 * k]);
            float w = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half>(conv1d_f16[c * 4 + k]))
                          .convert<float, sycl::rounding_mode::automatic>()[0];
            sum += x * w;
        }
        float act = sum / (1.0f + std::exp(-sum));
        ref_conv[c] = act;
        ref_result[c] = hidden[c] + ref_gated[c] + act;
    }

    // Allocate GPU buffers
    float* d_proj_key = sycl::malloc_device<float>(D, q);
    float* d_hidden = sycl::malloc_device<float>(D, q);
    float* d_value = sycl::malloc_device<float>(N, q);
    float* d_history = sycl::malloc_device<float>(history.size(), q);
    float* d_norm_key = sycl::malloc_device<float>(D, q);
    float* d_norm_query = sycl::malloc_device<float>(D, q);
    float* d_norm_conv = sycl::malloc_device<float>(D, q);
    uint16_t* d_conv1d = sycl::malloc_device<uint16_t>(conv1d_f16.size(), q);

    float* d_b_key = sycl::malloc_device<float>(D, q);
    float* d_b_query = sycl::malloc_device<float>(D, q);
    float* d_b_gate = sycl::malloc_device<float>(H, q);
    float* d_b_gated = sycl::malloc_device<float>(D, q);
    float* d_b_norm = sycl::malloc_device<float>(D, q);
    float* d_b_conv = sycl::malloc_device<float>(D, q);
    float* d_b_result = sycl::malloc_device<float>(D, q);

    q.memcpy(d_proj_key, proj_key.data(), D * sizeof(float)).wait();
    q.memcpy(d_hidden, hidden.data(), D * sizeof(float)).wait();
    q.memcpy(d_value, value.data(), N * sizeof(float)).wait();
    q.memcpy(d_history, history.data(), history.size() * sizeof(float)).wait();
    q.memcpy(d_norm_key, norm_key.data(), D * sizeof(float)).wait();
    q.memcpy(d_norm_query, norm_query.data(), D * sizeof(float)).wait();
    q.memcpy(d_norm_conv, norm_conv.data(), D * sizeof(float)).wait();
    q.memcpy(d_conv1d, conv1d_f16.data(), conv1d_f16.size() * sizeof(uint16_t)).wait();

    strata::kernels::PleWeights w{};
    w.norm_key = d_norm_key;
    w.norm_query = d_norm_query;
    w.norm_conv = d_norm_conv;
    w.conv1d_f16 = d_conv1d;

    strata::kernels::NativePlePostopsBuffers b{d_b_key, d_b_query, d_b_gate, d_b_gated, d_b_norm, d_b_conv, d_b_result};

    strata::kernels::native_ple_postops(d_proj_key, d_hidden, d_value, d_history, w, b, &q);
    q.wait();

    std::vector<float> act_result(D);
    q.memcpy(act_result.data(), d_b_result, D * sizeof(float)).wait();

    sycl::free(d_proj_key, q);
    sycl::free(d_hidden, q);
    sycl::free(d_value, q);
    sycl::free(d_history, q);
    sycl::free(d_norm_key, q);
    sycl::free(d_norm_query, q);
    sycl::free(d_norm_conv, q);
    sycl::free(d_conv1d, q);
    sycl::free(d_b_key, q);
    sycl::free(d_b_query, q);
    sycl::free(d_b_gate, q);
    sycl::free(d_b_gated, q);
    sycl::free(d_b_norm, q);
    sycl::free(d_b_conv, q);
    sycl::free(d_b_result, q);

    double max_rel = 0.0;
    int bad = 0;
    for (int i = 0; i < D; ++i) {
        double diff = std::abs(double(act_result[i]) - double(ref_result[i]));
        double denom = std::max(std::abs(double(ref_result[i])), 1e-6);
        double rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3) ++bad;
    }

    std::printf("  native_ple_postops: max_rel = %.3e, bad = %d / %d\n", max_rel, bad, D);
    return bad;
}

int test_native_ple_postops_batch(sycl::queue& q, int T) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> key((size_t)T * D), hidden((size_t)T * D), value((size_t)T * N);
    std::vector<float> history((size_t)D * HISTORY);
    std::vector<float> norm_key(D), norm_query(D), norm_conv(D);
    std::vector<uint16_t> conv1d_f16(D * 4);

    for (int i = 0; i < D; ++i) {
        norm_key[i] = 1.0f + 0.01f * dist(rng);
        norm_query[i] = 1.0f + 0.01f * dist(rng);
        norm_conv[i] = 1.0f + 0.01f * dist(rng);
        for (int k = 0; k < 4; ++k) {
            sycl::half h = sycl::vec<float, 1>(dist(rng) * 0.2f).convert<sycl::half, sycl::rounding_mode::rte>()[0];
            conv1d_f16[i * 4 + k] = sycl::bit_cast<uint16_t>(h);
        }
    }
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = dist(rng) * 0.1f;
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = dist(rng) * 0.1f;
    for (size_t i = 0; i < value.size(); ++i)
        value[i] = dist(rng) * 0.1f;
    for (size_t i = 0; i < history.size(); ++i)
        history[i] = dist(rng) * 0.05f;

    float* d_key = sycl::malloc_device<float>(key.size(), q);
    float* d_hidden = sycl::malloc_device<float>(hidden.size(), q);
    float* d_value = sycl::malloc_device<float>(value.size(), q);
    float* d_history = sycl::malloc_device<float>(history.size(), q);
    float* d_query_norm = sycl::malloc_device<float>((size_t)T * D, q);
    float* d_gated = sycl::malloc_device<float>((size_t)T * D, q);
    float* d_gate = sycl::malloc_device<float>((size_t)T * H, q);

    float* d_norm_key = sycl::malloc_device<float>(D, q);
    float* d_norm_query = sycl::malloc_device<float>(D, q);
    float* d_norm_conv = sycl::malloc_device<float>(D, q);
    uint16_t* d_conv1d = sycl::malloc_device<uint16_t>(conv1d_f16.size(), q);

    q.memcpy(d_key, key.data(), key.size() * sizeof(float)).wait();
    q.memcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float)).wait();
    q.memcpy(d_value, value.data(), value.size() * sizeof(float)).wait();
    q.memcpy(d_history, history.data(), history.size() * sizeof(float)).wait();
    q.memcpy(d_norm_key, norm_key.data(), D * sizeof(float)).wait();
    q.memcpy(d_norm_query, norm_query.data(), D * sizeof(float)).wait();
    q.memcpy(d_norm_conv, norm_conv.data(), D * sizeof(float)).wait();
    q.memcpy(d_conv1d, conv1d_f16.data(), conv1d_f16.size() * sizeof(uint16_t)).wait();

    strata::kernels::PleWeights w{};
    w.norm_key = d_norm_key;
    w.norm_query = d_norm_query;
    w.norm_conv = d_norm_conv;
    w.conv1d_f16 = d_conv1d;

    strata::kernels::native_ple_postops_batch(d_key, d_hidden, d_value, d_history, w, d_query_norm, d_gated, d_gate, T,
                                              &q);
    q.wait();

    std::vector<float> act_hidden(hidden.size());
    q.memcpy(act_hidden.data(), d_hidden, hidden.size() * sizeof(float)).wait();

    // Verify non-zero and finite
    int bad = 0;
    for (size_t i = 0; i < act_hidden.size(); ++i) {
        if (!std::isfinite(act_hidden[i]) || act_hidden[i] == 0.0f) ++bad;
    }

    sycl::free(d_key, q);
    sycl::free(d_hidden, q);
    sycl::free(d_value, q);
    sycl::free(d_history, q);
    sycl::free(d_query_norm, q);
    sycl::free(d_gated, q);
    sycl::free(d_gate, q);
    sycl::free(d_norm_key, q);
    sycl::free(d_norm_query, q);
    sycl::free(d_norm_conv, q);
    sycl::free(d_conv1d, q);

    std::printf("  native_ple_postops_batch (T=%d): bad = %d / %zu\n", T, bad, act_hidden.size());
    return bad;
}
}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    int bad = 0;
    bad += test_native_ple_postops(q);
    bad += test_native_ple_postops_batch(q, 1);
    bad += test_native_ple_postops_batch(q, 4);
    bad += test_native_ple_postops_batch(q, 16);

    if (bad != 0) {
        std::fprintf(stderr, "native_ple_postops test FAILED with %d bad elements\n", bad);
        return 1;
    }
    std::puts("native_ple_postops SYCL parity OK (0 failures)");
    return 0;
}
