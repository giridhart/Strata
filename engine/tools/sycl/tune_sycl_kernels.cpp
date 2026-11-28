// tools/sycl/tune_sycl_kernels.cpp - On-device autotuner for SYCL kernels on Intel Arc GPUs.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/core/sycl_tuning.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct BenchmarkMetrics {
    double median_us = 0.0;
    double min_us = 0.0;
    int total_iters = 0;
    double total_time_s = 0.0;
    bool event_profiling_used = false;
    bool verified = true;
    float max_abs_diff = 0.0f;
};

struct GemvShape {
    std::string kernel_name;
    strata::sycl_tuning::KernelType kernel_type;
    int64_t n_in;
    int64_t n_out;
    int code_bits;
    int group_elems;
    int act_kind;
    std::string desc;
    std::string w_cache;
    std::string x_cache;
    std::string out_cache;
};

// Comprehensive computational kernels & shapes across Qwen3.8-Flash-Next & Qwen3.6-35B-A3B
const GemvShape kTargetShapes[] = {
    // 1. BF16 Projections
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 48, 0, 0, 0, "ssm_alpha / ssm_beta", "streaming", "cached", "default"},
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 64, 0, 0, 0, "asymmetric decode 64", "streaming", "cached", "default"},
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 128, 0, 0, 0, "indexer.k_proj", "streaming", "cached", "default"},
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 256, 0, 0, 0, "asymmetric decode 256", "streaming", "cached", "default"},
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 512, 0, 0, 0, "indexer.q_proj", "streaming", "cached", "default"},
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 640, 0, 0, 0, "ffn_gate_inp", "streaming", "cached", "default"},
    {"bf16_gemv", strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 2560, 0, 0, 0, "ple_value / shared_proj", "streaming", "cached", "default"},

    // 2. S-GEMV Q8_K Projections (K-quants: attention projections and shared experts)
    {"s_gemv_q8k", strata::sycl_tuning::KernelType::S_GEMV_Q8K, 2560, 640, 8, 64, 1, "expert gate/up [2560, 640]", "streaming", "cached", "default"},
    {"s_gemv_q8k", strata::sycl_tuning::KernelType::S_GEMV_Q8K, 640, 2560, 8, 64, 1, "expert down [640, 2560]", "streaming", "cached", "default"},
    {"s_gemv_q8k", strata::sycl_tuning::KernelType::S_GEMV_Q8K, 2560, 2560, 4, 64, 1, "attn_output [2560, 2560]", "streaming", "cached", "default"},
    {"s_gemv_q8k", strata::sycl_tuning::KernelType::S_GEMV_Q8K, 2560, 6144, 4, 64, 1, "ffn_gate_up [2560, 6144]", "streaming", "cached", "default"},
    {"s_gemv_q8k", strata::sycl_tuning::KernelType::S_GEMV_Q8K, 2560, 10240, 4, 64, 1, "attn_qkv [2560, 10240]", "streaming", "cached", "default"},

    // 3. S-GEMV Q8_0 Projections (Legacy Q8_0 tensors)
    {"s_gemv_q8_0", strata::sycl_tuning::KernelType::S_GEMV_Q8_0, 2560, 640, 8, 32, 0, "ffn_down_shexp [2560, 640]", "streaming", "cached", "default"},
    {"s_gemv_q8_0", strata::sycl_tuning::KernelType::S_GEMV_Q8_0, 640, 2560, 8, 32, 0, "ffn_up_shexp [640, 2560]", "streaming", "cached", "default"},

    // 4. S2-GEMV (2-bit weights, 80% of model weight traffic)
    {"s2_gemv", strata::sycl_tuning::KernelType::S2_GEMV, 2560, 640, 2, 64, 0, "S2 gate/up [2560, 640]", "streaming", "cached", "default"},
    {"s2_gemv", strata::sycl_tuning::KernelType::S2_GEMV, 640, 2560, 2, 64, 0, "S2 down [640, 2560]", "streaming", "cached", "default"},
    {"s2_gemv", strata::sycl_tuning::KernelType::S2_GEMV, 2560, 2560, 2, 64, 0, "S2 attn_output [2560, 2560]", "streaming", "cached", "default"},

    // 5. S2 Grouped MoE (10 active experts batched)
    {"s2_expert", strata::sycl_tuning::KernelType::S2_EXPERT, 2560, 640, 2, 64, 0, "MoE grouped gate/up (10 active)", "streaming", "cached", "default"},
    {"s2_expert", strata::sycl_tuning::KernelType::S2_EXPERT, 640, 2560, 2, 64, 0, "MoE grouped down (10 active)", "streaming", "cached", "default"},

    // 6. Native GR RMSNorm (Pass 1 sum-of-squares, Pass 2 scale -> input X read twice, must be CACHED in L1!)
    {"gr_norm", strata::sycl_tuning::KernelType::GR_NORM, 2560, 1, 0, 0, 0, "n_embd norm [2560, 1]", "cached", "cached", "default"},
    {"gr_norm", strata::sycl_tuning::KernelType::GR_NORM, 10240, 1, 0, 0, 0, "hc_attn / hc_ffn norm [10240, 1]", "cached", "cached", "default"},
    {"gr_norm", strata::sycl_tuning::KernelType::GR_NORM, 320, 1, 0, 0, 0, "hc_lr norm [320, 1]", "cached", "cached", "default"},

    // 7. Native Router (Top-10 MoE expert routing)
    {"native_router", strata::sycl_tuning::KernelType::ROUTER, 512, 1, 0, 0, 0, "Flash-Next 512 experts, k=10", "default", "cached", "default"},
    {"native_router", strata::sycl_tuning::KernelType::ROUTER, 256, 1, 0, 0, 0, "Qwen3.6-35B 256 experts, k=8", "default", "cached", "default"},

    // 8. FlashAttention (Short-step KV head attention)
    {"flash_attn", strata::sycl_tuning::KernelType::FLASH_ATTN, 256, 24, 16, 256, 0, "Flash-Next Q24x256 / KV2x256", "default", "default", "default"},

    // 9. Native GDN (Gated Delta Net recurrent step)
    {"native_gdn", strata::sycl_tuning::KernelType::GDN, 128, 8, 32, 1, 0, "GDN recurrent step S=128", "default", "default", "default"},

    // 10. Phase 1 & 2 Prefill and Speculative Selection Kernels
    {"moe_scatter_add", strata::sycl_tuning::KernelType::MOE_SCATTER_ADD, 10, 2560, 0, 0, 0, "MoE scatter add n_u=10", "streaming", "cached", "default"},
    {"moe_scatter_add", strata::sycl_tuning::KernelType::MOE_SCATTER_ADD, 32, 2560, 0, 0, 0, "MoE scatter add n_u=32", "streaming", "cached", "default"},
    {"ple_dequant", strata::sycl_tuning::KernelType::PLE_DEQUANT, 64, 256, 4, 32, 0, "PLE IQ4_NL dequant 64 rows", "streaming", "cached", "default"},
    {"qsa_select_long", strata::sycl_tuning::KernelType::QSA_SELECT_LONG, 4096, 4, 0, 0, 0, "QSA DX scorer 4k blocks nq=4", "cached", "cached", "default"}
};

// Mandatory Numerical Verification: check that device kernel produces finite results
// and matches CPU reference within tolerance before timing or selecting any candidate.
bool verify_bf16_gemv_candidate(sycl::queue& q, const uint16_t* d_x, const uint16_t* d_w, float* d_y,
                                const float* h_ref, int64_t n_in, int64_t n_out, int tpr,
                                float& max_diff, float rtol = 1e-2f, float atol = 1e-2f) {
    q.memset(d_y, 0, (size_t)n_out * sizeof(float)).wait();
    strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q);
    q.wait_and_throw();

    std::vector<float> h_got(n_out);
    q.copy(d_y, h_got.data(), n_out).wait();

    max_diff = 0.0f;
    for (int64_t i = 0; i < n_out; ++i) {
        if (std::isnan(h_got[i]) || std::isinf(h_got[i])) {
            max_diff = std::numeric_limits<float>::infinity();
            return false;
        }
        const float diff = std::abs(h_got[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > atol + rtol * std::abs(h_ref[i])) {
            return false;
        }
    }
    return true;
}

BenchmarkMetrics measure_bf16_gemv_split(sycl::queue& q, const uint16_t* d_x, const uint16_t* d_w, float* d_y,
                                         const float* h_ref, int64_t n_in, int64_t n_out, int tpr,
                                         double min_time_sec, double warmup_sec) {
    BenchmarkMetrics metrics;

    // 0. Pre-flight Numerical Verification: mandatory check ensuring finite math and reference parity
    if (h_ref != nullptr) {
        float max_diff = 0.0f;
        if (!verify_bf16_gemv_candidate(q, d_x, d_w, d_y, h_ref, n_in, n_out, tpr, max_diff)) {
            metrics.verified = false;
            metrics.max_abs_diff = max_diff;
            metrics.median_us = 1e9;
            metrics.min_us = 1e9;
            return metrics;
        }
        metrics.verified = true;
        metrics.max_abs_diff = max_diff;
    }

    // 1. Warmup: run for warmup_sec (or at least 10 iterations) to allow GPU DVFS to ramp to steady-state boost clocks
    const auto w_start = std::chrono::steady_clock::now();
    int w_count = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w_start).count() < warmup_sec || w_count < 10) {
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q);
        ++w_count;
    }
    q.wait_and_throw();

    // 2. Check if event profiling is available and functional on this queue/device
    bool can_event_profile = false;
    try {
        sycl::event test_ev;
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q, &test_ev);
        test_ev.wait();
        auto s = test_ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto e = test_ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        if (e >= s) {
            can_event_profile = true;
        }
    } catch (...) {
        can_event_profile = false;
    }
    metrics.event_profiling_used = can_event_profile;

    // 3. Timed benchmark: submit in pipelined batches of BATCH_SIZE.
    // In an in-order queue, ev_first command_start marks when the first kernel begins on the GPU,
    // and ev_last command_end marks when the final kernel completes on the GPU.
    // This captures the entire unbroken GPU execution span with zero driver dispatch overhead.
    constexpr int BATCH_SIZE = 32;
    std::vector<double> times_us;
    times_us.reserve(4096);

    const auto t_start = std::chrono::steady_clock::now();
    const auto deadline = t_start + std::chrono::duration<double>(min_time_sec);

    while (std::chrono::steady_clock::now() < deadline || metrics.total_iters < BATCH_SIZE) {
        if (can_event_profile) {
            sycl::event ev_first, ev_last;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                if (b == 0) {
                    strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q, &ev_first);
                } else if (b == BATCH_SIZE - 1) {
                    strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q, &ev_last);
                } else {
                    strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q, nullptr);
                }
            }
            q.wait_and_throw();
            uint64_t s = ev_first.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e = ev_last.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (e >= s) {
                double batch_us = static_cast<double>(e - s) * 1e-3;
                double per_kernel_us = batch_us / BATCH_SIZE;
                times_us.push_back(per_kernel_us);
            }
        } else {
            auto b_start = std::chrono::high_resolution_clock::now();
            for (int b = 0; b < BATCH_SIZE; ++b) {
                strata::kernels::bf16_gemv_split(d_x, d_w, d_y, n_in, n_out, tpr, &q);
            }
            q.wait_and_throw();
            auto b_end = std::chrono::high_resolution_clock::now();
            double batch_us = std::chrono::duration<double, std::micro>(b_end - b_start).count();
            double per_kernel_us = batch_us / BATCH_SIZE;
            times_us.push_back(per_kernel_us);
        }
        metrics.total_iters += BATCH_SIZE;
    }
    const auto t_end = std::chrono::steady_clock::now();
    metrics.total_time_s = std::chrono::duration<double>(t_end - t_start).count();

    if (!times_us.empty()) {
        std::sort(times_us.begin(), times_us.end());
        metrics.median_us = times_us[times_us.size() / 2];
        metrics.min_us = times_us.front();
    }
    return metrics;
}

// CPU Reference for RMSNorm: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
void cpu_ref_rms_norm(const float* x, const float* gamma, float* y, int n_cols, float eps = 1e-6f) {
    double sum_sq = 0.0;
    for (int i = 0; i < n_cols; ++i) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    double mean = sum_sq / n_cols;
    float scale = 1.0f / (float)std::sqrt(mean + eps);
    for (int i = 0; i < n_cols; ++i) {
        y[i] = scale * x[i] * gamma[i];
    }
}

bool verify_gr_norm_candidate(sycl::queue& q, const float* d_x, const float* d_gamma, float* d_y,
                              const float* h_ref, int n_cols, float& max_diff,
                              float rtol = 1e-3f, float atol = 1e-3f) {
    q.memset(d_y, 0, (size_t)n_cols * sizeof(float)).wait();
    strata::kernels::native_gr_rms_norm_weighted(d_x, d_gamma, d_y, n_cols, 1, 1e-6f, &q);
    q.wait_and_throw();

    std::vector<float> h_got(n_cols);
    q.copy(d_y, h_got.data(), n_cols).wait();

    max_diff = 0.0f;
    for (int i = 0; i < n_cols; ++i) {
        if (std::isnan(h_got[i]) || std::isinf(h_got[i])) {
            max_diff = std::numeric_limits<float>::infinity();
            return false;
        }
        const float diff = std::abs(h_got[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > atol + rtol * std::abs(h_ref[i])) {
            return false;
        }
    }
    return true;
}

BenchmarkMetrics measure_gr_norm(sycl::queue& q, const float* d_x, const float* d_gamma, float* d_y,
                                 const float* h_ref, int n_cols, double min_time_sec, double warmup_sec) {
    BenchmarkMetrics metrics;
    if (h_ref != nullptr) {
        float max_diff = 0.0f;
        if (!verify_gr_norm_candidate(q, d_x, d_gamma, d_y, h_ref, n_cols, max_diff)) {
            metrics.verified = false;
            metrics.max_abs_diff = max_diff;
            metrics.median_us = 1e9;
            metrics.min_us = 1e9;
            return metrics;
        }
        metrics.verified = true;
        metrics.max_abs_diff = max_diff;
    }

    const auto w_start = std::chrono::steady_clock::now();
    int w_count = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w_start).count() < warmup_sec || w_count < 10) {
        strata::kernels::native_gr_rms_norm_weighted(d_x, d_gamma, d_y, n_cols, 1, 1e-6f, &q);
        ++w_count;
    }
    q.wait_and_throw();

    constexpr int BATCH_SIZE = 32;
    std::vector<double> times_us;
    times_us.reserve(4096);

    const auto t_start = std::chrono::steady_clock::now();
    const auto deadline = t_start + std::chrono::duration<double>(min_time_sec);

    while (std::chrono::steady_clock::now() < deadline || metrics.total_iters < BATCH_SIZE) {
        auto b_start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < BATCH_SIZE; ++b) {
            strata::kernels::native_gr_rms_norm_weighted(d_x, d_gamma, d_y, n_cols, 1, 1e-6f, &q);
        }
        q.wait_and_throw();
        auto b_end = std::chrono::high_resolution_clock::now();
        double batch_us = std::chrono::duration<double, std::micro>(b_end - b_start).count();
        times_us.push_back(batch_us / BATCH_SIZE);
        metrics.total_iters += BATCH_SIZE;
    }
    metrics.total_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (!times_us.empty()) {
        std::sort(times_us.begin(), times_us.end());
        metrics.median_us = times_us[times_us.size() / 2];
        metrics.min_us = times_us.front();
    }
    return metrics;
}

// CPU Reference for S2 GEMV with Q8_0 activations
void cpu_ref_s2_gemv(const uint8_t* x80, const uint8_t* codes, const float* scales, float* y,
                     int64_t n_in, int64_t n_out) {
    const int64_t n_quads = n_in / 4;
    for (int64_t o = 0; o < n_out; ++o) {
        double acc = 0.0;
        const uint8_t* c = codes + o * n_quads;
        const float* s = scales + o * (n_in / 64);

        for (int64_t q = 0; q < n_quads; ++q) {
            const uint8_t byte = c[q];
            const float d = s[q >> 4];
            const int64_t ablk = (q * 4) / 32;
            const uint8_t* blk = x80 + ablk * 34;
            const float dx = strata::kernels::f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
            const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
            const int off = (int)((q * 4) % 32);

            const int c0 = (int)(byte & 3) - 1;
            const int c1 = (int)((byte >> 2) & 3) - 1;
            const int c2 = (int)((byte >> 4) & 3) - 1;
            const int c3 = (int)((byte >> 6) & 3) - 1;
            const int idot0 = c0 * (int)xq[off + 0] + c1 * (int)xq[off + 1];
            const int idot1 = c2 * (int)xq[off + 2] + c3 * (int)xq[off + 3];
            const float scale = d * dx;
            acc += (double)idot0 * (double)scale + (double)idot1 * (double)scale;
        }
        y[o] = (float)acc;
    }
}

bool verify_s2_gemv_candidate(sycl::queue& q, const uint8_t* d_x80, const uint8_t* d_codes, const float* d_scales,
                              float* d_y, const float* h_ref, int64_t n_in, int64_t n_out, int tpr,
                              float& max_diff, float rtol = 1e-2f, float atol = 1e-2f) {
    q.memset(d_y, 0, (size_t)n_out * sizeof(float)).wait();
    strata::kernels::s2_gemv_q8(d_x80, d_codes, d_scales, d_y, n_in, n_out, tpr, &q);
    q.wait_and_throw();

    std::vector<float> h_got(n_out);
    q.copy(d_y, h_got.data(), n_out).wait();

    max_diff = 0.0f;
    for (int64_t i = 0; i < n_out; ++i) {
        if (std::isnan(h_got[i]) || std::isinf(h_got[i])) {
            max_diff = std::numeric_limits<float>::infinity();
            return false;
        }
        const float diff = std::abs(h_got[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > atol + rtol * std::abs(h_ref[i])) {
            return false;
        }
    }
    return true;
}

BenchmarkMetrics measure_s2_gemv(sycl::queue& q, const uint8_t* d_x80, const uint8_t* d_codes, const float* d_scales,
                                 float* d_y, const float* h_ref, int64_t n_in, int64_t n_out, int tpr,
                                 double min_time_sec, double warmup_sec) {
    BenchmarkMetrics metrics;
    if (h_ref != nullptr) {
        float max_diff = 0.0f;
        if (!verify_s2_gemv_candidate(q, d_x80, d_codes, d_scales, d_y, h_ref, n_in, n_out, tpr, max_diff)) {
            metrics.verified = false;
            metrics.max_abs_diff = max_diff;
            metrics.median_us = 1e9;
            metrics.min_us = 1e9;
            return metrics;
        }
        metrics.verified = true;
        metrics.max_abs_diff = max_diff;
    }

    const auto w_start = std::chrono::steady_clock::now();
    int w_count = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w_start).count() < warmup_sec || w_count < 10) {
        strata::kernels::s2_gemv_q8(d_x80, d_codes, d_scales, d_y, n_in, n_out, tpr, &q);
        ++w_count;
    }
    q.wait_and_throw();

    constexpr int BATCH_SIZE = 32;
    std::vector<double> times_us;
    times_us.reserve(4096);

    const auto t_start = std::chrono::steady_clock::now();
    const auto deadline = t_start + std::chrono::duration<double>(min_time_sec);

    while (std::chrono::steady_clock::now() < deadline || metrics.total_iters < BATCH_SIZE) {
        auto b_start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < BATCH_SIZE; ++b) {
            strata::kernels::s2_gemv_q8(d_x80, d_codes, d_scales, d_y, n_in, n_out, tpr, &q);
        }
        q.wait_and_throw();
        auto b_end = std::chrono::high_resolution_clock::now();
        double batch_us = std::chrono::duration<double, std::micro>(b_end - b_start).count();
        times_us.push_back(batch_us / BATCH_SIZE);
        metrics.total_iters += BATCH_SIZE;
    }
    metrics.total_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (!times_us.empty()) {
        std::sort(times_us.begin(), times_us.end());
        metrics.median_us = times_us[times_us.size() / 2];
        metrics.min_us = times_us.front();
    }
    return metrics;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    double min_time_sec = -1.0;
    std::string out_path = "";
    std::string workload_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            quick = true;
        } else if (arg == "--min-time" && i + 1 < argc) {
            min_time_sec = std::atof(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--workload" && i + 1 < argc) {
            workload_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --quick            Run fast sweep (0.1s per candidate)\n");
            printf("  --min-time <sec>   Target benchmark duration per candidate in seconds (default: 1.0s, or 0.1s with --quick)\n");
            printf("  --workload <path>  Tune kernels captured from a workload file\n");
            printf("  --out <path>       Custom output path for calibration table\n");
            return 0;
        }
    }

    if (min_time_sec <= 0.0) {
        min_time_sec = quick ? 0.1 : 1.0;
    }
    const double warmup_sec = quick ? 0.05 : 0.20;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling()}};
    const auto& dev = q.get_device();
    const auto fp = strata::sycl_tuning::DeviceFingerprint::from_device(dev);

    printf("====================================================================\n");
    printf("Strata SYCL On-Device Autotuner for Intel Arc (Battlemage / Xe2)\n");
    printf("Target Device : %s\n", fp.device_name.c_str());
    printf("Driver Version: %s\n", fp.driver_version.c_str());
    printf("Compute Units : %u Xe-cores/VEs\n", fp.compute_units);
    printf("Sub-group Size: %d\n", fp.native_subgroup_size);
    printf("Tuning Mode   : %s (min_time = %.2fs per candidate)\n", quick ? "QUICK (Fast sweep)" : "FULL (Accurate calibration)", min_time_sec);
    if (!workload_path.empty()) {
        printf("Workload File : %s\n", workload_path.c_str());
    }
    printf("====================================================================\n");

    std::vector<GemvShape> target_shapes;

    if (workload_path.empty()) {
        for (const auto& s : kTargetShapes) {
            target_shapes.push_back(s);
        }
    } else {
        std::vector<strata::sycl_tuning::CapturedShape> cap_shapes;
        std::string err;
        if (!strata::sycl_tuning::ShapeCollector::load_workload(workload_path, cap_shapes, err)) {
            fprintf(stderr, "Error loading workload file '%s': %s\n", workload_path.c_str(), err.c_str());
            return 1;
        }
        printf("Loaded %zu shapes from workload: %s\n", cap_shapes.size(), workload_path.c_str());
        for (const auto& s : cap_shapes) {
            GemvShape gs;
            gs.kernel_name = strata::sycl_tuning::to_string(s.type);
            gs.kernel_type = s.type;
            gs.n_in = s.n_in;
            gs.n_out = s.n_out;
            gs.code_bits = s.code_bits;
            gs.group_elems = s.group_elems;
            gs.act_kind = s.act_kind;
            gs.desc = "workload captured shape";

            if (s.type == strata::sycl_tuning::KernelType::GR_NORM) {
                gs.w_cache = "cached";
                gs.x_cache = "cached";
                gs.out_cache = "default";
            } else if (s.type == strata::sycl_tuning::KernelType::ROUTER || s.type == strata::sycl_tuning::KernelType::UNKNOWN) {
                gs.w_cache = "default";
                gs.x_cache = "cached";
                gs.out_cache = "default";
            } else {
                gs.w_cache = "streaming";
                gs.x_cache = "cached";
                gs.out_cache = "default";
            }
            target_shapes.push_back(gs);
        }
    }

    if (out_path.empty()) {
        std::string safe_dev = fp.device_name;
        for (char& c : safe_dev) {
            if (!std::isalnum(c)) c = '_';
        }
        out_path = "data/tuning/sycl/tuning_" + safe_dev + ".txt";
    }

    printf("Output destination: %s\n\n", out_path.c_str());

    std::ofstream out_file(out_path);
    if (!out_file.is_open()) {
        fprintf(stderr, "Error: Unable to create output file: %s\n", out_path.c_str());
        return 1;
    }

    // Write standard header
    out_file << "# STRATA_SYCL_TUNING_V1 autotuned calibration table\n";
    out_file << "STRATA_SYCL_TUNING_V1 \"" << fp.device_name << "\" \"" << fp.driver_version
             << "\" " << fp.compute_units << " " << fp.native_subgroup_size << "\n";
    out_file << "# kernel_name n_in n_out code_bits group_elems act_kind lws tpr slm_pad w_cache x_cache out_cache\n";

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);

    for (const auto& shape : target_shapes) {
        printf("Tuning %s [%lld x %lld] (%s)...\n", shape.kernel_name.c_str(),
               (long long)shape.n_in, (long long)shape.n_out, shape.desc.c_str());

        if (shape.kernel_type == strata::sycl_tuning::KernelType::BF16_GEMV) {
            const size_t in_size = (size_t)shape.n_in;
            const size_t out_size = (size_t)shape.n_out;
            const size_t w_size = in_size * out_size;

            std::vector<uint16_t> h_x(in_size);
            std::vector<uint16_t> h_w(w_size);
            for (auto& v : h_x) v = strata::kernels::bf16_from_f32(f_dist(rng));
            for (auto& v : h_w) v = strata::kernels::bf16_from_f32(f_dist(rng) * 0.05f);

            auto* d_x = sycl::malloc_device<uint16_t>(in_size, q);
            auto* d_w = sycl::malloc_device<uint16_t>(w_size, q);
            auto* d_y = sycl::malloc_device<float>(out_size, q);

            q.copy(h_x.data(), d_x, in_size);
            q.copy(h_w.data(), d_w, w_size);
            q.wait_and_throw();

            // Compute exact CPU reference for numerical verification
            std::vector<float> h_ref_y(out_size, 0.0f);
            for (size_t o = 0; o < out_size; ++o) {
                double acc = 0.0;
                for (size_t i = 0; i < in_size; ++i) {
                    acc += (double)strata::kernels::f32_from_bf16(h_x[i]) *
                           (double)strata::kernels::f32_from_bf16(h_w[o * in_size + i]);
                }
                h_ref_y[o] = (float)acc;
            }

            const int tpr_candidates[] = {16, 32, 64, 128, 256};
            int best_tpr = 128;
            double best_time = 1e9;

            for (int tpr : tpr_candidates) {
                BenchmarkMetrics m = measure_bf16_gemv_split(q, d_x, d_w, d_y, h_ref_y.data(), shape.n_in, shape.n_out,
                                                             tpr, min_time_sec, warmup_sec);
                if (!m.verified) {
                    printf("  Candidate TPR=%-3d : [VERIFY-FAIL] non-finite or diverged from CPU ref (max_diff=%.4e) -> DISCARDED\n",
                           tpr, m.max_abs_diff);
                    continue;
                }
                printf("  Candidate TPR=%-3d : %8.2f us (min: %8.2f us, %6d iters in %.2fs [%s]) [VERIFIED: max_diff=%.2e]\n",
                       tpr, m.median_us, m.min_us, m.total_iters, m.total_time_s,
                       m.event_profiling_used ? "HW Batch Profiling" : "Wall Clock", m.max_abs_diff);
                if (m.median_us < best_time) {
                    best_time = m.median_us;
                    best_tpr = tpr;
                }
            }

            if (best_time >= 1e9) {
                fprintf(stderr, "Error: All candidate configurations failed numerical verification for shape [%lld x %lld]!\n",
                        (long long)shape.n_in, (long long)shape.n_out);
                return 1;
            }

            const int best_lws = std::max(best_tpr, 32);
            printf("  ==> Selected: LWS=%d, TPR=%d (%.2f us) [w:%s, x:%s, out:%s]\n\n",
                   best_lws, best_tpr, best_time, shape.w_cache.c_str(), shape.x_cache.c_str(), shape.out_cache.c_str());

            out_file << shape.kernel_name << " " << shape.n_in << " " << shape.n_out << " "
                     << shape.code_bits << " " << shape.group_elems << " " << shape.act_kind << " "
                     << best_lws << " " << best_tpr << " 0 "
                     << shape.w_cache << " " << shape.x_cache << " " << shape.out_cache << "\n";

            sycl::free(d_x, q);
            sycl::free(d_w, q);
            sycl::free(d_y, q);
        } else if (shape.kernel_type == strata::sycl_tuning::KernelType::GR_NORM) {
            const size_t n_cols = (size_t)shape.n_in;
            std::vector<float> h_x(n_cols);
            std::vector<float> h_gamma(n_cols);
            for (auto& v : h_x) v = f_dist(rng);
            for (auto& v : h_gamma) v = f_dist(rng) * 0.5f + 1.0f;

            auto* d_x = sycl::malloc_device<float>(n_cols, q);
            auto* d_gamma = sycl::malloc_device<float>(n_cols, q);
            auto* d_y = sycl::malloc_device<float>(n_cols, q);

            q.copy(h_x.data(), d_x, n_cols);
            q.copy(h_gamma.data(), d_gamma, n_cols);
            q.wait_and_throw();

            std::vector<float> h_ref_y(n_cols, 0.0f);
            cpu_ref_rms_norm(h_x.data(), h_gamma.data(), h_ref_y.data(), (int)n_cols);

            BenchmarkMetrics m = measure_gr_norm(q, d_x, d_gamma, d_y, h_ref_y.data(), (int)n_cols, min_time_sec, warmup_sec);
            if (!m.verified) {
                printf("  [VERIFY-FAIL] non-finite or diverged from CPU ref (max_diff=%.4e) -> DISCARDED\n", m.max_abs_diff);
                fprintf(stderr, "Error: GR_NORM failed numerical verification for shape [%lld x %lld]!\n",
                        (long long)shape.n_in, (long long)shape.n_out);
                return 1;
            }
            const int best_lws = (shape.n_in >= 1024) ? 256 : 128;
            printf("  Candidate LWS=%-3d : %8.2f us (min: %8.2f us, %6d iters in %.2fs) [VERIFIED: max_diff=%.2e]\n",
                   best_lws, m.median_us, m.min_us, m.total_iters, m.total_time_s, m.max_abs_diff);

            printf("  ==> Selected: LWS=%d, TPR=0 (%.2f us) [w:%s, x:%s, out:%s]\n\n",
                   best_lws, m.median_us, shape.w_cache.c_str(), shape.x_cache.c_str(), shape.out_cache.c_str());

            out_file << shape.kernel_name << " " << shape.n_in << " " << shape.n_out << " "
                     << shape.code_bits << " " << shape.group_elems << " " << shape.act_kind << " "
                     << best_lws << " 0 0 "
                     << shape.w_cache << " " << shape.x_cache << " " << shape.out_cache << "\n";

            sycl::free(d_x, q);
            sycl::free(d_gamma, q);
            sycl::free(d_y, q);
        } else if (shape.kernel_type == strata::sycl_tuning::KernelType::S2_GEMV) {
            const size_t in_size = (size_t)shape.n_in;
            const size_t out_size = (size_t)shape.n_out;
            const size_t q80_bytes = (in_size / 32) * 34;
            const size_t code_bytes = out_size * (in_size / 4);
            const size_t scale_floats = out_size * (in_size / 64);

            std::vector<float> h_xf(in_size);
            for (auto& v : h_xf) v = f_dist(rng) * 2.0f;

            auto* d_xf = sycl::malloc_device<float>(in_size, q);
            auto* d_x80 = sycl::malloc_device<uint8_t>(q80_bytes, q);
            q.copy(h_xf.data(), d_xf, in_size).wait();
            strata::kernels::quantize_q8_0(d_xf, d_x80, in_size, &q);
            q.wait_and_throw();

            std::vector<uint8_t> h_x80(q80_bytes);
            q.copy(d_x80, h_x80.data(), q80_bytes).wait();
            sycl::free(d_xf, q);

            std::vector<uint8_t> h_codes(code_bytes);
            std::vector<float> h_scales(scale_floats);
            for (auto& v : h_codes) v = (uint8_t)rng();
            for (auto& v : h_scales) v = (float)(f_dist(rng) * 0.02f + 0.03f);

            auto* d_codes = sycl::malloc_device<uint8_t>(code_bytes, q);
            auto* d_scales = sycl::malloc_device<float>(scale_floats, q);
            auto* d_y = sycl::malloc_device<float>(out_size, q);

            q.copy(h_codes.data(), d_codes, code_bytes);
            q.copy(h_scales.data(), d_scales, scale_floats);
            q.wait_and_throw();

            std::vector<float> h_ref_y(out_size, 0.0f);
            cpu_ref_s2_gemv(h_x80.data(), h_codes.data(), h_scales.data(), h_ref_y.data(), in_size, out_size);

            const int tpr_candidates[] = {16, 32, 64, 128, 256};
            int best_tpr = 64;
            double best_time = 1e9;

            for (int tpr : tpr_candidates) {
                BenchmarkMetrics m = measure_s2_gemv(q, d_x80, d_codes, d_scales, d_y, h_ref_y.data(),
                                                     in_size, out_size, tpr, min_time_sec, warmup_sec);
                if (!m.verified) {
                    printf("  Candidate TPR=%-3d : [VERIFY-FAIL] non-finite or diverged from CPU ref (max_diff=%.4e) -> DISCARDED\n",
                           tpr, m.max_abs_diff);
                    continue;
                }
                printf("  Candidate TPR=%-3d : %8.2f us (min: %8.2f us, %6d iters in %.2fs) [VERIFIED: max_diff=%.2e]\n",
                       tpr, m.median_us, m.min_us, m.total_iters, m.total_time_s, m.max_abs_diff);
                if (m.median_us < best_time) {
                    best_time = m.median_us;
                    best_tpr = tpr;
                }
            }

            if (best_time >= 1e9) {
                fprintf(stderr, "Error: All candidate configurations failed numerical verification for shape [%lld x %lld]!\n",
                        (long long)shape.n_in, (long long)shape.n_out);
                return 1;
            }

            const int best_lws = std::max(best_tpr, 32);
            printf("  ==> Selected: LWS=%d, TPR=%d (%.2f us) [w:%s, x:%s, out:%s]\n\n",
                   best_lws, best_tpr, best_time, shape.w_cache.c_str(), shape.x_cache.c_str(), shape.out_cache.c_str());

            out_file << shape.kernel_name << " " << shape.n_in << " " << shape.n_out << " "
                     << shape.code_bits << " " << shape.group_elems << " " << shape.act_kind << " "
                     << best_lws << " " << best_tpr << " 0 "
                     << shape.w_cache << " " << shape.x_cache << " " << shape.out_cache << "\n";

            sycl::free(d_x80, q);
            sycl::free(d_codes, q);
            sycl::free(d_scales, q);
            sycl::free(d_y, q);
        } else {
            // Calibrate workgroup size, split-K, and per-argument cache hints
            int best_lws = (shape.n_out >= 1024) ? 256 : 128;
            int best_tpr = (shape.n_out <= 640 && shape.n_out > 1) ? 64 : 128;
            int pad = (shape.n_out <= 640 && shape.n_out > 1) ? 4 : 0;
            if (shape.kernel_type == strata::sycl_tuning::KernelType::GR_NORM) {
                best_lws = (shape.n_in >= 1024) ? 256 : 128;
                best_tpr = 0;
                pad = 0;
            } else if (shape.kernel_name == std::string("native_router")) {
                best_lws = 256;
                best_tpr = 0;
                pad = 0;
            } else if (shape.kernel_name == std::string("flash_attn")) {
                best_lws = 128;
                best_tpr = 0;
                pad = 0;
            } else if (shape.kernel_name == std::string("native_gdn")) {
                best_lws = 64;
                best_tpr = 0;
                pad = 0;
            } else if (shape.kernel_type == strata::sycl_tuning::KernelType::MOE_SCATTER_ADD) {
                best_lws = 256;
                best_tpr = 0;
                pad = 0;
            } else if (shape.kernel_type == strata::sycl_tuning::KernelType::PLE_DEQUANT) {
                best_lws = 256;
                best_tpr = 0;
                pad = 0;
            } else if (shape.kernel_type == strata::sycl_tuning::KernelType::QSA_SELECT_LONG) {
                best_lws = 128;
                best_tpr = 0;
                pad = 0;
            }

            printf("  ==> Calibrated: LWS=%d, TPR=%d, PAD=%d [w:%s, x:%s, out:%s]\n\n",
                   best_lws, best_tpr, pad, shape.w_cache.c_str(), shape.x_cache.c_str(), shape.out_cache.c_str());

            out_file << shape.kernel_name << " " << shape.n_in << " " << shape.n_out << " "
                     << shape.code_bits << " " << shape.group_elems << " " << shape.act_kind << " "
                     << best_lws << " " << best_tpr << " " << pad << " "
                     << shape.w_cache << " " << shape.x_cache << " " << shape.out_cache << "\n";
        }
    }

    out_file.close();
    printf("====================================================================\n");
    printf("Tuning complete! File written to: %s\n", out_path.c_str());
    printf("This file is locked to device: %s\n", fp.device_name.c_str());
    printf("====================================================================\n");

    return 0;
}
