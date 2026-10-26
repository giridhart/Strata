// tools/sycl/tune_sycl_kernels.cpp - On-device autotuner for SYCL kernels on Intel Arc GPUs.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/core/sycl_tuning.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_gr_norm.hpp"
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
    {"native_gdn", strata::sycl_tuning::KernelType::GDN, 128, 8, 32, 1, 0, "GDN recurrent step S=128", "default", "default", "default"}
};

BenchmarkMetrics measure_bf16_gemv_split(sycl::queue& q, const uint16_t* d_x, const uint16_t* d_w, float* d_y,
                                        int64_t n_in, int64_t n_out, int tpr, double min_time_sec, double warmup_sec) {
    BenchmarkMetrics metrics;

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

            const int tpr_candidates[] = {16, 32, 64, 128, 256};
            int best_tpr = 128;
            double best_time = 1e9;

            for (int tpr : tpr_candidates) {
                BenchmarkMetrics m = measure_bf16_gemv_split(q, d_x, d_w, d_y, shape.n_in, shape.n_out,
                                                             tpr, min_time_sec, warmup_sec);
                printf("  Candidate TPR=%-3d : %8.2f us (min: %8.2f us, %6d iters in %.2fs [%s])\n",
                       tpr, m.median_us, m.min_us, m.total_iters, m.total_time_s,
                       m.event_profiling_used ? "HW Batch Profiling" : "Wall Clock");
                if (m.median_us < best_time) {
                    best_time = m.median_us;
                    best_tpr = tpr;
                }
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
