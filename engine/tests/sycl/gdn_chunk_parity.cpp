// tests/sycl/gdn_chunk_parity.cpp - Parity verification for SYCL Chunked GDN recurrence
#include <sycl/sycl.hpp>
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gdn_chunk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int S = 128;
constexpr int HK = 16;
constexpr int HV = 48;
constexpr int C = 10240;
constexpr int ROW = HV * S;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

float half_to_float(uint16_t bits) {
    sycl::half h = sycl::bit_cast<sycl::half>(bits);
    return float(h);
}

struct TestInput {
    int64_t T;
    std::vector<float> h;
    std::vector<float> gate;
    std::vector<float> beta;
    std::vector<float> z;
    std::vector<float> gamma;
    std::vector<float> state_init;

    TestInput(int64_t tokens, uint32_t seed = 42) : T(tokens) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::uniform_real_distribution<float> unif(0.0f, 1.0f);

        h.resize((size_t)T * C);
        for (int64_t t = 0; t < T; ++t) {
            float* row = h.data() + t * C;
            // Q and K heads: normalize per head
            for (int head = 0; head < 2 * HK; ++head) {
                float norm_sq = 0.0f;
                for (int d = 0; d < S; ++d) {
                    float val = gauss(rng);
                    row[head * S + d] = val;
                    norm_sq += val * val;
                }
                float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-6f);
                for (int d = 0; d < S; ++d) {
                    row[head * S + d] *= inv_norm;
                }
            }
            // V heads
            for (int d = 2 * HK * S; d < C; ++d) {
                row[d] = gauss(rng);
            }
        }

        gate.resize((size_t)T * HV);
        for (auto& g : gate) {
            // Realistic negative log-decay
            g = -0.5f - 2.5f * unif(rng);
        }

        beta.resize((size_t)T * HV);
        for (auto& b : beta) {
            b = 0.1f + 0.8f * unif(rng);
        }

        z.resize((size_t)T * ROW);
        for (auto& zv : z) {
            zv = gauss(rng);
        }

        gamma.resize((size_t)S);
        for (auto& gm : gamma) {
            gm = 0.5f + unif(rng);
        }

        state_init.resize((size_t)S * ROW);
        for (auto& st : state_init) {
            st = 0.05f * gauss(rng);
        }
    }
};

struct TestOutput {
    std::vector<float> y;
    std::vector<uint16_t> y16;
    std::vector<float> state;
};

TestOutput run_kernel(sycl::queue& q, const TestInput& in, const char* chunked_env) {
    set_env("STRATA_GDN_CHUNKED", chunked_env);

    const size_t h_bytes = in.h.size() * sizeof(float);
    const size_t gate_bytes = in.gate.size() * sizeof(float);
    const size_t beta_bytes = in.beta.size() * sizeof(float);
    const size_t z_bytes = in.z.size() * sizeof(float);
    const size_t gamma_bytes = in.gamma.size() * sizeof(float);
    const size_t state_bytes = in.state_init.size() * sizeof(float);
    const size_t y_bytes = (size_t)in.T * ROW * sizeof(float);
    const size_t y16_bytes = (size_t)in.T * ROW * sizeof(uint16_t);

    float* d_h = sycl::malloc_device<float>(in.h.size(), q);
    float* d_gate = sycl::malloc_device<float>(in.gate.size(), q);
    float* d_beta = sycl::malloc_device<float>(in.beta.size(), q);
    float* d_z = sycl::malloc_device<float>(in.z.size(), q);
    float* d_gamma = sycl::malloc_device<float>(in.gamma.size(), q);
    float* d_state = sycl::malloc_device<float>(in.state_init.size(), q);
    float* d_y = sycl::malloc_device<float>((size_t)in.T * ROW, q);
    uint16_t* d_y16 = sycl::malloc_device<uint16_t>((size_t)in.T * ROW, q);

    q.memcpy(d_h, in.h.data(), h_bytes);
    q.memcpy(d_gate, in.gate.data(), gate_bytes);
    q.memcpy(d_beta, in.beta.data(), beta_bytes);
    q.memcpy(d_z, in.z.data(), z_bytes);
    q.memcpy(d_gamma, in.gamma.data(), gamma_bytes);
    q.memcpy(d_state, in.state_init.data(), state_bytes);
    q.wait();

    strata::prefill::gdn_recurrence(d_state, d_h, d_gate, d_beta, d_z, d_gamma, 1e-6f, d_y, d_y16, in.T, &q, ROW);
    q.wait();

    TestOutput out;
    out.y.resize((size_t)in.T * ROW);
    out.y16.resize((size_t)in.T * ROW);
    out.state.resize(in.state_init.size());

    q.memcpy(out.y.data(), d_y, y_bytes);
    q.memcpy(out.y16.data(), d_y16, y16_bytes);
    q.memcpy(out.state.data(), d_state, state_bytes);
    q.wait();

    sycl::free(d_h, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_z, q);
    sycl::free(d_gamma, q);
    sycl::free(d_state, q);
    sycl::free(d_y, q);
    sycl::free(d_y16, q);

    return out;
}

}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    const auto dev = q.get_device();
    std::printf("======================================================================\n");
    std::printf("  SYCL Chunked GDN Recurrence Parity Test (Xe2 / B580)\n");
    std::printf("  Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("======================================================================\n");

    const std::vector<int64_t> test_lengths = {64, 128, 256, 512, 1024, 2048};
    int failures = 0;

    for (int64_t T : test_lengths) {
        std::printf("\n--- Testing Prompt Length T = %lld ---\n", (long long)T);
        TestInput input(T);

        // 1. Reference run: STRATA_GDN_CHUNKED=0 (sequential recurrence)
        TestOutput ref = run_kernel(q, input, "0");

        // 2. Chunked candidate run: STRATA_GDN_CHUNKED=1
        TestOutput cand = run_kernel(q, input, "1");

        if (T < 128) {
            // Requirement: T < 128 must cleanly run the reference path bitwise
            int diff_y_bits = 0;
            int diff_y16_bits = 0;
            int diff_st_bits = 0;
            for (size_t i = 0; i < ref.y.size(); ++i) {
                if (std::memcmp(&ref.y[i], &cand.y[i], sizeof(float)) != 0) diff_y_bits++;
                if (ref.y16[i] != cand.y16[i]) diff_y16_bits++;
            }
            for (size_t i = 0; i < ref.state.size(); ++i) {
                if (std::memcmp(&ref.state[i], &cand.state[i], sizeof(float)) != 0) diff_st_bits++;
            }

            if (diff_y_bits == 0 && diff_y16_bits == 0 && diff_st_bits == 0) {
                std::printf("  [PASS] T = %lld (< 128): BITWISE EXACT match with reference recurrence (0 bit diffs)\n",
                            (long long)T);
            } else {
                std::printf("  [FAIL] T = %lld (< 128): Expected bitwise exact fallback! (diffs: y=%d, y16=%d, state=%d)\n",
                            (long long)T, diff_y_bits, diff_y16_bits, diff_st_bits);
                failures++;
            }
        } else {
            // Requirement: T >= 128 chunked recurrence numerical parity:
            // rel_err_y < 1e-4 and rel_err_state < 1e-4
            float max_ref_y = 0.0f;
            float max_diff_y = 0.0f;
            for (size_t i = 0; i < ref.y.size(); ++i) {
                max_ref_y = std::max(max_ref_y, std::fabs(ref.y[i]));
                max_diff_y = std::max(max_diff_y, std::fabs(cand.y[i] - ref.y[i]));
            }
            const float rel_err_y = max_diff_y / std::max(max_ref_y, 1e-6f);

            float max_ref_st = 0.0f;
            float max_diff_st = 0.0f;
            for (size_t i = 0; i < ref.state.size(); ++i) {
                max_ref_st = std::max(max_ref_st, std::fabs(ref.state[i]));
                max_diff_st = std::max(max_diff_st, std::fabs(cand.state[i] - ref.state[i]));
            }
            const float rel_err_st = max_diff_st / std::max(max_ref_st, 1e-6f);

            float max_diff_y16 = 0.0f;
            for (size_t i = 0; i < ref.y16.size(); ++i) {
                float rf = half_to_float(ref.y16[i]);
                float cf = half_to_float(cand.y16[i]);
                max_diff_y16 = std::max(max_diff_y16, std::fabs(cf - rf));
            }

            const bool ok_y = (rel_err_y < 1e-4f);
            const bool ok_st = (rel_err_st < 1e-4f);

            if (!ok_y) {
                // Find first token with diff > 1e-4
                for (int64_t t = 0; t < T; ++t) {
                    float tok_diff = 0.0f;
                    for (int h = 0; h < HV; ++h) {
                        for (int col = 0; col < S; ++col) {
                            size_t idx = (size_t)t * ROW + h * S + col;
                            tok_diff = std::max(tok_diff, std::fabs(cand.y[idx] - ref.y[idx]));
                        }
                    }
                    if (tok_diff > 1e-4f) {
                        std::printf("    First diff in y at token t=%lld: max_diff=%.4e\n", (long long)t, tok_diff);
                        break;
                    }
                }
            }

            std::printf("  y    max_ref=%.4e max_diff=%.4e rel_err=%.4e [%s]\n",
                        max_ref_y, max_diff_y, rel_err_y, ok_y ? "OK (< 1e-4)" : "FAIL");
            std::printf("  state max_ref=%.4e max_diff=%.4e rel_err=%.4e [%s]\n",
                        max_ref_st, max_diff_st, rel_err_st, ok_st ? "OK (< 1e-4)" : "FAIL");
            std::printf("  y16  max_diff=%.4e\n", max_diff_y16);

            if (ok_y && ok_st) {
                std::printf("  [PASS] T = %lld Chunked Recurrence Parity Validated\n", (long long)T);
            } else {
                std::printf("  [FAIL] T = %lld Chunked Recurrence Parity Exceeded Threshold\n", (long long)T);
                failures++;
            }
        }
    }

    // Additional check: verify STRATA_GDN_CHUNKED=0 explicitly at T = 256
    {
        std::printf("\n--- Testing STRATA_GDN_CHUNKED=0 Disable Check (T = 256) ---\n");
        TestInput input(256);
        TestOutput out1 = run_kernel(q, input, "0");
        TestOutput out2 = run_kernel(q, input, "0");

        int diff_bits = 0;
        for (size_t i = 0; i < out1.y.size(); ++i) {
            if (std::memcmp(&out1.y[i], &out2.y[i], sizeof(float)) != 0) diff_bits++;
        }
        for (size_t i = 0; i < out1.state.size(); ++i) {
            if (std::memcmp(&out1.state[i], &out2.state[i], sizeof(float)) != 0) diff_bits++;
        }

        if (diff_bits == 0) {
            std::printf("  [PASS] STRATA_GDN_CHUNKED=0 cleanly runs reference bitwise (0 bit diffs)\n");
        } else {
            std::printf("  [FAIL] STRATA_GDN_CHUNKED=0 produced inconsistent results! (%d diffs)\n", diff_bits);
            failures++;
        }
    }

    std::printf("\n======================================================================\n");
    if (failures == 0) {
        std::printf("  ALL GDN CHUNK PARITY TESTS PASSED (0 failures)\n");
        std::printf("======================================================================\n");
        return 0;
    } else {
        std::printf("  GDN CHUNK PARITY TESTS FAILED (%d failures)\n", failures);
        std::printf("======================================================================\n");
        return 1;
    }
}
