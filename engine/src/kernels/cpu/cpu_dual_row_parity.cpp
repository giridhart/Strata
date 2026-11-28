// src/kernels/cpu/cpu_dual_row_parity.cpp - correctness and microbenchmark suite for the P1 CPU kernels:
//   1. Q2_0 bitplane rows_bp<1> (`rows_bp4_nt1` + 12-row L1 prefetch) across aligned & unaligned row counts,
//      tail rows, slice ranges, and nt = 1, 2, 4, 8 vs legacy and double references.
//   2. AVX2 IQ kernels (iq256_gu_rows_v, iq256_rows_v, iq256_gu_rows_exact_one, native_gu_rows) including
//      IQ2_XS dual-row lockstep (`rows2_dot_iq2xs`), `Fmt32<17>::decode`, and IQ1_M (type 29) across
//      IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18), IQ3_S(21), IQ2_S(22), IQ4_XS(23), IQ1_M(29) at nt = 1, 2, 4, 8.
//   3. AVX-512 IQ kernels (iq512_gu_rows, iq512_rows, iq512_reference_gu_rows) with dual-row lockstep `rows2_dot`
//      and Singleton/exact-reduction preservation, dynamically executed when cpu_vnni_ok() && cpu_avx512bw_ok()
//      passes (e.g. Xeon w7-3465X), and skipped gracefully on AVX2-only machines.
//   4. `--bench [--mb 128] [--reps 5]` measures latency (us/call) and effective bandwidth (GB/s) from DRAM.

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int64_t kBenchH = 2560;
constexpr int64_t kBenchFF = 640;
constexpr int kMaxT = 8;
constexpr int kNtList[] = {1, 2, 4, 8};

int g_fail = 0;

#define CHECK(cond, ...)                                                                        \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                           \
            std::fprintf(stderr, __VA_ARGS__);                                                  \
            std::fprintf(stderr, "\n");                                                         \
            ++g_fail;                                                                           \
        }                                                                                       \
    } while (0)

uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fill_blocks(uint8_t* p, size_t bytes, int type, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = splitmix64(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        if (type == 29) {
            // IQ1_M packs f16 d into the upper nibbles of scales[8] at bytes 48..55; keep d finite & normal.
            uint16_t sc[4];
            std::memcpy(sc, p + o + 48, 8);
            sc[3] = (uint16_t) ((sc[3] & 0x0FFFu) | 0x2000u);   // exponent bits => finite normal ~0.002-0.01
            std::memcpy(p + o + 48, sc, 8);
        } else {
            const float d = (0.5f + (float) (splitmix64(seed) >> 40) * (1.f / 16777216.f)) * 1e-3f;
            const ggml_fp16_t h = ggml_fp32_to_fp16(d);
            std::memcpy(p + o, &h, 2);
        }
    }
}

std::vector<int> all_variants(int mask) {
    std::vector<int> out;
    for (int s = 0; s <= mask; ++s)
        if ((s & ~mask) == 0) out.push_back(s);
    return out;
}

float silu_mul(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }

double fp32_to_double_ref(const uint8_t* row, int nblocks, const cpu::ActQ& aq) {
    double sum = 0.0;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t d16;
        std::memcpy(&d16, blk, 2);
        const float d = ggml_fp16_to_fp32(d16);
        for (int c = 0; c < 2; ++c) {
            const float s = (float) d * aq.scale[b * 2 + c];
            const int8_t* q = aq.q + b * 64 + c * 32;
            int isum = 0;
            for (int i = 0; i < 8; ++i) {
                const uint8_t byte = blk[2 + c * 8 + i];
                for (int k = 0; k < 4; ++k) {
                    const int code = (int) ((byte >> (2 * k)) & 3) - 1;
                    isum += code * (int) q[4 * i + k];
                }
            }
            sum += (double) s * (double) isum;
        }
    }
    return sum;
}

void test_q2_bitplane() {
    if (!cpu::cpu_avx2_ok()) {
        std::printf("Q2_0 bitplane: SKIP (no AVX2)\n");
        return;
    }
    if (!cpu::q2_bitplane_enabled()) {
        std::printf("Q2_0 bitplane: SKIP (set STRATA_Q2_BITPLANE=1 to run)\n");
        return;
    }
    std::printf("[1/3] Q2_0 bitplane rows_bp (rows_bp4_nt1 + prefetch)...\n");

    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.f, 1.f);

    // Aligned multiples of 4 (16, 64), <16 prefetch boundary (12), and unaligned tail counts (37, 65).
    const int kRowCases[] = {12, 16, 37, 64, 65};
    for (int n : {128, 640, 2560}) {
        const int nblocks = n / 64;
        const size_t row_bytes = (size_t) nblocks * 18;
        static cpu::ActQ acts[kMaxT];
        const cpu::ActQ* ap[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x((size_t) n);
            for (float& v : x) v = nd(rng) * (t % 2 ? 4.f : 1.f);
            cpu::act_quant_q8_1_avx2(x.data(), n, acts[t]);
            CHECK(acts[t].bp_pairs == nblocks / 2, "n=%d expected bp_pairs=%d got %d", n, nblocks / 2, acts[t].bp_pairs);
            ap[t] = &acts[t];
        }
        for (int rows : kRowCases) {
            std::vector<uint8_t> w((size_t) rows * row_bytes);
            uint64_t seed = 0x920000ull + (uint64_t) n * 101 + (uint64_t) rows;
            fill_blocks(w.data(), w.size(), GGML_TYPE_Q2_0, seed);

            std::vector<std::vector<float>> ref(kMaxT, std::vector<float>((size_t) rows));
            float* rp[kMaxT];
            for (int t = 0; t < kMaxT; ++t) rp[t] = ref[(size_t) t].data();
            cpu::q2_0_gguf_rows_multi_avx2_legacy(w.data(), row_bytes, nblocks, ap, kMaxT, rp, 0, rows);

            std::vector<std::vector<float>> nt1_out(kMaxT, std::vector<float>((size_t) rows));
            for (int t = 0; t < kMaxT; ++t) {
                float* p1[1] = {nt1_out[(size_t) t].data()};
                cpu::q2_0_gguf_rows_multi_avx2(w.data(), row_bytes, nblocks, &ap[t], 1, p1, 0, rows);
                for (int r = 0; r < rows; ++r) {
                    const float got = nt1_out[(size_t) t][(size_t) r];
                    const double dref = fp32_to_double_ref(w.data() + (size_t) r * row_bytes, nblocks, acts[t]);
                    const float lref = ref[(size_t) t][(size_t) r];
                    const float tol = 1e-4f * (std::fabs(lref) + 1.f);
                    CHECK(std::fabs(got - lref) <= tol, "q2 bp vs legacy n=%d rows=%d t=%d r=%d: %g vs %g",
                          n, rows, t, r, got, lref);
                    CHECK(std::fabs((double) got - dref) <= (double) tol,
                          "q2 bp vs double n=%d rows=%d t=%d r=%d: %g vs %g", n, rows, t, r, got, dref);
                }
            }
            // Bitwise equality across nt = 2, 4, 8 and nt = 1 (`rows_bp4_nt1`), plus non-zero r0 slices.
            for (int nt : {2, 4, 8}) {
                std::vector<std::vector<float>> out(nt, std::vector<float>((size_t) rows));
                std::vector<float*> op(nt);
                for (int t = 0; t < nt; ++t) op[(size_t) t] = out[(size_t) t].data();
                cpu::q2_0_gguf_rows_multi_avx2(w.data(), row_bytes, nblocks, ap, nt, op.data(), 0, rows);
                for (int t = 0; t < nt; ++t) {
                    for (int r = 0; r < rows; ++r) {
                        CHECK(std::memcmp(&out[(size_t) t][(size_t) r], &nt1_out[(size_t) t][(size_t) r], 4) == 0,
                              "q2 bp nt=%d vs nt=1 bitwise diff n=%d rows=%d t=%d r=%d", nt, n, rows, t, r);
                    }
                }
            }
            if (rows > 7) {
                std::vector<float> sliced((size_t) rows, 0.f);
                float* sp[1] = {sliced.data()};
                cpu::q2_0_gguf_rows_multi_avx2(w.data(), row_bytes, nblocks, ap, 1, sp, 3, rows - 2);
                for (int r = 3; r < rows - 2; ++r) {
                    CHECK(std::memcmp(&sliced[(size_t) r], &nt1_out[0][(size_t) r], 4) == 0,
                          "q2 bp slice bitwise diff n=%d rows=%d r=%d", n, rows, r);
                }
            }
        }
    }
    std::printf("  Q2_0 bitplane: %s\n", g_fail == 0 ? "OK" : "FAIL");
}

void test_avx2_iq() {
    if (!cpu::cpu_avx2_ok()) {
        std::printf("AVX2 IQ checks: SKIP (no AVX2)\n");
        return;
    }
    std::printf("[2/3] AVX2 IQ gu_rows / dot_rows / IQ2_XS dual-row / IQ1_M checks...\n");
    const std::vector<int> types = {16, 17, 18, 21, 22, 23, 29};
    const std::vector<int> variants = all_variants(cpu::iq256_variants());
    const int n = 2560;
    const int row_cases[] = {37, 64, 65};

    for (int type : types) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        CHECK(tr != nullptr && tr->vec_dot != nullptr, "missing ggml traits for type %d", type);
        CHECK(cpu::iq256_supported(type), "iq256_supported(%d) should be true", type);
        if (type != 23) {
            CHECK(cpu::iq256x_supported(type), "iq256x_supported(%d) should be true", type);
        }

        for (int rows : row_cases) {
            cpu::NativeFmt f;
            std::string err;
            const bool fmt_ok = cpu::native_fmt(type, GGML_TYPE_Q8_0, n, 128, f, err);
            CHECK(fmt_ok, "native_fmt type=%d rows=%d: %s", type, rows, err.c_str());
            if (!fmt_ok) continue;

            std::vector<uint8_t> blob(f.bytes);
            uint64_t seed = 0x10203000ull + (uint64_t) type * 997 + (uint64_t) rows;
            fill_blocks(blob.data(), f.down_off, type, seed);

            std::mt19937 rng(300u + (unsigned) type + (unsigned) rows);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<std::vector<uint8_t>> acts(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
            const void* ap[kMaxT];
            for (int t = 0; t < kMaxT; ++t) {
                std::vector<float> x((size_t) n);
                for (float& v : x) v = nd(rng) * (t % 3 == 0 ? 3.f : 1.f);
                cpu::native_quant_act(f, x.data(), acts[(size_t) t].data());
                ap[t] = acts[(size_t) t].data();
            }

            // Reference gate/up and gate-only dot from ggml-cpu.
            std::vector<float> ref_dot((size_t) kMaxT * rows), ref_gu((size_t) kMaxT * rows);
            for (int t = 0; t < kMaxT; ++t) {
                for (int r = 0; r < rows; ++r) {
                    float g = 0.f, u = 0.f;
                    tr->vec_dot(n, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    tr->vec_dot(n, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    ref_dot[(size_t) t * rows + r] = g;
                    ref_gu[(size_t) t * rows + r] = silu_mul(g, u);
                }
            }

            // Baseline scalar variant at nt=1 for bitwise cross-nt and cross-variant comparison.
            std::vector<float> base_dot((size_t) kMaxT * rows), base_gu((size_t) kMaxT * rows);
            for (int t = 0; t < kMaxT; ++t) {
                float* bd = &base_dot[(size_t) t * rows];
                float* bg = &base_gu[(size_t) t * rows];
                cpu::iq256_rows_v(0, type, blob.data(), f.gu_row, n, &ap[t], 1, &bd, 0, rows);
                cpu::iq256_gu_rows_v(0, type, blob.data(), f.gu_row, f.up_off, n, &ap[t], 1, &bg, 0, rows);
                for (int r = 0; r < rows; ++r) {
                    const float rd = ref_dot[(size_t) t * rows + r];
                    const float rg = ref_gu[(size_t) t * rows + r];
                    CHECK(std::fabs(bd[r] - rd) <= 1e-4f * (std::fabs(rd) + 1.f),
                          "iq256 dot vs ggml type=%d rows=%d t=%d r=%d: %g vs %g", type, rows, t, r, bd[r], rd);
                    CHECK(std::fabs(bg[r] - rg) <= 1e-4f * (std::fabs(rg) + 1.f),
                          "iq256 gu vs ggml type=%d rows=%d t=%d r=%d: %g vs %g", type, rows, t, r, bg[r], rg);
                }
            }

            // Check every variant and nt = 1, 2, 4, 8 for bitwise invariance against base (v=0, nt=1).
            for (int v : variants) {
                for (int nt : kNtList) {
                    std::vector<std::vector<float>> out_d(nt, std::vector<float>((size_t) rows));
                    std::vector<std::vector<float>> out_gu(nt, std::vector<float>((size_t) rows));
                    std::vector<float*> od(nt), ogu(nt);
                    for (int t = 0; t < nt; ++t) {
                        od[(size_t) t] = out_d[(size_t) t].data();
                        ogu[(size_t) t] = out_gu[(size_t) t].data();
                    }
                    cpu::iq256_rows_v(v, type, blob.data(), f.gu_row, n, ap, nt, od.data(), 0, rows);
                    cpu::iq256_gu_rows_v(v, type, blob.data(), f.gu_row, f.up_off, n, ap, nt, ogu.data(), 0, rows);
                    for (int t = 0; t < nt; ++t) {
                        for (int r = 0; r < rows; ++r) {
                            CHECK(std::memcmp(&out_d[(size_t) t][(size_t) r], &base_dot[(size_t) t * rows + r], 4) == 0,
                                  "iq256_rows_v bitwise mismatch type=%d rows=%d v=%d nt=%d t=%d r=%d",
                                  type, rows, v, nt, t, r);
                            CHECK(std::memcmp(&out_gu[(size_t) t][(size_t) r], &base_gu[(size_t) t * rows + r], 4) == 0,
                                  "iq256_gu_rows_v bitwise mismatch type=%d rows=%d v=%d nt=%d t=%d r=%d",
                                  type, rows, v, nt, t, r);
                        }
                    }
                }
            }

            // Exact-1 check (including type 17 dual-row ggml exact and type 29 IQ1_M), plus iq256x_rows (EX=true).
            if (type != 23) {
                const bool exact_ggml = ggml_cpu_has_avx2() && !ggml_cpu_has_avx512();
                std::vector<float> ex((size_t) rows);
                cpu::iq256_gu_rows_exact_one(type, blob.data(), f.gu_row, f.up_off, n, ap[0], ex.data(), 0, rows);
                for (int r = 0; r < rows; ++r) {
                    const float rg = ref_gu[(size_t) r];
                    if (exact_ggml) {
                        CHECK(std::memcmp(&ex[(size_t) r], &rg, 4) == 0,
                              "exact_one bitwise vs ggml type=%d rows=%d r=%d: %g vs %g",
                              type, rows, r, ex[(size_t) r], rg);
                    } else {
                        CHECK(std::fabs(ex[(size_t) r] - rg) <= 1e-4f * (std::fabs(rg) + 1.f),
                              "exact_one vs ggml type=%d rows=%d r=%d: %g vs %g",
                              type, rows, r, ex[(size_t) r], rg);
                    }
                }
                for (int nt : kNtList) {
                    std::vector<std::vector<float>> ex_d(nt, std::vector<float>((size_t) rows));
                    std::vector<float*> ed(nt);
                    for (int t = 0; t < nt; ++t) ed[(size_t) t] = ex_d[(size_t) t].data();
                    cpu::iq256x_rows(type, blob.data(), f.gu_row, n, ap, nt, ed.data(), 0, rows);
                    for (int t = 0; t < nt; ++t) {
                        for (int r = 0; r < rows; ++r) {
                            const float rd = ref_dot[(size_t) t * rows + r];
                            if (exact_ggml) {
                                CHECK(std::memcmp(&ex_d[(size_t) t][(size_t) r], &rd, 4) == 0,
                                      "iq256x_rows bitwise vs ggml type=%d rows=%d nt=%d t=%d r=%d: %g vs %g",
                                      type, rows, nt, t, r, ex_d[(size_t) t][(size_t) r], rd);
                            } else {
                                CHECK(std::fabs(ex_d[(size_t) t][(size_t) r] - rd) <= 1e-4f * (std::fabs(rd) + 1.f),
                                      "iq256x_rows vs ggml type=%d rows=%d nt=%d t=%d r=%d: %g vs %g",
                                      type, rows, nt, t, r, ex_d[(size_t) t][(size_t) r], rd);
                            }
                        }
                    }
                }
            }
        }
    }

    // Verify native_gu_rows / native_gu_rows_invariant for IQ1_M across nt = 1, 2, 4, 8.
    {
        cpu::NativeFmt f;
        std::string err;
        CHECK(cpu::native_fmt(29, GGML_TYPE_Q8_0, n, 64, f, err), "native_fmt IQ1_M: %s", err.c_str());
        std::vector<uint8_t> blob(f.bytes);
        uint64_t seed = 0x771129ull;
        fill_blocks(blob.data(), f.down_off, 29, seed);
        std::mt19937 rng(929);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<std::vector<uint8_t>> acts(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
        const void* ap[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x((size_t) n);
            for (float& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), acts[(size_t) t].data());
            ap[t] = acts[(size_t) t].data();
        }
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) 29);
        for (int nt : kNtList) {
            std::vector<std::vector<float>> out(nt, std::vector<float>(64)), inv(nt, std::vector<float>(64));
            std::vector<float*> op(nt), ip(nt);
            for (int t = 0; t < nt; ++t) { op[(size_t) t] = out[(size_t) t].data(); ip[(size_t) t] = inv[(size_t) t].data(); }
            cpu::native_gu_rows(f, blob.data(), ap, nt, op.data(), 0, 64);
            cpu::native_gu_rows_invariant(f, blob.data(), ap, nt, ip.data(), 0, 64);
            for (int t = 0; t < nt; ++t) {
                for (int r = 0; r < 64; ++r) {
                    float g = 0.f, u = 0.f;
                    tr->vec_dot(n, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    tr->vec_dot(n, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    const float rg = silu_mul(g, u);
                    CHECK(std::fabs(out[(size_t) t][(size_t) r] - rg) <= 1e-4f * (std::fabs(rg) + 1.f),
                          "native_gu_rows IQ1_M nt=%d t=%d r=%d", nt, t, r);
                    CHECK(std::fabs(inv[(size_t) t][(size_t) r] - rg) <= 1e-4f * (std::fabs(rg) + 1.f),
                          "native_gu_rows_invariant IQ1_M nt=%d t=%d r=%d", nt, t, r);
                }
            }
        }
    }
    std::printf("  AVX2 IQ checks: %s\n", g_fail == 0 ? "OK" : "FAIL");
}

void test_avx512_iq() {
    if (!cpu::cpu_vnni_ok() || !cpu::cpu_avx512bw_ok()) {
        std::printf("[3/3] AVX-512 IQ checks: SKIP (no AVX-512 F/BW/VL/DQ + VNNI on this CPU)\n");
        return;
    }
    std::printf("[3/3] AVX-512 IQ dual-row gu_rows / dot_rows / Singleton reference checks (gather=%d)...\n",
                (int) cpu::iq512_gather_on());
    const int n = 2560;
    const int row_cases[] = {37, 64, 65};

    // Standard AVX-512 paths (Singleton = false): types 16, 17, 18, 21, 22.
    for (int type : {16, 17, 18, 21, 22}) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        CHECK(tr != nullptr && tr->vec_dot != nullptr, "missing ggml traits for type %d", type);
        CHECK(cpu::iq512_supported(type), "iq512_supported(%d) should be true", type);

        for (int rows : row_cases) {
            cpu::NativeFmt f;
            std::string err;
            const bool fmt_ok = cpu::native_fmt(type, GGML_TYPE_Q8_0, n, 128, f, err);
            CHECK(fmt_ok, "native_fmt type=%d rows=%d: %s", type, rows, err.c_str());
            if (!fmt_ok) continue;

            std::vector<uint8_t> blob(f.bytes);
            uint64_t seed = 0x51200000ull + (uint64_t) type * 131 + (uint64_t) rows;
            fill_blocks(blob.data(), f.down_off, type, seed);

            std::mt19937 rng(512u + (unsigned) type + (unsigned) rows);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<std::vector<uint8_t>> acts(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
            const void* ap[kMaxT];
            for (int t = 0; t < kMaxT; ++t) {
                std::vector<float> x((size_t) n);
                for (float& v : x) v = nd(rng) * (t % 2 ? 2.5f : 1.f);
                cpu::native_quant_act(f, x.data(), acts[(size_t) t].data());
                ap[t] = acts[(size_t) t].data();
            }

            // nt=8 forces single-row path (`NT > 4`) in gu_rows & dot_rows; compare dual-row nt=1,2,4 bitwise vs nt=8!
            std::vector<std::vector<float>> nt8_d(kMaxT, std::vector<float>((size_t) rows));
            std::vector<std::vector<float>> nt8_gu(kMaxT, std::vector<float>((size_t) rows));
            float* d8p[kMaxT];
            float* g8p[kMaxT];
            for (int t = 0; t < kMaxT; ++t) { d8p[t] = nt8_d[(size_t) t].data(); g8p[t] = nt8_gu[(size_t) t].data(); }
            // NOTE: dot_rows_nt splits nt=8 into two chunks of nt=4 (which uses rows2_dot), whereas gu_rows_nt<TY> dispatches nt=8 -> NT=8 (single-row)!
            // Also check single-row r0..r0+1 isolation for dot_rows tail verification.
            cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, n, ap, kMaxT, g8p, 0, rows);
            cpu::iq512_rows(type, blob.data(), f.gu_row, n, ap, kMaxT, d8p, 0, rows);

            for (int t = 0; t < kMaxT; ++t) {
                for (int r = 0; r < rows; ++r) {
                    float gg = 0.f, uu = 0.f;
                    tr->vec_dot(n, &gg, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    tr->vec_dot(n, &uu, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    const float rgu = silu_mul(gg, uu);
                    CHECK(std::fabs(nt8_d[(size_t) t][(size_t) r] - gg) <= 1e-3f * std::fabs(gg) + 1e-3f,
                          "iq512 dot vs ggml type=%d rows=%d t=%d r=%d: %g vs %g",
                          type, rows, t, r, nt8_d[(size_t) t][(size_t) r], gg);
                    CHECK(std::fabs(nt8_gu[(size_t) t][(size_t) r] - rgu) <= 1e-3f * std::fabs(rgu) + 1e-3f,
                          "iq512 gu vs ggml type=%d rows=%d t=%d r=%d: %g vs %g",
                          type, rows, t, r, nt8_gu[(size_t) t][(size_t) r], rgu);
                }
            }

            // Isolated 1-row `iq512_rows` (guarantees single-row `row_dot` even for nt <= 4) vs dual-row `nt = 1, 2, 4`!
            std::vector<std::vector<float>> isolated_d(4, std::vector<float>((size_t) rows));
            for (int r = 0; r < rows; ++r) {
                float* ip[4] = {isolated_d[0].data(), isolated_d[1].data(), isolated_d[2].data(), isolated_d[3].data()};
                cpu::iq512_rows(type, blob.data(), f.gu_row, n, ap, 4, ip, r, r + 1);
            }

            for (int nt : {1, 2, 4}) {
                std::vector<std::vector<float>> out_d(nt, std::vector<float>((size_t) rows));
                std::vector<std::vector<float>> out_gu(nt, std::vector<float>((size_t) rows));
                std::vector<float*> od(nt), ogu(nt);
                for (int t = 0; t < nt; ++t) { od[(size_t) t] = out_d[(size_t) t].data(); ogu[(size_t) t] = out_gu[(size_t) t].data(); }
                cpu::iq512_rows(type, blob.data(), f.gu_row, n, ap, nt, od.data(), 0, rows);
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, n, ap, nt, ogu.data(), 0, rows);
                for (int t = 0; t < nt; ++t) {
                    for (int r = 0; r < rows; ++r) {
                        CHECK(std::memcmp(&out_gu[(size_t) t][(size_t) r], &nt8_gu[(size_t) t][(size_t) r], 4) == 0,
                              "iq512 gu_rows dual-row vs single-row nt=%d diff type=%d rows=%d t=%d r=%d",
                              nt, type, rows, t, r);
                        CHECK(std::memcmp(&out_d[(size_t) t][(size_t) r], &isolated_d[(size_t) t][(size_t) r], 4) == 0,
                              "iq512 dot_rows dual-row vs single-row nt=%d diff type=%d rows=%d t=%d r=%d",
                              nt, type, rows, t, r);
                    }
                }
            }
        }
    }

    // Reference Singleton=true path (`iq512_reference_gu_rows`) for types 18, 21, 22, 23 across nt = 1, 2, 4, 8.
    for (int type : {18, 21, 22, 23}) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        cpu::NativeFmt f;
        std::string err;
        const bool fmt_ok = cpu::native_fmt(type, GGML_TYPE_Q8_0, n, 128, f, err);
        CHECK(fmt_ok, "native_fmt ref type=%d: %s", type, err.c_str());
        if (!fmt_ok) continue;

        std::vector<uint8_t> blob(f.bytes);
        uint64_t seed = 0x512F0000ull + (uint64_t) type;
        fill_blocks(blob.data(), f.down_off, type, seed);

        std::mt19937 rng(700u + (unsigned) type);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<std::vector<uint8_t>> acts(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
        const void* ap[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x((size_t) n);
            for (float& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), acts[(size_t) t].data());
            ap[t] = acts[(size_t) t].data();
        }
        std::vector<float> ref_gu(kMaxT * 37);
        for (int t = 0; t < kMaxT; ++t) {
            for (int r = 0; r < 37; ++r) {
                float g = 0.f, u = 0.f;
                tr->vec_dot(n, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                tr->vec_dot(n, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                ref_gu[(size_t) t * 37 + r] = silu_mul(g, u);
            }
        }
        std::vector<std::vector<float>> nt8(kMaxT, std::vector<float>(37));
        float* p8[kMaxT];
        for (int t = 0; t < kMaxT; ++t) p8[t] = nt8[(size_t) t].data();
        cpu::iq512_reference_gu_rows(type, blob.data(), f.gu_row, f.up_off, n, ap, kMaxT, p8, 0, 37);
        for (int nt : {1, 2, 4}) {
            std::vector<std::vector<float>> out(nt, std::vector<float>(37));
            std::vector<float*> op(nt);
            for (int t = 0; t < nt; ++t) op[(size_t) t] = out[(size_t) t].data();
            cpu::iq512_reference_gu_rows(type, blob.data(), f.gu_row, f.up_off, n, ap, nt, op.data(), 0, 37);
            for (int t = 0; t < nt; ++t) {
                for (int r = 0; r < 37; ++r) {
                    CHECK(std::memcmp(&out[(size_t) t][(size_t) r], &nt8[(size_t) t][(size_t) r], 4) == 0,
                          "iq512_reference_gu_rows dual vs single nt=%d diff type=%d t=%d r=%d", nt, type, t, r);
                    const float rg = ref_gu[(size_t) t * 37 + r];
                    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) {
                        CHECK(std::memcmp(&out[(size_t) t][(size_t) r], &rg, 4) == 0,
                              "iq512_reference_gu_rows vs ggml bitwise diff type=%d nt=%d t=%d r=%d",
                              type, nt, t, r);
                    } else {
                        CHECK(std::fabs(out[(size_t) t][(size_t) r] - rg) <= 1e-4f * (std::fabs(rg) + 1.f),
                              "iq512_reference_gu_rows vs ggml diff type=%d nt=%d t=%d r=%d", type, nt, t, r);
                    }
                }
            }
        }
    }
    std::printf("  AVX-512 IQ checks: %s\n", g_fail == 0 ? "OK" : "FAIL");
}

template <class Fn>
void run_ring_bench(const char* label, size_t nblobs, size_t blob_bytes, size_t bytes_per_call,
                    int reps, Fn&& fn) {
    // Warmup one full pass through the ring
    for (size_t i = 0; i < nblobs; ++i) fn(i);

    double best_us = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < nblobs; ++i) fn(i);
        const auto t1 = Clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (double) nblobs;
        best_us = std::min(best_us, us);
    }
    (void) blob_bytes;
    const double gbps = ((double) bytes_per_call / (best_us * 1e-6)) / 1e9;
    std::printf("  %-36s %8.2f us/call  %6.2f GB/s\n", label, best_us, gbps);
}

void run_benchmarks(int mb, int reps) {
    std::printf("\n=== CPU Dual-Row & Bitplane Microbenchmark (working set ~%d MB, reps=%d, h=%lld, ff=%lld) ===\n",
                mb, reps, (long long) kBenchH, (long long) kBenchFF);
    const size_t ring_bytes = (size_t) mb * 1024 * 1024;
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);

    // 1. Q2_0 down-proj (kBenchH rows of kBenchFF values)
    if (cpu::cpu_avx2_ok()) {
        std::printf("\n[Q2_0 down-proj (rows=%lld, n=%lld)]\n", (long long) kBenchH, (long long) kBenchFF);
        const int nblocks = (int) (kBenchFF / 64);
        const size_t row_bytes = (size_t) nblocks * 18;
        const size_t blob_bytes = (size_t) kBenchH * row_bytes;
        const size_t nblobs = std::max<size_t>(4, ring_bytes / blob_bytes);
        std::vector<uint8_t> ring(nblobs * blob_bytes);
        uint64_t seed = 0x2002ull;
        fill_blocks(ring.data(), ring.size(), GGML_TYPE_Q2_0, seed);

        static cpu::ActQ acts[kMaxT];
        const cpu::ActQ* ap[kMaxT];
        std::vector<std::vector<float>> out(kMaxT, std::vector<float>((size_t) kBenchH));
        float* op[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> h((size_t) kBenchFF);
            for (float& v : h) v = 0.05f * nd(rng);
            cpu::act_quant_q8_1_avx2(h.data(), (int) kBenchFF, acts[t]);
            ap[t] = &acts[t];
            op[t] = out[(size_t) t].data();
        }
        for (int nt : {1, 2, 4}) {
            char name[64];
            std::snprintf(name, sizeof name, "q2_0 legacy (nt=%d)", nt);
            run_ring_bench(name, nblobs, blob_bytes, blob_bytes, reps, [&](size_t idx) {
                cpu::q2_0_gguf_rows_multi_avx2_legacy(ring.data() + idx * blob_bytes, row_bytes, nblocks,
                                                      ap, nt, op, 0, (int) kBenchH);
            });
            if (cpu::q2_bitplane_enabled()) {
                std::snprintf(name, sizeof name, "q2_0 bitplane (nt=%d)", nt);
                run_ring_bench(name, nblobs, blob_bytes, blob_bytes, reps, [&](size_t idx) {
                    cpu::q2_0_gguf_rows_multi_avx2(ring.data() + idx * blob_bytes, row_bytes, nblocks,
                                                   ap, nt, op, 0, (int) kBenchH);
                });
            }
        }
        if (!cpu::q2_bitplane_enabled()) {
            std::printf("  (q2_0 bitplane skipped: set STRATA_Q2_BITPLANE=1 to benchmark)\n");
        }
    }

    // 2. IQ gu_rows & dot_rows on AVX2 (+ AVX-512 if present)
    const bool has_512 = cpu::cpu_vnni_ok() && cpu::cpu_avx512bw_ok();
    for (int type : {16, 17, 18, 21, 22, 23, 29}) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, GGML_TYPE_Q8_0, kBenchH, kBenchFF, f, err)) continue;
        const size_t gu_bytes = f.down_off;          // gate + up rows
        const size_t dot_bytes = f.up_off;           // gate rows only
        const size_t nblobs = std::max<size_t>(4, ring_bytes / gu_bytes);
        std::vector<uint8_t> ring(nblobs * gu_bytes);
        uint64_t seed = 0xB0000ull + (uint64_t) type;
        fill_blocks(ring.data(), ring.size(), type, seed);

        std::vector<std::vector<uint8_t>> acts(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
        const void* ap[kMaxT];
        std::vector<std::vector<float>> out(kMaxT, std::vector<float>((size_t) kBenchFF));
        float* op[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x((size_t) kBenchH);
            for (float& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), acts[(size_t) t].data());
            ap[t] = acts[(size_t) t].data();
            op[t] = out[(size_t) t].data();
        }
        std::printf("\n[%s (type=%d, rows=%lld, n=%lld)]\n",
                    ggml_type_name((ggml_type) type), type, (long long) kBenchFF, (long long) kBenchH);

        for (int nt : kNtList) {
            char name[64];
            if (cpu::cpu_avx2_ok()) {
                std::snprintf(name, sizeof name, "avx2 gu_rows (nt=%d)", nt);
                run_ring_bench(name, nblobs, gu_bytes, gu_bytes, reps, [&](size_t idx) {
                    cpu::iq256_gu_rows(type, ring.data() + idx * gu_bytes, f.gu_row, f.up_off,
                                       (int) kBenchH, ap, nt, op, 0, (int) kBenchFF);
                });
                std::snprintf(name, sizeof name, "avx2 dot_rows (nt=%d)", nt);
                run_ring_bench(name, nblobs, gu_bytes, dot_bytes, reps, [&](size_t idx) {
                    cpu::iq256_rows(type, ring.data() + idx * gu_bytes, f.gu_row,
                                    (int) kBenchH, ap, nt, op, 0, (int) kBenchFF);
                });
                if (nt == 1 && (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29)) {
                    std::snprintf(name, sizeof name, "avx2 exact_one (nt=1)");
                    run_ring_bench(name, nblobs, gu_bytes, gu_bytes, reps, [&](size_t idx) {
                        cpu::iq256_gu_rows_exact_one(type, ring.data() + idx * gu_bytes, f.gu_row, f.up_off,
                                                     (int) kBenchH, ap[0], op[0], 0, (int) kBenchFF);
                    });
                }
            }
            if (has_512 && cpu::iq512_supported(type)) {
                std::snprintf(name, sizeof name, "avx512 gu_rows (nt=%d)", nt);
                run_ring_bench(name, nblobs, gu_bytes, gu_bytes, reps, [&](size_t idx) {
                    cpu::iq512_gu_rows(type, ring.data() + idx * gu_bytes, f.gu_row, f.up_off,
                                       (int) kBenchH, ap, nt, op, 0, (int) kBenchFF);
                });
                std::snprintf(name, sizeof name, "avx512 dot_rows (nt=%d)", nt);
                run_ring_bench(name, nblobs, gu_bytes, dot_bytes, reps, [&](size_t idx) {
                    cpu::iq512_rows(type, ring.data() + idx * gu_bytes, f.gu_row,
                                    (int) kBenchH, ap, nt, op, 0, (int) kBenchFF);
                });
            }
            if (has_512 && (type == 18 || type == 21 || type == 22 || type == 23)) {
                std::snprintf(name, sizeof name, "avx512 ref_gu (nt=%d)", nt);
                run_ring_bench(name, nblobs, gu_bytes, gu_bytes, reps, [&](size_t idx) {
                    cpu::iq512_reference_gu_rows(type, ring.data() + idx * gu_bytes, f.gu_row, f.up_off,
                                                 (int) kBenchH, ap, nt, op, 0, (int) kBenchFF);
                });
            }
        }
    }
    if (!has_512) {
        std::printf("\nAVX-512 bench: SKIP (no AVX-512 F/BW/VL/DQ + VNNI on this CPU)\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    bool bench = false;
    int mb = 128;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--mb") == 0 && i + 1 < argc) mb = std::max(8, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    }

    std::printf("cpu_dual_row_parity: avx2=%d avxvnni=%d avx512bw=%d vnni512=%d iq256_var=%d\n",
                (int) cpu::cpu_avx2_ok(), (int) cpu::cpu_avxvnni_ok(),
                (int) cpu::cpu_avx512bw_ok(), (int) cpu::cpu_vnni_ok(), cpu::iq256_variant());

    test_q2_bitplane();
    test_avx2_iq();
    test_avx512_iq();

    if (g_fail == 0 && bench) run_benchmarks(mb, reps);

    if (g_fail != 0) {
        std::fprintf(stderr, "cpu_dual_row_parity: FAILED (%d errors)\n", g_fail);
        return 1;
    }
    std::printf("cpu_dual_row_parity: ALL CHECKS PASSED\n");
    return 0;
}
