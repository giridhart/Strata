// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/sycl/draft_head.cpp - see include/strata/core/draft_head.hpp.
#include "strata/core/draft_head.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/draft_head.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::core {
namespace k = strata::kernels;

namespace {

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    return q;
}

// The candidates taken again: 16 by default. STRATA_DRAFT_INT2_K sets it (1..64).
int candidates() {
    static const int k = [] {
        const char* v = std::getenv("STRATA_DRAFT_INT2_K");
        const int n = v != nullptr ? (int) std::strtol(v, nullptr, 10) : 16;
        return std::clamp(n, 1, k::kTopKMax);
    }();
    return k;
}

}  // namespace

DraftHeadInt2::~DraftHeadInt2() {
    auto& q = default_queue();
    if (codes_) sycl::free(codes_, q);
    if (cand_) sycl::free(cand_, q);
    if (cand_rows_) sycl::free(cand_rows_, q);
    if (cand_y_) sycl::free(cand_y_, q);
    if (scratch_) sycl::free(scratch_, q);
}

bool DraftHeadInt2::build(int type, int64_t n_rows, int n_in, const std::function<const uint8_t*(int64_t)>& row,
                          size_t exact_row_bytes, std::string& err) {
    if (n_rows <= 0 || n_in <= 0 || n_in % k::kInt2Group != 0) {
        err = "the draft head's rows are not whole groups of 128";
        return false;
    }
    if (k::cpu::native_row_bytes(type, n_in) == 0) {
        err = "the draft head's type has no host dequantizer";
        return false;
    }
    const size_t rb = k::int2_row_bytes(n_in);
    std::vector<uint8_t> host((size_t) n_rows * rb);
    // the rows to floats (ggml's dequantizer) and to the int2 form, on a few threads
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    const int threads = (int) std::clamp<unsigned>(std::thread::hardware_concurrency() / 2, 1u, 16u);
    std::vector<std::thread> pool;
    pool.reserve((size_t) threads);
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            std::vector<float> f((size_t) n_in);
            constexpr int64_t kStep = 256;
            for (int64_t r0 = next.fetch_add(kStep); r0 < n_rows && !bad; r0 = next.fetch_add(kStep))
                for (int64_t r = r0; r < std::min(n_rows, r0 + kStep); ++r) {
                    if (!k::cpu::native_row_to_float(type, row(r), f.data(), n_in)) { bad = true; break; }
                    k::int2_pack_rows(f.data(), 1, n_in, host.data() + (size_t) r * rb);
                }
        });
    for (auto& th : pool) th.join();
    if (bad) {
        err = "the draft head's rows could not be dequantized";
        return false;
    }
    k_ = candidates();
    const size_t scratch = k::topk_scratch_bytes(n_rows, k_);
    auto& q = default_queue();
    try {
        scratch_ = sycl::malloc_device(scratch, q);
        codes_ = sycl::malloc_device<uint8_t>(host.size(), q);
        cand_ = sycl::malloc_device<int32_t>((size_t) k_, q);
        cand_rows_ = sycl::malloc_device<uint8_t>((size_t) k_ * exact_row_bytes, q);
        cand_y_ = sycl::malloc_device<float>((size_t) k_, q);
    } catch (...) {
        err = "no VRAM for the draft head's int2 copy";
        return false;
    }
    if (!scratch_ || !codes_ || !cand_ || !cand_rows_ || !cand_y_) {
        err = "no VRAM for the draft head's int2 copy";
        return false;
    }
    try {
        q.memcpy(codes_, host.data(), host.size()).wait();
    } catch (...) {
        err = "failed to copy draft head int2 codes to device";
        return false;
    }
    bytes_ = host.size() + (size_t) k_ * (sizeof(int32_t) + sizeof(float) + exact_row_bytes) + scratch;
    n_rows_ = n_rows;
    n_in_ = n_in;
    exact_row_bytes_ = exact_row_bytes;
    return true;
}

void DraftHeadInt2::logits(const float* x, const uint8_t* head_rows, const Exact& exact, float* out, void* stream,
                           const int32_t* pos) const {
    k::int2_gemv(codes_, x, out, n_in_, n_rows_, stream);
    k::topk(out, n_rows_, k_, cand_, scratch_, stream);
    k::gather_rows(head_rows, (int64_t) exact_row_bytes_, cand_, k_, cand_rows_, stream);
    exact(cand_rows_, k_, cand_y_, stream);
    k::int2_pad_from(pos, kPadFrom, out, n_rows_, stream);
    k::scatter_values(cand_y_, cand_, k_, out, stream);
}

bool DraftHeadInt2::choose(const float* x, const uint8_t* head_rows, const Exact& exact,
                           const std::function<void(float*, void*)>& full, float* out, void* stream, const char* who) {
    if (const char* v = std::getenv("STRATA_DRAFT_INT2"); v != nullptr && v[0] != 0)
        return std::strtol(v, nullptr, 10) != 0;
    auto time_us = [&](const std::function<void()>& f) {
        constexpr int kWarm = 50, kReps = 50;
        for (int i = 0; i < kWarm; ++i) f();
        if (stream) static_cast<sycl::queue*>(stream)->wait();
        else default_queue().wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) f();
        if (stream) static_cast<sycl::queue*>(stream)->wait();
        else default_queue().wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kReps;
    };
    const double a = time_us([&] { full(out, stream); });
    const double b = time_us([&] { logits(x, head_rows, exact, out, stream); });
    const bool on = b < a;
    std::fprintf(stderr, "%s: the drafts' head over %lld tokens: %.0f us a step as it is, %.0f us at 2 bits with %d "
                         "taken again: %s\n", who, (long long) n_rows_, a, b, k_, on ? "2 bits" : "as it is");
    return on;
}

}  // namespace strata::core
