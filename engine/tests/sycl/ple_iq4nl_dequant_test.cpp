// tests/sycl/ple_iq4nl_dequant_test.cpp - STRATA_PLE_COMPACT: ple_iq4nl_dequant
// (raw IQ4_NL PLE rows to floats on the device) against iq4nl_dequant_row on host.
// Bit for bit equivalence check and on-device dequantization throughput microbenchmark.

#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
using namespace strata::kernels;

int check(sycl::queue& q, const std::vector<uint8_t>& raw, int64_t rows, const char* what) {
    float* dout = nullptr;
    uint8_t* draw = nullptr;
    try {
        dout = sycl::malloc_device<float>((size_t)rows * PLE_HEAD_DIM, q);
        draw = sycl::malloc_device<uint8_t>(raw.size(), q);
        if (!dout || !draw) {
            std::printf("SYCL malloc_device failed for %s\n", what);
            if (dout) sycl::free(dout, q);
            if (draw) sycl::free(draw, q);
            return 1;
        }
        q.memcpy(draw, raw.data(), raw.size()).wait();
        q.fill(dout, 0.0f, (size_t)rows * PLE_HEAD_DIM).wait();
    } catch (const std::exception& e) {
        std::printf("SYCL setup failed for %s: %s\n", what, e.what());
        if (dout) sycl::free(dout, q);
        if (draw) sycl::free(draw, q);
        return 1;
    }

    try {
        ple_iq4nl_dequant(draw, rows, dout, &q);
        q.wait();
    } catch (const std::exception& e) {
        std::printf("SYCL error for %s: %s\n", what, e.what());
        sycl::free(dout, q);
        sycl::free(draw, q);
        return 1;
    }

    std::vector<float> got((size_t)rows * PLE_HEAD_DIM), want(got.size());
    q.memcpy(got.data(), dout, got.size() * sizeof(float)).wait();
    sycl::free(dout, q);
    sycl::free(draw, q);

    for (int64_t r = 0; r < rows; ++r) {
        iq4nl_dequant_row(raw.data() + (size_t)r * PLE_ROW_BYTES, want.data() + (size_t)r * PLE_HEAD_DIM);
    }

    long long diff = 0, first = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        if (std::isnan(got[i]) && std::isnan(want[i])) continue; // a NaN scale: any payload is NaN
        if (a != b) {
            if (first < 0) first = (long long)i;
            ++diff;
        }
    }
    std::printf("  %-52s %lld rows: %s\n", what, (long long)rows, diff ? "*** DIFFERS ***" : "bitwise equal");
    if (diff) {
        std::printf("      %lld values differ, first at %lld\n", diff, first);
    }
    return diff ? 1 : 0;
}

void benchmark_throughput(sycl::queue& q, int64_t rows, int iters) {
    const size_t raw_bytes = (size_t)rows * PLE_ROW_BYTES;
    const size_t out_floats = (size_t)rows * PLE_HEAD_DIM;
    const size_t out_bytes = out_floats * sizeof(float);

    std::vector<uint8_t> h_raw(raw_bytes);
    std::mt19937 rng(42);
    for (auto& b : h_raw) b = (uint8_t)(rng() & 0xff);

    uint8_t* d_raw = sycl::malloc_device<uint8_t>(raw_bytes, q);
    float* d_out = sycl::malloc_device<float>(out_floats, q);
    if (!d_raw || !d_out) {
        std::printf("Benchmark alloc failed\n");
        if (d_raw) sycl::free(d_raw, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }
    q.memcpy(d_raw, h_raw.data(), raw_bytes).wait();

    // Warmup
    for (int i = 0; i < 5; ++i) {
        ple_iq4nl_dequant(d_raw, rows, d_out, &q);
    }
    q.wait();

    // Timed runs
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ple_iq4nl_dequant(d_raw, rows, d_out, &q);
    }
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iters;
    double gb_in = (double)raw_bytes / 1e9;
    double gb_out = (double)out_bytes / 1e9;
    double bandwidth_gb_s = (gb_in + gb_out) / (avg_ms / 1e3);
    double mrows_per_s = (double)rows / (avg_ms / 1e3) / 1e6;

    std::printf("\n=== Microbenchmark: on-device ple_iq4nl_dequant ===\n");
    std::printf("  Batch rows: %lld (input %.2f MB, output %.2f MB)\n", (long long)rows,
                (double)raw_bytes / (1024.0 * 1024.0), (double)out_bytes / (1024.0 * 1024.0));
    std::printf("  Iterations: %d | Mean latency: %.3f ms\n", iters, avg_ms);
    std::printf("  Throughput: %.2f Mrows/s | Combined bandwidth: %.2f GB/s\n\n", mrows_per_s, bandwidth_gb_s);

    sycl::free(d_raw, q);
    sycl::free(d_out, q);
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running ple_iq4nl_dequant test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int bad = 0;
    std::mt19937 rng(20261026u);

    {   // All fp16 patterns: 65,536 scales over 5 blocks a row = 13,108 rows
        const int64_t rows = (65536 + 4) / 5;
        std::vector<uint8_t> raw((size_t)rows * PLE_ROW_BYTES);
        for (auto& b : raw) b = (uint8_t)(rng() & 0xff);
        uint32_t d = 0;
        for (int64_t r = 0; r < rows; ++r) {
            for (int b = 0; b < PLE_HEAD_DIM / 32; ++b, ++d) {
                const uint16_t h = (uint16_t)(d & 0xffff);
                std::memcpy(&raw[(size_t)r * PLE_ROW_BYTES + (size_t)b * 18], &h, 2);
            }
        }
        bad += check(q, raw, rows, "every fp16 scale pattern, random codes");
    }

    {   // Random rows with odd row count
        const int64_t rows = 4096 + 37;
        std::vector<uint8_t> raw((size_t)rows * PLE_ROW_BYTES);
        for (auto& b : raw) b = (uint8_t)(rng() & 0xff);
        bad += check(q, raw, rows, "random bytes (odd row count)");
    }

    {   // Out-of-table row of PleTable (scale 0, code 8)
        const int64_t rows = 16;
        std::vector<uint8_t> raw((size_t)rows * PLE_ROW_BYTES);
        for (int64_t r = 0; r < rows; ++r) {
            for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
                uint8_t* blk = &raw[(size_t)r * PLE_ROW_BYTES + (size_t)b * 18];
                std::memset(blk, 0, 2);
                std::memset(blk + 2, 0x88, 16);
            }
        }
        bad += check(q, raw, rows, "out-of-table row (scale 0, code 8)");

        std::vector<float> want(PLE_HEAD_DIM);
        iq4nl_dequant_row(raw.data(), want.data());
        for (float v : want) {
            uint32_t a;
            std::memcpy(&a, &v, 4);
            if (a != 0) {
                std::printf("  *** the out-of-table row does not decode to +0.0f\n");
                ++bad;
                break;
            }
        }
    }

    // Microbenchmark
    benchmark_throughput(q, 100000, 50);

    std::printf("ple_iq4nl_dequant_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
