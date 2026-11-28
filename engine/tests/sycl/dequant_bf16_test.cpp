#include <sycl/sycl.hpp>
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/artifact/dequant.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool cpu_dequant_row(int type, const uint8_t* row_bytes, int cols, float* out) {
    using namespace strata;
    switch (type) {
        case 2:  // Q4_0 (32 elems / 18 bytes)
            for (int c = 0; c < cols; c += 32)
                dequantize_q4_0(row_bytes + (size_t)(c / 32) * 18, out + c);
            return true;
        case 6:  // Q5_0 (32 elems / 22 bytes)
            for (int c = 0; c < cols; c += 32)
                dequantize_q5_0(row_bytes + (size_t)(c / 32) * 22, out + c);
            return true;
        case 8:  // Q8_0 (32 elems / 34 bytes)
            for (int c = 0; c < cols; c += 32)
                dequantize_q8_0(row_bytes + (size_t)(c / 32) * 34, out + c);
            return true;
        case 20:  // IQ4_NL (32 elems / 18 bytes)
            for (int c = 0; c < cols; c += 32)
                dequantize_iq4_nl(row_bytes + (size_t)(c / 32) * 18, out + c);
            return true;
        case 11:  // Q3_K (256 elems / 110 bytes)
            for (int c = 0; c < cols; c += 256)
                dequantize_q3_K(row_bytes + (size_t)(c / 256) * 110, out + c);
            return true;
        case 12:  // Q4_K (256 elems / 144 bytes)
            for (int c = 0; c < cols; c += 256)
                dequantize_q4_K(row_bytes + (size_t)(c / 256) * 144, out + c);
            return true;
        case 13:  // Q5_K (256 elems / 176 bytes)
            for (int c = 0; c < cols; c += 256)
                dequantize_q5_K(row_bytes + (size_t)(c / 256) * 176, out + c);
            return true;
        case 14:  // Q6_K (256 elems / 210 bytes)
            for (int c = 0; c < cols; c += 256)
                dequantize_q6_K(row_bytes + (size_t)(c / 256) * 210, out + c);
            return true;
        case 23:  // IQ4_XS (256 elems / 136 bytes)
            for (int c = 0; c < cols; c += 256)
                dequantize_iq4_xs(row_bytes + (size_t)(c / 256) * 136, out + c);
            return true;
        case 42:  // Q2_0 (64 elems / 18 bytes)
            for (int c = 0; c < cols; c += 64)
                dequantize_q2_0(row_bytes + (size_t)(c / 64) * 18, out + c);
            return true;
        default:
            return false;
    }
}

float bf16_to_f32_val(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running dequant_bf16_test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(42);
    int total_failures = 0;

    const std::vector<int> test_types = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42};
    const int rows = 4;
    const int cols = 512;  // Multiple of 256 and 64

    for (int type : test_types) {
        if (!strata::kernels::dequant_bf16_supported(type)) {
            std::printf("Type %d not supported, skipping\n", type);
            continue;
        }

        // Determine block geometry
        int be = 0, bb = 0;
        // From dequant_bf16:
        switch (type) {
            case 2:
                be = 32;
                bb = 18;
                break;
            case 6:
                be = 32;
                bb = 22;
                break;
            case 8:
                be = 32;
                bb = 34;
                break;
            case 20:
                be = 32;
                bb = 18;
                break;
            case 11:
                be = 256;
                bb = 110;
                break;
            case 12:
                be = 256;
                bb = 144;
                break;
            case 13:
                be = 256;
                bb = 176;
                break;
            case 14:
                be = 256;
                bb = 210;
                break;
            case 23:
                be = 256;
                bb = 136;
                break;
            case 42:
                be = 64;
                bb = 18;
                break;
            default:
                continue;
        }

        const size_t row_bytes = (cols / be) * bb;
        const size_t total_bytes = rows * row_bytes;
        const size_t total_elements = rows * cols;

        std::vector<uint8_t> h_blocks(total_bytes);
        for (size_t i = 0; i < total_bytes; ++i) {
            h_blocks[i] = static_cast<uint8_t>(rng() & 0xFF);
        }

        // Set non-zero, finite FP16 scale for each block
        for (size_t r = 0; r < rows; ++r) {
            for (size_t b = 0; b < (size_t)(cols / be); ++b) {
                size_t offset = r * row_bytes + b * bb;
                // scale position depends on format:
                // For Q6_K, d is at byte 208. For Q3_K, d is at byte 108.
                // For Q4_K/Q5_K, d is at byte 0, dmin at byte 2.
                // For others, d is at byte 0.
                if (type == 14) {
                    // Q6_K: d is at offset + 208
                    h_blocks[offset + 208] = 0x00;
                    h_blocks[offset + 209] = 0x3c;  // 1.0 in fp16
                } else if (type == 11) {
                    // Q3_K: d is at offset + 108
                    h_blocks[offset + 108] = 0x00;
                    h_blocks[offset + 109] = 0x3c;
                } else if (type == 12 || type == 13) {
                    // Q4_K / Q5_K: d at 0, dmin at 2
                    h_blocks[offset + 0] = 0x00;
                    h_blocks[offset + 1] = 0x3c;
                    h_blocks[offset + 2] = 0x00;
                    h_blocks[offset + 3] = 0x38;  // 0.5 in fp16
                } else {
                    h_blocks[offset + 0] = 0x00;
                    h_blocks[offset + 1] = 0x3c;  // 1.0 in fp16
                }
            }
        }

        // Compute CPU reference
        std::vector<float> cpu_ref(total_elements);
        for (int r = 0; r < rows; ++r) {
            cpu_dequant_row(type, h_blocks.data() + r * row_bytes, cols, cpu_ref.data() + r * cols);
        }

        // Allocate GPU memory
        uint8_t* d_blocks = sycl::malloc_device<uint8_t>(total_bytes, q);
        float* d_out_f32 = sycl::malloc_device<float>(total_elements, q);
        uint16_t* d_out_bf16 = sycl::malloc_device<uint16_t>(total_elements, q);

        q.memcpy(d_blocks, h_blocks.data(), total_bytes).wait();

        // Test dequant_f32
        strata::kernels::dequant_f32(type, d_blocks, 0, rows, cols, d_out_f32, &q);
        q.wait();

        std::vector<float> gpu_f32(total_elements);
        q.memcpy(gpu_f32.data(), d_out_f32, total_elements * sizeof(float)).wait();

        int f32_mismatches = 0;
        for (size_t i = 0; i < total_elements; ++i) {
            float diff = std::fabs(gpu_f32[i] - cpu_ref[i]);
            float tol = 1e-5f * std::max(1.0f, std::fabs(cpu_ref[i]));
            if (diff > tol) {
                if (f32_mismatches < 3) {
                    std::printf("  [Type %d F32 mismatch at %zu] GPU: %f, CPU: %f (diff %f)\n", type, i, gpu_f32[i],
                                cpu_ref[i], diff);
                }
                f32_mismatches++;
            }
        }

        // Test dequant_bf16
        strata::kernels::dequant_bf16(type, d_blocks, 0, rows, cols, d_out_bf16, &q);
        q.wait();

        std::vector<uint16_t> gpu_bf16_raw(total_elements);
        q.memcpy(gpu_bf16_raw.data(), d_out_bf16, total_elements * sizeof(uint16_t)).wait();

        int bf16_mismatches = 0;
        for (size_t i = 0; i < total_elements; ++i) {
            float gpu_val = bf16_to_f32_val(gpu_bf16_raw[i]);
            float diff = std::fabs(gpu_val - cpu_ref[i]);
            float tol = 0.01f * std::max(1.0f, std::fabs(cpu_ref[i]));  // BF16 has ~8-bit precision
            if (diff > tol) {
                if (bf16_mismatches < 3) {
                    std::printf("  [Type %d BF16 mismatch at %zu] GPU: %f, CPU: %f (diff %f)\n", type, i, gpu_val,
                                cpu_ref[i], diff);
                }
                bf16_mismatches++;
            }
        }

        if (f32_mismatches == 0 && bf16_mismatches == 0) {
            std::printf("Type %2d: PASS (f32: 0 mismatches, bf16: 0 mismatches across %zu elements)\n", type,
                        total_elements);
        } else {
            std::printf("Type %2d: FAIL (f32: %d mismatches, bf16: %d mismatches)\n", type, f32_mismatches,
                        bf16_mismatches);
            total_failures += (f32_mismatches + bf16_mismatches);
        }

        sycl::free(d_blocks, q);
        sycl::free(d_out_f32, q);
        sycl::free(d_out_bf16, q);
    }

    std::printf("\ndequant_bf16_test: %s (total failures: %d)\n", total_failures == 0 ? "PASSED" : "FAILED",
                total_failures);
    return total_failures == 0 ? 0 : 1;
}
