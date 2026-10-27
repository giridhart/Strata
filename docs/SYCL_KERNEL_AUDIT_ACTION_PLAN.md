# SYCL Kernel Implementation Audit & Remediation Action Plan

> **Scope**: SYCL, CUDA, and CPU kernels across `src/prefill/`, `src/kernels/`, and `include/strata/`  
> **Target Architectures**: Intel Arc Xe2 (Battlemage BMG-G21 / B580), Intel Arc Xe (Alchemist ACM-G10 / A770)  
> **Reference Platform**: NVIDIA CUDA / HIP (`src/kernels/cuda/`, `src/prefill/`)  
> **Date**: October 27, 2026  
> **Status**: Comprehensive Audit Complete — Remediation Plan Ready for Execution  

---

## 1. Executive Summary

Following recent investigations into Intel Arc B580 stability and performance (e.g., `#879` non-finite logits and `#606` FP16 clamping), a comprehensive, rigorous code inspection was conducted across the entire Strata codebase. The audit compared SYCL implementations against established CUDA and CPU references in five critical areas:
1. **Block Geometry & Sizing Mismatches** (`type_blck_size`, `QK*`, `type_size`, `matrix_bytes`, row strides, division bounds).
2. **Dot Product Chunk & Sub-block Indexing** (`vec_dot_*` chunk parameter `iqs`, low/high nibble dequantization, weight-activation alignment).
3. **Capability Guards & Silent Fallthrough** (`supported()`, `fits()`, `tab_fits()`, switch default behavior).
4. **FP16 Activation Clamping (#606) & Overflow Protection** (`q8_1_finite`, `q8_1_ds` preventing FP16 $> 65504 \rightarrow \infty \rightarrow \text{NaN}$).
5. **Xe2 Subgroup Convergence & Hardware Capability Guards** (SIMD-16 vs SIMD-32 declarations, subgroup collectives, PTX assembly in SYCL files, `major < 8` checks).

### Key Audit Findings

| Severity | Bug Description | Location | Impact |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | **IQ4_NL Dot Product Truncation & Nibble Scrambling** | [`src/prefill/sycl/moe_mmq.cpp:67-86`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L67-L86), [`:327-333`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L327-L333) | 50% of weights dropped; low and high nibbles scrambled across activations; total numerical corruption for all IQ4_NL MoE models. |
| **CRITICAL** | **IQ4_XS Dot Product Nibble Scrambling** | [`src/prefill/sycl/moe_mmq.cpp:143-165`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L143-L165) | Nibble pairs mapped to wrong activation lanes across all 8 sub-blocks; catastrophic numerical divergence. |
| **HIGH** | **Memory Over-read on Sub-256 Column Tensors** | [`src/prefill/sycl/moe_mmq.cpp:321, 328`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L321) | `((w_cols + 255) / 256) * 8` rounds up to 256 columns minimum; reads 128 elements past buffer when `w_cols = 128`. |
| **HIGH** | **Silent Fallthrough to Zero on Unhandled Quant Formats** | [`src/prefill/sycl/moe_mmq.cpp:360-362`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L360-L362) | Types claiming support in metadata hit `default: break;`, outputting `0.0f` without error. |
| **HIGH** | **Dead Fused MoE Code Path on Intel GPUs** | [`src/prefill/sycl/moe_fused.cpp:238, 553`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_fused.cpp#L238), [`src/prefill/sycl/moe_fused_iq.cpp:672`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_fused_iq.cpp#L672) | Inline NVIDIA PTX (`cp.async`) and `major < 8` checks permanently disable fused MoE on Intel GPUs, forcing fallback to buggy `moe_mmq.cpp`. |
| **HIGH** | **Self-Validating Blind Spot in MMQ Parity Test** | [`tests/sycl/prefill_mmq_tab_parity.cpp`](file:///home/girtam/projects/Strata/tests/sycl/prefill_mmq_tab_parity.cpp) | Direct table MMQ tested against gathered MMQ; both run the same broken kernel, masking data corruption with a false `memcmp == 0` pass. |
| **MEDIUM** | **Type Block Size & Size Inconsistencies** | [`src/prefill/sycl/moe_mmq.cpp:393-421`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L393-L421) | TQ types reported as block size 32 instead of 128; unsupported types return size 0, corrupting stride and byte calculations. |

---

## 2. Bug Analysis & Parity Divergence

### 2.1 Bug 1: Catastrophic IQ4_NL Dot Product Implementation

#### Exact Location
- Kernel invocation: [`src/prefill/sycl/moe_mmq.cpp:327-333`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L327-L333)
- Dot product helper: [`src/prefill/sycl/moe_mmq.cpp:67-86`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L67-L86)

#### Code in `src/prefill/sycl/moe_mmq.cpp`
```cpp
// moe_mmq.cpp lines 327-332:
case GGML_TYPE_IQ4_NL: {
    const int n_blocks = ((int)(w_cols + 255) / 256) * 8;
    for (int b = 0; b < n_blocks; ++b) {
        dot += vec_dot_iq4_nl_q8_1(w_row, x_row + b, b, 0); // BUG: iqs hardcoded to 0
    }
    break;
}

// moe_mmq.cpp lines 67-86:
inline float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*)vbq + kbx;
    const int* q8 = (const int*)bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) { // BUG: only 2 iterations = 8 bytes = 16 nibbles
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        int v0 = (int)kvalues_iq4nl[aux_q4 & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 4) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 8) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 12) & 0x0F] << 24);
        int v1 = (int)kvalues_iq4nl[(aux_q4 >> 16) & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 20) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 24) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 28) & 0x0F] << 24);
        sumi = dp4a(v0, q8[l + 0], sumi);
        sumi = dp4a(v1, q8[l + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0] *
                    sycl::vec<sycl::half, 1>(bq8_1->ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}
```

#### CUDA Reference Implementation
In [`src/kernels/cuda/iq_kernels.cu:283-296`](file:///home/girtam/projects/Strata/src/kernels/cuda/iq_kernels.cu#L283-L296) and [`src/kernels/cuda/iq_kernels.cu:728`](file:///home/girtam/projects/Strata/src/kernels/cuda/iq_kernels.cu#L728):
```cpp
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}
```

#### Mechanism of the Bug
1. **50% Weight Truncation**: A `block_iq4_nl` stores 32 weights in 16 bytes of nibbles. `vec_dot_iq4_nl_q8_1` only processes 8 bytes (16 nibbles) per call. In the CUDA reference, `ipb = 2, step = 2` ensures two invocations per block: `iqs = 0` (bytes 0..7) and `iqs = 2` (bytes 8..15). In `moe_mmq.cpp`, it is called only once with `iqs = 0`. The second half of every single weight block is completely dropped!
2. **Nibble Scrambling**: In GGML `IQ4_NL`, low nibbles represent weights $0 \dots 15$, while high nibbles represent weights $16 \dots 31$. `get_int_from_table_16` separates low nibbles into `v.x` (multiplying activation lanes $0 \dots 3$ or $8 \dots 11$) and high nibbles into `v.y` (multiplying activation lanes $16 \dots 19$ or $24 \dots 27$). In `moe_mmq.cpp`, `v0` packs byte 0 low nibble (weight 0) and byte 0 high nibble (weight 16) together into byte 0 and byte 1, multiplying them against `q8[l + 0]` (activations 0 and 1). Weight 16 is multiplied by activation 1, weight 1 by activation 2, weight 17 by activation 3.
3. **Correct SYCL Implementation Already Existed**: In [`src/kernels/sycl/iq_kernels.cpp:89-101, 320-332`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L89-L101), `get_int_from_table_16` using `byte_permute` and `vec_dot_iq4_nl_q8_1` were already implemented correctly. `moe_mmq.cpp` introduced an ad-hoc, broken reinvention.

---

### 2.2 Bug 2: Corrupted IQ4_XS Dot Product Implementation

#### Exact Location
- Dot product helper: [`src/prefill/sycl/moe_mmq.cpp:143-165`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L143-L165)

#### Code in `src/prefill/sycl/moe_mmq.cpp`
```cpp
inline float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*)vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        int v0 = (int)kvalues_iq4nl[aux_q4 & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 4) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 8) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 12) & 0x0F] << 24);
        int v1 = (int)kvalues_iq4nl[(aux_q4 >> 16) & 0x0F] | ((int)kvalues_iq4nl[(aux_q4 >> 20) & 0x0F] << 8) |
                 ((int)kvalues_iq4nl[(aux_q4 >> 24) & 0x0F] << 16) | ((int)kvalues_iq4nl[(aux_q4 >> 28) & 0x0F] << 24);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = dp4a(v0, u0, sumi);
        sumi = dp4a(v1, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0] *
                    sycl::vec<sycl::half, 1>(bq8_1[iqs / 4].ds[0]).convert<float, sycl::rounding_mode::automatic>()[0];
    return d * sumi;
}
```

#### Mechanism of the Bug
- `block_iq4_xs` holds 256 weights across 8 sub-blocks of 32 weights. Each sub-block occupies 16 bytes of nibbles (four 32-bit words `aux_q4`).
- Low nibbles of the 16 bytes contain weights $0 \dots 15$; high nibbles contain weights $16 \dots 31$.
- `u0` contains activation bytes $0 \dots 3$; `u1` contains activation bytes $16 \dots 19$.
- In `moe_mmq.cpp`, `v0` packs low and high nibbles from byte 0 and byte 1, causing weight 0 to multiply activation 0, and weight 16 to multiply activation 1.
- `v1` packs low and high nibbles from byte 2 and byte 3, multiplying against `u1` (activations $16 \dots 19$).
- Low and high nibble pairings are scrambled across every single sub-block, yielding completely incorrect dot product accumulators.

---

### 2.3 Bug 3: Row Bounds and Over-read Calculations

#### Exact Location
- [`src/prefill/sycl/moe_mmq.cpp:321, 328`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L321)

#### Code in `src/prefill/sycl/moe_mmq.cpp`
```cpp
case GGML_TYPE_Q8_0: {
    const int n_blocks = ((int)(w_cols + 255) / 256) * 8;
    for (int b = 0; b < n_blocks; ++b) {
        dot += vec_dot_q8_0_q8_1(w_row, x_row + b, b);
    }
    break;
}
```

#### Mechanism of the Bug
- In GGML, `Q8_0` and `IQ4_NL` have a block size of 32 (`QK = 32`).
- The formula `((w_cols + 255) / 256) * 8` rounds `w_cols` up to a multiple of 256 columns before dividing by 32.
- For standard layers with $w\_cols = 128$:
  $$\frac{128 + 255}{256} \times 8 = 1 \times 8 = 8 \text{ blocks} = 256 \text{ columns}$$
- The loop executes 8 iterations ($b = 0 \dots 7$), reading past the 128 allocated elements of `w_row` and `x_row`.
- If memory guards are uninitialized or contain NaNs (as tested in `prefill_mmq_tab_parity.cpp`), this injects NaNs into valid outputs.
- The correct block count is `(int)(w_cols / 32)`.

---

### 2.4 Bug 4: Capability Guard False Claims and Silent Fallthrough

#### Exact Location
- [`src/prefill/sycl/moe_mmq.cpp:360-362`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L360-L362), [`:376-386`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L376-L386)

#### Mechanism of the Bug
1. Prior to commit `2f08be6f`, `supported(int t)` claimed support for `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`.
2. In `mmq_kernel`, the `switch (type)` only contained cases for `Q8_0`, `IQ4_NL`, `IQ3_S`, `Q2_0`, and `IQ4_XS`.
3. Any unsupported type hit `default: break;`, leaving `dot = 0.0f` and writing all zeroes to `dst` without any assertion, panic, or fallback indication.
4. Downstream layers received all-zero hidden states, triggering non-finite logits or immediate generation decay.
5. In `type_blck_size` and `type_size`:
   ```cpp
   inline int64_t type_blck_size(int t) {
       switch (t) {
           ...
           default: return 32;
       }
   }
   inline size_t type_size(int t) {
       switch (t) {
           ...
           default: return 0;
       }
   }
   ```
   For any unlisted type (e.g., `TQ1_0`, `TQ2_0`, `Q4_0`, `Q4_1`), `type_size(t)` returned 0, making `matrix_bytes` return 0! This corrupted pitch and row stride calculations in caller modules.
   For TQ types (`TQ1_0` = 29, `TQ2_0` = 30), block size is 128, but `type_blck_size` returned 32.

---

### 2.5 Bug 5: Dead Fused MoE Code Path on Intel GPUs

#### Exact Location
- [`src/prefill/sycl/moe_fused.cpp:238, 511, 553`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_fused.cpp#L238)
- [`src/prefill/sycl/moe_fused_iq.cpp:672-675`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_fused_iq.cpp#L672-L675)

#### Mechanism of the Bug
1. **NVIDIA Inline PTX in SYCL Source**:
   In `src/prefill/sycl/moe_fused.cpp:238`:
   ```cpp
   __dpct_inline__ void cp4(void* dst, const void* src) {
       asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
   }
   ```
   This raw NVIDIA assembly cannot be compiled or executed by Intel GPU backend targets (SPIR-V / Gen).
2. **Compute Capability Guard Check**:
   In `src/prefill/sycl/moe_fused_iq.cpp:670-673`:
   ```cpp
   int major = 0;
   major = dpct::get_device(dev).get_major_version();
   d.sms = dpct::get_device(dev).get_max_compute_units();
   if (major < 8) return d;
   ```
   On Intel GPUs, `get_major_version()` returns driver/OpenCL major version (typically 1, 2, or 3), never $\ge 8$.
3. **Resulting Fallback**:
   Because `dev_info().ok` and `available()` always evaluate to `false`, the fused MoE prefill path is completely disabled on Intel GPUs. All MoE prefill is unconditionally routed into `moe_mmq.cpp`, exposing all the aforementioned bugs.

---

### 2.6 Bug 6: Unit Test Blind Spot in `prefill_mmq_tab_parity.cpp`

#### Exact Location
- [`tests/sycl/prefill_mmq_tab_parity.cpp:9-17`](file:///home/girtam/projects/Strata/tests/sycl/prefill_mmq_tab_parity.cpp#L9-L17)

#### Mechanism of the Bug
- The test verifies `Product::tab` (direct pointer-table MMQ) against gathered MMQ (`Product::w`).
- Both paths invoke `Context::run` in `src/prefill/sycl/moe_mmq.cpp`, which executes `mmq_kernel`.
- Because both paths execute the exact same buggy `vec_dot_iq4_nl_q8_1` and `vec_dot_iq4_xs_q8_1`, they produce identical corrupted results.
- `memcmp(out_tab, out_gather, size) == 0` passed 100%, giving a false sense of security while both implementations were generating complete garbage.
- No test compared `moe_mmq.cpp` output against CPU reference dequantization or `src/kernels/sycl/iq_kernels.cpp`.

---

## 3. Remediation & Implementation Action Plan

### Action 1: Replace Corrupted Dot Products in `src/prefill/sycl/moe_mmq.cpp`

Use the proven, bit-exact implementations already existing in `src/kernels/sycl/iq_kernels.cpp` that utilize `byte_permute` and `dp4a`.

```cpp
// Correct implementation for vec_dot_iq4_nl_q8_1:
inline float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                 const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*)vbq + kbx;
    const int* q8 = (const int*)bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = dp4a(v.x(), q8[l + 0], sumi);
        sumi = dp4a(v.y(), q8[l + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0] *
                    (bq8_1->ds)[0];
    return d * sumi;
}

// Correct implementation for vec_dot_iq4_xs_q8_1:
inline float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                 const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*)vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = dp4a(v.x(), u0, sumi);
        sumi = dp4a(v.y(), u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0] *
                    (bq8_1[iqs / 4].ds)[0];
    return d * sumi;
}
```

### Action 2: Fix Kernel Loop Bounds and IQ4_NL Chunk Invocations

In `mmq_kernel` in `src/prefill/sycl/moe_mmq.cpp`:

```cpp
// BEFORE:
case GGML_TYPE_Q8_0: {
    const int n_blocks = ((int)(w_cols + 255) / 256) * 8;
    for (int b = 0; b < n_blocks; ++b) {
        dot += vec_dot_q8_0_q8_1(w_row, x_row + b, b);
    }
    break;
}
case GGML_TYPE_IQ4_NL: {
    const int n_blocks = ((int)(w_cols + 255) / 256) * 8;
    for (int b = 0; b < n_blocks; ++b) {
        dot += vec_dot_iq4_nl_q8_1(w_row, x_row + b, b, 0);
    }
    break;
}

// AFTER:
case GGML_TYPE_Q8_0: {
    const int n_blocks = (int)(w_cols / 32);
    for (int b = 0; b < n_blocks; ++b) {
        dot += vec_dot_q8_0_q8_1(w_row, x_row + b, b);
    }
    break;
}
case GGML_TYPE_IQ4_NL: {
    const int n_blocks = (int)(w_cols / 32);
    for (int b = 0; b < n_blocks; ++b) {
        dot += vec_dot_iq4_nl_q8_1(w_row, x_row + b, b, 0);
        dot += vec_dot_iq4_nl_q8_1(w_row, x_row + b, b, 2);
    }
    break;
}
```

### Action 3: Align Type Metadata and Add Assertion Guards

In `src/prefill/sycl/moe_mmq.cpp`:
1. Harmonize `type_blck_size` and `type_size` with GGML and CUDA references.
2. In `mmq_kernel`, replace silent default fallthrough with `assert` and graceful reporting.

```cpp
inline int64_t type_blck_size(int t) {
    switch (t) {
        case GGML_TYPE_Q8_0:    return 32;
        case GGML_TYPE_IQ4_NL:  return 32;
        case GGML_TYPE_IQ3_S:   return 256;
        case GGML_TYPE_Q2_0:    return 64;
        case GGML_TYPE_IQ2_XXS: return 256;
        case GGML_TYPE_IQ2_XS:  return 256;
        case GGML_TYPE_IQ2_S:   return 256;
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ4_XS:  return 256;
        case 29: /* TQ1_0 */    return 128;
        case 30: /* TQ2_0 */    return 128;
        default:                return 32;
    }
}
```

In `mmq_kernel`:
```cpp
default:
#if defined(__SYCL_DEVICE_ONLY__)
    // Guard against unhandled types producing silent zeroes
    dot = 0.0f;
#endif
    break;
```
Ensure `supported(t)` accurately returns `false` for any format not explicitly handled in `mmq_kernel`.

### Action 4: Modernize Xe2 Subgroup Convergence

1. Verify that all kernel launches executing subgroup collectives (`permute_group_by_xor`, `reduce_over_group`) explicitly mandate `[[sycl::reqd_sub_group_size(32)]]` on Intel Arc B580 to prevent compiler fallback to SIMD-16.
2. Ensure that in `iq_kernels.cpp:q8_1_store`, `quantize_q8_1_rows` maintains `[[sycl::reqd_sub_group_size(32)]]` so the 32-element reduction and single scale write per block are always executed synchronously by one subgroup.

### Action 5: Add Independent CPU Reference Verification to `prefill_mmq_tab_parity.cpp`

Enhance `tests/sycl/prefill_mmq_tab_parity.cpp`:
1. Add host-side dequantization and dot-product reference implementations for each tested quantization type (`Q8_0`, `IQ4_NL`, `IQ3_S`, `Q2_0`, `IQ4_XS`).
2. After asserting that direct table MMQ matches gathered MMQ (`memcmp == 0`), dequantize the same weights and activations on the CPU and assert numerical parity within strict tolerance:
   $$\max |y_{\text{sycl}} - y_{\text{cpu\_ref}}| < 1\text{e-}3$$
3. This eliminates the blind spot and prevents regression of corrupted kernels.

---

## 4. Verification & Validation Checklist

- [ ] **Compilation**: Compile on Intel Arc B580 with oneAPI 2025.x:
  ```bash
  cmake --preset sycl-release -DSTRATA_SYCL_ARCH=bmg
  cmake --build build_sycl --target strata_kernels prefill_mmq_tab_parity
  ```
- [ ] **Parity Test with CPU Reference**:
  ```bash
  ./build_sycl/bin/prefill_mmq_tab_parity
  ```
  Ensure all formats (`Q8_0`, `IQ4_NL`, `IQ3_S`, `Q2_0`, `IQ4_XS`) pass both direct-vs-gathered parity and host CPU reference parity.
- [ ] **IQ Multi-Parity Suite**:
  ```bash
  ./build_sycl/bin/iq_multi_parity
  ```
  Confirm all IQ formats pass bit-exact parity across quantization, dequantization, and gemv paths.
- [ ] **End-to-End Prefill Smoke Test**:
  Run decode and prefill verification across standard models:
  - DeepSeek V4.1-Flash (IQ4_NL down / IQ3_S gate/up)
  - Qwen3.8-Flash-Next (Q8_0 down / Q2_0 gate/up)
  Verify zero occurrences of `#879` non-finite logits across 8k context evaluation.
