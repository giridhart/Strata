# Intel Arc B580: Non-Finite Logits (#879) on IQ3_S Models - Diagnostic & Action Plan

> **Target Platform**: Intel Arc B580 (Xe2-HPG, BMG-G21, 20 Xe-cores, 160 VEs)  
> **Model Quantization**: `IQ3_S` (Format ID 21) MoE / Dense architectures  
> **Symptom**: Intermittent `non-finite logits (#879)` during prompt evaluation or decode verification  
> **Status**: Ready for implementation in subsequent session  

---

## Executive Summary

When running `IQ3_S` models on Intel Arc B580, the server/verifier intermittently reports:
```
strata serve: non-finite logits (#879) in a decode/first decode window at position ...
```
This diagnostic audit identified four root causes in the SYCL kernel implementation:
1. **Missing `#606` FP16 Overflow Clamp in `native_mmvq.cpp`**: Direct cast to `sycl::half2` in `native_quantize_q8_1_kernel` produces `+Infinity` whenever 32-element activation sums exceed `65504.0f`. Subsequent dot products with zero weights produce `inf * 0 = NaN`.
2. **Subgroup Size Mismatch & Illegal SIMD-16 XOR Shuffle in `quantize_q8_1_rows`**: In `iq_kernels.cpp`, `quantize_q8_1_kernel` lacks `[[sycl::reqd_sub_group_size(32)]]`. When the compiler defaults to SIMD-16 on B580, `q8_1_store` executes `permute_group_by_xor(sg, ..., 16)` (an out-of-bounds shuffle that reads dirty register state) and fails to write scales for lanes 16..31.
3. **Non-Uniform Subgroup Collective Divergence in MoE Expert Kernels**: In `native_gu_multi_kernel`, `expdec_gu_kernel`, and `native_down_multi_kernel`, row/channel bounds checks cause subwarps inside a 32-lane subgroup to exit or break early while other subwarps execute `permute_group_by_xor` / `sub16_sum`, causing undefined behavior on Xe2.
4. **Hardcoded 32-lane XOR Reduction in `warp_sum`**: In `iq_kernels.cpp`, `warp_sum` hardcodes a 5-step XOR loop starting at `o = 16`, causing an out-of-bounds shuffle if executed with subgroup size 16 or 8.

---

## Action 1: Add Missing FP16 Clamping (`#606`) to `src/kernels/sycl/native_mmvq.cpp`

### Context
In [`src/core/sycl/verify.cpp#L1172`](file:///home/girtam/projects/Strata/src/core/sycl/verify.cpp#L1172), `native_quantize_q8_1(head_mixed_, xq_, N, T, cs)` quantizes the hidden states immediately before `native_mmvq` computes the head logits.

In CUDA ([`src/kernels/cuda/native_mmvq.cu#L166`](file:///home/girtam/projects/Strata/src/kernels/cuda/native_mmvq.cu#L166)), `q8_1_finite` and `q8_1_ds` clamp the scale and sum to `kHalfMax = 65504.0f`. The SYCL implementation omitted this, casting directly to `sycl::half2`, which rounds values $> 65504$ to `+Infinity`.

### Implementation
1. Add include to [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp):
   ```cpp
   #include "strata/kernels/q8_1_finite.hpp"
   ```
2. In `native_quantize_q8_1_kernel` (around line 195):
   ```cpp
   // BEFORE:
   const float d = amax / 127.0f;
   const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
   y[i / Q8K].qs[i % Q8K] = q;
   if (i % Q8K == 0) y[i / Q8K].ds = sycl::half2(d, sum);

   // AFTER:
   const float d = strata::kernels::q8_1_finite(amax / 127.0f);
   const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
   y[i / Q8K].qs[i % Q8K] = q;
   if (i % Q8K == 0) y[i / Q8K].ds = strata::kernels::q8_1_ds(d, sum);
   ```
3. In `native_swiglu_quantize_q8_1_kernel` (around line 208):
   ```cpp
   // BEFORE:
   const float d = amax / 127.0f;
   const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
   y[i / Q8K].qs[i % Q8K] = q;
   if (i % Q8K == 0) y[i / Q8K].ds = sycl::half2(d, sum);

   // AFTER:
   const float d = strata::kernels::q8_1_finite(amax / 127.0f);
   const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
   y[i / Q8K].qs[i % Q8K] = q;
   if (i % Q8K == 0) y[i / Q8K].ds = strata::kernels::q8_1_ds(d, sum);
   ```

---

## Action 2: Fix Subgroup Size and Out-of-Bounds Shuffle in `src/kernels/sycl/iq_kernels.cpp`

### Context
`Q8_1` quantization blocks have 32 elements. In [`src/kernels/sycl/iq_kernels.cpp#L3437`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L3437), `quantize_q8_1_rows` launches `quantize_q8_1_kernel` without `[[sycl::reqd_sub_group_size(32)]]`.

When compiled on B580 (where native SIMD is 16), the compiler may target SIMD-16. `q8_1_store` executes `permute_group_by_xor(sg, ..., 16)`, which is out-of-bounds for a 16-lane subgroup. Furthermore, only lane 0 writes `y[ib].ds`, leaving lanes 16..31 dequantized with an incorrect scale in [`Split<21>::apply`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L1114).

### Implementation
1. Enforce 32-lane subgroup execution on `quantize_q8_1_rows` ([`src/kernels/sycl/iq_kernels.cpp#L3437`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L3437)):
   ```cpp
   // BEFORE:
   cgh.parallel_for(
       sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) * sycl::range<3>(1, 1, 256),
                         sycl::range<3>(1, 1, 256)),
       [=](sycl::nd_item<3> /*item_ct1*/) { quantize_q8_1_kernel(x, (block_q8_1*)y, n); });

   // AFTER:
   cgh.parallel_for(
       sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) * sycl::range<3>(1, 1, 256),
                         sycl::range<3>(1, 1, 256)),
       [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
           quantize_q8_1_kernel(x, (block_q8_1*)y, n);
       });
   ```
2. Also enforce `[[sycl::reqd_sub_group_size(32)]]` on the SwiGLU fallback quantizer at line 3654:
   ```cpp
   cgh.parallel_for(
       sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((nh + 255) / 256)) * sycl::range<3>(1, 1, 256),
                         sycl::range<3>(1, 1, 256)),
       [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
           quantize_q8_1_kernel(h, hq, nh);
       });
   ```
3. Guard `q8_1_store` against executing out-of-bounds shuffles if ever called in smaller subgroups:
   ```cpp
   inline void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
       float amax = sycl::fabs(xi), sum = xi;
       auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
       const uint32_t sg_size = sg.get_local_range()[0];
   #pragma unroll
       for (int o = 16; o > 0; o >>= 1) {
           if (o < (int)sg_size) {
               amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
               sum += sycl::permute_group_by_xor(sg, sum, o);
           }
       }
       const float d = q8_1_finite(amax / 127.0f);
       const int8_t q = q8_1_quant(xi, d, amax);
       const long long ib = i / 32, iqs = i % 32;
       y[ib].qs[iqs] = q;
       if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
   }
   ```

---

## Action 3: Eliminate Subgroup Divergence in Native MoE Kernels

### Context
In `native_gu_multi_kernel`, `expdec_gu_kernel`, and `native_down_multi_kernel` ([`src/kernels/sycl/iq_kernels.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp)), work items compute bounds that vary per subwarp (`sw = tid >> 4` or `tid >> 3`). Boundary subwarps exiting early while other subwarps call `sycl::permute_group_by_xor` violates SYCL 2020 collective convergence rules.

### Implementation
Following the exact pattern used in commit `42a2c94b` for `s2_gemv_q8`:
1. In `native_gu_multi_kernel` ([`iq_kernels.cpp#L1794-L1830`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L1794-L1830)):
   - Instead of `if (row >= 2 * L.n_ff) return;`, set `const bool valid_row = (row < 2 * L.n_ff);`.
   - Ensure inactive lanes contribute `0.0f` to dot products and participate in `sub16_sum`, skipping only memory stores (`if (valid_row && t == 0) dst[...] = s * sc;`).
2. In `expdec_gu_kernel` ([`iq_kernels.cpp#L2219-L2220`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L2219-L2220)):
   - Avoid breaking the loop per-subwarp when `row0 >= nrow`. Keep the subgroup uniform by executing dummy passes with zeroed accumulator output for inactive subwarps.
3. In `native_down_multi_kernel` ([`iq_kernels.cpp#L1916-L1933`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L1916-L1933)):
   - Ensure all subwarps enter `row_dot_sub8` uniformly; mask `s_hq` or set input to `0.0f` when `!valid_r` so `sub8_sum` executes uniformly across the entire subgroup.

---

## Action 4: Make `warp_sum` in `iq_kernels.cpp` Subgroup-Agnostic

### Context
[`src/kernels/sycl/native_mmvq.cpp#L185`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L185) uses:
```cpp
inline float warp_sum(float x) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::reduce_over_group(sg, x, sycl::plus<float>());
}
```
In contrast, [`src/kernels/sycl/iq_kernels.cpp#L806`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp#L806) hardcodes `for (int o = 16; o > 0; o >>= 1)`.

### Implementation
Update `warp_sum` in `iq_kernels.cpp`:
```cpp
// BEFORE:
inline float warp_sum(float v) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += sycl::permute_group_by_xor(sg, v, o);
    }
    return v;
}

// AFTER:
inline float warp_sum(float v) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::reduce_over_group(sg, v, sycl::plus<float>());
}
```

---

## Action 5: Fix Batched Prefill Quantization in `src/prefill/sycl/moe_mmq.cpp`

### Context: The "Prompt > 69 Tokens" Threshold Discovery
The user observed that non-finite logits appear specifically when **prompt length > 69 tokens**.

In [`src/program/sycl/generate.cpp#L575`](file:///home/girtam/projects/Strata/src/program/sycl/generate.cpp#L575):
```cpp
int64_t short_read = 64;
...
auto windows_ok = [&](int64_t a, int64_t b) -> bool {
    if (no_short || b - a > o.short_read) return false;
    return true;
};
```
1. **Prompt <= 69**: With template overhead (~5 tokens), the un-reused tokens `b - a <= 64` (`short_read`). `windows_ok` is **true**. The prompt is processed incrementally via `read_windows` using the decode/verify engine in small 8-token windows. The batched prefill engine is completely bypassed.
2. **Prompt > 69**: `b - a > 64`. `windows_ok` flips to **false**. Strata switches execution to **`sp.run` (the batched SYCL prefill engine in `src/prefill/sycl/prefill.cpp`)**.
3. In `sp.run`, activations are quantized via `mmq::quantize` in [`src/prefill/sycl/moe_mmq.cpp#L487-L500`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L487-L500):
   ```cpp
   // BEFORE:
   const float d = amax / 127.0f;
   ...
   bq->ds[0] = sycl::vec<float, 1>(d).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
   bq->ds[1] = sycl::vec<float, 1>(d * sum).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
   ```
   Like `native_mmvq.cpp`, `moe_mmq.cpp` lacks `q8_1_finite` clamping. When `d * sum > 65504`, `bq->ds[1]` overflows to `+Infinity`, causing NaNs in MoE down projections that poison the recurrent GDN state and KV cache.
4. When `sp.run` completes, Strata runs the first decode window on token $n - 1$ ([`generate.cpp#L6708`](file:///home/girtam/projects/Strata/src/program/sycl/generate.cpp#L6708)), which immediately reads the poisoned state and triggers `#879`.

### Implementation
In [`src/prefill/sycl/moe_mmq.cpp#L487-L500`](file:///home/girtam/projects/Strata/src/prefill/sycl/moe_mmq.cpp#L487-L500):
```cpp
// Include q8_1_finite.hpp:
#include "strata/kernels/q8_1_finite.hpp"

// In quantize():
const float d = strata::kernels::q8_1_finite(amax / 127.0f);
const float id = 1.0f / d;
float sum = 0.0f;
for (int i = 0; i < 32; ++i) {
    const int64_t col = b * 32 + i;
    const float val = col < cols ? src_row[col] : 0.0f;
    int q0 = (int)sycl::round(val * id);
    q0 = sycl::clamp(q0, -128, 127);
    bq->qs[i] = (int8_t)q0;
    sum += (float)q0;
}
bq->ds = strata::kernels::q8_1_ds(d, d * sum);
```

### Diagnostic Verification
To immediately verify this on B580 before code changes:
- Pass `--short-read 128` (or `--short-read 256`) to keep prompts up to 128 tokens in the `read_windows` path.
- Or pass `--short-read 0` to force even 1-token prompts into `sp.run` to confirm immediate reproduction.

---

## Action 6: Validation & Verification Steps

After applying the changes in the new session:

1. **Compile SYCL Kernels**:
   ```sh
   cmake -B build_sycl -GNinja -DSTRATA_ENABLE_SYCL=ON
   ninja -C build_sycl strata_kernels sycl_verify_kernels sycl_native_mmvq
   ```

2. **Run Parity and Divergence Suites**:
   ```sh
   ./build_sycl/tests/sycl/sycl_subgroup_divergence_test
   ./build_sycl/src/kernels/iq_multi_parity
   ./build_sycl/tests/sycl/numerics_test
   ```

3. **Verify Target B580 Execution with NaN Injection**:
   ```sh
   # Verify the guard detects injected NaNs as expected:
   STRATA_NAN_INJECT=1 ./build_sycl/strata launch --family deepseek41 --model ist-das-2.00bpw --plan
   # Run without injection to confirm clean generation:
   ./build_sycl/strata launch --family deepseek41 --model ist-das-2.00bpw --yes --no-tui
   ```
