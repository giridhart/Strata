# Intel Compiler Optimization Analysis: Assembly Breakdown of `-fp-model=fast` vs `-fp-model=precise`

In numerical software and inference engines like **Strata**, floating-point reproducibility and bit-exact parity against reference implementations (CUDA, ggml, IEEE-754) are critical. 

Unlike GCC and Clang—which default to strict IEEE-754 semantics unless `-ffast-math` is explicitly passed—the **Intel oneAPI DPC++/C++ Compiler (`icx` / `icpx`) defaults to `-fp-model=fast`**.

This document breaks down the assembly transformations that occur when `-fp-model=precise` is **omitted** under Intel oneAPI, compares them against `-fp-model=precise`, and maps each transformation to observed engine parity failures in Strata.

---

## 1. Summary of Optimizations Enabled by Default under `icpx`

```mermaid
flowchart TD
    subgraph Default ["Default icpx: -fp-model=fast"]
        F1["FMA Contraction: -ffp-contract=fast<br/>Fuses mul + add into vfmadd"]
        F2["Reciprocal Hoisting: Inverts divisor outside loop,<br/>replaces vdivps with vmulps"]
        F3["Vector Math Library: Calls SVML (__svml_expf8)<br/>polynomial approximations"]
        F4["Reassociation: Reorders arithmetic<br/>(a + b) + c -> a + (b + c)"]
    end

    subgraph Impact ["Parity Impact in Strata"]
        I1["q8k_quant_parity Mismatches<br/>(Mantissa LSB drift vs un-fused baseline)"]
        I2["quantize_act_parity: 303k mismatches<br/>(amax / 127.0f reciprocal error)"]
        I3["Issue #152 Parity Breach<br/>(Drift between single-token nt=1 and multi-token nt=4/8)"]
    end

    F1 --> I1
    F2 --> I2
    F3 --> I3
```

| Optimization Category | Without `-fp-model=precise` (`-fp-model=fast`) | With `-fp-model=precise` | Impact on Strata |
| :--- | :--- | :--- | :--- |
| **FMA Contraction** | Fuses adjacent `*` and `+` into `vfmadd` without intermediate rounding. | Discretely rounds intermediate products via `vmulps` then `vaddps`. | Fails [`q8k_quant_parity`](../src/kernels/cpu/q8k_quant_parity.cpp). Fixed via `-fp-model=precise` or `__asm__("" : "+x"(p))`. |
| **Division / Scaling** | Algebraic reassociation: `x / (y / 127)` &rarr; `x * (127 * (1 / y))`. Replaces `vdivps` with `vmulps`. | Strictly evaluates `scale = y / 127.0f`, followed by IEEE-754 `vdivps`. | Without precise division, [`quantize_act_parity`](../src/kernels/sycl/quantize_act_parity.cpp) produced **303,000 mismatches**. |
| **Transcendentals (`std::exp`)** | Replaces scalar `expf` with Intel SVML (`__svml_expf8`) polynomial approximations. | Retains scalar `expf` with IEEE-754 round-to-nearest semantics. | In SwiGLU, SVML introduces 1–2 ULP approximation drift between scalar `nt=1` and vectorized `nt=4..8` batches ([Issue #152](../CMakeLists.txt#L1740-L1742)). |
| **Associativity** | Reorders reductions and associative chains for maximum throughput. | Preserves sequential evaluation order. | Thread/lane reduction non-determinism. |

---

## 2. Deep Dive: Assembly Breakdown with Annotations

All assembly listings below were compiled with **Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1** targeting `x86-64-v3` (`-mavx2 -mfma -O3`).

---

### Case 1: Multiply-Add Contraction (`mul_ps` followed by `add_ps`)

In quantized kernels such as `q8k_quant`, reference kernels assume discrete rounding of the product before adding into the accumulator or biased round.

#### C++ Source
```cpp
void mul_add(const float* a, const float* b, const float* c, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float prod = a[i] * b[i]; // Reference expects IEEE-754 rounding here
        out[i] = prod + c[i];
    }
}
```

#### Annotated Assembly Comparison

````carousel
```nasm
# -------------------------------------------------------------
# COMPILED WITH: -fp-model=precise
# -------------------------------------------------------------
.L_precise_loop:
    vmovups ymm0, ymmword ptr [rdi + 4*r10]      # Load a[i..i+7]
    vmovups ymm1, ymmword ptr [rdi + 4*r10 + 32] # Load a[i+8..i+15]

    # Step 1: Discrete SIMD multiplication
    # Exact IEEE-754 rounding to single precision (24-bit mantissa) occurs HERE:
    vmulps  ymm0, ymm0, ymmword ptr [rsi + 4*r10]
    vmulps  ymm1, ymm1, ymmword ptr [rsi + 4*r10 + 32]

    # Step 2: Discrete SIMD addition
    # Adds rounded products with c[i..i+15]:
    vaddps  ymm0, ymm0, ymmword ptr [rdx + 4*r10]
    vaddps  ymm1, ymm1, ymmword ptr [rdx + 4*r10 + 32]

    vmovups ymmword ptr [rcx + 4*r10], ymm0      # Store out[i..i+7]
    vmovups ymmword ptr [rcx + 4*r10 + 32], ymm1 # Store out[i+8..i+15]
    add     r10, 16
    cmp     r10, r8
    jb      .L_precise_loop
```
<!-- slide -->
```nasm
# -------------------------------------------------------------
# COMPILED WITHOUT -fp-model=precise (Default: -fp-model=fast)
# -------------------------------------------------------------
.L_fast_loop:
    vmovups ymm0, ymmword ptr [rsi + 4*r10]      # Load b[i..i+7]
    vmovups ymm1, ymmword ptr [rsi + 4*r10 + 32] # Load b[i+8..i+15]
    vmovups ymm2, ymmword ptr [rdi + 4*r10]      # Load a[i..i+7]
    vmovups ymm3, ymmword ptr [rdi + 4*r10 + 32] # Load a[i+8..i+15]

    # OPTIMIZATION: Fused Multiply-Add Contraction (FMA)
    # Replaces discrete vmulps + vaddps with vfmadd213ps:
    # ymm2 = (ymm0 * ymm2) + c[i]
    # NOTE: The product (a * b) is calculated with infinite precision
    # (no intermediate rounding) and rounded ONLY ONCE at the addition.
    vfmadd213ps ymm2, ymm0, ymmword ptr [rdx + 4*r10]
    vfmadd213ps ymm3, ymm1, ymmword ptr [rdx + 4*r10 + 32]

    vmovups ymmword ptr [rcx + 4*r10], ymm2      # Store out[i..i+7]
    vmovups ymmword ptr [rcx + 4*r10 + 32], ymm3 # Store out[i+8..i+15]
    add     r10, 16
    cmp     r10, r8
    jb      .L_fast_loop
```
````

> [!WARNING]
> **Why this causes parity failure:** 
> When reference code in GGML / CUDA computes `round(a * b) + c`, but Intel `icpx` emits `round((a * b) + c)`, the least significant bit (LSB) of the floating-point mantissa can differ whenever the intermediate multiplication would have rounded across a bit boundary. In Strata's integer quantizers, this 1-bit drift can change an integer bucket from `q` to `q + 1`.

---

### Case 2: Quantizer Scaling (Reciprocal Hoisting & Division Elimination)

In activation quantization, values are scaled by `scale = amax / 127.0f`:

#### C++ Source
```cpp
void quantize_scale(const float* in, float* out, float amax, int n) {
    float scale = amax / 127.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = in[i] / scale;
    }
}
```

#### Annotated Assembly Comparison

````carousel
```nasm
# -------------------------------------------------------------
# COMPILED WITH: -fp-model=precise
# -------------------------------------------------------------
    # 1. Compute scale = amax / 127.0f using exact division:
    vdivss  xmm0, xmm0, dword ptr [rip + .LCPI_127]  # xmm0 = amax / 127.0f
    ...
    # 2. Broadcast scale to vector register:
    vbroadcastss ymm1, xmm0                          # ymm1 = [scale, scale, ...]

.L_div_loop:
    vmovups ymm2, ymmword ptr [rdi + 4*r8]           # Load in[i..i+7]
    vmovups ymm3, ymmword ptr [rdi + 4*r8 + 32]      # Load in[i+8..i+15]

    # 3. Exact IEEE-754 vector division executed per vector:
    # Guaranteed correct rounding within 0.5 ULP.
    vdivps  ymm2, ymm2, ymm1                         # in[i..i+7] / scale
    vdivps  ymm3, ymm3, ymm1                         # in[i+8..i+15] / scale

    vmovups ymmword ptr [rsi + 4*r8], ymm2
    vmovups ymmword ptr [rsi + 4*r8 + 32], ymm3
    add     r8, 16
    cmp     r8, rcx
    jb      .L_div_loop
```
<!-- slide -->
```nasm
# -------------------------------------------------------------
# COMPILED WITHOUT -fp-model=precise (Default: -fp-model=fast)
# -------------------------------------------------------------
    # OPTIMIZATION 1: Algebraic Inversion Outside Loop
    # Transforms: in[i] / (amax / 127.0f) --> in[i] * (127.0f * (1.0f / amax))
    vmovss  xmm1, dword ptr [rip + .LCPI_1]          # xmm1 = 1.0f
    vdivss  xmm0, xmm1, xmm0                         # xmm0 = 1.0f / amax (INVERTED!)
    ...
    # Pre-multiply reciprocal by 127.0f outside the loop:
    vmulss  xmm1, xmm0, dword ptr [rip + .LCPI_127]  # xmm1 = (1.0f / amax) * 127.0f
    vbroadcastss ymm1, xmm1                          # Broadcast inv_scale to ymm1

.L_mul_loop:
    # OPTIMIZATION 2: Division replaced by Multiplication
    # Divides are eliminated; fast vector multiplies replace vdivps:
    vmulps  ymm2, ymm1, ymmword ptr [rdi + 4*r8]     # in[i..i+7] * inv_scale
    vmulps  ymm3, ymm1, ymmword ptr [rdi + 4*r8 + 32]# in[i+8..i+15] * inv_scale

    vmovups ymmword ptr [rsi + 4*r8], ymm2
    vmovups ymmword ptr [rsi + 4*r8 + 32], ymm3
    add     r8, 16
    cmp     r8, rcx
    jb      .L_mul_loop
```
````

> [!IMPORTANT]
> **Why `quantize_act_parity` had 303,000 mismatches:**
> In floating-point arithmetic, $x / (y / 127) \neq x \times (127 / y)$. The pre-computation of $1.0 / amax$ incurs a rounding error, and multiplying by $127$ incurs another. When iterating over thousands of activation coordinates, boundary values right at quantization thresholds round in the opposite direction.

---

### Case 3: Transcendentals & SwiGLU (`__svml_expf8` vs `expf`)

In SwiGLU activation layers ([`native_expert.cpp:194`](../src/kernels/cpu/native_expert.cpp#L194)), gates are computed via:
$$ff[t][r] = \left(\frac{g}{1.0 + \exp(-g)}\right) \cdot u$$

#### Assembly Behavior: Scalar vs Auto-Vectorized SVML

When compiled **without** `-fp-model=precise`:
```nasm
# Fast Loop: Compiler vectorizes the loop and links against Intel SVML:
    vxorps  ymm0, ymm0, ymmword ptr [rip + .LCPI_NEG_ZERO] # Negate: -g
    # Calls Intel Short Vector Math Library 8-wide AVX2 exponential:
    call    qword ptr [rip + __svml_expf8_mask_l9@GOTPCREL]
    vaddps  ymm0, ymm0, ymmword ptr [rip + .LCPI_ONE]      # 1.0 + exp(-g)
    vdivps  ymm0, ymm1, ymm0                               # g / (1.0 + exp(-g))
    vmulps  ymm0, ymm0, ymmword ptr [r14]                  # silu * u
```

When compiled **with** `-fp-model=precise`:
```nasm
# Precise Loop: Transcendentals are NOT vectorized into SVML polynomials:
    vmovss  xmm1, dword ptr [r15 + rbp]                    # Scalar load: g
    vxorps  xmm0, xmm1, xmmword ptr [rip + .LCPI_NEG_ZERO] # -g
    call    expf                                           # Standard libm expf
    vaddss  xmm0, xmm0, dword ptr [rip + .LCPI_ONE]        # 1.0 + exp(-g)
    vdivss  xmm0, xmm1, xmm0                               # g / (1.0 + exp(-g))
    vmulss  xmm0, xmm0, dword ptr [r14 + rbp]              # silu * u
    vmovss  dword ptr [rbx + rbp], xmm0
```

> [!NOTE]
> **Issue #152 Parity Contract:**
> Intel's `__svml_expf8` uses an 8-way SIMD polynomial minimax approximation optimized for throughput. Its rounding error characteristics differ by ~0.5–1.5 ULP from scalar standard `expf`.
> If single-token decode (`nt=1`) executes the scalar `expf` path while multi-token speculative verification (`nt=2..8`) executes the unrolled/vectorized SVML path, the model's logits diverge based on drafting length. Passing `-fp-model=precise` guarantees identical evaluation across batch sizes.

---

## 3. Explaining the Compiler Error: `#define -fp-model precise`

During CMake configuration, passing the flag improperly can trigger this build error:
```text
<command line> : 6:9 error: macro name must be an identifier, #define -fp-model precise
```

### Root Cause
In C/C++ compiler drivers (`clang`, `gcc`, `icx`, `icpx`):
- The `-D` flag defines a preprocessor macro: `-DNAME=VALUE` translates internally to `#define NAME VALUE`.
- If a compiler option like `-fp-model=precise` is passed to CMake's **`target_compile_definitions()`** or `COMPILE_DEFINITIONS` instead of **`target_compile_options()`**, CMake automatically prepends `-D` to it.
- The command line then receives:
  ```bash
  icpx -D -fp-model=precise ...
  # or
  icpx -D-fp-model=precise ...
  ```
- The C preprocessor parses `-fp-model` as the macro name:
  ```c
  #define -fp-model precise
  ```
- Because `-fp-model` starts with a hyphen `-` (which is not a valid C identifier `[a-zA-Z_][a-zA-Z0-9_]*`), the compiler emits:
  `error: macro name must be an identifier`.

### Correct CMake Usage

```cmake
# WRONG (causes "#define -fp-model precise"):
target_compile_definitions(my_target PRIVATE -fp-model=precise)
set_property(SOURCE file.cpp APPEND PROPERTY COMPILE_DEFINITIONS "-fp-model=precise")

# CORRECT:
target_compile_options(my_target PRIVATE -fp-model=precise)
set_property(SOURCE file.cpp APPEND PROPERTY COMPILE_OPTIONS "-fp-model=precise")
```

---

## 4. Key Takeaways for Strata on Intel Architectures

1. **Host CPU Engine (`strata_kernels_cpu`):** Must compile with `-fp-model=precise` on `expert.cpp`, `iq_avx512.cpp`, `iq_avx2.cpp`, and `iq_avx1.cpp` ([CMakeLists.txt:1742-1745](../CMakeLists.txt#L1742-L1745)) to prevent SVML polynomial drift and FMA contraction.
2. **Device GPU Engine (Intel Arc / Xe2 SYCL):** Must compile with `-fp-model=precise` for host glue and `-cl-fp32-correctly-rounded-divide-sqrt` for device kernels ([docs/INTEL.md:185](../docs/INTEL.md#L185)) to guarantee 0.5 ULP division on Arc GPUs.
3. **Explicit ASM Barriers:** Where compiler options cannot be overridden per file, use inline assembly barriers:
   ```cpp
   #if defined(__GNUC__) || defined(__clang__)
   __asm__("" : "+x"(p)); // Prevent FMA contraction across mul and add
   #endif
   ```
