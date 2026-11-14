# Strata SYCL B580 Fork & Rebase Guide

This guide describes how to fork [Niko1221/Strata](https://github.com/Niko1221/Strata) to your personal GitHub account (`<your-github-username>/Strata`), configure Git remotes, maintain a regular upstream rebasing workflow, transfer changes to the Intel Arc B580 target machine, and resume kernel porting in future sessions.

---

## 1. Architecture & Repository Topology

- **Upstream (Main Repository)**: `https://github.com/Niko1221/Strata.git` (`upstream`)
- **Your Personal Fork**: `https://github.com/<your-github-username>/Strata.git` (`origin`)
- **Development Workstation**: `/path/to/Strata`
  - Current state: branch `main` (or feature branch `sycl-b580`), ahead of upstream `origin/main` by 26 commits.
- **Target Test Rig**: Intel Arc B580 GPU machine.

---

## 2. Setting Up Your Fork and Pushing

### Step 1: Fork on GitHub
1. Open [https://github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) in your browser.
2. Click **Fork** (top right) and create the fork under your GitHub account (`<your-github-username>`) with repository name **`Strata`**.

### Step 2: Configure Remotes on Your Dev Machine
Run the following commands in your local Strata directory (`/path/to/Strata`):

```bash
# Rename current 'origin' (points to Niko1221/Strata) to 'upstream'
git remote rename origin upstream

# Add your fork as the new 'origin'
# (Use SSH if you have SSH keys configured, or HTTPS)
git remote add origin git@github.com:<your-github-username>/Strata.git
# Or: git remote add origin https://github.com/<your-github-username>/Strata.git

# Verify remotes
git remote -v
```
You should see:
```text
origin    git@github.com:<your-github-username>/Strata.git (fetch)
origin    git@github.com:<your-github-username>/Strata.git (push)
upstream  https://github.com/Niko1221/Strata.git (fetch)
upstream  https://github.com/Niko1221/Strata.git (push)
```

### Step 3: Set Up a Dedicated Branch (`sycl-b580`)
To keep upstream `main` clean and isolate all B580/SYCL development on a dedicated branch:

```bash
# Create and switch to sycl-b580 from your current main
git checkout -b sycl-b580

# Reset local main to track upstream/main cleanly (optional, but recommended)
git branch -f main upstream/main
```

### Step 4: Push to Your Fork
```bash
# Push your sycl-b580 branch to your fork
git push -u origin sycl-b580

# (Optional) Push the clean main branch if desired
git push -u origin main
```

---

## 3. Recurring Upstream Rebase Workflow

When upstream (`Niko1221/Strata`) releases new features, fixes, or optimizations, follow this procedure to keep `sycl-b580` up to date:

```bash
# 1. Fetch latest upstream commits
git fetch upstream

# 2. Switch to your development branch
git checkout sycl-b580

# 3. Rebase your commits on top of latest upstream main
git rebase upstream/main
```

### If Conflicts Occur During Rebase:
1. Inspect conflicting files:
   ```bash
   git status
   ```
2. Open conflicted files, resolve conflicts (preserving both upstream improvements and SYCL Xe2 implementations), and stage them:
   ```bash
   git add <conflicted_file>
   ```
3. Continue the rebase:
   ```bash
   git rebase --continue
   ```
   *(Note: You can abort at any time with `git rebase --abort` if needed)*

### Verify After Rebase:
Always run the quiet build and test scripts before pushing:
```bash
./tools/build_sycl.sh strata
./tools/test_sycl.sh
```

### Push Rebased Branch to Your Fork:
Because rebasing rewrites commit hashes, use `--force-with-lease` (safer than `--force`):
```bash
git push origin sycl-b580 --force-with-lease
```

---

## 4. Deploying to the Intel Arc B580 Target Machine

Once your changes are pushed to GitHub or packed locally, you can synchronize your target machine using one of three methods:

### Method A: Direct Git Pull from Your Fork (Recommended)
On the B580 target machine:
```bash
# If cloned from upstream, add your fork:
git remote add fork https://github.com/<your-github-username>/Strata.git
git fetch fork
git checkout -B sycl-b580 fork/sycl-b580

# Future updates:
git pull --rebase fork sycl-b580
```

### Method B: Git Bundle (Offline / Direct SCP)
On your dev machine, a verified bundle is already generated:
```bash
# Generate bundle containing all commits ahead of upstream
git bundle create strata-sycl-b580.bundle upstream/main..HEAD
git bundle verify strata-sycl-b580.bundle
```
Transfer `strata-sycl-b580.bundle` to the target machine (via scp/sftp/usb), then on the target machine:
```bash
git fetch /path/to/strata-sycl-b580.bundle HEAD:sycl-b580
git checkout sycl-b580
```

### Method C: Git Patch
A cumulative patch is also generated:
```bash
git diff upstream/main..HEAD > strata-sycl-b580.patch
```
On the target machine:
```bash
git apply --check strata-sycl-b580.patch
git apply strata-sycl-b580.patch
```

---

## 5. Status & Next Session Work Plan

### Current Status (Completed & Tested)
- **Commit `9b352762`**: Added SYCL expert cache and Arc B580 optimization guide (`docs/SYCL_B580_EXPERT_CACHE_GUIDE.md`).
- **Commit `2c52ddee`**: Wired up host-mirror table (`table_host_mirror`, `seed_host_mirrors`, `apply_swaps_host`) and ring budget calculation for `STRATA_VERIFY_NO_HOST` in `src/core/sycl/verify.cpp` and `src/program/sycl/generate.cpp`.
- **Commit `55599055`**: Ported Intel Xe2 B70 optimizations and benchmarks (`xmx_prefill_gemm_kernel`, native MMQ SYCL prefill, batched KV/indexer appends, and cache-conditioned routing prior).
- **Commit `3c964b13`**: Upstream Merge v0.2.12 (`146df3c4`), bringing 95 upstream commits (DeepSeek-V4.1-Flash presets, run manifest telemetry, MMQ margin over-read protection, QSA two-pass scorer bounds, prompt chunk split rules, and frontend tool alias extensions).
- **Commit `3537f122`**: SYCL Port of v0.2.12 optimizations:
  - Ordered DMA completion via `raise_flag_b_ordered` in `Verifier` (`verify.hpp`, `verify.cpp`) preventing Flag B from racing ahead of in-flight DMA on empty/lean steps.
  - MMQ Stream-K over-read margin sizing (`mmq_ids_margin.hpp`) in `Prefill` (`prefill.cpp`), allocating sufficient rows to prevent Level Zero GPU page faults.
  - Two-pass selection diagnostic hook (`tp_status.on("SYCL")`) and F16 MoE stubs (`want_dm_f16`, `dm_f16_used`).
  - STRATA_MANIFEST telemetry hooks on ready and request loops in `src/program/sycl/generate.cpp`.
  - STRATA_FORCE_FAULT token-forcing verification hook.
- **Commit `9ed9888b`**: Upstream Merge v0.2.11 (`9f7d977d`), bringing 208 upstream commits, Unsloth v3.1 Qwen3.8-Flash-Next quant definitions, Q5_0 quantization activation sum fix (-16 offset with quantized sum) ported to SYCL kernels, and full conflict resolution in `CMakeLists.txt` and `server_intel.py`.
- **Commit `0962e1bc`**: MoE 16-Lane Subgroup GEMV (`s2_expert_grouped.cpp`, Xe2 SIMD16 subgroup matrix multiplication).
- **Commit `03aaa697`**: Fused Per-Head RMSNorm + RoPE (`src/kernels/sycl/native_rope.cpp`, `rope_parity.cpp`, benchmarks).
- **Commit `a576809e`**: Multi-Token Router (`native_router_top10_multi`, 256 threads / 8 subgroups) and Residency-Biased Routing (`native_route_resident`, `route_resident_k`).
- **Commit `dde46cd1`**: Batched KV/indexer appends (`kv_append_q8_steps`, `kv_append_q4_steps`, `native_qsa_indexer_append_steps`) and cache-conditioned routing prior (`route_prior_top10`).
- **All Host/Emulation & SYCL Native Tests Pass**: Verified 100% pass rate via `./tools/test_sycl.sh cpu` (93/93 passed) and `./tools/build_sycl.sh sycl-test` (36/36 passed).

---

### Next Session: Upstream `v0.2.13` / `v0.2.13.1+` (`13eedf552`) Porting Plan

Upstream `main` has advanced by 379 commits since `v0.2.12` (`146df3c46` -> `13eedf552`, engine `0.1.59`).

#### B580 Safety Guardrails (MUST FOLLOW)
1. **Never overwrite `src/*/sycl/` with upstream `sycl/` DPCT files**:
   - Upstream maintains a separate DPCT-migrated tree under `sycl/`. Our native B580 Xe2 implementation lives under `src/core/sycl/`, `src/kernels/sycl/`, `src/prefill/sycl/`, and `src/program/sycl/`.
2. **Preserve `CMakeLists.txt` Native SYCL Targets During Modular CMake Merge (`067b4dda5`, `8b1667701`)**:
   - Upstream split `CMakeLists.txt` into `cmake/strata_ggml.cmake`, `cmake/strata_gpu_engine.cmake`, and `cmake/backends/{cpu,cuda,hip,sycl,metal}.cmake`, and moved vendored ggml sources into `third_party/strata-ggml/`.
   - When merging `CMakeLists.txt`, ensure our `STRATA_ENABLE_SYCL` target definitions (`strata_sycl_kernels`, `strata`, `sycl-test`, `perf-sycl`) remain intact and link against `third_party/strata-ggml/`.
3. **Verify Every Step on B580 / SYCL Test Suite**:
   - Run `./tools/build_sycl.sh strata && ./tools/build_sycl.sh sycl-test` after each ported commit.

#### Priority 1: Safe Bug Fixes & Memory/CLI Guardrails (Low Risk, High Value for B580)
1. **Additional MMQ Row-ID Over-Read Margins (`697b6d67a`, `14dd746f6`)**:
   - **Files**: `src/prefill/sycl/prefill.cpp`, `src/prefill/sycl/gemm.cpp`
   - **Change**: Size the DeepSeek MTP draft step's MMQ `ident` row-id table (`B * hc` rows) and dense MMQ row-id table with `mmq_ids_table_rows(...)` margin (`include/strata/prefill/mmq_ids_margin.hpp`) so `mul_mat_q` and `mul_mat_q_stream_k_fixup` never read past the end of the scratch allocation.
2. **PLE Direct Probe Buffer Overflow Fix (`dcaf4755c`)**:
   - **File**: `src/program/sycl/generate.cpp`
   - **Change**: In `ple_direct_rows_per_s`, size the probe output buffer as `PLE_N_HEADS * PLE_HEAD_DIM` floats per token instead of hardcoded `2560`, preventing a 24 KB heap overflow on hidden > 2560 models.
3. **CLI `--geometry` and `--families` Support (`77f7835a8`, `febc0226f`)**:
   - **File**: `src/program/sycl/generate.cpp`
   - **Change**: Handle `--geometry` (prints `kGeometryName`, `flash_next` by default) and `--families` in the SYCL `main()` entry point so `strata launch` and control-plane checks can inspect the SYCL binary.
4. **Q5_K / Q6_K Down Expert Format Entries (`cd14b976a`)**:
   - **Files**: `src/kernels/sycl/iq_kernels.cpp`, `src/kernels/sycl/native_grouped_parity.cpp`
   - **Change**: Add format IDs `13` (`Q5_K`) and `14` (`Q6_K`) to `STRATA_D_FMTS` so Unsloth UD packs with Q5_K/Q6_K down projections pass `native_expert_supported` on Intel Arc. Does not alter `IQ3_S` execution paths.
5. **Explicit Error on Failed Free-Memory Query & Parked Conversation Cache Release (`e008709d7`, `8c71c4927`, `feaa928f9`, `28bd0185f`)**:
   - **Files**: `src/core/sycl/expert_cache.cpp`, `src/program/sycl/generate.cpp`, `include/strata/core/conversation_cache.hpp`, `include/strata/core/ple_residency.hpp`
   - **Change**: Report Level Zero free-memory query failures explicitly instead of silently returning `0 free`, release RAM back to the OS when parked conversations are evicted, and apply Linux file-cache credit in `ple_residency.hpp`.

#### Priority 2: Diagnostics & Non-Flash-Next Shape Support (Medium Priority)
1. **FP32 Generic Router Fallback for GPUs without FP64 (`3801ec40a`)**:
   - **Files**: `src/kernels/sycl/router_top10.cpp`, `src/kernels/sycl/router_top10_parity.cpp`
   - **Change**: Query `q->get_device().has(sycl::aspect::fp64)` in the generic router (`router_top10_generic`, used by non-Flash-Next shapes such as top-8 / 256-expert models) and dispatch an FP32 kernel when FP64 is unavailable on Intel Arc. Keep the Flash-Next `top-10` fast path untouched.
2. **Watchdog Prefill Compute Stream Stall Reporting (`711a7e937`)**:
   - **Files**: `include/strata/core/stream_watch.hpp`, `src/prefill/sycl/prefill.cpp`, `src/program/sycl/generate.cpp`
   - **Change**: Register the prefill compute queue with `stream_watch` during prompt processing so `stall_report` logs whether the prefill queue has completed or is still executing if a watchdog timeout occurs.

#### Priority 3: Low Priority / Skip (Do Not Risk B580 Regressions)
1. **oneMKL Skinny BF16 GEMM Workaround (`0f3ae296b`, `646826312`) — LOW PRIORITY / NOT NEEDED**:
   - **Why**: Upstream's `sycl/src/prefill/gemm.dp.cpp` uses `oneapi::mkl::blas::gemm`, which is non-deterministic across launches for skinny shapes ($M \le 64$) on Arc. Our native B580 port (`src/prefill/sycl/gemm.cpp`) **does not use oneMKL at all** — it already uses deterministic native kernels (`bf16_gemv` for $T=1$, Xe2 hardware XMX DPAS `xmx_gemm` / `xmx_prefill_gemm_kernel` for $T>1$, and scalar FMA fallback). Neither `IQ3_S` nor BF16 projections on B580 invoke oneMKL.
2. **Full `kCompiledGeometry` Template Refactor Across All Native SYCL Kernels (`af7759b69` .. `c59866283`) — LOW PRIORITY / CAUTION**:
   - **Why**: Upstream parameterized kernel constants via `kCompiledGeometry` for multi-shape builds (`q4_lite`, `q4_mid`, `q4_big`, `q4_deep`), which stay identical to Flash-Next for the default build. Avoid mass-editing our B580-tuned kernel templates in `src/kernels/sycl/` until multi-shape SYCL builds are explicitly needed, and validate each kernel individually against `sycl-test` and `perf-sycl`.

---

## 6. Daily Cheat Sheet

```bash
# Build Strata binary quietly
./tools/build_sycl.sh strata

# Build benchmarks quietly
./tools/build_sycl.sh perf-sycl

# Run full test suite quietly
./tools/test_sycl.sh

# Run specific test (e.g. router parity)
./build_sycl/router_top10_parity

# Run speculative verify benchmarks
./build_sycl/perf_speculative_verify
```
