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
- **Commit `0962e1bc`**: MoE 16-Lane Subgroup GEMV (`s2_expert_grouped.cpp`, Xe2 SIMD16 subgroup matrix multiplication).
- **Commit `03aaa697`**: Fused Per-Head RMSNorm + RoPE (`src/kernels/sycl/native_rope.cpp`, `rope_parity.cpp`, benchmarks).
- **Commit `a576809e`**: Multi-Token Router (`native_router_top10_multi`, 256 threads / 8 subgroups) and Residency-Biased Routing (`native_route_resident`, `route_resident_k`).
- **All 43 Host/Emulation Tests Pass**: Verified 100% pass rate via `./tools/test_sycl.sh`.

### Next Session Kernel Porting Tasks

#### Priority 5: Batched K/V and Indexer Appends
In speculative verification and multi-token drafting (MTP), tokens are verified in batches ($T = 4$ to $8$). Currently, SYCL executes a sequential `for (int t = tb; t < te; ++t)` loop submitting individual kernel commands.

1. **`src/kernels/sycl/kv_q8.cpp`**:
   - Add template `kv_append_q8_kernel<bool MULTI = false>` supporting `step_stride` and `cur_stride`.
   - Implement `kv_append_q8_steps(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table, const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride, int n_tok, const QsaShapes& s, void* stream, const KvHostPools* host)`.
2. **`src/kernels/sycl/kv_q4.cpp`**:
   - Update `kv_append_q4_kernel` with `step_stride` and `planes` (`planes = 1` for hybrid/folded, `2` for paired K/V).
   - Implement `kv_append_q4_steps(...)`.
3. **`src/kernels/sycl/native_qsa_indexer.cpp`**:
   - Update `append` kernel to loop over `n_steps` with `pos_stride`.
   - Implement `native_qsa_indexer_append_steps(...)`.
4. **Wire into `src/core/sycl/verify.cpp` & `src/core/sycl/mtp.cpp`**:
   - Replace token loops with single batched kernel invocations:
     - `kv_append_q8_steps`
     - `kv_append_q4_steps`
     - `native_qsa_indexer_append_steps`
5. **Parity & Benchmarks**:
   - Add SYCL version of `kv_steps_parity.cpp`.
   - Add benchmark in `perf/sycl/perf_speculative_verify.cpp` comparing batched vs unbatched append times.

#### Priority 6: Cache-Conditioned Routing Prior (`STRATA_ROUTE_PRIOR`)
1. **`src/kernels/sycl/route_prior.cpp`**:
   - Implement `route_prior_top10` (MoCCE arXiv:2412.00099) using SYCL subgroups and warp shuffles.
2. **`src/core/sycl/verify.cpp`**:
   - Wire `route_prior_top10` when `STRATA_ROUTE_PRIOR > 0.0f`.
3. **Register in `CMakeLists.txt`**.

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
