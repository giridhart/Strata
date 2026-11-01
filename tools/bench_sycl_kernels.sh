#!/usr/bin/env bash
# tools/bench_sycl_kernels.sh: Standalone microbenchmark suite for Intel SYCL kernel variants.
# Tests and benchmarks prompt attention, MMVQ decode, and recurrent kernels without running a full model.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build_sycl"

# Source oneAPI environment
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

echo "================================================================================"
echo "          Strata Intel SYCL Standalone Microbenchmark Suite"
echo "================================================================================"
DEVICE_NAME=$(sycl-ls 2>/dev/null | grep -E "level_zero:gpu|opencl:gpu" | head -n 1 || echo "Intel Graphics")
echo "Target Device: ${DEVICE_NAME}"
echo ""

# 1. Build benchmark binaries if needed
echo "[1/5] Ensuring benchmark binaries are up to date..."
make -C "${BUILD_DIR}" qsa_prompt_attn_parity mmvq_bench mmvq_sg_bench fused_gr_bench sycl_model_shapes -j"$(nproc)" >/dev/null 2>&1
echo "Binaries built successfully."
echo ""

# 2. Prompt Attention Benchmarks
echo "================================================================================"
echo "  [2/5] Benchmarking Prompt Attention Variants (qsa_prompt_attn_parity)"
echo "================================================================================"
echo "--- Variant A: Pure SYCL XMX Joint Matrix (STRATA_PROMPT_ATTN_ESIMD=0) ---"
STRATA_PROMPT_ATTN_ESIMD=0 "${BUILD_DIR}/qsa_prompt_attn_parity" 4096 128 2
echo ""
echo "--- Variant B: Xe2 Hardware int8 DPAS (STRATA_PROMPT_ATTN_ESIMD=1, LV=3 Default) ---"
STRATA_PROMPT_ATTN_ESIMD=1 STRATA_ESIMD_LV=3 "${BUILD_DIR}/qsa_prompt_attn_parity" 4096 128 2
echo ""
echo "--- Variant C: Xe2 Hardware int8 DPAS (STRATA_PROMPT_ATTN_ESIMD=1, LV=2 Fast) ---"
STRATA_PROMPT_ATTN_ESIMD=1 STRATA_ESIMD_LV=2 "${BUILD_DIR}/qsa_prompt_attn_parity" 4096 128 2 || true
echo ""

# 3. Dense MMVQ Benchmarks
echo "================================================================================"
echo "  [3/5] Benchmarking Dense Q6_K MMVQ: GGUF vs Wide vs Planar (mmvq_bench)"
echo "================================================================================"
echo "--- Single-Token Decode (ncols=1, 2560x2560) ---"
"${BUILD_DIR}/mmvq_bench" 2560 2560 1
echo ""
echo "--- Speculative Decode / Small Batch (ncols=4, 2560x2560) ---"
"${BUILD_DIR}/mmvq_bench" 2560 2560 4
echo ""

# 4. MMVQ Sub-Group and Register Pressure Benchmarks
echo "================================================================================"
echo "  [4/5] Benchmarking Sub-Group Width & Large GRF (mmvq_sg_bench)"
echo "================================================================================"
echo "--- Mode 1: Standard SIMD32 (STRATA_MMVQ_SG=32) ---"
STRATA_MMVQ_SG=32 "${BUILD_DIR}/mmvq_sg_bench" 100
echo ""
echo "--- Mode 2: Native Xe2 SIMD16 Sub-groups (STRATA_MMVQ_SG=16) ---"
STRATA_MMVQ_SG=16 "${BUILD_DIR}/mmvq_sg_bench" 100
echo ""

# 5. Comprehensive Model Shapes & High-Priority IQ3_S Architecture Test
echo "================================================================================"
echo "  [5/5] Comprehensive Production Model Shapes & IQ3_S Parity (sycl_model_shapes)"
echo "================================================================================"
"${BUILD_DIR}/sycl_model_shapes"
echo ""

echo "================================================================================"
echo "  Microbenchmark Run Complete."
echo "================================================================================"
