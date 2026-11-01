#!/usr/bin/env bash
# tools/check_b580_grf.sh - Empirical A/B testing of 128-GRF vs 256-GRF on Intel Arc Battlemage (B580/Xe2)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_sycl"

echo "================================================================="
echo "  Intel Arc Xe2 / B580 MMVQ Decode (ncols=1) GRF A/B Benchmark   "
echo "================================================================="

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    set +u +e
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
    set -eu
fi

if [ ! -d "${BUILD_DIR}" ] || [ ! -x "${BUILD_DIR}/mmvq_bench" ]; then
    echo "Error: ${BUILD_DIR}/mmvq_bench not found. Build with build_sycl first."
    exit 1
fi

DUMP_IGC=0
DUMP_DIR="/tmp/igc_grf_dump_$$"
if [ "${1:-}" == "--dump-igc" ]; then
    DUMP_IGC=1
    mkdir -p "${DUMP_DIR}"
    echo "IGC Shader dump enabled -> ${DUMP_DIR}"
fi

echo ""
echo "--- 1. Testing Square Hidden Projection (2560 x 2560, ncols=1) ---"
if [ "${DUMP_IGC}" -eq 1 ]; then
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="${DUMP_DIR}" "${BUILD_DIR}/mmvq_bench" 2560 2560 1
else
    "${BUILD_DIR}/mmvq_bench" 2560 2560 1
fi

echo ""
echo "--- 2. Testing MLP Expansion (2560 x 10240, ncols=1) ---"
"${BUILD_DIR}/mmvq_bench" 2560 10240 1

echo ""
echo "--- 3. Testing Attention Projection (6144 x 2560, ncols=1) ---"
"${BUILD_DIR}/mmvq_bench" 6144 2560 1

echo ""
echo "--- 4. Full Shape & Quantization Suite (IQ3_S, Q6_K, Q5_K, Q4_K, IQ4_XS) ---"
"${BUILD_DIR}/mmvq_sg_bench" 150

if [ "${DUMP_IGC}" -eq 1 ]; then
    echo ""
    echo "--- IGC Register Spill & GRF Allocation Summary ---"
    grep -rn "Total GRF\|Spill\|Fill" "${DUMP_DIR}" || echo "No spill info found in ${DUMP_DIR}"
fi

echo ""
echo "================================================================="
echo "  Benchmark complete.                                            "
echo "================================================================="
