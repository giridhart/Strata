#!/usr/bin/env bash
# tools/build_sycl.sh: Quiet build wrapper for Intel oneAPI SYCL on Strata
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build_sycl"
LOG_FILE="${BUILD_DIR}/build.log"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Source oneAPI environment if available
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

RECONFIGURE=false
TARGETS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--reconfigure)
            RECONFIGURE=true
            shift
            ;;
        --clean)
            echo "[INFO] Cleaning ${BUILD_DIR}..."
            rm -rf "${BUILD_DIR}"
            RECONFIGURE=true
            shift
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("strata")
fi

mkdir -p "${BUILD_DIR}"

# Ahead-Of-Time (AOT) compilation flag for Battlemage (B580)
# Default is JIT (SPIR-V), which is portable across Xe2 devices (B580, Lunar Lake, etc.).
# Set AOT_BMG=1 to compile directly to BMG machine code (eliminates JIT startup time).
CMAKE_AOT_FLAG="-DSTRATA_AOT_BMG=OFF"
if [[ "${AOT_BMG:-0}" == "1" ]]; then
    CMAKE_AOT_FLAG="-DSTRATA_AOT_BMG=ON"
fi

# Ensure CMake is configured (or reconfigured if requested)
if [[ ! -f "${BUILD_DIR}/Makefile" || "${RECONFIGURE}" == "true" ]]; then
    echo "[INFO] Configuring CMake in ${BUILD_DIR} (AOT_BMG=${AOT_BMG:-0}, XMX_SG16=1)..."
    cmake -B "${BUILD_DIR}" -S "${REPO_DIR}" \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_SYCL=ON \
        -DSTRATA_TARGET_XE2=ON \
        -DSTRATA_XMX_SG16=ON \
        ${CMAKE_AOT_FLAG} \
        -DCMAKE_CXX_FLAGS="-DSTRATA_XMX_SG16=1" \
        -DSTRATA_BUILD_PERF=ON \
        -DSTRATA_BUILD_TESTS=ON > "${LOG_FILE}" 2>&1 || {
            echo "[FAILED] CMake configure failed. Log excerpt:"
            tail -n 15 "${LOG_FILE}"
            exit 1
        }
fi

echo "[INFO] Building targets: ${TARGETS[*]} with ${JOBS} jobs..."
if make -C "${BUILD_DIR}" -j"${JOBS}" "${TARGETS[@]}" >> "${LOG_FILE}" 2>&1; then
    WARN_COUNT=$(grep -ci "warning:" "${LOG_FILE}" || true)
    echo "[OK] Build succeeded: targets '${TARGETS[*]}' (warnings: ${WARN_COUNT}, log: ${LOG_FILE})"
    exit 0
else
    echo "[FAILED] Build failed for targets '${TARGETS[*]}'. Compiler error excerpt:"
    grep -E "error:|Error [0-9]+" "${LOG_FILE}" | head -n 10 || tail -n 15 "${LOG_FILE}"
    exit 1
fi
