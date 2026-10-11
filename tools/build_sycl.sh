#!/usr/bin/env bash
# tools/build_sycl.sh: Quiet build wrapper for Intel oneAPI SYCL on Strata
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build_sycl"
TARGET="${1:-strata}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
LOG_FILE="${BUILD_DIR}/build.log"

mkdir -p "${BUILD_DIR}"

# Source oneAPI environment if available
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

# Ensure CMake is configured
if [[ ! -f "${BUILD_DIR}/Makefile" ]]; then
    echo "[INFO] Configuring CMake in ${BUILD_DIR}..." >> "${LOG_FILE}"
    cmake -B "${BUILD_DIR}" -S "${REPO_DIR}" \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_SYCL=ON \
        -DSTRATA_TARGET_XE2=ON \
        -DSTRATA_BUILD_PERF=ON \
        -DSTRATA_BUILD_TESTS=ON > "${LOG_FILE}" 2>&1 || {
            echo "[FAILED] CMake configure failed. Log excerpt:"
            tail -n 15 "${LOG_FILE}"
            exit 1
        }
fi

if [[ $# -eq 0 ]]; then
    TARGETS=("strata")
elif [[ "$1" == "all" ]]; then
    TARGETS=()
else
    TARGETS=("$@")
fi

if make -C "${BUILD_DIR}" -j"${JOBS}" "${TARGETS[@]}" > "${LOG_FILE}" 2>&1; then
    WARN_COUNT=$(grep -ci "warning:" "${LOG_FILE}" || true)
    echo "[OK] Build succeeded: targets '${TARGETS[*]:-all}' (warnings: ${WARN_COUNT}, log: ${LOG_FILE})"
    exit 0
else
    echo "[FAILED] Build failed for targets '${TARGETS[*]:-all}'. Compiler error excerpt:"
    grep -E "error:|Error [0-9]+" "${LOG_FILE}" | head -n 10 || tail -n 15 "${LOG_FILE}"
    exit 1
fi
