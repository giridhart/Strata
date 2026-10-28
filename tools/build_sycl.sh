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
BUILD_TYPE="${BUILD_TYPE:-Release}"
EXTRA_CXX_FLAGS=""
TARGETS=()

print_help() {
    cat <<EOF
Usage: $(basename "$0") [options] [targets...]

Options:
  -r, --reconfigure        Force CMake reconfiguration
  --clean                  Remove build directory before building
  -p, --profile, -g,
  --debug-info             Build with RelWithDebInfo and debug symbols (-gline-tables-only,
                           -fno-omit-frame-pointer) for Intel VTune profiling
  --build-type TYPE        Specify CMake build type (Release, RelWithDebInfo, Debug)
                           (default: Release)
  -h, --help               Show this help message and exit

Environment Variables:
  BUILD_TYPE=...           Set default build type (e.g. RelWithDebInfo)
  PROFILE=1                Equivalent to --profile
  AOT_BMG=1                Enable Ahead-Of-Time compilation for Intel Arc B580 (bmg)
  JOBS=N                   Parallel compilation jobs (default: nproc or 4)

Examples:
  $(basename "$0")                          # Standard optimized Release build
  $(basename "$0") --profile                # Build with debug line tables for VTune
  $(basename "$0") -r verify_parity         # Reconfigure and build verify_parity
EOF
    exit 0
}

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
        -p|--profile|--debug-info|-g)
            BUILD_TYPE="RelWithDebInfo"
            RECONFIGURE=true
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            RECONFIGURE=true
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

if [[ "${PROFILE:-0}" == "1" ]]; then
    BUILD_TYPE="RelWithDebInfo"
fi

if [[ "${BUILD_TYPE}" == "RelWithDebInfo" ]]; then
    # -fno-omit-frame-pointer: full CPU and host worker call-stack unwinding in VTune
    # Note: CMAKE_BUILD_TYPE=RelWithDebInfo natively passes -O2 -g -DNDEBUG to both host and SYCL device
    EXTRA_CXX_FLAGS="-fno-omit-frame-pointer"
fi

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
    echo "[INFO] Configuring CMake in ${BUILD_DIR} (BUILD_TYPE=${BUILD_TYPE}, AOT_BMG=${AOT_BMG:-0}, XMX_SG16=1)..."
    CXX_FLAGS="-DSTRATA_XMX_SG16=1"
    if [[ -n "${EXTRA_CXX_FLAGS}" ]]; then
        CXX_FLAGS="${CXX_FLAGS} ${EXTRA_CXX_FLAGS}"
    fi

    cmake -B "${BUILD_DIR}" -S "${REPO_DIR}" \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DSTRATA_ENABLE_SYCL=ON \
        -DSTRATA_TARGET_XE2=ON \
        -DSTRATA_XMX_SG16=ON \
        ${CMAKE_AOT_FLAG} \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
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
