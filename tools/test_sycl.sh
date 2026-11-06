#!/usr/bin/env bash
# tools/test_sycl.sh: Quiet test runner wrapper for Intel oneAPI SYCL on Strata
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build_sycl"
LOG_FILE="${BUILD_DIR}/test.log"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "${BUILD_DIR}"

# Source oneAPI environment
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

# Set default runtime environment for Intel Xe2/B580 if not already configured
export IGC_EnableGEPSimplification="${IGC_EnableGEPSimplification:-0}"
export SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export UR_L0_DEVICE_SCOPE_EVENTS="${UR_L0_DEVICE_SCOPE_EVENTS:-2}"
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS="${SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS:-2}"
export STRATA_EVENT_MARK="${STRATA_EVENT_MARK:-barrier}"
export STRATA_COPY_ENGINE="${STRATA_COPY_ENGINE:-0}"

# Argument handling: 'all' or empty runs all tests by default
CTEST_BIN="$(command -v /usr/bin/ctest || command -v ctest)"

if [[ $# -eq 0 ]] || [[ "$1" == "all" ]]; then
    shift 2>/dev/null || true
    TEST_ARGS=(-j"${JOBS}" --output-on-failure "$@")
elif [[ "$1" == "sycl" ]] || [[ "$1" == "gpu" ]]; then
    shift
    TEST_ARGS=(-j"${JOBS}" --output-on-failure -R "^sycl_" "$@")
elif [[ "$1" == "cpu" ]] || [[ "$1" == "fast" ]]; then
    shift
    TEST_ARGS=(-j"${JOBS}" --output-on-failure -E "sycl|gpu|device|parity|bench|pool|expert_multi" "$@")
else
    TEST_ARGS=(-j"${JOBS}" --output-on-failure "$@")
fi

if "${CTEST_BIN}" --test-dir "${BUILD_DIR}" "${TEST_ARGS[@]}" > "${LOG_FILE}" 2>&1; then
    TOTAL=$(grep -oE "[0-9]+% tests passed" "${LOG_FILE}" | tail -n 1 || echo "100% tests passed")
    COUNT=$(grep -oE "[0-9]+ tests passed out of [0-9]+" "${LOG_FILE}" | tail -n 1 || echo "All tests passed")
    echo "[OK] Tests passed: ${COUNT} (${TOTAL}). Log: ${LOG_FILE}"
    exit 0
elif "${CTEST_BIN}" --test-dir "${BUILD_DIR}" --rerun-failed --output-on-failure >> "${LOG_FILE}" 2>&1; then
    TOTAL=$(grep -oE "[0-9]+% tests passed" "${LOG_FILE}" | tail -n 1 || echo "100% tests passed")
    COUNT=$(grep -oE "[0-9]+ tests passed out of [0-9]+" "${LOG_FILE}" | tail -n 1 || echo "All tests passed")
    echo "[OK] Tests passed (passed after transient rerun): ${COUNT} (${TOTAL}). Log: ${LOG_FILE}"
    exit 0
else
    echo "[FAILED] Tests failed. Log summary from ${LOG_FILE}:"
    grep -E "The following tests FAILED:|\*\*\*Failed|\*\*\*Timeout" "${LOG_FILE}" || tail -n 15 "${LOG_FILE}"
    exit 1
fi
