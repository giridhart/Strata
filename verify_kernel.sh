#!/bin/bash
# Usage: ./verify_kernel.sh <kernel_name_without_extension>
# Example: ./verify_kernel.sh ple

TARGET=$1
FILE_PATH="src/kernels/sycl/${TARGET}.cpp"

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh >/dev/null 2>&1
source /opt/intel/oneapi/mkl/2025.3/env/vars.sh >/dev/null 2>&1

if [ -z "$TARGET" ]; then
  echo "[ERROR] Provide a kernel target name."
  exit 1
fi

echo "--- [1/2] BUILDING: $TARGET ---"
# Pipe stderr to stdout and capture it. Use ninja or make depending on your setup.
BUILD_OUT=$(make ${TARGET}_parity 2>&1)
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
  echo "[BUILD_FAIL] Compilation failed for $TARGET."
  echo "--- TRUNCATED COMPILER ERRORS (Top 20 lines) ---"
  echo "$BUILD_OUT" | grep -A 2 -E "error:" | head -n 30
  if ! echo "$BUILD_OUT" | grep -q "error:"; then
    echo "$BUILD_OUT" | tail -n 25
  fi

  echo "--- REVERTING $FILE_PATH ---"
  TS=$(date +%Y%m%d_%H%M%S) && cp "$FILE_PATH" "${FILE_PATH}.failed.${TS}" && git checkout -- "$FILE_PATH"
  exit 1
fi

echo "--- [2/2] TESTING: ${TARGET}_parity ---"
# Assuming the built parity binary goes to build_sycl/ or bin/
TEST_OUT=$(./build_sycl/${TARGET}_parity 2>&1)
TEST_STATUS=$?

if [ $TEST_STATUS -ne 0 ]; then
  echo "[PARITY_FAIL] Math mismatch or runtime crash for $TARGET."
  echo "--- TRUNCATED TEST OUTPUT (First failure) ---"
  echo "$TEST_OUT" | head -n 15

  echo "--- REVERTING $FILE_PATH ---"
  TS=$(date +%Y%m%d_%H%M%S) && cp "$FILE_PATH" "${FILE_PATH}.failed.${TS}" && git checkout -- "$FILE_PATH"
  exit 2
fi

echo "[SUCCESS] $TARGET compiled and passed parity. Changes kept."
exit 0
