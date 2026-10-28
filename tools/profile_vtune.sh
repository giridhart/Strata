#!/usr/bin/env bash
# ==============================================================================
# profile_vtune.sh - Profile Strata Inference with Intel VTune Profiler
#
# Prompt: "write a cpp program to compute Nth fibonacci number"
#
# This script configures the Intel oneAPI environment, launches Strata,
# runs the prompt through Strata CLI / API, and captures profiling data
# (GPU Hotspots, CPU Hotspots, or Level Zero tracing) using Intel VTune.
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration & Defaults
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prompt requested by user
PROMPT="write a cpp program to compute Nth fibonacci number"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# VTune settings
ANALYSIS_TYPE="${ANALYSIS_TYPE:-gpu-hotspots}"   # gpu-hotspots, hotspots, xpu-offload, performance-snapshot
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/vtune_profiles/strata_fibonacci_${ANALYSIS_TYPE}_${TIMESTAMP}}"
DATA_LIMIT="${DATA_LIMIT:-1000}"                 # Limit in MB (0 = unlimited)

# Strata server settings
SERVER_PORT="${SERVER_PORT:-8080}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-q2_0}"

# Execution mode: "api" (via Strata server/client) or "engine" (direct SYCL binary)
RUN_MODE="${RUN_MODE:-api}"

# ------------------------------------------------------------------------------
# Step 1: Initialize Intel oneAPI and Python Environment
# ------------------------------------------------------------------------------
echo "=== [1/5] Initializing oneAPI & Python Environment ==="

ONEAPI_SETVARS="/opt/intel/oneapi/setvars.sh"
if [[ -f "${ONEAPI_SETVARS}" ]]; then
    echo "[oneAPI] Sourcing ${ONEAPI_SETVARS}..."
    # shellcheck disable=SC1090
    source "${ONEAPI_SETVARS}" --force >/dev/null 2>&1 || true
else
    echo "[oneAPI] Note: ${ONEAPI_SETVARS} not found, relying on current PATH."
fi

# Ensure Python virtual environment if present
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "[Python] Activating .venv from ${REPO_ROOT}/.venv..."
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Verify VTune availability
VTUNE_BIN="$(which vtune 2>/dev/null || true)"
if [[ -z "${VTUNE_BIN}" ]]; then
    if [[ -x "/opt/intel/oneapi/vtune/latest/bin64/vtune" ]]; then
        VTUNE_BIN="/opt/intel/oneapi/vtune/latest/bin64/vtune"
    else
        echo "[-] Error: 'vtune' not found in PATH or /opt/intel/oneapi/vtune."
        echo "    Ensure Intel VTune Profiler is installed and setvars.sh is sourced."
        exit 1
    fi
fi
echo "[VTune] Found VTune at: ${VTUNE_BIN}"
"${VTUNE_BIN}" --version | head -n 2

# Set GPU execution environment for Intel Arc (Level Zero)
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_ENABLE_TRACING="${ZE_ENABLE_TRACING:-1}"
export STRATA_VERIFY_GRAPH="${STRATA_VERIFY_GRAPH:-1}"

# Check for WSL2 environment virtualization note
if grep -qi "microsoft" /proc/version 2>/dev/null; then
    IS_WSL2=1
    echo "[Environment] Detected WSL2 virtualized Linux environment."
else
    IS_WSL2=0
fi

# ------------------------------------------------------------------------------
# Step 2: Prepare Profiling Destination
# ------------------------------------------------------------------------------
echo "=== [2/5] Profiling Destination ==="
mkdir -p "$(dirname "${RESULT_DIR}")"
echo "[Output] Target result directory: ${RESULT_DIR}"
echo "[Config] Analysis Type: ${ANALYSIS_TYPE}"
echo "[Config] Prompt: \"${PROMPT}\""

# ------------------------------------------------------------------------------
# Step 3: Check / Start Strata
# ------------------------------------------------------------------------------
echo "=== [3/5] Checking Strata Inference Service ==="

SERVER_PID=""
STARTED_OURSELVES=0

is_server_ready() {
    curl -s --connect-timeout 2 "${SERVER_URL}/v1/models" >/dev/null 2>&1
}

if [[ "${RUN_MODE}" == "api" ]]; then
    if is_server_ready; then
        echo "[Strata] Existing Strata server detected responding on ${SERVER_URL}."
    else
        echo "[Strata] No running server found on ${SERVER_URL}."
        echo "[Strata] Launching Strata server in background..."
        
        # Launch server in background
        python3 "${REPO_ROOT}/strata.py" serve \
            --port "${SERVER_PORT}" \
            >/tmp/strata_vtune_server.log 2>&1 &
        SERVER_PID=$!
        STARTED_OURSELVES=1
        
        echo "[Strata] Waiting for server to initialize (PID: ${SERVER_PID})..."
        WAIT_SECONDS=0
        MAX_WAIT=60
        until is_server_ready; do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                echo "[-] Strata server failed to start. Logs:"
                cat /tmp/strata_vtune_server.log | tail -n 20
                exit 1
            fi
            sleep 1
            WAIT_SECONDS=$((WAIT_SECONDS + 1))
            if [[ ${WAIT_SECONDS} -ge ${MAX_WAIT} ]]; then
                echo "[-] Timeout waiting for Strata server on ${SERVER_URL}."
                exit 1
            fi
        done
        echo "[Strata] Server ready at ${SERVER_URL}."
    fi
fi

# Cleanup handler
cleanup() {
    if [[ "${STARTED_OURSELVES}" -eq 1 && -n "${SERVER_PID}" ]]; then
        echo "[Cleanup] Stopping background Strata server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# Step 4: Run VTune Profiling
# ------------------------------------------------------------------------------
echo "=== [4/5] Executing Prompt and Capturing VTune Profile ==="

# Locate active engine or server process PID
TARGET_PID=""
if [[ -n "${SERVER_PID}" ]]; then
    # Look for child engine process if spawned, otherwise the server PID
    CHILD_ENGINE_PID="$(pgrep -P "${SERVER_PID}" -f "strata|generate" 2>/dev/null | head -n 1 || true)"
    TARGET_PID="${CHILD_ENGINE_PID:-${SERVER_PID}}"
else
    # Auto-detect running strata engine or python server
    TARGET_PID="$(pgrep -f "build_sycl/strata|serve/server.py|strata.py serve" | head -n 1 || true)"
fi

echo "[Profiling] Target Process PID: ${TARGET_PID:-N/A}"

# Helper function to send the inference prompt
send_prompt_request() {
    echo "[Prompt Request] Sending: \"${PROMPT}\" to ${SERVER_URL}/v1/chat/completions..."
    
    # Generate JSON payload
    local payload
    payload="$(cat <<EOF
{
  "model": "strata",
  "messages": [
    {
      "role": "user",
      "content": "${PROMPT}"
    }
  ],
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMPERATURE},
  "stream": true
}
EOF
)"

    echo "--- Model Output Stream ---"
    local t0 t1 elapsed
    t0=$(date +%s%N)
    
    curl -sN -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "${payload}" | while IFS= read -r line; do
            if [[ "${line}" =~ ^data:\ (.*) ]]; then
                chunk="${BASH_REMATCH[1]}"
                if [[ "${chunk}" != "[DONE]" ]]; then
                    # Extract delta content using python
                    content="$(python3 -c "import json,sys; obj=json.loads(sys.argv[1]); delta=obj.get('choices',[{}])[0].get('delta',{}); print(delta.get('content',''), end='')" "${chunk}" 2>/dev/null || true)"
                    printf "%s" "${content}"
                fi
            fi
        done
    echo ""
    echo "--- End Model Output ---"
    t1=$(date +%s%N)
    elapsed=$(python3 -c "print(f'{(${t1} - ${t0}) / 1e9:.3f}')")
    echo "[Timing] Request completed in ${elapsed} s."
}

# Profiling Execution
if [[ -n "${TARGET_PID}" ]]; then
    echo "[VTune] Starting VTune collection attached to PID ${TARGET_PID}..."
    
    # Run vtune in background attached to target PID
    "${VTUNE_BIN}" -collect "${ANALYSIS_TYPE}" \
        -result-dir "${RESULT_DIR}" \
        -data-limit "${DATA_LIMIT}" \
        -target-pid "${TARGET_PID}" >/tmp/vtune_collector.log 2>&1 &
    VTUNE_PID=$!
    
    # Give collector a moment to initialize probes
    sleep 2
    
    # Execute the user prompt
    send_prompt_request
    
    # Tell VTune to stop collection
    echo "[VTune] Stopping VTune collection..."
    "${VTUNE_BIN}" -r "${RESULT_DIR}" -command stop >/dev/null 2>&1 || true
    wait "${VTUNE_PID}" 2>/dev/null || true
else
    echo "[VTune] Wrapping prompt generation script under VTune..."
    
    # Create self-contained runner script
    RUNNER_SCRIPT="/tmp/run_prompt_${TIMESTAMP}.sh"
    cat <<EOF >"${RUNNER_SCRIPT}"
#!/usr/bin/env bash
curl -sN -X POST "${SERVER_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"strata","messages":[{"role":"user","content":"${PROMPT}"}],"max_tokens":${MAX_TOKENS},"temperature":${TEMPERATURE},"stream":true}'
EOF
    chmod +x "${RUNNER_SCRIPT}"

    "${VTUNE_BIN}" -collect "${ANALYSIS_TYPE}" \
        -result-dir "${RESULT_DIR}" \
        -data-limit "${DATA_LIMIT}" \
        -- "${RUNNER_SCRIPT}" || {
            echo "[-] VTune collection reported non-zero status."
            if [[ ${IS_WSL2} -eq 1 ]]; then
                echo "[WSL2 Diagnostic] VTune hardware sampling drivers (SEP) require bare-metal PMU access."
                echo "                  In WSL2, use User-Mode Sampling, Intel PTI tracing, or VTune on Windows host."
            fi
        }
    rm -f "${RUNNER_SCRIPT}"
fi

# ------------------------------------------------------------------------------
# Step 5: Summary Report & Inspection
# ------------------------------------------------------------------------------
echo "=== [5/5] Profiling Results ==="

if [[ -d "${RESULT_DIR}" ]]; then
    echo "[Success] VTune profiling results saved to: ${RESULT_DIR}"
    
    # Attempt to print VTune summary report
    echo "--- VTune Summary ---"
    "${VTUNE_BIN}" -report summary -result-dir "${RESULT_DIR}" 2>/dev/null || true
    
    echo ""
    echo "To view detailed results interactively in VTune GUI:"
    echo "  vtune-gui ${RESULT_DIR}"
else
    echo "[-] VTune result directory was not created. Check /tmp/vtune_collector.log:"
    tail -n 25 /tmp/vtune_collector.log 2>/dev/null || true
fi

echo "=== Done ==="
