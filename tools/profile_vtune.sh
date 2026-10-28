#!/usr/bin/env bash
# ==============================================================================
# profile_vtune.sh - Profile Strata Inference with Intel VTune Profiler
#
# Designed for bare-metal Linux machines equipped with Intel Arc / Xe2 GPUs.
# Profiles prompt prefill and token generation with Intel VTune Profiler
# without polluting traces with model loading / startup disk I/O.
#
# Default Prompt: "write a cpp program to compute Nth fibonacci number"
#
# Usage:
#   # 1. Profile using default settings (GPU Hotspots on prompt):
#   ./tools/profile_vtune.sh
#
#   # 2. Profile an already-running Strata server:
#   ./tools/profile_vtune.sh --url http://127.0.0.1:8080
#
#   # 3. Launch with a specific model family / path:
#   ./tools/profile_vtune.sh --family qwen --model q2_0
#   ./tools/profile_vtune.sh --model-path /path/to/model.gguf
#
#   # 4. Run CPU Hotspots or XPU Offload analysis:
#   ./tools/profile_vtune.sh --analysis hotspots
#   ./tools/profile_vtune.sh --analysis xpu-offload
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default Settings
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROMPT="write a cpp program to compute Nth fibonacci number"
MAX_TOKENS=256
TEMPERATURE=0.0
ANALYSIS_TYPE="gpu-hotspots"  # gpu-hotspots, hotspots, xpu-offload, performance-snapshot
DATA_LIMIT=1000               # Result size limit in MB (0 = unlimited)

SERVER_HOST="127.0.0.1"
SERVER_PORT=8080
SERVER_URL=""
CUSTOM_OUT_DIR=""
MODEL_FAMILY=""
MODEL_SIZE=""
MODEL_PATH=""
EXTRA_SERVER_ARGS=""

# ------------------------------------------------------------------------------
# CLI Argument Parsing
# ------------------------------------------------------------------------------
print_help() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -p, --prompt TEXT        Inference prompt to profile
                           (default: "${PROMPT}")
  -a, --analysis TYPE      VTune analysis type: gpu-hotspots, hotspots,
                           xpu-offload, performance-snapshot (default: gpu-hotspots)
  -u, --url URL            URL of an already-running Strata server (skips server launch)
  -m, --model NAME         Model size/preset (e.g. q2_0, iq3_xxs)
  -f, --family NAME        Model family (e.g. qwen, deepseek41)
      --model-path PATH    Path to local model weights (directory of shards or .gguf)
      --port PORT          Port for Strata server if launched by this script (default: 8080)
      --max-tokens N       Maximum new tokens to generate (default: 256)
      --temperature T      Sampling temperature (default: 0.0)
  -o, --out-dir DIR        Custom output directory for VTune results
  -h, --help               Show this help message and exit

Examples:
  $(basename "$0")
  $(basename "$0") --analysis hotspots --max-tokens 512
  $(basename "$0") --url http://127.0.0.1:8080
  $(basename "$0") --model-path /data/models/qwen --analysis gpu-hotspots
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -a|--analysis)
            ANALYSIS_TYPE="$2"
            shift 2
            ;;
        -u|--url)
            SERVER_URL="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -f|--family)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -o|--out-dir)
            CUSTOM_OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage details." >&2
            exit 1
            ;;
    esac
done

if [[ -z "${SERVER_URL}" ]]; then
    SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
fi

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
RESULT_DIR="${CUSTOM_OUT_DIR:-${REPO_ROOT}/vtune_profiles/strata_fibonacci_${ANALYSIS_TYPE}_${TIMESTAMP}}"

echo "========================================================================"
echo "           Strata Inference Profiling with Intel VTune                  "
echo "========================================================================"
echo "Target Prompt   : \"${PROMPT}\""
echo "Analysis Type   : ${ANALYSIS_TYPE}"
echo "Server Endpoint : ${SERVER_URL}"
echo "Result Directory: ${RESULT_DIR}"
echo "========================================================================"

# ------------------------------------------------------------------------------
# 1. Environment & Preflight Checks
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/5] Preflight & Environment Configuration..."

# 1.1 Source oneAPI setvars if present
ONEAPI_SETVARS="${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh"
if [[ -f "${ONEAPI_SETVARS}" ]]; then
    echo "  [oneAPI] Sourcing ${ONEAPI_SETVARS}..."
    # shellcheck disable=SC1090
    source "${ONEAPI_SETVARS}" --force >/dev/null 2>&1 || true
else
    echo "  [oneAPI] Info: ${ONEAPI_SETVARS} not found, relying on current PATH."
fi

# 1.2 Locate VTune binary
VTUNE_BIN="$(which vtune 2>/dev/null || true)"
if [[ -z "${VTUNE_BIN}" ]]; then
    for candidate in \
        "/opt/intel/oneapi/vtune/latest/bin64/vtune" \
        "/opt/intel/oneapi/vtune/2026.6/bin64/vtune" \
        "/opt/intel/oneapi/vtune/2025.0/bin64/vtune"; do
        if [[ -x "${candidate}" ]]; then
            VTUNE_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${VTUNE_BIN}" ]]; then
    echo "  [-] Error: Intel VTune binary ('vtune') not found." >&2
    echo "      Install Intel oneAPI Base Toolkit or standalone VTune Profiler." >&2
    exit 1
fi
echo "  [VTune] Using binary: ${VTUNE_BIN}"
"${VTUNE_BIN}" --version | head -n 1

# 1.3 Linux Kernel Profiling Permissions (for bare-metal execution)
if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
    PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
    if [[ "${PARANOID}" -gt 0 ]]; then
        echo "  [Warning] /proc/sys/kernel/perf_event_paranoid is currently ${PARANOID}."
        echo "            For hardware event-based profiling, run on target machine:"
        echo "              sudo sysctl -w kernel.perf_event_paranoid=0"
    else
        echo "  [Kernel] perf_event_paranoid is set to ${PARANOID} (OK)."
    fi
fi

if [[ -f /proc/sys/dev/i915/perf_stream_paranoid ]]; then
    I915_PARANOID="$(cat /proc/sys/dev/i915/perf_stream_paranoid)"
    if [[ "${I915_PARANOID}" -ne 0 ]]; then
        echo "  [Warning] /proc/sys/dev/i915/perf_stream_paranoid is ${I915_PARANOID}."
        echo "            For Intel GPU metrics, run:"
        echo "              sudo sysctl -w dev.i915.perf_stream_paranoid=0"
    fi
fi

if [[ -f /proc/sys/dev/xe/observation_paranoid ]]; then
    XE_PARANOID="$(cat /proc/sys/dev/xe/observation_paranoid)"
    if [[ "${XE_PARANOID}" -ne 0 ]]; then
        echo "  [Warning] /proc/sys/dev/xe/observation_paranoid is ${XE_PARANOID}."
        echo "            For Intel Xe2 GPU metrics, run:"
        echo "              sudo sysctl -w dev.xe.observation_paranoid=0"
    fi
fi

# 1.4 GPU & Graph Environment
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_ENABLE_TRACING="${ZE_ENABLE_TRACING:-1}"
export STRATA_VERIFY_GRAPH="${STRATA_VERIFY_GRAPH:-1}"

# 1.5 Activate local Python virtualenv if available
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "  [Python] Activating virtual environment: ${REPO_ROOT}/.venv"
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# ------------------------------------------------------------------------------
# 2. Check or Launch Strata Inference Server
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/5] Connecting to Strata Server..."

SERVER_PID=""
STARTED_SERVER=0

is_server_ready() {
    curl -s --connect-timeout 2 "${SERVER_URL}/v1/models" >/dev/null 2>&1
}

cleanup() {
    local sig="${1:-EXIT}"
    if [[ "${STARTED_SERVER}" -eq 1 && -n "${SERVER_PID}" ]]; then
        echo ""
        echo "[Cleanup] Shutting down Strata server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap 'cleanup INT' INT
trap 'cleanup TERM' TERM
trap 'cleanup EXIT' EXIT

if is_server_ready; then
    echo "  [Strata] Active server detected at ${SERVER_URL}."
else
    echo "  [Strata] No running server found at ${SERVER_URL}."
    echo "  [Strata] Launching new Strata instance..."

    LAUNCH_ARGS=("--port" "${SERVER_PORT}" "--no-tui")
    if [[ -n "${MODEL_PATH}" ]]; then
        LAUNCH_ARGS+=("--model-path" "${MODEL_PATH}")
    fi
    if [[ -n "${MODEL_FAMILY}" ]]; then
        LAUNCH_ARGS+=("--family" "${MODEL_FAMILY}")
    fi
    if [[ -n "${MODEL_SIZE}" ]]; then
        LAUNCH_ARGS+=("--model" "${MODEL_SIZE}")
    fi

    # Start server with strata.py launch or serve
    if [[ -f "${REPO_ROOT}/strata.py" ]]; then
        python3 "${REPO_ROOT}/strata.py" launch "${LAUNCH_ARGS[@]}" \
            >/tmp/strata_vtune_server.log 2>&1 &
        SERVER_PID=$!
        STARTED_SERVER=1
    else
        echo "  [-] Error: strata.py not found in ${REPO_ROOT}." >&2
        exit 1
    fi

    echo "  [Strata] Server starting in background (PID: ${SERVER_PID}). Waiting for ready..."
    WAIT_COUNT=0
    MAX_WAIT=120
    until is_server_ready; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "  [-] Error: Strata server terminated unexpectedly. Log output:" >&2
            cat /tmp/strata_vtune_server.log | tail -n 25 >&2
            exit 1
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [[ ${WAIT_COUNT} -ge ${MAX_WAIT} ]]; then
            echo "  [-] Error: Timed out waiting for Strata server on ${SERVER_URL}." >&2
            exit 1
        fi
    done
    echo "  [Strata] Server is online and ready for inference."
fi

# ------------------------------------------------------------------------------
# 3. Locate Target Engine Process for Precision Profiling
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/5] Locating Active Engine Process for Precision Attachment..."

# By attaching to the engine right before sending the prompt and detaching
# immediately when generation ends, we isolate 100% of the profile to inference
# rather than polluting the trace with initial model loading / memory mapping.
TARGET_PID=""
for candidate in \
    $(pgrep -f "build_sycl/strata" 2>/dev/null || true) \
    $(pgrep -f "strata_engine" 2>/dev/null || true) \
    $(pgrep -f "serve/server.py" 2>/dev/null || true); do
    if kill -0 "${candidate}" 2>/dev/null; then
        TARGET_PID="${candidate}"
        break
    fi
done

if [[ -z "${TARGET_PID}" && -n "${SERVER_PID}" ]]; then
    TARGET_PID="${SERVER_PID}"
fi

echo "  [Target] Selected Process PID for profiling: ${TARGET_PID:-None (Wrapper mode)}"

# ------------------------------------------------------------------------------
# 4. Run VTune Collection During Prompt Inference
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/5] Executing Prompt & Capturing VTune Trace..."
mkdir -p "$(dirname "${RESULT_DIR}")"

run_prompt_inference() {
    echo ""
    echo "--------------------------- Model Generation ---------------------------"
    local t0 t1 elapsed
    t0=$(date +%s%N)

    # Send request and stream tokens
    local payload
    payload=$(python3 -c "import json, sys; print(json.dumps({'model': 'strata', 'messages': [{'role': 'user', 'content': sys.argv[1]}], 'max_tokens': int(sys.argv[2]), 'temperature': float(sys.argv[3]), 'stream': True}))" "${PROMPT}" "${MAX_TOKENS}" "${TEMPERATURE}")

    curl -sN -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "${payload}" | while IFS= read -r line; do
            if [[ "${line}" =~ ^data:\ (.*) ]]; then
                chunk="${BASH_REMATCH[1]}"
                if [[ "${chunk}" != "[DONE]" ]]; then
                    content="$(python3 -c "import json, sys; obj=json.loads(sys.argv[1]); print(obj.get('choices',[{}])[0].get('delta',{}).get('content',''), end='')" "${chunk}" 2>/dev/null || true)"
                    printf "%s" "${content}"
                fi
            fi
        done

    echo ""
    echo "------------------------- End Model Generation -------------------------"
    t1=$(date +%s%N)
    elapsed=$(python3 -c "print(f'{(${t1} - ${t0}) / 1e9:.3f}')")
    echo "  [Timing] Inference duration: ${elapsed} seconds."
}

if [[ -n "${TARGET_PID}" ]]; then
    echo "  [VTune] Attaching collector to PID ${TARGET_PID}..."
    "${VTUNE_BIN}" -collect "${ANALYSIS_TYPE}" \
        -result-dir "${RESULT_DIR}" \
        -data-limit "${DATA_LIMIT}" \
        -target-pid "${TARGET_PID}" >/tmp/vtune_collector.log 2>&1 &
    VTUNE_PID=$!

    # Wait briefly for collector initialization
    sleep 2

    # Execute the requested prompt
    run_prompt_inference

    # Cleanly stop data collection
    echo "  [VTune] Generation finished. Stopping data collection..."
    "${VTUNE_BIN}" -r "${RESULT_DIR}" -command stop >/dev/null 2>&1 || true
    wait "${VTUNE_PID}" 2>/dev/null || true
else
    # Standalone command wrapper mode
    echo "  [VTune] Wrapping prompt execution under VTune..."
    WRAPPER_SCRIPT="/tmp/run_prompt_inference_${TIMESTAMP}.sh"
    cat <<EOF >"${WRAPPER_SCRIPT}"
#!/usr/bin/env bash
curl -sN -X POST "${SERVER_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"strata","messages":[{"role":"user","content":"${PROMPT}"}],"max_tokens":${MAX_TOKENS},"temperature":${TEMPERATURE},"stream":true}'
EOF
    chmod +x "${WRAPPER_SCRIPT}"

    "${VTUNE_BIN}" -collect "${ANALYSIS_TYPE}" \
        -result-dir "${RESULT_DIR}" \
        -data-limit "${DATA_LIMIT}" \
        -- "${WRAPPER_SCRIPT}"
    rm -f "${WRAPPER_SCRIPT}"
fi

# ------------------------------------------------------------------------------
# 5. Summarize & Export Profile Reports
# ------------------------------------------------------------------------------
echo ""
echo "[Step 5/5] Generating Summary Report..."

if [[ -d "${RESULT_DIR}" ]]; then
    SUMMARY_FILE="${RESULT_DIR}/vtune_summary.txt"
    "${VTUNE_BIN}" -report summary -result-dir "${RESULT_DIR}" -format text -report-output "${SUMMARY_FILE}" 2>/dev/null || true
    
    echo "========================================================================"
    echo "VTune Profiling Complete!"
    echo "Results stored at : ${RESULT_DIR}"
    if [[ -f "${SUMMARY_FILE}" ]]; then
        echo "Summary report at : ${SUMMARY_FILE}"
        echo ""
        echo "--- VTune Profile Summary ---"
        head -n 35 "${SUMMARY_FILE}"
        echo "-----------------------------"
    fi
    echo ""
    echo "To inspect kernels, timeline, and memory bandwidth interactively in GUI:"
    echo "  vtune-gui ${RESULT_DIR}"
    echo "========================================================================"
else
    echo "[-] VTune collection failed to produce result directory: ${RESULT_DIR}" >&2
    if [[ -f /tmp/vtune_collector.log ]]; then
        echo "Collector log tail:" >&2
        cat /tmp/vtune_collector.log | tail -n 25 >&2
    fi
    exit 1
fi
