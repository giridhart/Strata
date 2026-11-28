#!/usr/bin/env bash
# build_inkling_sycl.sh - Build strata-inkling-sycl for Intel Arc Alchemist (dg2) & Battlemage (bmg_g21, bmg_g31 B70)
# CRITICAL: Zero reqd_sub_group_size(8) anywhere -> clean IGC AOT/JIT compilation on Battlemage B70!
# 100% MIT License.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENG_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/build}"
TARGET_MODE="${2:-fat}"   # fat | bmg | acm | jit

mkdir -p "${OUT_DIR}"

ICPX="${ICPX:-icpx}"
CC="${CC:-icx}"

echo "[build_inkling_sycl] ICPX=${ICPX} TARGET_MODE=${TARGET_MODE} OUT_DIR=${OUT_DIR}"

# 1. Compile vendored zstd-dec + strata_reader.cpp
ZSTD_DIR="${ENG_ROOT}/third_party/zstd-dec"
ZSTD_SRCS=(
  "${ZSTD_DIR}/common/debug.c"
  "${ZSTD_DIR}/common/entropy_common.c"
  "${ZSTD_DIR}/common/error_private.c"
  "${ZSTD_DIR}/common/fse_decompress.c"
  "${ZSTD_DIR}/common/xxhash.c"
  "${ZSTD_DIR}/common/zstd_common.c"
  "${ZSTD_DIR}/decompress/huf_decompress.c"
  "${ZSTD_DIR}/decompress/zstd_ddict.c"
  "${ZSTD_DIR}/decompress/zstd_decompress.c"
  "${ZSTD_DIR}/decompress/zstd_decompress_block.c"
)

OBJS=()
for f in "${ZSTD_SRCS[@]}"; do
  base="$(basename "${f%.c}")"
  obj="${OUT_DIR}/zstd_${base}.o"
  "${CC}" -O2 -DZSTD_DISABLE_ASM=1 -I"${ZSTD_DIR}" -I"${ZSTD_DIR}/common" -I"${ZSTD_DIR}/decompress" -c "${f}" -o "${obj}"
  OBJS+=("${obj}")
done

"${ICPX}" -std=c++20 -O2 -I"${ENG_ROOT}/include" -I"${ZSTD_DIR}" \
  -c "${ENG_ROOT}/src/artifact/strata_reader.cpp" -o "${OUT_DIR}/strata_reader.o"
OBJS+=("${OUT_DIR}/strata_reader.o")

# Compile native AVX-512/AVX2 S_K/SF4/MX8 CPU expert kernels for Strata Predictive CPU/RAM Hybrid pool
CPU_ISA_FLAGS=()
if [ "$(uname -m)" = "x86_64" ]; then
  CPU_ISA_FLAGS=(-mavx2 -mfma -mf16c)
fi
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" \
  -c "${ENG_ROOT}/src/quant/sk_codecs.cpp" -o "${OUT_DIR}/sk_codecs.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" \
  -c "${ENG_ROOT}/src/quant/dequant.cpp" -o "${OUT_DIR}/dequant.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" \
  -c "${ENG_ROOT}/src/quant/types.cpp" -o "${OUT_DIR}/types.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" -I"${ENG_ROOT}/src/quant" \
  -c "${ENG_ROOT}/src/quant/isa.cpp" -o "${OUT_DIR}/isa.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" -I"${ENG_ROOT}/src/quant" \
  -c "${ENG_ROOT}/src/quant/dispatch.cpp" -o "${OUT_DIR}/dispatch.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" -I"${ENG_ROOT}/src/quant" \
  -c "${ENG_ROOT}/src/quant/selfcheck.cpp" -o "${OUT_DIR}/selfcheck.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -ffp-contract=off -I"${ENG_ROOT}/include" -I"${ENG_ROOT}/src/quant" \
  -c "${ENG_ROOT}/src/quant/act_ref.cpp" -o "${OUT_DIR}/act_ref.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -ffp-contract=off -I"${ENG_ROOT}/include" -I"${ENG_ROOT}/src/quant" \
  -c "${ENG_ROOT}/src/quant/dot_generic.cpp" -o "${OUT_DIR}/dot_generic.o"
"${ICPX}" -std=c++20 -O3 "${CPU_ISA_FLAGS[@]}" -I"${ENG_ROOT}/include" \
  -c "${ENG_ROOT}/src/inkling2/inkling2_loop.cpp" -o "${OUT_DIR}/inkling2_loop.o"
OBJS+=("${OUT_DIR}/sk_codecs.o" "${OUT_DIR}/dequant.o" "${OUT_DIR}/types.o" "${OUT_DIR}/isa.o" "${OUT_DIR}/dispatch.o" "${OUT_DIR}/selfcheck.o" "${OUT_DIR}/act_ref.o" "${OUT_DIR}/dot_generic.o" "${OUT_DIR}/inkling2_loop.o")

# 2. Target flags (Alchemist dg2 + Battlemage bmg_g21 B580 + Arc Pro B70 bmg_g31 + portable spir64 JIT!)
SYCL_FLAGS=(-fsycl -std=c++20 -O3 -fno-fast-math)
HAS_OCLOC=0
if command -v ocloc >/dev/null 2>&1; then
  HAS_OCLOC=1
fi

TARGETS_CSV=""
if [ -n "${SYCL_TARGETS_OVERRIDE:-}" ]; then
  TARGETS_CSV="${SYCL_TARGETS_OVERRIDE}"
elif [ -n "${AOT_DEVICE:-}" ]; then
  if [ "${HAS_OCLOC}" = "1" ]; then
    case "${AOT_DEVICE}" in
      intel_gpu_*|spir64*) TARGETS_CSV="${AOT_DEVICE},spir64" ;;
      *) TARGETS_CSV="intel_gpu_${AOT_DEVICE},spir64" ;;
    esac
  else
    echo "[build_inkling_sycl] WARNING: ocloc not found; falling back to spir64 JIT for AOT_DEVICE=${AOT_DEVICE}" >&2
    TARGETS_CSV="spir64"
  fi
else
  case "${TARGET_MODE}" in
    fat)
      if [ "${HAS_OCLOC}" = "1" ] && ocloc ids bmg_g31 >/dev/null 2>&1; then
        TARGETS_CSV="spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21,intel_gpu_bmg_g31"
      elif [ "${HAS_OCLOC}" = "1" ]; then
        TARGETS_CSV="spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21"
      else
        echo "[build_inkling_sycl] WARNING: ocloc not found; falling back to spir64 JIT" >&2
        TARGETS_CSV="spir64"
      fi
      ;;
    bmg)
      if [ "${HAS_OCLOC}" = "1" ] && ocloc ids bmg_g31 >/dev/null 2>&1; then
        TARGETS_CSV="intel_gpu_bmg_g21,intel_gpu_bmg_g31,spir64"
      elif [ "${HAS_OCLOC}" = "1" ]; then
        TARGETS_CSV="intel_gpu_bmg_g21,spir64"
      else
        echo "[build_inkling_sycl] WARNING: ocloc not found; falling back to spir64 JIT for TARGET_MODE=bmg" >&2
        TARGETS_CSV="spir64"
      fi
      ;;
    acm)
      if [ "${HAS_OCLOC}" = "1" ]; then
        TARGETS_CSV="intel_gpu_acm_g10,spir64"
      else
        echo "[build_inkling_sycl] WARNING: ocloc not found; falling back to spir64 JIT for TARGET_MODE=acm" >&2
        TARGETS_CSV="spir64"
      fi
      ;;
    jit)
      TARGETS_CSV="spir64"
      ;;
    *)
      echo "Unknown TARGET_MODE=${TARGET_MODE}" >&2
      exit 1
      ;;
  esac
fi

SYCL_FLAGS+=("-fsycl-targets=${TARGETS_CSV}")

# Enable SPV_KHR_integer_dot_product (dp4a) for every target triple, plus optional GRF_MODE=128|256 AOT override
IFS=',' read -r -a _TARGET_ARR <<< "${TARGETS_CSV}"
for _tgt in "${_TARGET_ARR[@]}"; do
  SYCL_FLAGS+=("-Xspirv-translator=${_tgt}" "--spirv-ext=+SPV_KHR_integer_dot_product")
  if [ "${_tgt}" != "spir64" ] && [ -n "${GRF_MODE:-}" ]; then
    case "${GRF_MODE}" in
      256)
        SYCL_FLAGS+=("-Xsycl-target-backend=${_tgt}" "-options -ze-opt-large-register-file")
        ;;
      128)
        SYCL_FLAGS+=("-Xsycl-target-backend=${_tgt}" "-options -ze-opt-regular-grf-kernel")
        ;;
      *)
        echo "Unknown GRF_MODE=${GRF_MODE} (expected 128 or 256)" >&2
        exit 1
        ;;
    esac
  fi
done
if [ -n "${GRF_MODE:-}" ]; then
  echo "[build_inkling_sycl] GRF_MODE=${GRF_MODE} enabled"
fi

MKL_FLAGS=()
MKL_LIBS=()
if [ "${STRATA_SYCL_USE_ONEMKL:-0}" = "1" ] && [ -n "${MKLROOT:-}" ] && [ -f "${MKLROOT}/include/oneapi/mkl/blas.hpp" ]; then
  echo "[build_inkling_sycl] Enabling optional legacy Intel oneMKL XMX GEMM (${MKLROOT})"
  MKL_FLAGS=(-DINK_SYCL_USE_ONEMKL=1 "-I${MKLROOT}/include")
  MKL_LIBS=("-L${MKLROOT}/lib" "-Wl,-rpath,${MKLROOT}/lib" -lmkl_sycl_blas -lmkl_intel_lp64 -lmkl_sequential -lmkl_core)
else
  echo "[build_inkling_sycl] Zero-oneMKL build (native SG16 joint_matrix XMX)"
fi

"${ICPX}" "${SYCL_FLAGS[@]}" "${MKL_FLAGS[@]}" \
  -I"${SCRIPT_DIR}" \
  -I"${ENG_ROOT}/include" \
  -I"${ENG_ROOT}/src/hy4flash/run" \
  -I"${ENG_ROOT}/src/hy4flash/sycl" \
  -I"${ENG_ROOT}/third_party/stb" \
  -I"${ZSTD_DIR}" \
  "${SCRIPT_DIR}/ink_sy_main.cpp" \
  "${OBJS[@]}" \
  "${MKL_LIBS[@]}" \
  -lpthread \
  -o "${OUT_DIR}/strata-inkling-sycl"

ls -lh "${OUT_DIR}/strata-inkling-sycl"
"${OUT_DIR}/strata-inkling-sycl" --version
