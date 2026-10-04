source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
cd build_sycl
for f in *_parity; do
  echo "=== Running $f ==="
  ./$f
  if [ $? -ne 0 ]; then
    echo "FAILED: $f"
  fi
done
