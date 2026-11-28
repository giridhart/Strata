// tests/sycl/igc_workaround_test.cpp - verifies IGC 2.28 GEP simplification workaround
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "strata/core/device.hpp"

int main() {
    std::printf("Running igc_workaround_test...\n");

    // Initialize devices and trigger compiler workaround
    int count = strata::core::device_count();
    std::printf("Device count: %d\n", count);

    // 1. Verify SYCL persistent code cache is set
    const char* cache = std::getenv("SYCL_CACHE_PERSISTENT");
    if (!cache || std::strcmp(cache, "1") != 0) {
        std::fprintf(stderr, "FAIL: SYCL_CACHE_PERSISTENT not set to 1 (got '%s')\n", cache ? cache : "(null)");
        return 1;
    }
    std::printf("PASS: SYCL_CACHE_PERSISTENT is 1\n");

    // 2. Verify IGC GEP simplification is disabled for Xe2 / IGC 2.28
    const char* gep = std::getenv("IGC_EnableGEPSimplification");
    if (!gep || std::strcmp(gep, "0") != 0) {
        std::fprintf(stderr, "FAIL: IGC_EnableGEPSimplification not set to 0 (got '%s')\n", gep ? gep : "(null)");
        return 1;
    }
    std::printf("PASS: IGC_EnableGEPSimplification is 0\n");

    // 3. Verify compiler_workaround returns detected version
    std::string ver = strata::core::compiler_workaround();
    std::printf("compiler_workaround() version: %s\n", ver.c_str());
    if (ver.empty()) {
        std::fprintf(stderr, "FAIL: compiler_workaround() returned empty version string\n");
        return 1;
    }
    std::printf("PASS: compiler_workaround() version is '%s'\n", ver.c_str());

    std::printf("All IGC workaround verification checks PASSED.\n");
    return 0;
}
