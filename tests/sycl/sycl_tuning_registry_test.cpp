// tests/sycl/sycl_tuning_registry_test.cpp - Unit tests for SYCL tuning registry.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_tuning.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void test_fingerprint_matching(const sycl::device& dev) {
    printf("[1/5] Testing device fingerprinting and matching...\n");
    auto fp = strata::sycl_tuning::DeviceFingerprint::from_device(dev);
    assert(!fp.device_name.empty());
    assert(fp.compute_units > 0);
    assert(fp.matches(dev));

    // Mismatched device name must reject
    auto bad_fp_name = fp;
    bad_fp_name.device_name = "Nonexistent GPU 9000";
    assert(!bad_fp_name.matches(dev));

    // Mismatched compute units must reject
    auto bad_fp_cu = fp;
    bad_fp_cu.compute_units = fp.compute_units + 42;
    assert(!bad_fp_cu.matches(dev));

    printf("      Device fingerprint matching: PASSED\n");
}

void test_table_parsing_and_lookup(const sycl::device& dev) {
    printf("[2/5] Testing tuning table line parsing and exact lookup...\n");
    auto fp = strata::sycl_tuning::DeviceFingerprint::from_device(dev);

    // Format:
    // STRATA_SYCL_TUNING_V1 <device_name> <driver_version> <compute_units> <subgroup_size>
    // kernel n_in n_out code_bits group_elems act_kind lws tpr slm_pad cache_hint
    std::string valid_table =
        "STRATA_SYCL_TUNING_V1 \"" + fp.device_name + "\" \"" + fp.driver_version + "\" " +
        std::to_string(fp.compute_units) + " " + std::to_string(fp.native_subgroup_size) + "\n" +
        "s_gemv_q8k 2560 640 8 64 1 128 64 4 streaming\n" +
        "s_gemv_q8k 2560 2560 4 64 1 256 128 0 cached\n" +
        "bf16_gemv 2560 48 0 0 0 128 128 0 streaming\n" +
        "bf16_gemv 2560 640 0 0 0 64 64 0 streaming cached default\n";

    strata::sycl_tuning::SyclTuningRegistry reg;
    std::string err;
    bool ok = reg.load_from_string(valid_table, dev, err);
    if (!ok) {
        fprintf(stderr, "Failed to load valid table: %s\n", err.c_str());
        std::exit(1);
    }
    assert(reg.is_loaded());
    assert(reg.entry_count() == 4);

    // Exact lookup for s_gemv_q8k [2560, 640]
    uint64_t key1 = strata::sycl_tuning::make_kernel_key(
        strata::sycl_tuning::KernelType::S_GEMV_Q8K, 2560, 640, 8, 64, 1);
    auto p1 = reg.get_or_fallback(key1, {256, 0, 0, strata::sycl_tuning::CachePolicy::Default, {}});
    assert(p1.lws == 128);
    assert(p1.tpr == 64);
    assert(p1.slm_pad == 4);
    assert(p1.cache == strata::sycl_tuning::CachePolicy::StreamingL3);

    // Exact lookup for bf16_gemv [2560, 48]
    uint64_t key2 = strata::sycl_tuning::make_kernel_key(
        strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 48, 0, 0, 0);
    auto p2 = reg.get_or_fallback(key2, {256, 0, 0, strata::sycl_tuning::CachePolicy::Default, {}});
    assert(p2.lws == 128);
    assert(p2.tpr == 128);
    assert(p2.cache == strata::sycl_tuning::CachePolicy::StreamingL3);

    // Exact lookup for bf16_gemv [2560, 640] with per-argument cache hints
    uint64_t key3 = strata::sycl_tuning::make_kernel_key(
        strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 640, 0, 0, 0);
    auto p3 = reg.get_or_fallback(key3, {256, 0, 0, strata::sycl_tuning::CachePolicy::Default, {}});
    assert(p3.lws == 64);
    assert(p3.tpr == 64);
    assert(p3.hints.weight == strata::sycl_tuning::CachePolicy::StreamingL3);
    assert(p3.hints.activation == strata::sycl_tuning::CachePolicy::CachedL1L3);
    assert(p3.hints.output == strata::sycl_tuning::CachePolicy::Default);

    printf("      Table parsing and exact lookup: PASSED\n");
}

void test_fallback_behavior(const sycl::device& dev) {
    (void)dev;
    printf("[3/5] Testing fallback resolution on missing shapes...\n");
    strata::sycl_tuning::SyclTuningRegistry reg;
    // Registry not loaded: must always return provided fallback
    uint64_t key = strata::sycl_tuning::make_kernel_key(
        strata::sycl_tuning::KernelType::BF16_GEMV, 9999, 9999, 0, 0, 0);
    strata::sycl_tuning::LaunchParams fallback{256, 128, 8, strata::sycl_tuning::CachePolicy::Default};
    auto res = reg.get_or_fallback(key, fallback);
    assert(res.lws == 256);
    assert(res.tpr == 128);
    assert(res.slm_pad == 8);

    printf("      Fallback resolution: PASSED\n");
}

void test_device_mismatch_rejection(const sycl::device& dev) {
    printf("[4/5] Testing rejection of tuning table from wrong GPU device...\n");
    // Header claims to be a different device (e.g. Arc A770 vs B580)
    std::string foreign_table =
        "STRATA_SYCL_TUNING_V1 \"Intel(R) Arc(TM) A770 Graphics\" \"1.3.26\" 512 16\n"
        "s_gemv_q8k 2560 640 8 64 1 64 32 0 default\n";

    strata::sycl_tuning::SyclTuningRegistry reg;
    std::string err;
    bool ok = reg.load_from_string(foreign_table, dev, err);
    assert(!ok);
    assert(!err.empty());
    assert(!reg.is_loaded());

    printf("      Device mismatch rejection: PASSED (Properly rejected with: %s)\n", err.c_str());
}

void test_malformed_input_safety(const sycl::device& dev) {
    printf("[5/5] Testing malformed input resilience...\n");
    strata::sycl_tuning::SyclTuningRegistry reg;
    std::string err;

    // Empty string
    assert(!reg.load_from_string("", dev, err));

    // Garbage contents
    assert(!reg.load_from_string("NOT_A_VALID_HEADER random tokens here\nfoo bar", dev, err));

    // Incomplete rows
    auto fp = strata::sycl_tuning::DeviceFingerprint::from_device(dev);
    std::string incomplete_table =
        "STRATA_SYCL_TUNING_V1 \"" + fp.device_name + "\" \"" + fp.driver_version + "\" " +
        std::to_string(fp.compute_units) + " " + std::to_string(fp.native_subgroup_size) + "\n" +
        "s_gemv_q8k 2560 640\n";  // Missing columns
    assert(!reg.load_from_string(incomplete_table, dev, err));

    printf("      Malformed input resilience: PASSED\n");
}

void test_shape_collector() {
    printf("[6/6] Testing automatic kernel shape capture & workload export...\n");
    auto& sc = strata::sycl_tuning::ShapeCollector::instance();
    sc.clear();
    sc.enable(true);

    STRATA_SYCL_RECORD_SHAPE(strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 48, 0, 0, 0);
    STRATA_SYCL_RECORD_SHAPE(strata::sycl_tuning::KernelType::S_GEMV_Q8K, 2560, 640, 8, 64, 1);
    STRATA_SYCL_RECORD_SHAPE(strata::sycl_tuning::KernelType::BF16_GEMV, 2560, 48, 0, 0, 0); // duplicate must deduplicate

    assert(sc.count() == 2);

    const std::string tmp_path = "test_captured_workload.txt";
    std::string err;
    bool dump_ok = sc.dump_workload(tmp_path, err);
    assert(dump_ok);

    std::vector<strata::sycl_tuning::CapturedShape> loaded_shapes;
    bool load_ok = strata::sycl_tuning::ShapeCollector::load_workload(tmp_path, loaded_shapes, err);
    assert(load_ok);
    assert(loaded_shapes.size() == 2);
    assert(loaded_shapes[0].n_in == 2560 || loaded_shapes[1].n_in == 2560);

    std::remove(tmp_path.c_str());
    sc.enable(false);
    printf("      Automatic kernel shape capture & export: PASSED\n");
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v};
    const auto& dev = q.get_device();
    printf("====================================================\n");
    printf("SYCL Tuning Registry Unit Test Suite\n");
    printf("Active Target Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("====================================================\n");

    test_fingerprint_matching(dev);
    test_table_parsing_and_lookup(dev);
    test_fallback_behavior(dev);
    test_device_mismatch_rejection(dev);
    test_malformed_input_safety(dev);
    test_shape_collector();

    printf("====================================================\n");
    printf("ALL 6 TUNING REGISTRY TESTS PASSED SUCCESSFULLY!\n");
    printf("====================================================\n");
    return 0;
}
