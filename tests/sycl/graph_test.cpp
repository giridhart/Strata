#include <sycl/sycl.hpp>
#include "strata/core/graph.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running graph_test on: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    int total_failures = 0;

    // Test 1: CapturedGraph recording and multiple replays
    {
        constexpr size_t N = 1024;
        int* data = sycl::malloc_shared<int>(N, q);
        for (size_t i = 0; i < N; ++i) data[i] = 1;

        strata::core::CapturedGraph g;
        std::string err;

        if (!g.begin(&q, err)) {
            std::printf("FAIL: CapturedGraph::begin failed: %s\n", err.c_str());
            total_failures++;
        }

        // Submit 2 parallel_for kernels during recording
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            data[idx] += 10;
        });
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            data[idx] *= 2;
        });

        if (!g.end(&q, err)) {
            std::printf("FAIL: CapturedGraph::end failed: %s\n", err.c_str());
            total_failures++;
        }

        if (!g.valid()) {
            std::printf("FAIL: CapturedGraph is not valid after end()\n");
            total_failures++;
        }

        if (g.nodes() < 2) {
            std::printf("FAIL: Expected at least 2 nodes, got %zu\n", g.nodes());
            total_failures++;
        } else {
            std::printf("CapturedGraph: captured %zu nodes\n", g.nodes());
        }

        // At this point, the kernels should have been captured, not executed on data!
        // In SYCL queue recording, data may still be initial 1.
        // Let's launch the graph for execution!
        // Initial state before replay:
        for (size_t i = 0; i < N; ++i) data[i] = 1;

        // Replay 1: (1 + 10) * 2 = 22
        if (!g.launch(&q, err)) {
            std::printf("FAIL: CapturedGraph::launch 1 failed: %s\n", err.c_str());
            total_failures++;
        }
        if (!g.wait_ms(1000)) {
            std::printf("FAIL: CapturedGraph::wait_ms timed out on replay 1\n");
            total_failures++;
        }
        if (data[0] != 22) {
            std::printf("FAIL: Replay 1 expected 22, got %d\n", data[0]);
            total_failures++;
        }

        // Replay 2: (22 + 10) * 2 = 64
        if (!g.launch(&q, err)) {
            std::printf("FAIL: CapturedGraph::launch 2 failed: %s\n", err.c_str());
            total_failures++;
        }
        if (!g.wait_ms(1000)) {
            std::printf("FAIL: CapturedGraph::wait_ms timed out on replay 2\n");
            total_failures++;
        }
        if (data[0] != 64) {
            std::printf("FAIL: Replay 2 expected 64, got %d\n", data[0]);
            total_failures++;
        }

        std::printf("CapturedGraph replay test: %s\n", total_failures == 0 ? "PASS" : "FAIL");

        sycl::free(data, q);
    }

    // Test 2: GraphRegistry caching and execution
    {
        strata::core::GraphRegistry reg(&q);
        std::string err;

        constexpr size_t N = 512;
        int* val = sycl::malloc_shared<int>(N, q);
        for (size_t i = 0; i < N; ++i) val[i] = 5;

        int body_invocations = 0;
        auto body = [&]() {
            body_invocations++;
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                val[idx] += 100;
            });
        };

        // First record: body should be called
        if (!reg.record(strata::core::LayerType::GDN, 1, body, err)) {
            std::printf("FAIL: GraphRegistry::record 1 failed: %s\n", err.c_str());
            total_failures++;
        }

        if (body_invocations != 1) {
            std::printf("FAIL: Expected 1 body invocation, got %d\n", body_invocations);
            total_failures++;
        }

        // Second record for same key: body should NOT be called (cached)
        if (!reg.record(strata::core::LayerType::GDN, 1, body, err)) {
            std::printf("FAIL: GraphRegistry::record 2 failed: %s\n", err.c_str());
            total_failures++;
        }

        if (body_invocations != 1) {
            std::printf("FAIL: Expected cached body to not run again, but invocations = %d\n", body_invocations);
            total_failures++;
        }

        if (reg.size() != 1 || reg.captures() != 1) {
            std::printf("FAIL: Expected size=1, captures=1, got size=%zu, captures=%zu\n",
                        reg.size(), reg.captures());
            total_failures++;
        }

        // Launch from registry
        for (size_t i = 0; i < N; ++i) val[i] = 5;

        if (!reg.launch(strata::core::LayerType::GDN, 1, 1000, err)) {
            std::printf("FAIL: GraphRegistry::launch failed: %s\n", err.c_str());
            total_failures++;
        }

        if (val[0] != 105) {
            std::printf("FAIL: Expected val[0] == 105 after registry launch, got %d\n", val[0]);
            total_failures++;
        }

        // Record a second key (QSA)
        auto qsa_body = [&]() {
            body_invocations++;
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                val[idx] *= 2;
            });
        };

        if (!reg.record(strata::core::LayerType::QSA, 1, qsa_body, err)) {
            std::printf("FAIL: GraphRegistry::record QSA failed: %s\n", err.c_str());
            total_failures++;
        }

        if (reg.size() != 2 || reg.captures() != 2) {
            std::printf("FAIL: Expected size=2, captures=2, got size=%zu, captures=%zu\n",
                        reg.size(), reg.captures());
            total_failures++;
        }

        // Launch QSA: 105 * 2 = 210
        if (!reg.launch(strata::core::LayerType::QSA, 1, 1000, err)) {
            std::printf("FAIL: GraphRegistry::launch QSA failed: %s\n", err.c_str());
            total_failures++;
        }

        if (val[0] != 210) {
            std::printf("FAIL: Expected val[0] == 210 after QSA launch, got %d\n", val[0]);
            total_failures++;
        }

        std::printf("GraphRegistry test: %s\n", total_failures == 0 ? "PASS" : "FAIL");
        sycl::free(val, q);
    }

    std::printf("\ngraph_test: %s (total failures: %d)\n",
                total_failures == 0 ? "PASSED" : "FAILED", total_failures);
    return total_failures == 0 ? 0 : 1;
}
