#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
#include <chrono>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

inline double ms_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main() {
    sycl::queue q{sycl::property::queue::in_order{}};
    std::cout << "[GRAPH_LIMIT] Target Device: " 
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const std::vector<int> counts = {100, 300, 500, 700, 1000, 1300, 1500, 2000, 3000};
    int* dev_buf = sycl::malloc_device<int>(1024, q);
    int* host_buf = sycl::malloc_host<int>(1024, q);
    for (int i = 0; i < 1024; ++i) {
        host_buf[i] = i;
    }

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "--- PHASE 1: Simple dummy parallel_for kernels ---" << std::endl;
    std::cout << "=======================================================" << std::endl;

    for (int n : counts) {
        std::cout << "\n[TEST PHASE 1] Graph size: " << n << " nodes" << std::endl;
        std::cout.flush();

        const auto t_start = Clock::now();
        syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

        graph.begin_recording(q);
        for (int i = 0; i < n; ++i) {
            q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
                dev_buf[idx] += 1;
            });
        }
        graph.end_recording(q);

        const double rec_ms = ms_since(t_start);
        std::cout << "[TEST PHASE 1] Recorded " << n << " nodes in " << rec_ms << " ms" << std::endl;
        std::cout << "[TEST PHASE 1] Calling graph.finalize() for " << n << " nodes..." << std::endl;
        std::cout.flush();

        const auto t_fin = Clock::now();
        auto exec = graph.finalize();
        const double fin_ms = ms_since(t_fin);

        std::cout << "[TEST PHASE 1] SUCCESS: graph.finalize() completed in " 
                  << fin_ms << " ms for " << n << " nodes." << std::endl;
        std::cout.flush();

        q.ext_oneapi_graph(exec);
        q.wait();
        std::cout << "[TEST PHASE 1] Replay completed for " << n << " nodes." << std::endl;
        std::cout.flush();
    }

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "--- PHASE 2: Heterogeneous kernels with USM host memory ---" << std::endl;
    std::cout << "=======================================================" << std::endl;

    for (int n : counts) {
        std::cout << "\n[TEST PHASE 2] Graph size: " << n << " nodes (with USM host memory + multiple kernels)" << std::endl;
        std::cout.flush();

        const auto t_start = Clock::now();
        syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

        graph.begin_recording(q);
        for (int i = 0; i < n; ++i) {
            if (i % 3 == 0) {
                // Host memory read to device
                q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
                    dev_buf[idx] = host_buf[idx];
                });
            } else if (i % 3 == 1) {
                // Arithmetic on device
                q.parallel_for(sycl::nd_range<1>{sycl::range<1>{1024}, sycl::range<1>{128}}, [=](sycl::nd_item<1> it) {
                    int idx = it.get_global_id(0);
                    dev_buf[idx] = dev_buf[idx] * 2 + 1;
                });
            } else {
                // Device to host write
                q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
                    host_buf[idx] = dev_buf[idx];
                });
            }
        }
        graph.end_recording(q);

        const double rec_ms = ms_since(t_start);
        std::cout << "[TEST PHASE 2] Recorded " << n << " nodes in " << rec_ms << " ms" << std::endl;
        std::cout << "[TEST PHASE 2] Calling graph.finalize() for " << n << " nodes..." << std::endl;
        std::cout.flush();

        const auto t_fin = Clock::now();
        auto exec = graph.finalize();
        const double fin_ms = ms_since(t_fin);

        std::cout << "[TEST PHASE 2] SUCCESS: graph.finalize() completed in " 
                  << fin_ms << " ms for " << n << " nodes." << std::endl;
        std::cout.flush();

        q.ext_oneapi_graph(exec);
        q.wait();
        std::cout << "[TEST PHASE 2] Replay completed for " << n << " nodes." << std::endl;
        std::cout.flush();
    }

    sycl::free(dev_buf, q);
    sycl::free(host_buf, q);
    std::cout << "\n[GRAPH_LIMIT] Benchmark complete!" << std::endl;
    return 0;
}
