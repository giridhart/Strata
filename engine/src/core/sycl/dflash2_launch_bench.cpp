// src/core/sycl/dflash2_launch_bench.cpp - what a dependent kernel costs inside a captured graph on this GPU: N tiny kernels in a chain, and N kernels of 256 blocks.
//
//     dflash2_launch_bench [n=100]
#include <sycl/sycl.hpp>
#if __has_include("strata/core/mock_cuda_graph.hpp")
#include "strata/core/mock_cuda_graph.hpp"
#else
#include "../../../include/strata/core/mock_cuda_graph.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100;
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

    constexpr size_t total_bytes = 1 << 24;
    float* x = sycl::malloc_device<float>(total_bytes / sizeof(float), q);
    q.memset(x, 0, total_bytes).wait();

    for (int blocks : {1, 64, 256, 1024}) {
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        cudaStreamBeginCapture(&q, cudaStreamCaptureModeThreadLocal);
        for (int i = 0; i < n; ++i) {
            q.parallel_for(sycl::nd_range<1>(blocks * 256, 256), [=](sycl::nd_item<1> item) {
                const int idx = item.get_global_id(0);
                if (idx < blocks * 256) x[idx] = x[idx] * 1.0001f + 1.0f;
            });
        }
        cudaStreamEndCapture(&q, &g);
        cudaGraphInstantiate(&ge, g, 0);
        cudaGraphLaunch(ge, &q);
        q.wait();

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < 50; ++it) cudaGraphLaunch(ge, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("graph of %d chained kernels, %4d blocks: %.2f us per kernel\n", n, blocks, 1000.0 * ms / 50 / n);

        // one launch from an idle stream: event, launch, event, wait (what the drafter does each round)
        {
            double tot = 0, wall = 0;
            for (int it = 0; it < 20; ++it) {
                q.wait();
                auto start = std::chrono::steady_clock::now();
                cudaGraphLaunch(ge, &q);
                q.wait();
                double m = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                wall += m;
                tot += m;
            }
            std::printf("single launch from idle, %d kernels, %4d blocks: events %.3f ms, wall %.3f ms\n", n, blocks, tot / 20, wall / 20);
        }

        // the same without a graph
        t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < 5; ++it) {
            for (int i = 0; i < n; ++i) {
                q.parallel_for(sycl::nd_range<1>(blocks * 256, 256), [=](sycl::nd_item<1> item) {
                    const int idx = item.get_global_id(0);
                    if (idx < blocks * 256) x[idx] = x[idx] * 1.0001f + 1.0f;
                });
            }
        }
        q.wait();
        t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("stream of %d chained kernels, %4d blocks: %.2f us per kernel\n", n, blocks, 1000.0 * ms / 5 / n);

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }

    sycl::free(x, q);
    return 0;
}
