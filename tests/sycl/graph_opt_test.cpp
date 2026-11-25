// tests/sycl/graph_opt_test.cpp
// Correctness tests and A/B microbenchmark for SYCL command_graph runtime options
// (STRATA_SYCL_GRAPH_OPT, STRATA_SYCL_GRAPH_NATIVE, STRATA_SYCL_GRAPH_NO_CYCLE_CHECK, STRATA_SYCL_GRAPH_END_ALL),
// multi-queue fork/join recording, whole-graph update, and warmup-before-capture.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include "strata/core/graph.hpp"
#include "strata/core/mock_cuda_graph.hpp"
#include "strata/core/sycl_graph_env.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

void set_env_var(const char* name, const char* val) {
#if defined(_WIN32)
    _putenv_s(name, val ? val : "");
#else
    if (val) {
        setenv(name, val, 1);
    } else {
        unsetenv(name);
    }
#endif
}

void clear_graph_envs() {
    set_env_var("STRATA_SYCL_GRAPH_OPT", nullptr);
    set_env_var("STRATA_SYCL_GRAPH_NATIVE", nullptr);
    set_env_var("STRATA_SYCL_GRAPH_NO_CYCLE_CHECK", nullptr);
    set_env_var("STRATA_SYCL_GRAPH_END_ALL", nullptr);
}

struct ModeConfig {
    const char* name;
    const char* opt;
    const char* native;
    const char* no_cycle;
    const char* end_all;
};

void apply_config(const ModeConfig& cfg) {
    clear_graph_envs();
    set_env_var("STRATA_SYCL_GRAPH_OPT", cfg.opt);
    set_env_var("STRATA_SYCL_GRAPH_NATIVE", cfg.native);
    set_env_var("STRATA_SYCL_GRAPH_NO_CYCLE_CHECK", cfg.no_cycle);
    set_env_var("STRATA_SYCL_GRAPH_END_ALL", cfg.end_all);
}

int test_env_parsing() {
    int fails = 0;

    clear_graph_envs();
    auto o = strata::sycl_graph::read_options();
    if (o.native_recording || o.no_cycle_check || o.end_all_queues) {
        std::printf("FAIL [env]: default options should all be false\n");
        ++fails;
    }

    set_env_var("STRATA_SYCL_GRAPH_OPT", "1");
    o = strata::sycl_graph::read_options();
    if (!o.native_recording || !o.no_cycle_check || !o.end_all_queues) {
        std::printf("FAIL [env]: STRATA_SYCL_GRAPH_OPT=1 should enable all options\n");
        ++fails;
    }

    set_env_var("STRATA_SYCL_GRAPH_NATIVE", "0");
    o = strata::sycl_graph::read_options();
    if (o.native_recording || !o.no_cycle_check || !o.end_all_queues) {
        std::printf("FAIL [env]: STRATA_SYCL_GRAPH_NATIVE=0 should override OPT=1\n");
        ++fails;
    }

    clear_graph_envs();
    std::printf("Test 1 (env parsing): %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
}

int test_modes_correctness(sycl::queue& q) {
    int fails = 0;
    const ModeConfig configs[] = {
        {"default", nullptr, nullptr, nullptr, nullptr},
        {"no_cycle_check", nullptr, nullptr, "1", nullptr},
        {"native_recording", nullptr, "1", nullptr, nullptr},
        {"opt_all", "1", nullptr, nullptr, nullptr},
    };

    constexpr size_t N = 1024;
    int* buf = sycl::malloc_shared<int>(N, q);

    for (const auto& cfg : configs) {
        apply_config(cfg);

        // A. CapturedGraph
        {
            for (size_t i = 0; i < N; ++i) buf[i] = 3;
            strata::core::CapturedGraph g;
            std::string err;
            if (!g.begin(&q, err)) {
                std::printf("FAIL [%s/CapturedGraph]: begin: %s\n", cfg.name, err.c_str());
                ++fails;
                continue;
            }
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { buf[idx] += 7; });
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { buf[idx] *= 3; });
            if (!g.end(&q, err)) {
                std::printf("FAIL [%s/CapturedGraph]: end: %s\n", cfg.name, err.c_str());
                ++fails;
                continue;
            }
            for (size_t i = 0; i < N; ++i) buf[i] = 3;
            // Replay 1: (3 + 7) * 3 = 30; Replay 2: (30 + 7) * 3 = 111
            if (!g.launch(&q, err) || !g.wait_ms(2000) || buf[0] != 30 || buf[N - 1] != 30) {
                std::printf("FAIL [%s/CapturedGraph]: replay 1 expected 30, got %d (err=%s)\n", cfg.name, buf[0], err.c_str());
                ++fails;
            }
            if (!g.launch(&q, err) || !g.wait_ms(2000) || buf[0] != 111 || buf[N - 1] != 111) {
                std::printf("FAIL [%s/CapturedGraph]: replay 2 expected 111, got %d (err=%s)\n", cfg.name, buf[0], err.c_str());
                ++fails;
            }
        }

        // B. mock_cuda_graph.hpp
        {
            for (size_t i = 0; i < N; ++i) buf[i] = 2;
            if (cudaStreamBeginCapture(&q, cudaStreamCaptureModeThreadLocal) != 0) {
                std::printf("FAIL [%s/mock_cuda_graph]: begin failed\n", cfg.name);
                ++fails;
                continue;
            }
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { buf[idx] += 5; });
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { buf[idx] *= 4; });
            cudaGraph_t graph = nullptr;
            if (cudaStreamEndCapture(&q, &graph) != 0 || !graph) {
                std::printf("FAIL [%s/mock_cuda_graph]: end failed\n", cfg.name);
                ++fails;
                continue;
            }
            cudaGraphExec_t exec = nullptr;
            const int ie = cudaGraphInstantiate(&exec, graph, 0);
            cudaGraphDestroy(graph);
            if (ie != 0 || !exec) {
                std::printf("FAIL [%s/mock_cuda_graph]: instantiate failed\n", cfg.name);
                ++fails;
                continue;
            }
            for (size_t i = 0; i < N; ++i) buf[i] = 2;
            // (2 + 5) * 4 = 28
            if (cudaGraphLaunch(exec, &q) != 0) {
                std::printf("FAIL [%s/mock_cuda_graph]: launch failed\n", cfg.name);
                ++fails;
            }
            q.wait();
            if (buf[0] != 28 || buf[N - 1] != 28) {
                std::printf("FAIL [%s/mock_cuda_graph]: expected 28, got %d\n", cfg.name, buf[0]);
                ++fails;
            }
            cudaGraphExecDestroy(exec);
        }
    }

    sycl::free(buf, q);
    clear_graph_envs();
    std::printf("Test 2 (modes correctness): %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
}

int test_multiqueue_fork_join(sycl::queue& q_main) {
    int fails = 0;
    sycl::queue q_side(q_main.get_context(), q_main.get_device(), sycl::property::queue::in_order{});

    constexpr size_t N = 512;
    int* a = sycl::malloc_shared<int>(N, q_main);
    int* b = sycl::malloc_shared<int>(N, q_main);
    int* out = sycl::malloc_shared<int>(N, q_main);

    // Enable END_ALL so forked q_side exits recording cleanly via g->end_recording().
    apply_config({"end_all", nullptr, nullptr, "1", "1"});

    if (cudaStreamBeginCapture(&q_main, cudaStreamCaptureModeThreadLocal) != 0) {
        std::printf("FAIL [fork_join]: begin failed\n");
        sycl::free(a, q_main);
        sycl::free(b, q_main);
        sycl::free(out, q_main);
        clear_graph_envs();
        return 1;
    }

    // Main queue initializes a, forks to q_side (mirrors STRATA_PCIE_EARLY pf_cs_ pattern)
    q_main.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { a[idx] += 10; });
    sycl::event ev_fork = q_main.ext_oneapi_submit_barrier();
    q_side.ext_oneapi_submit_barrier({ev_fork});
    q_side.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { b[idx] += 20; });
    sycl::event ev_join = q_side.ext_oneapi_submit_barrier();
    q_main.ext_oneapi_submit_barrier({ev_join});
    q_main.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { out[idx] = a[idx] + b[idx]; });

    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(&q_main, &graph) != 0 || !graph) {
        std::printf("FAIL [fork_join]: end_recording failed\n");
        ++fails;
    } else {
        cudaGraphExec_t exec = nullptr;
        const int ie = cudaGraphInstantiate(&exec, graph, 0);
        cudaGraphDestroy(graph);
        if (ie != 0 || !exec) {
            std::printf("FAIL [fork_join]: instantiate failed\n");
            ++fails;
        } else {
            for (size_t i = 0; i < N; ++i) {
                a[i] = 1;
                b[i] = 2;
                out[i] = 0;
            }
            if (cudaGraphLaunch(exec, &q_main) != 0) {
                std::printf("FAIL [fork_join]: launch failed\n");
                ++fails;
            }
            q_main.wait();
            if (out[0] != 33 || out[N - 1] != 33) {
                std::printf("FAIL [fork_join]: expected 33, got %d\n", out[0]);
                ++fails;
            }

            // Crucial check: q_side must have exited recording state so eager work runs immediately.
            q_side.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { b[idx] = 99; }).wait();
            if (b[0] != 99) {
                std::printf("FAIL [fork_join]: q_side remained stuck in recording state (b[0]=%d)\n", b[0]);
                ++fails;
            }

            cudaGraphExecDestroy(exec);
        }
    }

    sycl::free(a, q_main);
    sycl::free(b, q_main);
    sycl::free(out, q_main);
    clear_graph_envs();
    std::printf("Test 3 (multi-queue fork/join with END_ALL=1): %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
}

int test_whole_graph_update(sycl::queue& q) {
    namespace sycl_ext = sycl::ext::oneapi::experimental;
    int fails = 0;
    constexpr size_t N = 1024;
    int* ptrA = sycl::malloc_shared<int>(N, q);
    int* ptrB = sycl::malloc_shared<int>(N, q);
    for (size_t i = 0; i < N; ++i) {
        ptrA[i] = 10;
        ptrB[i] = 100;
    }

    auto record_chain = [&](int* ptr) {
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { ptr[idx] += 5; });
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) { ptr[idx] *= 2; });
    };

    try {
        sycl_ext::command_graph<sycl_ext::graph_state::modifiable> gA(
            q.get_context(), q.get_device(), {sycl_ext::property::graph::no_cycle_check{}});
        gA.begin_recording(q);
        record_chain(ptrA);
        gA.end_recording();

        auto exec = gA.finalize({sycl_ext::property::graph::updatable{}});
        q.ext_oneapi_graph(exec).wait();
        if (ptrA[0] != 30 || ptrB[0] != 100) {
            std::printf("FAIL [whole_graph_update]: before update expected A=30 B=100, got A=%d B=%d\n", ptrA[0], ptrB[0]);
            ++fails;
        }

        sycl_ext::command_graph<sycl_ext::graph_state::modifiable> gB(
            q.get_context(), q.get_device(), {sycl_ext::property::graph::no_cycle_check{}});
        gB.begin_recording(q);
        record_chain(ptrB);
        gB.end_recording();

        exec.update(gB);
        q.ext_oneapi_graph(exec).wait();
        if (ptrA[0] != 30 || ptrB[0] != 210) {
            std::printf("FAIL [whole_graph_update]: after update expected A=30 B=210, got A=%d B=%d\n", ptrA[0], ptrB[0]);
            ++fails;
        }
    } catch (const std::exception& e) {
        std::printf("FAIL [whole_graph_update]: exception: %s\n", e.what());
        ++fails;
    }

    sycl::free(ptrA, q);
    sycl::free(ptrB, q);
    std::printf("Test 4 (whole-graph update): %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
}

struct BenchRow {
    std::string label;
    int nodes = 0;
    double record_ms = 0.0;
    double finalize_ms = 0.0;
    double replay_median_us = 0.0;
    double replay_mean_us = 0.0;
    bool ok = true;
};

BenchRow bench_eager(sycl::queue& q, int n_nodes, int iters, float* buf, size_t elems) {
    BenchRow r;
    r.label = "eager (direct)";
    r.nodes = n_nodes;

    auto run_once = [&]() {
        for (int k = 0; k < n_nodes; ++k) {
            const float c = 1.0001f;
            q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
                buf[idx] = buf[idx] * c + 0.001f;
            });
        }
        q.wait();
    };

    // Warmup
    for (int w = 0; w < 3; ++w) run_once();

    std::vector<double> samples;
    samples.reserve((size_t)iters);
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        run_once();
        const double us = us_since(t0);
        samples.push_back(us);
        sum += us;
    }
    std::sort(samples.begin(), samples.end());
    r.replay_median_us = samples[samples.size() / 2];
    r.replay_mean_us = sum / (double)iters;
    return r;
}

BenchRow bench_graph_mode(sycl::queue& q, const ModeConfig& cfg, int n_nodes, int iters, float* buf, size_t elems) {
    BenchRow r;
    r.label = cfg.name;
    r.nodes = n_nodes;
    apply_config(cfg);

    const auto t_rec0 = Clock::now();
    if (cudaStreamBeginCapture(&q, cudaStreamCaptureModeThreadLocal) != 0) {
        r.ok = false;
        clear_graph_envs();
        return r;
    }
    for (int k = 0; k < n_nodes; ++k) {
        const float c = 1.0001f;
        q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
            buf[idx] = buf[idx] * c + 0.001f;
        });
    }
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(&q, &graph) != 0 || !graph) {
        r.ok = false;
        clear_graph_envs();
        return r;
    }
    r.record_ms = ms_since(t_rec0);

    const auto t_fin0 = Clock::now();
    cudaGraphExec_t exec = nullptr;
    const int ie = cudaGraphInstantiate(&exec, graph, 0);
    r.finalize_ms = ms_since(t_fin0);
    cudaGraphDestroy(graph);

    if (ie != 0 || !exec) {
        r.ok = false;
        clear_graph_envs();
        return r;
    }

    // Warmup replays
    for (int w = 0; w < 5; ++w) {
        cudaGraphLaunch(exec, &q);
        q.wait();
    }

    std::vector<double> samples;
    samples.reserve((size_t)iters);
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        cudaGraphLaunch(exec, &q);
        q.wait();
        const double us = us_since(t0);
        samples.push_back(us);
        sum += us;
    }
    std::sort(samples.begin(), samples.end());
    r.replay_median_us = samples[samples.size() / 2];
    r.replay_mean_us = sum / (double)iters;

    cudaGraphExecDestroy(exec);
    clear_graph_envs();
    return r;
}

void run_benchmarks(sycl::queue& q) {
    constexpr size_t elems = 4096;
    float* buf = sycl::malloc_device<float>(elems, q);
    q.memset(buf, 0, elems * sizeof(float)).wait();

    // Warm up the kernel binary once before timing so cold JIT doesn't skew the first row.
    q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
        buf[idx] = buf[idx] * 1.0001f + 0.001f;
    }).wait();

    const ModeConfig configs[] = {
        {"graph_default", nullptr, nullptr, nullptr, nullptr},
        {"graph_no_cycle", nullptr, nullptr, "1", nullptr},
        {"graph_native", nullptr, "1", nullptr, nullptr},
        {"graph_opt_all", "1", nullptr, nullptr, nullptr},
    };

    const int node_counts[] = {64, 1200};
    const int iters_for[] = {50, 15};

    std::printf("\n=== SYCL Graph A/B Microbenchmark (%s) ===\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-18s | %5s | %9s | %11s | %12s | %12s\n",
                "Mode", "Nodes", "Record ms", "Finalize ms", "Median us", "Mean us");
    std::printf("-------------------+-------+-----------+-------------+--------------+-------------\n");

    for (size_t ci = 0; ci < 2; ++ci) {
        const int n = node_counts[ci];
        const int iters = iters_for[ci];
        auto er = bench_eager(q, n, iters, buf, elems);
        std::printf("%-18s | %5d | %9s | %11s | %12.1f | %12.1f\n",
                    er.label.c_str(), er.nodes, "-", "-", er.replay_median_us, er.replay_mean_us);

        for (const auto& cfg : configs) {
            auto gr = bench_graph_mode(q, cfg, n, iters, buf, elems);
            if (!gr.ok) {
                std::printf("%-18s | %5d | %9s | %11s | %12s | %12s\n",
                            gr.label.c_str(), gr.nodes, "FAIL", "FAIL", "FAIL", "FAIL");
            } else {
                std::printf("%-18s | %5d | %9.2f | %11.2f | %12.1f | %12.1f\n",
                            gr.label.c_str(), gr.nodes, gr.record_ms, gr.finalize_ms,
                            gr.replay_median_us, gr.replay_mean_us);
            }
        }
        std::printf("-------------------+-------+-----------+-------------+--------------+-------------\n");
    }

    sycl::free(buf, q);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
    }

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::printf("Running sycl_graph_opt_test on: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    int total_failures = 0;
    total_failures += test_env_parsing();
    total_failures += test_modes_correctness(q);
    total_failures += test_multiqueue_fork_join(q);
    total_failures += test_whole_graph_update(q);

    if (do_bench && total_failures == 0) {
        run_benchmarks(q);
    }

    std::printf("\nsycl_graph_opt_test: %s (total failures: %d)\n",
                total_failures == 0 ? "PASSED" : "FAILED", total_failures);
    return total_failures == 0 ? 0 : 1;
}
