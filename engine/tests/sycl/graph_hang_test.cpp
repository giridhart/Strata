#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include "strata/core/sycl_doorbell.hpp"
#include "strata/kernels/dp4a.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace syclex = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

inline double ms_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Helper for multidimensional SLM rank-0 accessor (matching native_mmvq)
inline void slm_2d_write(float slm[4][32], int lid) {
    if (lid < 32) {
        slm[0][lid] = 1.0f;
        slm[1][lid] = 2.0f;
        slm[2][lid] = 3.0f;
        slm[3][lid] = 4.0f;
    }
}

inline float slm_2d_read(float slm[4][32]) {
    return slm[0][0] + slm[1][0] + slm[2][0] + slm[3][0];
}

inline void composite_slm_write(float slm[4][32], int lid, const float* act) {
    if (lid < 32) {
        slm[0][lid] = act[lid];
        slm[1][lid] = act[lid + 32];
        slm[2][lid] = act[lid + 64];
        slm[3][lid] = act[lid + 96];
    }
}

// -----------------------------------------------------------------------------
// Test Config 1: Shared Local Memory (SLM) Accessors
// Multi-dimensional local_accessor<float[4][32], 0> and 1D local_accessor<float, 1>
// -----------------------------------------------------------------------------
bool test_config_1(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 1] Shared Local Memory (SLM) Accessors" << std::endl;
    std::cout << "Testing multidimensional local_accessor<float[4][32], 0> and local_accessor<float, 1>" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    float* d_out = sycl::malloc_device<float>(1024, q);
    q.fill(d_out, 0.0f, 1024).wait();

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        if (i % 2 == 0) {
            // Rank-0 multidimensional local accessor simulating native_mmvq / native_expert_grouped
            q.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float[4][32], 0> slm_2d(cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{128}, sycl::range<1>{128}},
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                        int lid = it.get_local_id(0);
                        slm_2d_write(slm_2d, lid);
                        sycl::group_barrier(it.get_group());
                        if (lid == 0) {
                            d_out[0] += slm_2d_read(slm_2d);
                        }
                    });
            });
        } else {
            // 1D local accessor simulating reduction scratch
            q.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> slm_1d(sycl::range<1>(128), cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{128}, sycl::range<1>{128}},
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                        int lid = it.get_local_id(0);
                        slm_1d[lid] = (float)lid;
                        sycl::group_barrier(it.get_group());
                        if (lid == 0) {
                            d_out[1] += slm_1d[127];
                        }
                    });
            });
        }
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 1] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 1] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 1] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 1] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 1] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(d_out, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 2: System-Scope Atomic Spin Loops
// strata::sys_load(flag) and strata_spin_pause() in wait_flag_ge
// -----------------------------------------------------------------------------
bool test_config_2(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 2] System-Scope Atomic Spin Loops" << std::endl;
    std::cout << "Testing strata::sys_load(flag) + atomic_fence(system) + strata_spin_pause()" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    uint32_t* host_flag = sycl::malloc_host<uint32_t>(1, q);
    uint32_t* dev_skip = sycl::malloc_device<uint32_t>(1, q);
    *host_flag = 1000;
    q.fill(dev_skip, 0u, 1).wait();

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        uint32_t target_val = 100;
        uint32_t spin_max = 10000;
        if (i % 2 == 0) {
            // wait_flag_ge_kernel
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                [=](sycl::nd_item<1>) {
                    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(host_flag) < target_val; ++spin) {
                        strata_spin_pause();
                    }
                    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
                });
        } else {
            // wait_flag_ge_or_kernel
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                [=](sycl::nd_item<1>) {
                    if (strata::sys_load(dev_skip) == target_val) return;
                    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(host_flag) < target_val; ++spin) {
                        strata_spin_pause();
                    }
                    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
                });
        }
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 2] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 2] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 2] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 2] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 2] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(host_flag, q);
    sycl::free(dev_skip, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 3: Explicit Subgroup Directives
// [[sycl::reqd_sub_group_size(32)]] on kernels inside command graph
// -----------------------------------------------------------------------------
bool test_config_3(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 3] Explicit Subgroup Directives" << std::endl;
    std::cout << "Testing [[sycl::reqd_sub_group_size(32)]] and sub_group operations" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    float* d_buf = sycl::malloc_device<float>(1024, q);
    q.fill(d_buf, 1.0f, 1024).wait();

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        q.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{256}, sycl::range<1>{64}},
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = it.get_sub_group();
                int gid = it.get_global_id(0);
                float val = d_buf[gid % 1024];
                float sum = sycl::reduce_over_group(sg, val, sycl::plus<float>{});
                if (sg.get_local_id()[0] == 0) {
                    d_buf[gid % 1024] = sum;
                }
            });
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 3] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 3] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 3] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 3] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 3] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(d_buf, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 4: Intel Cache-Control Annotated Pointers
// sycl::ext::intel::experimental::read_hint / write_hint with cache_mode::uncached
// -----------------------------------------------------------------------------
bool test_config_4(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 4] Intel Cache-Control Annotated Pointers" << std::endl;
    std::cout << "Testing doorbell_uncached_read and doorbell_uncached_write in command graph" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    uint32_t* host_val = sycl::malloc_host<uint32_t>(1024, q);
    for (int i = 0; i < 1024; ++i) host_val[i] = i;

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        if (i % 2 == 0) {
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{64}, sycl::range<1>{64}},
                [=](sycl::nd_item<1> it) {
                    int id = it.get_global_id(0);
                    uint32_t v = strata::sys_load(&host_val[id]);
                    if (v == 999999) host_val[id] = 0;
                });
        } else {
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{64}, sycl::range<1>{64}},
                [=](sycl::nd_item<1> it) {
                    int id = it.get_global_id(0);
                    strata::sys_store(&host_val[id], (uint32_t)id);
                });
        }
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 4] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 4] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 4] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 4] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 4] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(host_val, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 5: Layer Stacking (Scale-Up)
// Realistic composite layer combining GEMVs + SLM + atomics for L in {1, 5, 10, 20, 61}
// -----------------------------------------------------------------------------
bool test_config_5(sycl::queue& q, const std::vector<int>& layers = {1, 5, 10, 20, 61}) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 5] Layer Stacking (Scale-Up)" << std::endl;
    std::cout << "Testing full layer composite (SLM + Atomics + SG32 + Mapped Memory)" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    const int N = 2048;
    float* dev_act = sycl::malloc_device<float>(N, q);
    float* dev_w = sycl::malloc_device<float>(N, q);
    uint32_t* host_flag = sycl::malloc_host<uint32_t>(128, q);
    int32_t* host_plan = sycl::malloc_host<int32_t>(256, q);
    int32_t* dev_plan = sycl::malloc_device<int32_t>(256, q);

    for (int i = 0; i < 128; ++i) host_flag[i] = 1000;
    for (int i = 0; i < 256; ++i) host_plan[i] = i;
    q.fill(dev_act, 1.0f, N);
    q.fill(dev_w, 0.5f, N);
    q.wait();

    for (int num_layers : layers) {
        std::cout << "\n--- Testing " << num_layers << " stacked layers ---" << std::endl;
        std::cout.flush();

        const auto t_start = Clock::now();
        syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

        graph.begin_recording(q);

        for (int l = 0; l < num_layers; ++l) {
            // Kernel 1: gr_read / norm (GEMV + reduce)
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{256}, sycl::range<1>{64}},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    int gid = it.get_global_id(0);
                    dev_act[gid % N] = dev_act[gid % N] * dev_w[gid % N];
                });

            // Kernel 2: native_mmvq (multidimensional SLM + SG32)
            q.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float[4][32], 0> slm_2d(cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{128}, sycl::range<1>{128}},
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                        int lid = it.get_local_id(0);
                        composite_slm_write(slm_2d, lid, dev_act);
                        sycl::group_barrier(it.get_group());
                        if (lid == 0) {
                            dev_act[0] += slm_2d_read(slm_2d);
                        }
                    });
            });

            // Kernel 3: wait_flag_ge(flagA)
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                [=](sycl::nd_item<1>) {
                    for (uint32_t s = 0; s < 1000 && strata::sys_load(&host_flag[l % 128]) < 1; ++s) {
                        strata_spin_pause();
                    }
                    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
                });

            // Kernel 4: copy_i32_from_mapped (copy plan from host)
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{128}, sycl::range<1>{128}},
                [=](sycl::nd_item<1> it) {
                    for (int i = it.get_local_id(0); i < 256; i += it.get_local_range(0)) {
                        dev_plan[i] = host_plan[i];
                    }
                });

            // Kernel 5: native_expert_grouped (down kernel with 1D SLM)
            q.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<uint32_t, 1> s_buf(sycl::range<1>(180), cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{256}, sycl::range<1>{256}},
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                        int lid = it.get_local_id(0);
                        if (lid < 180) s_buf[lid] = dev_plan[lid % 256];
                        sycl::group_barrier(it.get_group());
                        if (lid == 0) dev_act[1] += (float)s_buf[0];
                    });
            });

            // Kernel 6: wait_flag_ge(flagB)
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                [=](sycl::nd_item<1>) {
                    for (uint32_t s = 0; s < 1000 && strata::sys_load(&host_flag[(l + 1) % 128]) < 1; ++s) {
                        strata_spin_pause();
                    }
                    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
                });

            // Kernel 7: moe_combine / gr_write
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{256}, sycl::range<1>{64}},
                [=](sycl::nd_item<1> it) {
                    int gid = it.get_global_id(0);
                    dev_act[gid % N] += 1.0f;
                });
        }

        graph.end_recording(q);

        const double rec_ms = ms_since(t_start);
        int total_nodes = num_layers * 7;
        std::cout << "[CONFIG 5] Recorded " << total_nodes << " nodes (" << num_layers 
                  << " layers) in " << rec_ms << " ms" << std::endl;
        std::cout << "[CONFIG 5] Calling graph.finalize() for " << num_layers << " layers..." << std::endl;
        std::cout.flush();

        const auto t_fin = Clock::now();
        auto exec = graph.finalize();
        const double fin_ms = ms_since(t_fin);
        std::cout << "[CONFIG 5] SUCCESS: " << num_layers << " layers finalized in " << fin_ms << " ms" << std::endl;
        std::cout.flush();

        q.ext_oneapi_graph(exec);
        q.wait();
        std::cout << "[CONFIG 5] Replay completed for " << num_layers << " layers." << std::endl;
        std::cout.flush();
    }

    sycl::free(dev_act, q);
    sycl::free(dev_w, q);
    sycl::free(host_flag, q);
    sycl::free(host_plan, q);
    sycl::free(dev_plan, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 6: this_work_item::get_nd_item<3>() and get_nd_item<1>()
// -----------------------------------------------------------------------------
bool test_config_6(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 6] this_work_item::get_nd_item in Command Graph" << std::endl;
    std::cout << "Testing sycl::ext::oneapi::this_work_item::get_nd_item<3>() and get_nd_item<1>()" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    int* d_out = sycl::malloc_device<int>(1024, q);
    q.fill(d_out, 0, 1024).wait();

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        if (i % 2 == 0) {
            q.parallel_for(
                sycl::nd_range<3>{sycl::range<3>{1, 4, 256}, sycl::range<3>{1, 1, 256}},
                [=](sycl::nd_item<3> /*unused*/) {
                    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
                    size_t gid = it.get_global_linear_id();
                    d_out[gid % 1024] += (int)it.get_local_id(2);
                });
        } else {
            q.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{256}, sycl::range<1>{256}},
                [=](sycl::nd_item<1> /*unused*/) {
                    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
                    size_t gid = it.get_global_id(0);
                    d_out[gid % 1024] += (int)it.get_local_id(0);
                });
        }
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 6] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 6] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 6] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 6] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 6] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(d_out, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 7: Heterogeneous Ranges (3D nd_range, 2D range, 1D nd_range)
// -----------------------------------------------------------------------------
bool test_config_7(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 7] Heterogeneous Multi-Dimensional Ranges" << std::endl;
    std::cout << "Testing mixed 3D nd_range<3>, 2D basic range<2>, and 1D nd_range<1>" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    float* d_buf = sycl::malloc_device<float>(2048, q);
    q.fill(d_buf, 1.0f, 2048).wait();

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        if (i % 3 == 0) {
            // 3D nd_range (simulating native_gu_kernel / native_down_kernel)
            q.parallel_for(
                sycl::nd_range<3>{sycl::range<3>{1, 2, 512}, sycl::range<3>{1, 1, 256}},
                [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                    size_t gid = it.get_global_linear_id();
                    d_buf[gid % 2048] += 0.5f;
                });
        } else if (i % 3 == 1) {
            // 2D basic range (simulating verify.cpp line 892 & 914)
            q.parallel_for(sycl::range<2>{16, 128}, [=](sycl::id<2> id) {
                size_t r = id[0];
                size_t c = id[1];
                d_buf[r * 128 + c] *= 1.001f;
            });
        } else {
            // 1D nd_range
            q.parallel_for(sycl::nd_range<1>{1024, 128}, [=](sycl::nd_item<1> it) {
                size_t gid = it.get_global_id(0);
                d_buf[gid % 2048] += 1.0f;
            });
        }
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 7] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 7] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 7] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 7] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 7] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(d_buf, q);
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 8: High Argument Count Kernels (12-16 parameters)
// -----------------------------------------------------------------------------
struct KernelArgs {
    float* p1; float* p2; float* p3; float* p4;
    int32_t* i1; int32_t* i2; int32_t* i3; int32_t* i4;
    uint32_t* u1; uint32_t* u2;
    int n1; int n2; int n3; float s1;
};

bool test_config_8(sycl::queue& q, int num_nodes = 100) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 8] Kernels with 14 Pointer/Scalar Arguments" << std::endl;
    std::cout << "Testing kernels with many arguments simulating expert_grouped & shared_expert" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    float* f_bufs[4];
    int32_t* i_bufs[4];
    uint32_t* u_bufs[2];
    for (int k = 0; k < 4; ++k) {
        f_bufs[k] = sycl::malloc_device<float>(256, q);
        i_bufs[k] = sycl::malloc_device<int32_t>(256, q);
    }
    for (int k = 0; k < 2; ++k) {
        u_bufs[k] = sycl::malloc_device<uint32_t>(256, q);
    }
    q.wait();

    KernelArgs ka;
    ka.p1 = f_bufs[0]; ka.p2 = f_bufs[1]; ka.p3 = f_bufs[2]; ka.p4 = f_bufs[3];
    ka.i1 = i_bufs[0]; ka.i2 = i_bufs[1]; ka.i3 = i_bufs[2]; ka.i4 = i_bufs[3];
    ka.u1 = u_bufs[0]; ka.u2 = u_bufs[1];
    ka.n1 = 256; ka.n2 = 128; ka.n3 = 64; ka.s1 = 0.5f;

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);
    for (int i = 0; i < num_nodes; ++i) {
        q.parallel_for(
            sycl::nd_range<1>{256, 128},
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                int id = it.get_global_id(0);
                ka.p1[id] = ka.p2[id] + ka.p3[id] * ka.s1;
                ka.i1[id] = ka.i2[id] + ka.i3[id] + ka.n1;
                ka.u1[id] = ka.u2[id] + (uint32_t)id;
            });
    }
    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    std::cout << "[CONFIG 8] Recorded " << num_nodes << " nodes in " << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 8] Calling graph.finalize()..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 8] SUCCESS in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 8] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 8] Replay completed successfully." << std::endl;
    std::cout.flush();

    for (int k = 0; k < 4; ++k) {
        sycl::free(f_bufs[k], q);
        sycl::free(i_bufs[k], q);
    }
    for (int k = 0; k < 2; ++k) {
        sycl::free(u_bufs[k], q);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Test Config 9: Full Real-World Topology (1,500+ nodes, 61 layers)
// -----------------------------------------------------------------------------
bool test_config_9(sycl::queue& q) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "[CONFIG 9] Full Realistic Verify Window Topology (61 layers, ~1,500 nodes)" << std::endl;
    std::cout << "Testing full 61-layer sequence with 3D/2D ranges, this_work_item, atomics, USM" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout.flush();

    const int N = 2560;
    float* d_R = sycl::malloc_device<float>(N * 4, q);
    float* d_act = sycl::malloc_device<float>(N, q);
    float* d_w = sycl::malloc_device<float>(N, q);
    uint32_t* h_flagA = sycl::malloc_host<uint32_t>(128, q);
    uint32_t* h_flagB = sycl::malloc_host<uint32_t>(128, q);
    int32_t* h_plan = sycl::malloc_host<int32_t>(256, q);
    int32_t* d_plan = sycl::malloc_device<int32_t>(256, q);

    for (int i = 0; i < 128; ++i) { h_flagA[i] = 1000; h_flagB[i] = 1000; }
    for (int i = 0; i < 256; ++i) h_plan[i] = i;
    q.fill(d_R, 0.0f, N * 4);
    q.fill(d_act, 1.0f, N);
    q.fill(d_w, 0.5f, N);
    q.wait();

    const auto t_start = Clock::now();
    syclex::command_graph<syclex::graph_state::modifiable> graph{q.get_context(), q.get_device()};

    graph.begin_recording(q);

    for (int l = 0; l < 61; ++l) {
        // 1. fused_gr_read_multi
        q.parallel_for(sycl::nd_range<1>{256, 64}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            int gid = it.get_global_id(0);
            d_act[gid % N] = d_R[gid % N] + d_w[gid % N];
        });

        // 2. native_quantize_q8_1
        q.parallel_for(sycl::nd_range<1>{128, 64}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_act[gid % N] = d_act[gid % N] * 0.125f;
        });

        // 3. native_mmvq (multidimensional SLM + SG32)
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[4][32], 0> slm_2d(cgh);
            cgh.parallel_for(
                sycl::nd_range<1>{128, 128},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    int lid = it.get_local_id(0);
                    composite_slm_write(slm_2d, lid, d_act);
                    sycl::group_barrier(it.get_group());
                    if (lid == 0) d_act[0] += slm_2d_read(slm_2d);
                });
        });

        // 4. GDN conv
        q.parallel_for(sycl::nd_range<1>{64, 64}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_act[gid % N] += 0.1f;
        });

        // 5. GDN gate
        q.parallel_for(sycl::nd_range<1>{64, 64}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_act[gid % N] *= 0.9f;
        });

        // 6. GDN step norm
        q.parallel_for(sycl::nd_range<1>{64, 64}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_act[gid % N] += 0.2f;
        });

        // 7. Router GEMV
        q.parallel_for(sycl::nd_range<1>{512, 128}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            int gid = it.get_global_id(0);
            d_act[gid % N] += 0.01f;
        });

        // 8. Router top10
        q.parallel_for(sycl::nd_range<1>{64, 32}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            int gid = it.get_global_id(0);
            d_act[gid % N] += 0.02f;
        });

        // 9. shared_expert
        q.parallel_for(sycl::nd_range<1>{256, 128}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_act[gid % N] += 0.05f;
        });

        // 10. wait_flag_ge(flagA)
        q.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
            for (uint32_t s = 0; s < 1000 && strata::sys_load(&h_flagA[l % 128]) < 1; ++s) strata_spin_pause();
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        });

        // 11. copy_i32_from_mapped with this_work_item
        q.parallel_for(sycl::nd_range<1>{128, 128}, [=](sycl::nd_item<1> /*it*/) {
            auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
            for (int i = item.get_local_id(0); i < 256; i += item.get_local_range(0)) d_plan[i] = h_plan[i];
        });

        // 12. native_expert_grouped: launch_gu (3D nd_range)
        q.parallel_for(
            sycl::nd_range<3>{sycl::range<3>{1, 2, 512}, sycl::range<3>{1, 1, 256}},
            [=](sycl::nd_item<3> /*it*/) [[sycl::reqd_sub_group_size(32)]] {
                auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
                size_t gid = it.get_global_linear_id();
                d_act[gid % N] += 0.001f;
            });

        // 13. native_expert_grouped: swiglu (3D nd_range)
        q.parallel_for(
            sycl::nd_range<3>{sycl::range<3>{1, 1, 256}, sycl::range<3>{1, 1, 256}},
            [=](sycl::nd_item<3> /*it*/) [[sycl::reqd_sub_group_size(32)]] {
                auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
                size_t gid = it.get_global_linear_id();
                d_act[gid % N] += 0.002f;
            });

        // 14. native_expert_grouped: launch_down with SLM (3D nd_range)
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> s_buf(sycl::range<1>(180), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>{sycl::range<3>{1, 2, 512}, sycl::range<3>{1, 1, 256}},
                [=](sycl::nd_item<3> /*it*/) [[sycl::reqd_sub_group_size(32)]] {
                    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
                    int lid = it.get_local_id(2);
                    if (lid < 180) s_buf[lid] = d_plan[lid % 256];
                    it.barrier(sycl::access::fence_space::local_space);
                    if (lid == 0) d_act[1] += (float)s_buf[0];
                });
        });

        // 15. wait_flag_ge(flagB)
        q.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
            for (uint32_t s = 0; s < 1000 && strata::sys_load(&h_flagB[l % 128]) < 1; ++s) strata_spin_pause();
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        });

        // 16. PCIe native_expert_grouped: launch_gu (3D nd_range)
        q.parallel_for(
            sycl::nd_range<3>{sycl::range<3>{1, 1, 256}, sycl::range<3>{1, 1, 256}},
            [=](sycl::nd_item<3> /*it*/) [[sycl::reqd_sub_group_size(32)]] {
                auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
                size_t gid = it.get_global_linear_id();
                d_act[gid % N] += 0.003f;
            });

        // 17. PCIe native_expert_grouped: launch_down (3D nd_range)
        q.parallel_for(
            sycl::nd_range<3>{sycl::range<3>{1, 1, 256}, sycl::range<3>{1, 1, 256}},
            [=](sycl::nd_item<3> /*it*/) [[sycl::reqd_sub_group_size(32)]] {
                auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
                size_t gid = it.get_global_linear_id();
                d_act[gid % N] += 0.004f;
            });

        // 18. 2D range copy (verify.cpp line 892 & 914)
        q.parallel_for(sycl::range<2>{4, 128}, [=](sycl::id<2> id) {
            size_t r = id[0]; size_t c = id[1];
            d_act[r * 128 + c] += 0.005f;
        });

        // 19. moe_combine
        q.parallel_for(sycl::nd_range<1>{256, 64}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_act[gid % N] += 0.01f;
        });

        // 20. gr_write
        q.parallel_for(sycl::nd_range<1>{256, 64}, [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            d_R[gid % (N * 4)] += d_act[gid % N];
        });
    }

    graph.end_recording(q);

    const double rec_ms = ms_since(t_start);
    int total_nodes = 61 * 20;
    std::cout << "[CONFIG 9] Recorded " << total_nodes << " nodes (61 layers x 20 kernels) in " 
              << rec_ms << " ms" << std::endl;
    std::cout << "[CONFIG 9] Calling graph.finalize() for 1,220 nodes..." << std::endl;
    std::cout.flush();

    const auto t_fin = Clock::now();
    auto exec = graph.finalize();
    const double fin_ms = ms_since(t_fin);
    std::cout << "[CONFIG 9] SUCCESS: 1,220 nodes finalized in " << fin_ms << " ms" << std::endl;
    std::cout.flush();

    std::cout << "[CONFIG 9] Replaying graph..." << std::endl;
    std::cout.flush();
    q.ext_oneapi_graph(exec);
    q.wait();
    std::cout << "[CONFIG 9] Replay completed successfully." << std::endl;
    std::cout.flush();

    sycl::free(d_R, q);
    sycl::free(d_act, q);
    sycl::free(d_w, q);
    sycl::free(h_flagA, q);
    sycl::free(h_flagB, q);
    sycl::free(h_plan, q);
    sycl::free(d_plan, q);
    return true;
}

// -----------------------------------------------------------------------------
// Main CLI Driver
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    sycl::queue q{sycl::property::queue::in_order{}};
    std::cout << "=== SYCL Command Graph Hang Investigation Harness ===" << std::endl;
    std::cout << "Target Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Driver Version: " << q.get_device().get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout.flush();

    int config = 0;
    bool run_all = false;

    if (argc == 1) {
        run_all = true;
    } else {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
                config = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--all") == 0) {
                run_all = true;
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                std::cout << "Usage: " << argv[0] << " [--config <1-9>] [--all]" << std::endl;
                return 0;
            }
        }
    }

    if (!run_all && (config < 1 || config > 9)) {
        std::cout << "Please specify --config <1-9> or --all. Use -h for help." << std::endl;
        return 1;
    }

    bool success = true;
    if (run_all || config == 1) success &= test_config_1(q);
    if (run_all || config == 2) success &= test_config_2(q);
    if (run_all || config == 3) success &= test_config_3(q);
    if (run_all || config == 4) success &= test_config_4(q);
    if (run_all || config == 5) success &= test_config_5(q);
    if (run_all || config == 6) success &= test_config_6(q);
    if (run_all || config == 7) success &= test_config_7(q);
    if (run_all || config == 8) success &= test_config_8(q);
    if (run_all || config == 9) success &= test_config_9(q);

    std::cout << "\n=======================================================" << std::endl;
    std::cout << (success ? "All tested configs passed successfully!" : "Some configs failed or hung.") << std::endl;
    std::cout << "=======================================================" << std::endl;
    return success ? 0 : 1;
}
