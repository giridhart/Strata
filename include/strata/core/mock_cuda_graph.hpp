#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <cstddef>
#include <map>
#include <memory>

#define cudaStreamCaptureModeThreadLocal 0

namespace strata_graph_detail {
namespace sycl_ext = sycl::ext::oneapi::experimental;
using mod_graph_t = sycl_ext::command_graph<sycl_ext::graph_state::modifiable>;
using exec_graph_t = sycl_ext::command_graph<sycl_ext::graph_state::executable>;

inline std::map<void*, mod_graph_t*>& get_active_recording() {
    static thread_local std::map<void*, mod_graph_t*> active;
    return active;
}
} // namespace strata_graph_detail

inline int cudaStreamBeginCapture(void* stream, int = 0) {
    if (!stream) return 1;
    auto* q = static_cast<sycl::queue*>(stream);
    try {
        auto* g = new strata_graph_detail::mod_graph_t(q->get_context(), q->get_device());
        g->begin_recording(*q);
        strata_graph_detail::get_active_recording()[stream] = g;
    } catch (...) {
        return 1;
    }
    return 0;
}

inline int cudaStreamEndCapture(void* stream, void** out_graph) {
    if (!stream || !out_graph) return 1;
    auto& active = strata_graph_detail::get_active_recording();
    auto it = active.find(stream);
    if (it == active.end()) return 1;
    auto* q = static_cast<sycl::queue*>(stream);
    auto* g = it->second;
    active.erase(it);
    try {
        g->end_recording(*q);
        *out_graph = static_cast<void*>(g);
    } catch (...) {
        delete g;
        *out_graph = nullptr;
        return 1;
    }
    return 0;
}

inline int cudaGraphGetNodes(void* graph, void*, size_t* count) {
    if (!graph || !count) return 1;
    auto* g = static_cast<strata_graph_detail::mod_graph_t*>(graph);
    try {
        *count = g->get_nodes().size();
    } catch (...) {
        *count = 0;
        return 1;
    }
    return 0;
}

inline int cudaGraphInstantiate(void** out_exec, void* graph, ...) {
    if (!graph || !out_exec) return 1;
    auto* g = static_cast<strata_graph_detail::mod_graph_t*>(graph);
    try {
        auto* exec = new strata_graph_detail::exec_graph_t(g->finalize());
        *out_exec = static_cast<void*>(exec);
    } catch (...) {
        *out_exec = nullptr;
        return 1;
    }
    return 0;
}

inline int cudaGraphLaunch(void* exec, void* stream) {
    if (!exec || !stream) return 1;
    auto* exec_g = static_cast<strata_graph_detail::exec_graph_t*>(exec);
    auto* q = static_cast<sycl::queue*>(stream);
    try {
        q->ext_oneapi_graph(*exec_g);
    } catch (...) {
        return 1;
    }
    return 0;
}

inline int cudaGraphDestroy(void* graph) {
    if (graph) delete static_cast<strata_graph_detail::mod_graph_t*>(graph);
    return 0;
}

inline int cudaGraphExecDestroy(void* exec) {
    if (exec) delete static_cast<strata_graph_detail::exec_graph_t*>(exec);
    return 0;
}

inline int cudaGraphUpload(void*, void*) {
    return 0;
}

typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaGraphNode_t;
typedef int cudaGraphNodeType;
#define cudaGraphNodeTypeKernel 0
#define cudaGraphNodeTypeMemcpy 1
#define cudaGraphNodeTypeMemset 2
struct cudaKernelNodeParams { int x; };
inline int cudaGraphNodeGetType(void*, int* b) { if(b) *b = 0; return 0; }
inline int cudaGraphKernelNodeGetParams(void*, void*) { return 0; }
inline int cudaGraphLaunch_mock(void* a, void* b) { return cudaGraphLaunch(a, b); }
inline int cudaGraphExecDestroy_mock(void* a) { return cudaGraphExecDestroy(a); }
#define cudaDeviceScheduleSpin 1
#define cudaDeviceMapHost 2
inline int cudaInitDevice(int, int, int) { return 0; }
