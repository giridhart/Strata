// src/core/sycl/graph.cpp.dp.cpp - P2.S5: Pure SYCL 2020 command_graph implementation.
#include "strata/core/graph.hpp"

#include <chrono>
#include <cstdio>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::core {

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = std::move(o.graph_);
        exec_ = std::move(o.exec_);
        done_ = std::move(o.done_);
        nodes_ = o.nodes_;
        o.nodes_ = 0;
    }
    return *this;
}

void CapturedGraph::reset() {
    exec_.reset();
    graph_.reset();
    done_.reset();
    nodes_ = 0;
}

bool CapturedGraph::begin(void* stream, std::string& err) {
    if (graph_ || exec_) {
        err = "begin: this CapturedGraph is already recorded";
        return false;
    }
    if (!stream) {
        err = "begin: null stream/queue pointer";
        return false;
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        graph_ = std::make_unique<modifiable_graph_t>(q->get_context(), q->get_device());
        graph_->begin_recording(*q);
    } catch (const sycl::exception& e) {
        graph_.reset();
        err = std::string("begin (begin_recording): ") + e.what();
        return false;
    }
    return true;
}

bool CapturedGraph::end(void* stream, std::string& err) {
    if (!graph_) {
        err = "end: no graph is currently being recorded";
        return false;
    }
    if (!stream) {
        err = "end: null stream/queue pointer";
        return false;
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        graph_->end_recording(*q);
        nodes_ = graph_->get_nodes().size();
        if (nodes_ == 0) {
            err = "end: the capture recorded ZERO nodes - the body launched nothing";
            reset();
            return false;
        }
        exec_ = std::make_unique<executable_graph_t>(graph_->finalize());
    } catch (const sycl::exception& e) {
        reset();
        err = std::string("end (finalize): ") + e.what();
        return false;
    }
    return true;
}

bool CapturedGraph::launch(void* stream, std::string& err) const {
    if (!exec_) {
        err = "launch: not recorded";
        return false;
    }
    if (!stream) {
        err = "launch: null stream/queue pointer";
        return false;
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        done_ = std::make_unique<sycl::event>(q->ext_oneapi_graph(*exec_));
    } catch (const sycl::exception& e) {
        err = std::string("launch (ext_oneapi_graph): ") + e.what();
        return false;
    }
    return true;
}

bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!done_) return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        try {
            auto status = done_->get_info<sycl::info::event::command_execution_status>();
            if (status == sycl::info::event_command_status::complete) {
                done_->wait();
                return true;
            }
        } catch (...) {
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        for (int i = 0; i < 64; ++i) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }
}

bool GraphRegistry::record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err) {
    const Key k{(int) type, n_tokens};
    if (graphs_.count(k)) return true;

    CapturedGraph g;
    if (!g.begin(stream_, err)) return false;
    body();
    if (!g.end(stream_, err)) return false;

    graphs_.emplace(k, std::move(g));
    ++captures_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType type, int n_tokens) const {
    const auto it = graphs_.find(Key{(int) type, n_tokens});
    return it == graphs_.end() ? nullptr : &it->second;
}

bool GraphRegistry::launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const {
    const CapturedGraph* g = find(type, n_tokens);
    if (!g) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "launch: no graph recorded for %s n=%d", to_string(type), n_tokens);
        err = buf;
        return false;
    }
    if (!g->launch(stream_, err)) return false;
    if (!g->wait_ms(timeout_ms)) {
        err = "launch: timed out waiting for the graph to complete";
        return false;
    }
    return true;
}

}  // namespace strata::core
