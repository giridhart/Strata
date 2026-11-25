// include/strata/core/sycl_graph_env.hpp
// Opt-in runtime environment controls for SYCL command_graph recording & finalization.
// All options default to OFF so the default path stays identical to prior releases.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <cstdlib>

namespace strata::sycl_graph {

struct GraphOptions {
    bool native_recording = false;
    bool no_cycle_check = false;
    bool end_all_queues = false;
};

inline bool env_flag(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    return v[0] != '0';
}

inline GraphOptions read_options() {
    const bool opt = env_flag("STRATA_SYCL_GRAPH_OPT", false);
    GraphOptions o;
    o.native_recording = env_flag("STRATA_SYCL_GRAPH_NATIVE", opt);
    o.no_cycle_check = env_flag("STRATA_SYCL_GRAPH_NO_CYCLE_CHECK", opt);
    o.end_all_queues = env_flag("STRATA_SYCL_GRAPH_END_ALL", opt);
    return o;
}

inline sycl::property_list make_properties(const GraphOptions& o) {
    namespace pg = sycl::ext::oneapi::experimental::property::graph;
    if (o.native_recording && o.no_cycle_check) {
        return sycl::property_list{pg::enable_native_recording{}, pg::no_cycle_check{}};
    }
    if (o.native_recording) {
        return sycl::property_list{pg::enable_native_recording{}};
    }
    if (o.no_cycle_check) {
        return sycl::property_list{pg::no_cycle_check{}};
    }
    return sycl::property_list{};
}

}  // namespace strata::sycl_graph
