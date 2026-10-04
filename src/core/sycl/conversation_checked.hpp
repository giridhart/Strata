#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace strata::core::conversation_detail {
inline bool add(size_t& total, size_t n) {
    /*
    DPCT1007:96: Migration of std::numeric_limits::max is not supported.
    */
    if (n > std::numeric_limits<size_t>::max() - total) return false;
    total += n;
    return true;
}
inline bool product(size_t& out, std::initializer_list<uint64_t> factors) {
    size_t value = 1;
    for (uint64_t factor : factors) {
        /*
        DPCT1007:97: Migration of std::numeric_limits::max is not supported.
        */
        if (factor > std::numeric_limits<size_t>::max() ||
            /*
            DPCT1007:98: Migration of std::numeric_limits::max is not supported.
            */
            (factor &&
             value > std::numeric_limits<size_t>::max() / factor)) return false;
        value *= (size_t) factor;
    }
    out = value;
    return true;
}
} // namespace strata::core::conversation_detail
