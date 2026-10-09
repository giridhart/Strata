// The default CPU share is not armed when every expert is resident (#1595): no GPU, no model.
#include <cstdio>

#include "strata/prefill/share_rules.hpp"

using strata::prefill::share_rules::default_share_has_work;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const long long pairs = 48 * 512;
    CHECK(default_share_has_work(pairs - 1, pairs));       // one expert streams: the share has something to take
    CHECK(default_share_has_work(1500, pairs));            // a 12 GB card: most of them stream
    CHECK(!default_share_has_work(pairs, pairs));          // every expert resident: nothing to take
    CHECK(!default_share_has_work(pairs + 100, pairs));    // a cache with spare slots
    CHECK(default_share_has_work(0, pairs));               // empty cache
    CHECK(default_share_has_work(100000, 0));              // total unknown: keep today's behaviour
    static_assert(!default_share_has_work(10, 10), "constexpr");
    if (failed == 0) std::printf("share_rules_test: ok\n");
    return failed ? 1 : 0;
}
