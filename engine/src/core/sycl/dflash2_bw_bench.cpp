// read-bandwidth probe: sum of a 256 MB buffer with float4 loads, a few launch shapes
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>

static void rd(const sycl::float4* __restrict__ p, size_t n, float* out, sycl::nd_item<1> item) {
    float s = 0;
    const size_t stride = item.get_global_range(0);
    for (size_t i = item.get_global_id(0); i < n; i += stride) {
        sycl::float4 v = p[i];
        s += v.x() + v.y() + v.z() + v.w();
    }
    if (s == 1234.5f) out[0] = s;
}

static void rd_unroll(const sycl::int4* __restrict__ p, size_t n, int* out, sycl::nd_item<1> item) {
    int s = 0;
    const size_t stride = item.get_global_range(0);
    size_t i = item.get_global_id(0);
    for (; i + 3 * stride < n; i += 4 * stride) {
        sycl::int4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        s += a.x() ^ a.y() ^ a.z() ^ a.w() ^ b.x() ^ b.y() ^ b.z() ^ b.w() ^ c.x() ^ c.y() ^ c.z() ^ c.w() ^ d.x() ^ d.y() ^ d.z() ^ d.w();
    }
    if (s == 1234) out[0] = s;
}

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const size_t bytes = 256u << 20, n = bytes / 16;
    void* p = sycl::malloc_device(bytes, q);
    float* o = sycl::malloc_device<float>(1, q);
    q.memset(p, 1, bytes).wait();
    q.memset(o, 0, sizeof(float)).wait();

    for (int blocks : {112, 224, 448, 896, 3584, 14336}) {
        for (int k = 0; k < 2; ++k) {
            q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 20; ++it) {
                if (k == 0) {
                    q.parallel_for(sycl::nd_range<1>(blocks * 256, 256), [=](sycl::nd_item<1> item) {
                        rd((const sycl::float4*) p, n, o, item);
                    });
                } else {
                    q.parallel_for(sycl::nd_range<1>(blocks * 256, 256), [=](sycl::nd_item<1> item) {
                        rd_unroll((const sycl::int4*) p, n, (int*) o, item);
                    });
                }
            }
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("%s blocks %5d: %.1f GB/s\n", k == 0 ? "loop  " : "unroll", blocks, 20.0 * bytes / (ms / 1000.0) / 1e9);
        }
    }

    sycl::free(p, q);
    sycl::free(o, q);
    return 0;
}
