// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include "strata/kernels/cvec.hpp"

#include <stdexcept>

namespace strata::kernels {
namespace {

static sycl::queue default_queue;

constexpr int THREADS = 256;
constexpr int MAXK = 16;  // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;  // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables {
    float* dir = nullptr;
    float* s = nullptr;
    int* on = nullptr;
};
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() { return 0; }
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;

    t.dir = sycl::malloc_device<float>(g_dir_host.size(), default_queue);
    t.s = sycl::malloc_device<float>(g_s_host.size(), default_queue);
    t.on = sycl::malloc_device<int>(1, default_queue);
    if (!t.dir || !t.s || !t.on) {
        err = "failed";
        t = DevTables{};
        return false;
    }
    default_queue.memcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float)).wait();
    default_queue.memcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float)).wait();
    default_queue.memcpy(t.on, &flag, sizeof(int)).wait();
    return true;
}

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }

// one block per (stream, token): the pending write, then h . v over the stream, then the update
/*
DPCT1110:2: The total declared local variable size in device function
cvec_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                 const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                 const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj, int64_t inj_ld, int write,
                 float* part, const sycl::nd_item<1>& item) {
    const int c = item.get_group(0) % hc;
    const int64_t t = item.get_group(0) / hc;
    float* r = R + t * r_ld + (int64_t)c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;  // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float)hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item.get_local_id(0) + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = sycl::fma((float)(b[d]), (float)w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = sycl::fma(xv, (float)(v[d]), dot);
        }
    }
    if (steer && mode == 0) {
        dot = sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), dot, sycl::plus<float>());
        if ((item.get_local_id(0) & 31) == 0) part[item.get_local_id(0) >> 5] = dot;
        item.barrier(sycl::access::fence_space::local_space);
        if (item.get_local_id(0) < 32) {
            float p = item.get_local_id(0) < THREADS / 32 ? part[item.get_local_id(0)] : 0.0f;
            p = sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), p, sycl::plus<float>());
            if (item.get_local_id(0) == 0) part[0] = p;
        }
        /*
        DPCT1118:4: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:215: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item.barrier();
        dot = part[0] * s;  // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item.get_local_id(0) + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? sycl::fma(-dot, (float)(v[d]), xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t)THREADS * MAXK) {
        err = "control vector: unsupported n_embd";
        return false;
    }
    if (s.empty() || dir.size() != s.size() * (size_t)n_embd) {
        err = "control vector: bad table sizes";
        return false;
    }
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        sycl::free(g_dev[d].dir, default_queue);
        sycl::free(g_dev[d].s, default_queue);
        sycl::free(g_dev[d].on, default_queue);
        g_dev[d] = DevTables{};
    }
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l)
        g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        default_queue.wait();
        default_queue.memcpy(g_dev[d].on, &v, sizeof(int)).wait();
    }
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr)
        throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const unsigned blocks = ((unsigned)g_cvec.hc) * ((unsigned)T);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part_acc(sycl::range<1>(THREADS / 32), cgh);

        auto g_cvec_mode = g_cvec.mode;
        auto g_cvec_n_embd = (int)g_cvec.n_embd;
        auto g_cvec_hc = (int)g_cvec.hc;

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * THREADS), sycl::range<1>(THREADS)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                             cvec_kernel(R, t.dir, t.s, t.on, g_cvec_mode, layer, g_cvec_n_embd, g_cvec_hc, r_ld, bo,
                                         bo_ld, inj, inj_ld, write ? 1 : 0,
                                         part_acc.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });
    if (!stream) default_queue.wait();
}

}  // namespace strata::kernels
