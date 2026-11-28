// src/kernels/cuda/elementwise.cu - P2.S5's glue kernels.  See the header for why each exists.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/core/sycl_doorbell.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

void embedding_gather_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                             const float* __restrict__ offsets, int64_t n, int code_bits, int code_bias,
                             int group_elems, float* __restrict__ out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    volatile float product = (float)(code + code_bias) * scales[group];
    out[i] = product + (offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
///
/// The branch is not cosmetic. `exp(89)` overflows f32 and `exp(20)` is already 4.85e8 where `log1p` loses
/// relative precision; above 20 the function is `x` to within f32 anyway.
inline float softplus_dev(float x) { return x > 20.0f ? x : sycl::log1p(sycl::native::exp(x)); }

void gdn_gate_kernel(const float* __restrict__ alpha, const float* __restrict__ dt, const float* __restrict__ ssm_a,
                     float* __restrict__ gate, int64_t h_v) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i >= h_v) return;
    const int64_t t = i / h_v;  // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void)t;
}

void scale_kernel(float* __restrict__ x, int64_t n, float s) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i < n) {
        volatile float vi = x[i];
        volatile float vs = s;
        x[i] = vi * vs;
    }
}

// `dst[i] += src[i]`.  See the header: this is what lets R4's GPU half and CPU half run at the same
// time and still add up to one `parts` buffer.
void add_kernel(float* __restrict__ dst, const float* __restrict__ src, long long n) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const long long i = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i < n) dst[i] += src[i];
}

void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i < n) y[i] = f16_from_f32(x[i]);
}

void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i < n) y[i] = bf16_from_f32(x[i]);
}

void silu_kernel(float* __restrict__ x, int64_t n) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double)x[i];
    x[i] = (float)(v / (1.0 + sycl::exp(-v)));
}

/// A bounded launch: a zero-length grid is illegal, and `n == 0` is a real call (an empty selection).
inline unsigned grid_for(int64_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

/// One WARP per row, reduced through shuffles.  The rows here are short and few - (24, 256), (2, 256),
/// (4, 128) - so a block-per-row tree would spend its time in `__syncthreads` for 8 warps of 32, and the whole
/// call is 30 rows.  A warp reduction with NO shared memory and NO barrier is the shape that fits.
void rms_norm_weighted_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t rows, int64_t cols,
                              float eps) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int lane = item.get_local_id(0) & 31;
    const int64_t row = ((int64_t)item.get_group(0) * (item.get_local_range(0) >> 5)) + (item.get_local_id(0) >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
    for (int64_t c = lane; c < cols; c += 32)
        acc += r[c] * r[c];
    acc = sycl::reduce_over_group(item.get_sub_group(), acc, sycl::plus<float>());
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = sycl::rsqrt(acc / (float)cols + eps);
    inv = sycl::select_from_group(item.get_sub_group(), inv, 0);
    for (int64_t c = lane; c < cols; c += 32)
        r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

bool check_launch([[maybe_unused]] const char* what) { return true; }

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> /*item*/) {
                        embedding_gather_kernel(codes, scales, offsets, n, code_bits, code_bias, group_elems, out);
                    });
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens, int64_t h_v,
              void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> /*item*/) { gdn_gate_kernel(alpha, dt, ssm_a, gate, h_v); });
    check_launch("gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> /*item*/) { scale_kernel(x, n, s); });
    check_launch("scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> /*item*/) { add_kernel(dst, src, n); });
    check_launch("add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> /*item*/) { to_f16_kernel(x, y, n); });
    check_launch("f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> /*item*/) { to_bf16_kernel(x, y, n); });
    check_launch("f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    {
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid_for(n)) * (THREADS)), sycl::range<1>(THREADS)),
                        [=](sycl::nd_item<1> /*item*/) { silu_kernel(x, n); });
    }
    check_launch("silu_inplace");
}

/// THE DOORBELL.  One thread, one INCREMENT - the cost is the launch, and inside a graph that is paid once.
///
/// **IT INCREMENTS THE MEMORY, AND IT DOES NOT TAKE THE VALUE AS AN ARGUMENT.**  The first version did -
/// doorbell_ring(d_seq, *(h_seq) + 1u) - and that is a CLONED LITERAL: the expression is evaluated on the HOST
/// at CAPTURE time, so the graph rings 1 on every replay and a host waiting for a change waits forever.  It
/// would have looked like a working doorbell on the first token, which is the worst way for it to be wrong.
/// **THE FENCE IS NOT DECORATION, AND ITS ABSENCE WAS COSTING ~10 ms PER TOKEN ON THE HOST SIDE.**
///
/// The ring PUBLISHES three buffers the host is about to read - `h_x_f`, `h_ids`, `h_weights` - so the increment
/// must be ordered after their writes, or the host can observe the ring and then read a payload that has not
/// landed.  `__threadfence_system()` is what orders them, and it covers the HOST as well as the device, which is
/// the whole point: the reader is the CPU.
///
/// Without it the host loop was compensating with a driver call.  `session_loop` polled with `cudaEventQuery`
/// on EVERY spin iteration, because round 195 had measured that a memory-only spin never saw the datum flip -
/// and that measurement was right about the symptom and wrong about the cause.  The cause is this missing fence:
/// the write was not ordered into host-visible memory, so no amount of reading it would show it, and the driver
/// call was flushing the whole pipeline enough to make it appear.  A 10-22 us driver call per iteration is a
/// very expensive substitute for one fence instruction.
///
/// **THE STORE IS VOLATILE**, like `doorbell_publish_kernel`'s.  On RDNA4 (gfx1201) a plain store to mapped pinned
/// memory stays in the GPU's L2 until the stream is synchronized - the host never saw the ring (tests/hip/handoff:
/// 0 of 100 rings seen without a sync; volatile, a system-scope atomic store or a fence after the store: 100 of 100).
void doorbell_ring_kernel(uint32_t* seq) {
    strata::sys_store(seq, strata::sys_load(seq) + 1u);
}

void doorbell_wait_kernel(const volatile uint32_t* flag, const volatile uint32_t* seq, uint32_t spin_max) {
    const uint32_t want = strata::sys_load(seq);
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) != want; ++spin)
        strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (d_flag == nullptr || d_seq == nullptr) return;
    const uint32_t spin_max = strata::spin_max(*q);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                    [=](sycl::nd_item<1> /*item*/) { doorbell_wait_kernel(d_flag, d_seq, spin_max); });
    check_launch("doorbell_wait");
}

void copy_from_mapped_kernel(sycl::float4* __restrict__ dst, const sycl::float4* src, int64_t n4) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    for (int64_t i = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); i < n4;
         i += (int64_t)item.get_group_range(0) * item.get_local_range(0)) {
        const sycl::float4 v = const_cast<const sycl::float4*>(src)[i];
        dst[i] = v;
    }
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): block = row, the plan's hit rows `dst[0, *count)`.
void copy_rows_from_mapped_kernel(sycl::float4* __restrict__ dst, const sycl::float4* src, int64_t row4,
                                  const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count, int& hit) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int row = item.get_group(0);

    if (item.get_local_id(0) == 0) {
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i)
            h |= hit_rows[i] == row;
        hit = h;
    }
    item.barrier(sycl::access::fence_space::local_space);
    sycl::float4* d = dst + (int64_t)row * row4;
    if (hit) {
        for (int64_t i = item.get_local_id(0); i < row4; i += item.get_local_range(0))
            d[i] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const sycl::float4* sr = src + (int64_t)row * row4;
        for (int64_t i = item.get_local_id(0); i < row4; i += item.get_local_range(0))
            d[i] = const_cast<const sycl::float4*>(sr)[i];
    }
}
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t)dst & 15) != 0 || ((uintptr_t)src & 15) != 0) {
        std::fprintf(stderr,
                     "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 0> hit_acc_ct1(cgh);

        auto width_ct2 = width / 4;

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(((unsigned)rows) * (128)), sycl::range<1>(128)),
                         [=](sycl::nd_item<1> /*item*/) {
                             copy_rows_from_mapped_kernel((sycl::float4*)dst, (const sycl::float4*)src, width_ct2,
                                                          hit_rows, count, hit_acc_ct1);
                         });
    });
}
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t)dst & 15) != 0 || ((uintptr_t)src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int blocks = (int)((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    q->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((blocks) * (256)), sycl::range<1>(256)),
        [=](sycl::nd_item<1> /*item*/) { copy_from_mapped_kernel((sycl::float4*)dst, (const sycl::float4*)src, n4); });
    check_launch("copy_from_mapped");
}

void doorbell_publish_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, const float* __restrict__ w,
                             int n, int k, float* x_out, int32_t* ids_out, float* w_out, uint32_t* seq) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    for (int i = item.get_local_id(0); i < n; i += item.get_local_range(0))
        x_out[i] = x[i];
    if ((int)item.get_local_id(0) < k) {
        ids_out[item.get_local_id(0)] = ids[item.get_local_id(0)];
        w_out[item.get_local_id(0)] = w[item.get_local_id(0)];
    }
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    item.barrier();
    if (item.get_local_id(0) == 0) {
        strata::sys_store(seq, strata::sys_load(seq) + 1u);
    }
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (k > 1024) {
        std::fprintf(stderr, "doorbell_publish: k too large\n");
        std::exit(1);
    }
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> /*item*/) {
        doorbell_publish_kernel(x, ids, weights, (int)n, (int)k, x_out, ids_out, weights_out, d_seq);
    });
    check_launch("doorbell_publish");
}

void doorbell_publish_res_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                 const int32_t* __restrict__ d_res, int n_expert, int n, int k,
                                 float* x_out, int32_t* ids_out, uint32_t* seq,
                                 int* s_miss, sycl::nd_item<1> item) {
    if (item.get_local_id(0) == 0) {
        *s_miss = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);
    if ((int)item.get_local_id(0) < k) {
        const int32_t id = ids[item.get_local_id(0)];
        ids_out[item.get_local_id(0)] = id;
        if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>(*s_miss).store(1);
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (*s_miss) {
        for (int i = item.get_local_id(0); i < n; i += item.get_local_range(0))
            x_out[i] = x[i];
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    } else if ((int)item.get_local_id(0) < k) {
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    }
    item.barrier();
    if (item.get_local_id(0) == 0) {
        strata::sys_store(seq, strata::sys_load(seq) + 1u);
    }
}

void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                          float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (k > 1024) {
        std::fprintf(stderr, "doorbell_publish_res: k too large\n");
        std::exit(1);
    }
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 0> s_miss(cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)),
                         [=](sycl::nd_item<1> item) {
                             doorbell_publish_res_kernel(x, ids, d_res, n_expert, (int)n, (int)k,
                                                         x_out, ids_out, d_seq,
                                                         s_miss.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                         item);
                         });
    });
    check_launch("doorbell_publish_res");
}

void doorbell_publish_value_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                   const float* __restrict__ w, int n, int k, float* x_out, int32_t* ids_out,
                                   float* w_out, uint32_t* seq, uint32_t value) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    for (int i = item.get_local_id(0); i < n; i += item.get_local_range(0))
        x_out[i] = x[i];
    if ((int)item.get_local_id(0) < k) {
        ids_out[item.get_local_id(0)] = ids[item.get_local_id(0)];
        w_out[item.get_local_id(0)] = w[item.get_local_id(0)];
    }
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    item.barrier();
    if (item.get_local_id(0) == 0) {
        strata::sys_store(seq, value);
    }
}

void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                            float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value,
                            void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (k > 1024) {
        std::fprintf(stderr, "doorbell_publish_value: k too large\n");
        std::exit(1);
    }
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> /*item*/) {
        doorbell_publish_value_kernel(x, ids, weights, (int)n, (int)k, x_out, ids_out, weights_out, d_seq, value);
    });
    check_launch("doorbell_publish_value");
}

void copy_i32_from_mapped_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    for (int i = item.get_local_id(0); i < n; i += item.get_local_range(0))
        dst[i] = src[i];
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](sycl::nd_item<1> /*item*/) {
        copy_i32_from_mapped_kernel(dst, (const volatile int32_t*)src, (int)n);
    });
    check_launch("copy_i32_from_mapped");
}

void doorbell_ring(uint32_t* d_seq, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (d_seq == nullptr) return;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                    [=](sycl::nd_item<1> /*item*/) { doorbell_ring_kernel(d_seq); });
    check_launch("doorbell_ring");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (rows <= 0 || cols <= 0) return;
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned)((rows + warps_per_block - 1) / warps_per_block);
    q->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((grid) * (warps_per_block * 32)), sycl::range<1>(warps_per_block * 32)),
        [=](sycl::nd_item<1> /*item*/)
            [[sycl::reqd_sub_group_size(32)]] { rms_norm_weighted_kernel(x, w, rows, cols, eps); });
    check_launch("rms_norm_weighted");
}

}  // namespace strata::kernels
