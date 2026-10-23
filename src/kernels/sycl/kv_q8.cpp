// src/kernels/cuda/kv_q8.cu - see include/strata/kernels/kv_q8.hpp.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

static sycl::queue default_queue;

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long)s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

// One block = one 64-value group of one KV head of K (blockIdx.z = 0) or V (1); 64 threads, one value each.
// MULTI (S26 STRATA_LFUSE): blockIdx.z = 2 * token + (K / V); token j reads step + j * step_stride and its rows at
// kcur / vcur + j * cur_stride - every token's code is the single launch's
template <bool MULTI = false>
void kv_append_q8_kernel(int8_t* __restrict__ k_q, int8_t* __restrict__ v_q, uint16_t* __restrict__ k_scale,
                         uint16_t* __restrict__ v_scale, const int32_t* __restrict__ table,
                         const int32_t* __restrict__ step, const float* __restrict__ kcur,
                         const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size, KvHostPools host,
                         int step_stride, int cur_stride,
                         float* warp_max, const sycl::nd_item<3>& item) {
    if constexpr (MULTI) {
        const int j = item.get_group(0) >> 1;
        step += (size_t)j * step_stride;
        kcur += (size_t)j * cur_stride;
        vcur += (size_t)j * cur_stride;
    }
    const long long pos = (long long)*(step + kStepPos);
    const int h = item.get_group(2), g = item.get_group(1), t = item.get_local_id(2);
    const bool is_v = MULTI ? ((item.get_group(0) & 1) == 1) : (item.get_group(0) == 1);
    const int groups = head_dim / KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_Q8_GROUP + t];
    // max |x| over the 64 values: two warps, then combine through shared memory in a fixed order
    const float a = sycl::reduce_over_group(item.get_sub_group(), sycl::fabs(x), sycl::maximum<float>());

    if ((t & 31) == 0) warp_max[t >> 5] = a;
    item.barrier(sycl::access::fence_space::local_space);
    const float amax = sycl::fmax(warp_max[0], warp_max[1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);  // quantize against the STORED scale
    int q = 0;
    if (sf > 0.0f) {
        q = sycl::vec<float, 1>{(x / sf)}.convert<int, sycl::rounding_mode::rte>()[0];
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    // KV streaming: the host copy (identity layout) always, the VRAM page only if the block is resident
    const long long page = (long long)table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t)q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_q : host.k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t)q;
        if (t == 0) (is_v ? host.v_scale : host.k_scale)[row * groups + g] = sbits;
    }
}

// One thread = 4 consecutive values of one cell and head (as the FP16 gather does with uint2).
void kv_gather_q8_kernel(const int8_t* __restrict__ k_q, const int8_t* __restrict__ v_q,
                         const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                         const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                         const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                         uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                         const sycl::nd_item<1>& item) {
    const long long n_ids = (long long)*(step + kStepWidth);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long)per);
    const int rem = (int)(i % (kv_heads * (long long)per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long)table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const sycl::char4 kc = reinterpret_cast<const sycl::char4*>(k_q + row * head_dim)[q4];
    const sycl::char4 vc = reinterpret_cast<const sycl::char4*>(v_q + row * head_dim)[q4];
    sycl::ushort4 ko, vo;
    ko.x() = f16_from_f32((float)kc.x() * ks);
    ko.y() = f16_from_f32((float)kc.y() * ks);
    ko.z() = f16_from_f32((float)kc.z() * ks);
    ko.w() = f16_from_f32((float)kc.w() * ks);
    vo.x() = f16_from_f32((float)vc.x() * vs);
    vo.y() = f16_from_f32((float)vc.y() * vs);
    vo.z() = f16_from_f32((float)vc.z() * vs);
    vo.w() = f16_from_f32((float)vc.w() * vs);
    const long long dst = (id * kv_heads + h) * (long long)per + q4;
    reinterpret_cast<sycl::ushort4*>(k_scratch)[dst] = ko;
    reinterpret_cast<sycl::ushort4*>(v_scratch)[dst] = vo;
}

// One thread = 8 consecutive values of one cell and head: an 8-byte load of codes and a 16-byte
// store of the dequantized fp16. The 8 values share one 64-value group, so one scale is loaded per thread and
// the arithmetic stays `(float) code * scale -> f16_from_f32`, i.e. bitwise identical to the 4-wide kernel.
// On Intel Xe2 (Lunar Lake and Battlemage B580), a SIMD16 subgroup executing 8-wide loads reads 16 * 8 = 128 bytes,
// perfectly saturating a 128-byte hardware cache-line transaction on B580's GDDR6 memory bus.
void kv_gather_q8_wide_kernel(const int8_t* __restrict__ k_q, const int8_t* __restrict__ v_q,
                              const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                              const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                              const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                              uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                              const sycl::nd_item<1>& item) {
    const long long n_ids = (long long)*(step + kStepWidth);
    const int per = head_dim / 8;
    const long long total = n_ids * kv_heads * per;
    const long long i = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long)per);
    const int rem = (int)(i % (kv_heads * (long long)per));
    const int h = rem / per, q8 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long)table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q8 * 8;
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const sycl::vec<int8_t, 8> kc = reinterpret_cast<const sycl::vec<int8_t, 8>*>(k_q + row * head_dim)[q8];
    const sycl::vec<int8_t, 8> vc = reinterpret_cast<const sycl::vec<int8_t, 8>*>(v_q + row * head_dim)[q8];
    sycl::vec<uint16_t, 8> ko, vo;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        ko[k] = f16_from_f32((float)kc[k] * ks);
        vo[k] = f16_from_f32((float)vc[k] * vs);
    }
    const long long dst = (id * kv_heads + h) * (long long)per + q8;
    reinterpret_cast<sycl::vec<uint16_t, 8>*>(k_scratch)[dst] = ko;
    reinterpret_cast<sycl::vec<uint16_t, 8>*>(v_scratch)[dst] = vo;
}

}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const sycl::range<3> grid(2, (unsigned)(s.head_dim / KV_Q8_GROUP), (unsigned)s.n_head_kv);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> warp_max_acc(sycl::range<1>(2), cgh);
        KvHostPools host_pools = host ? *host : KvHostPools{};

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, KV_Q8_GROUP), sycl::range<3>(1, 1, KV_Q8_GROUP)),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                             kv_append_q8_kernel<false>(k_q, v_q, k_scale, v_scale, page_table, step, kcur, vcur,
                                                        (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, host_pools,
                                                        0, 0,
                                                        warp_max_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                        item);
                         });
    });
    if (!stream) default_queue.wait();
}

void kv_append_q8_steps(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride,
                        int n_tok, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append_q8 (steps)");
    if (n_tok < 1) return;
    const sycl::range<3> grid((unsigned)(2 * n_tok), (unsigned)(s.head_dim / KV_Q8_GROUP), (unsigned)s.n_head_kv);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> warp_max_acc(sycl::range<1>(2), cgh);
        KvHostPools host_pools = host ? *host : KvHostPools{};

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, KV_Q8_GROUP), sycl::range<3>(1, 1, KV_Q8_GROUP)),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                             kv_append_q8_kernel<true>(k_q, v_q, k_scale, v_scale, page_table, step, kcur, vcur,
                                                       (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, host_pools,
                                                       step_stride, cur_stride,
                                                       warp_max_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                       item);
                         });
    });
    if (!stream) default_queue.wait();
}

void kv_gather_q8_wide_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                            const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                            const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8_wide");
    if (max_ids <= 0) return;
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 8);
    const unsigned blocks = (unsigned)((total + 255) / 256);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        kv_gather_q8_wide_kernel(k_q, v_q, k_scale, v_scale, page_table, ids, step, (int)s.n_head_kv, (int)s.head_dim,
                                 (int)s.page_size, k_scratch, v_scratch, item);
    });
    if (!stream) default_queue.wait();
}

void kv_gather_q8_standard_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                                const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                                const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8_standard");
    if (max_ids <= 0) return;
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    const unsigned blocks = (unsigned)((total + 255) / 256);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        kv_gather_q8_kernel(k_q, v_q, k_scale, v_scale, page_table, ids, step, (int)s.n_head_kv, (int)s.head_dim,
                            (int)s.page_size, k_scratch, v_scratch, item);
    });
    if (!stream) default_queue.wait();
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;

    const char* env = std::getenv("STRATA_Q8_GATHER_WIDE");
    const bool wide_enabled = (env == nullptr || env[0] != '0');

    if (wide_enabled && (s.head_dim % 8 == 0)) {
        kv_gather_q8_wide_step(k_q, v_q, k_scale, v_scale, page_table, ids, step, max_ids, s, k_scratch, v_scratch, stream);
    } else {
        kv_gather_q8_standard_step(k_q, v_q, k_scale, v_scale, page_table, ids, step, max_ids, s, k_scratch, v_scratch, stream);
    }
}

}  // namespace strata::kernels
