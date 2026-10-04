import re
import sys

def refactor_gr(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove dpct header
    content = content.replace('#include <dpct/dpct.hpp>\n', '')

    # 2. Add static default queue
    helper_def = '''
static sycl::queue default_queue{sycl::property::queue::in_order()};
'''
    content = content.replace('namespace strata::kernels {\nnamespace {\n', 'namespace strata::kernels {\nnamespace {\n' + helper_def)

    # 3. __dpct_inline__ -> inline
    content = content.replace('__dpct_inline__', 'inline')

    # 4. dpct::queue_ptr -> sycl::queue*
    content = content.replace('dpct::queue_ptr st = stream ? (dpct::queue_ptr)stream : &dpct::get_default_queue();',
                              'sycl::queue* st = stream ? static_cast<sycl::queue*>(stream) : &default_queue;')
    content = content.replace('(stream ? (sycl::queue*)stream : &dpct::get_default_queue())',
                              '(stream ? static_cast<sycl::queue*>(stream) : &default_queue)')
    content = content.replace('&dpct::get_default_queue()', '&default_queue')
    content = content.replace('dpct::queue_ptr', 'sycl::queue*')

    # 5. Remove dpct::has_capability_or_fail
    content = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', content)

    # 6. check function
    old_check = '''void check(const char* what) {
    const dpct::err0 e = 0;
    if (e != 0) { std::fprintf(stderr, "%s: %d\n", what, (int) e); std::exit(1); }
    if (std::getenv("STRATA_SYNC_KERNELS") != nullptr) {
        const dpct::err0 se = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
        if (se != 0) { std::fprintf(stderr, "%s (sync): %d\n", what, (int) se); std::exit(1); }
    }
}'''
    new_check = '''void check(const char* /*what*/) {
}'''
    content = content.replace(old_check, new_check)

    # 7. In gr_read, replace passing stream with st to downstream functions
    content = content.replace('native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);',
                              'native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, st);')
    content = content.replace('bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);',
                              'bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, st);')
    content = content.replace('native_gr_down_silu(ws.lo, hc_lr, hc, stream);',
                              'native_gr_down_silu(ws.lo, hc_lr, hc, st);')
    content = content.replace('bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);',
                              'bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, st);')
    content = content.replace('native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);',
                              'native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, st);')
    content = content.replace('native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, stream);',
                              'native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, stream);')

    # Add wait if !stream at end of gr_read and gr_write
    content = content.replace('check("gr_read");\n}', 'check("gr_read");\n    if (!stream) default_queue.wait();\n}')

    # In gr_write:
    content = re.sub(r'if \(stream == nullptr\) \{[\s\S]*?\}', 'if (!stream) default_queue.wait();', content)

    # 8. warp_sum and warp_sumf
    old_warp_sum = '''inline double warp_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, off);
    /*
    DPCT1023:308: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=sub_group_mask" to use the experimental class
    "sycl::ext::oneapi::sub_group_mask".
    */
    return dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}
inline float warp_sumf(float v) {
#pragma unroll
    /*
    DPCT1096:309: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int off = 16; off > 0; off >>= 1) v += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, off);
    /*
    DPCT1023:316: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=sub_group_mask" to use the experimental class
    "sycl::ext::oneapi::sub_group_mask".
    */
    /*
    DPCT1096:317: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    return dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}'''

    new_warp_sum = '''inline double warp_sum(sycl::sub_group sg, double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += sycl::shift_group_left(sg, v, off);
    }
    return sycl::select_from_group(sg, v, 0);
}
inline float warp_sumf(sycl::sub_group sg, float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += sycl::shift_group_left(sg, v, off);
    }
    return sycl::select_from_group(sg, v, 0);
}'''
    content = content.replace(old_warp_sum, new_warp_sum)

    # 9. block_sumf and block_sum
    old_block_sumf = '''float block_sumf(float v, double* scratch_raw) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *scratch = (float *)scratch_raw;
    // The leading barrier is not decoration - see the note on `block_sum` above: the result is read straight out
    // of `scratch[0]` by every thread and a later call reuses the array, so without it a fast thread can
    // overwrite `scratch[0]` before a slow one has read the previous result.
    /*
    DPCT1065:310: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    v = warp_sumf(v);
    if (lane == 0) scratch[warp] = v;
    /*
    DPCT1065:311: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    v = (item_ct1.get_local_id(2) < nw) ? scratch[item_ct1.get_local_id(2)]
                                        : 0.0f;
    if (warp == 0) v = warp_sumf(v);
    if (item_ct1.get_local_id(2) == 0) scratch[0] = v;
    /*
    DPCT1065:312: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return scratch[0];
}

/// Block-wide sum in DOUBLE, broadcast to every thread.
///
/// Double because `ref/gr.py` reduces in float64 for the norm and the bottleneck, and because a block-wide
/// f32 sum of 2560 squares is a different number from the reference's; the shuffle tree here also reorders
/// the additions, so the only defence is enough precision that the order stops mattering.
///
/// The leading `__syncthreads()` is not decoration: the result is read straight out of `scratch[0]` by every
/// thread, and a later call reuses the same array.  Without a barrier on entry a fast thread can overwrite
/// `scratch[0]` before a slow one has read the previous result - a race that is invisible in most runs.
double block_sum(double v, double* scratch) {
    /*
    DPCT1065:313: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    /*
    DPCT1065:314: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    v = (item_ct1.get_local_id(2) < nw) ? scratch[item_ct1.get_local_id(2)] : 0.0;
    if (warp == 0) v = warp_sum(v);
    if (item_ct1.get_local_id(2) == 0) scratch[0] = v;
    /*
    DPCT1065:315: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return scratch[0];
}'''

    new_block_sumf = '''inline float block_sumf(sycl::nd_item<1> item, float v, double* scratch_raw) {
    float *scratch = (float *)scratch_raw;
    sycl::group_barrier(item.get_group());
    auto sg = item.get_sub_group();
    const int tid = item.get_local_id(0);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    v = warp_sumf(sg, v);
    if (lane == 0) scratch[warp] = v;
    sycl::group_barrier(item.get_group());
    const int nw = ((int)item.get_local_range(0) + 31) >> 5;
    v = (tid < nw) ? scratch[tid] : 0.0f;
    if (warp == 0) v = warp_sumf(sg, v);
    if (tid == 0) scratch[0] = v;
    sycl::group_barrier(item.get_group());
    return scratch[0];
}

inline double block_sum(sycl::nd_item<1> item, double v, double* scratch) {
    sycl::group_barrier(item.get_group());
    auto sg = item.get_sub_group();
    const int tid = item.get_local_id(0);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    v = warp_sum(sg, v);
    if (lane == 0) scratch[warp] = v;
    sycl::group_barrier(item.get_group());
    const int nw = ((int)item.get_local_range(0) + 31) >> 5;
    v = (tid < nw) ? scratch[tid] : 0.0;
    if (warp == 0) v = warp_sum(sg, v);
    if (tid == 0) scratch[0] = v;
    sycl::group_barrier(item.get_group());
    return scratch[0];
}'''
    content = content.replace(old_block_sumf, new_block_sumf)

    # 10. Update gr_norm_kernel signature & calls
    content = content.replace(
        'void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w_norm, float eps,\n                               int n_embd, float* __restrict__ xn, uint16_t* __restrict__ xq,\n                               double *scratch) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n\n    const int c = item_ct1.get_group(2);',
        'void gr_norm_kernel(sycl::nd_item<1> item, const float* __restrict__ R, const float* __restrict__ w_norm, float eps,\n                               int n_embd, float* __restrict__ xn, uint16_t* __restrict__ xq,\n                               double *scratch) {\n    const int c = item.get_group(0);'
    )
    content = content.replace('item_ct1.get_local_id(2)', 'item.get_local_id(0)')
    content = content.replace('item_ct1.get_local_range(2)', 'item.get_local_range(0)')
    content = content.replace('block_sumf(ss, scratch)', 'block_sumf(item, ss, scratch)')
    content = content.replace('block_sum(ss, scratch)', 'block_sum(item, ss, scratch)')

    # 11. Update gr_down_kernel signature & calls
    content = content.replace(
        'void gr_down_kernel(const Activation* __restrict__ xq, const uint16_t* __restrict__ w_down, int hc_dim,\n                               int hc_lr, int hc, Activation* __restrict__ lq,\n                               float *part) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    const int k = item_ct1.get_group(2);',
        'void gr_down_kernel(sycl::nd_item<1> item, const Activation* __restrict__ xq, const uint16_t* __restrict__ w_down, int hc_dim,\n                               int hc_lr, int hc, Activation* __restrict__ lq,\n                               float *part) {\n    const int k = item.get_group(0);'
    )
    content = content.replace('item_ct1.barrier(sycl::access::fence_space::local_space);', 'sycl::group_barrier(item.get_group());')

    # In gr_down_kernel, warp_sumf(acc) -> warp_sumf(sg, acc)
    content = content.replace('warp_sumf(acc);', 'warp_sumf(item.get_sub_group(), acc);')
    content = content.replace('warp_sumf(t);', 'warp_sumf(item.get_sub_group(), t);')

    # 12. Update gr_gate_kernel signature & calls
    content = content.replace(
        'void gr_gate_kernel(const Activation* __restrict__ lq, const uint16_t* __restrict__ w_up,\n                               const float* __restrict__ xn, int hc_dim, int hc_lr, float* __restrict__ gated) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    const int i = item_ct1.get_group(2) * WARPS + (item_ct1.get_local_id(2) >> 5);',
        'void gr_gate_kernel(sycl::nd_item<1> item, const Activation* __restrict__ lq, const uint16_t* __restrict__ w_up,\n                               const float* __restrict__ xn, int hc_dim, int hc_lr, float* __restrict__ gated) {\n    const int i = item.get_group(0) * WARPS + (item.get_local_id(0) >> 5);'
    )

    # 13. Update gr_mean_kernel signature & calls
    content = content.replace(
        'void gr_mean_kernel(const float* __restrict__ gated, int n_embd, int hc, float* __restrict__ mixed) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    const int d = item_ct1.get_group(2) * item_ct1.get_local_range(2) +\n                  item_ct1.get_local_id(2);',
        'void gr_mean_kernel(sycl::nd_item<1> item, const float* __restrict__ gated, int n_embd, int hc, float* __restrict__ mixed) {\n    const int d = item.get_global_id(0);'
    )

    # 14. Update gr_inject_kernel signature & calls
    content = content.replace(
        'void gr_inject_kernel(const Activation* __restrict__ xq, const uint16_t* __restrict__ w_inject,\n                                 int hc_dim, int hc, float* __restrict__ inject) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    const int c = item_ct1.get_local_id(2) >> 5;',
        'void gr_inject_kernel(sycl::nd_item<1> item, const Activation* __restrict__ xq, const uint16_t* __restrict__ w_inject,\n                                 int hc_dim, int hc, float* __restrict__ inject) {\n    const int c = item.get_local_id(0) >> 5;'
    )

    # 15. Update gr_write_kernel signature & calls
    content = content.replace(
        'void gr_write_kernel(const float* __restrict__ R, const float* __restrict__ block_out,\n                                const float* __restrict__ inject, int n_embd, int hc, float* __restrict__ out,\n                                uint8_t *dpct_local) {\n    // w = 2*sigmoid(inject/hc), computed once per stream in shared rather than per element.\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    auto smem_raw = (unsigned char *)dpct_local;\n    float* w = (float*) smem_raw;\n    if ((int)item_ct1.get_local_id(2) < hc) w[item_ct1.get_local_id(2)] =\n        2.0f * sigmoid_f(inject[item_ct1.get_local_id(2)] / (float)hc);\n    item_ct1.barrier(sycl::access::fence_space::local_space);\n    const long long n = (long long) hc * n_embd;\n    for (long long i =\n             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +\n             item_ct1.get_local_id(2);\n         i < n; i += (long long)item_ct1.get_group_range(2) *\n                     item_ct1.get_local_range(2)) {',
        'void gr_write_kernel(sycl::nd_item<1> item, const float* __restrict__ R, const float* __restrict__ block_out,\n                                const float* __restrict__ inject, int n_embd, int hc, float* __restrict__ out,\n                                float* w) {\n    // w = 2*sigmoid(inject/hc), computed once per stream in shared rather than per element.\n    if ((int)item.get_local_id(0) < hc) w[item.get_local_id(0)] =\n        2.0f * sigmoid_f(inject[item.get_local_id(0)] / (float)hc);\n    sycl::group_barrier(item.get_group());\n    const long long n = (long long) hc * n_embd;\n    for (long long i = item.get_global_id(0); i < n; i += item.get_group_range(0) * item.get_local_range(0)) {'
    )

    # 16. Convert launches to 1D ranges
    # gr_norm_kernel launch
    content = re.sub(
        r'sycl::nd_range<3>\(sycl::range<3>\(1, 1, hc\) \* sycl::range<3>\(1, 1, THREADS\),\s*sycl::range<3>\(1, 1, THREADS\)\)',
        r'sycl::nd_range<1>(sycl::range<1>((size_t)hc * THREADS), sycl::range<1>(THREADS))',
        content
    )
    content = re.sub(
        r'\[=\]\(sycl::nd_item<3> item_ct1\)[^\{]*\{\s*gr_norm_kernel<([^>]+)>\(\s*R, w_norm, eps, n_embd, ws\.xn, ws\.xq,\s*scratch_acc_ct1[^;]*\);\s*\}\)',
        r'[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] { gr_norm_kernel<\1>(item, R, w_norm, eps, n_embd, ws.xn, ws.xq, scratch_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get()); })',
        content
    )

    # gr_down_kernel launch
    content = re.sub(
        r'sycl::nd_range<3>\(sycl::range<3>\(1, 1, hc_lr\) \* sycl::range<3>\(1, 1, THREADS\),\s*sycl::range<3>\(1, 1, THREADS\)\)',
        r'sycl::nd_range<1>(sycl::range<1>((size_t)hc_lr * THREADS), sycl::range<1>(THREADS))',
        content
    )
    content = re.sub(
        r'\[=\]\(sycl::nd_item<3> item_ct1\)[^\{]*\{\s*gr_down_kernel<([^>]+)>\(\s*([^;]+),\s*part_acc_ct1[^;]*\);\s*\}\)',
        r'[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] { gr_down_kernel<\1>(item, \2, part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get()); })',
        content
    )

    # gr_gate_kernel launch
    content = re.sub(
        r'sycl::nd_range<3>\(\s*sycl::range<3>\(1, 1, \(hc_dim \+ WARPS - 1\) / WARPS\) \*\s*sycl::range<3>\(1, 1, THREADS\),\s*sycl::range<3>\(1, 1, THREADS\)\)',
        r'sycl::nd_range<1>(sycl::range<1>((size_t)((hc_dim + WARPS - 1) / WARPS) * THREADS), sycl::range<1>(THREADS))',
        content
    )
    content = re.sub(
        r'\[=\]\(sycl::nd_item<3> item_ct1\)[^\{]*\{\s*gr_gate_kernel<([^>]+)>\(([^;]+)\);\s*\}\)',
        r'[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] { gr_gate_kernel<\1>(item, \2); })',
        content
    )

    # gr_mean_kernel launch
    content = re.sub(
        r'sycl::nd_range<3>\(\s*sycl::range<3>\(1, 1, \(n_embd \+ THREADS - 1\) / THREADS\) \*\s*sycl::range<3>\(1, 1, THREADS\),\s*sycl::range<3>\(1, 1, THREADS\)\)',
        r'sycl::nd_range<1>(sycl::range<1>((size_t)((n_embd + THREADS - 1) / THREADS) * THREADS), sycl::range<1>(THREADS))',
        content
    )
    content = re.sub(
        r'\[=\]\(sycl::nd_item<3> item_ct1\)[^\{]*\{\s*gr_mean_kernel\(([^;]+)\);\s*\}\)',
        r'[=](sycl::nd_item<1> item) { gr_mean_kernel(item, \1); })',
        content
    )

    # gr_inject_kernel launch
    content = re.sub(
        r'sycl::nd_range<3>\(\s*sycl::range<3>\(1, 1, 32 \* hc\),\s*sycl::range<3>\(1, 1, 32 \* hc\)\)',
        r'sycl::nd_range<1>(sycl::range<1>(32 * hc), sycl::range<1>(32 * hc))',
        content
    )
    content = re.sub(
        r'\[=\]\(sycl::nd_item<3> item_ct1\)[^\{]*\{\s*gr_inject_kernel<([^>]+)>\(([^;]+)\);\s*\}\)',
        r'[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] { gr_inject_kernel<\1>(item, \2); })',
        content
    )

    # gr_write launch
    content = re.sub(
        r'sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1\(\s*sycl::range<1>\(\(size_t\)s\.hc \* sizeof\(float\)\), cgh\);',
        r'sycl::local_accessor<float, 1> w_acc(sycl::range<1>((size_t)s.hc), cgh);',
        content
    )
    content = re.sub(
        r'sycl::nd_range<3>\(sycl::range<3>\(1, 1, blocks\) \*\s*sycl::range<3>\(1, 1, THREADS\),\s*sycl::range<3>\(1, 1, THREADS\)\)',
        r'sycl::nd_range<1>(sycl::range<1>((size_t)blocks * THREADS), sycl::range<1>(THREADS))',
        content
    )
    content = re.sub(
        r'\[=\]\(sycl::nd_item<3> item_ct1\)\s*\{\s*gr_write_kernel\(\s*R, block_out, inject, \(int\)s\.n_embd, \(int\)s\.hc,\s*R_out,\s*dpct_local_acc_ct1[^;]*\);\s*\}',
        r'[=](sycl::nd_item<1> item) { gr_write_kernel(item, R, block_out, inject, (int)s.n_embd, (int)s.hc, R_out, w_acc.get_multi_ptr<sycl::access::decorated::no>().get()); }',
        content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_gr('src/kernels/sycl/gr.dp.cpp')
    print('Refactoring gr.dp.cpp completed.')
