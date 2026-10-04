import re

with open('src/kernels/sycl/elementwise.dp.cpp', 'r') as f:
    text = f.read()

# Headers
text = text.replace('#include <dpct/dpct.hpp>', '#include <cstdint>')

# Macros
text = text.replace('__dpct_inline__', 'inline')

# Items
text = text.replace('sycl::nd_item<3> item_ct1', 'sycl::nd_item<1> item')
text = text.replace('item_ct1.get_group(2)', 'item.get_group(0)')
text = text.replace('item_ct1.get_local_range(2)', 'item.get_local_range(0)')
text = text.replace('item_ct1.get_local_id(2)', 'item.get_local_id(0)')
text = text.replace('item_ct1.get_group_range(2)', 'item.get_group_range(0)')
text = text.replace('item_ct1', 'item')
text = text.replace('sycl::ext::oneapi::this_work_item::get_nd_item<3>()', 'sycl::ext::oneapi::this_work_item::get_nd_item<1>()')
text = text.replace('dpct::select_from_sub_group', 'sycl::select_from_group')
text = text.replace('dpct::shift_sub_group_left', 'sycl::shift_group_left')
text = text.replace('sycl::ext::oneapi::this_work_item::get_sub_group()', 'item.get_sub_group()')

# Dimensions (3D to 1D)
text = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1,\s*1,\s*(.*?)\)\s*\*\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)

text = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>(\1), sycl::range<1>(\2))', text)

def fix_func(func_name, kernel_name, extra_args=[]):
    global text
    # pattern: void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) { ... }
    # replace queue launch and errors
    
    # 1. find queue launch
    text = re.sub(r'(?:\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->\s*parallel_for|\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->parallel_for)', 'q->parallel_for', text)
    
    # 2. Add queue declaration at the start of functions
    text = re.sub(r'(void ' + func_name + r'\(.*?\) \{)\n(\s*if \(.*?\) return;\n)', 
                  r'\1\n\2    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text)

fix_func('embedding_gather', 'embedding_gather_kernel')
fix_func('gdn_gate', 'gdn_gate_kernel')
fix_func('scale_inplace', 'scale_kernel')
fix_func('add_inplace', 'add_kernel')
fix_func('f32_to_f16_bulk', 'to_f16_kernel')
fix_func('f32_to_bf16_bulk', 'to_bf16_kernel')

# fix silu_inplace
text = re.sub(r'void silu_inplace\(float\* x, int64_t n, void\* stream\) \{[\s\S]*?check_launch\("silu_inplace"\);\n\s*sync_if_needed\(stream, "silu_inplace"\);\n\}', 
r'''void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_for(n) * THREADS), sycl::range<1>(THREADS)), [=](sycl::nd_item<1> item) {
        silu_kernel(x, n);
    });
    if (stream == nullptr) q->wait();
}''', text)

text = re.sub(r'void copy_rows_from_mapped\(.*?\) \{[\s\S]*?\}\);\n\}',
r'''void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows, const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 1> hit_acc_ct1(sycl::range<1>(1), cgh);
        auto width_ct2 = width / 4;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((unsigned)rows * 128), sycl::range<1>(128)), [=](sycl::nd_item<1> item) {
            copy_rows_from_mapped_kernel((sycl::float4 *)dst, (const sycl::float4 *)src, width_ct2, hit_rows, count, hit_acc_ct1[0]);
        });
    });
}''', text)

fix_func('copy_from_mapped', 'copy_from_mapped_kernel')

# fix doorbell_publish
text = re.sub(r'void doorbell_publish\(.*?\) \{[\s\S]*?\}\);\n\s*check_launch\("doorbell_publish"\);\n\}',
r'''void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> item) {
        doorbell_publish_kernel(x, ids, weights, (int)n, (int)k, x_out, ids_out, weights_out, d_seq);
    });
}''', text)

fix_func('copy_i32_from_mapped', 'copy_i32_from_mapped_kernel')

# fix doorbell_ring
text = re.sub(r'void doorbell_ring\(uint32_t\* d_seq, void\* stream\) \{[\s\S]*?sync_if_needed\(stream, "doorbell_ring"\);\n\}',
r'''void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1> item) {
        doorbell_ring_kernel(d_seq);
    });
    if (stream == nullptr) q->wait();
}''', text)

# fix doorbell_wait
text = re.sub(r'void doorbell_wait\(const uint32_t\* d_flag, const uint32_t\* d_seq, void\* stream\) \{[\s\S]*?check_launch\("doorbell_wait"\);\n\}',
r'''void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1> item) {
        doorbell_wait_kernel(d_flag, d_seq);
    });
}''', text)

# fix rms_norm_weighted
text = re.sub(r'void rms_norm_weighted\(.*?\) \{[\s\S]*?sync_if_needed\(stream, "rms_norm_weighted"\);\n\}',
r'''void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * warps_per_block * 32), sycl::range<1>(warps_per_block * 32)), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        rms_norm_weighted_kernel(x, w, rows, cols, eps);
    });
    if (stream == nullptr) q->wait();
}''', text)

# Cleanup
text = re.sub(r'check_launch\(".*?"\);\n\s*sync_if_needed\(stream, ".*?"\);', 'if (stream == nullptr) q->wait();', text)
text = re.sub(r'check_launch\(".*?"\);', '', text)
text = re.sub(r'void sync_if_needed\(.*?\) try \{[\s\S]*?\}\n\n', '', text)
text = re.sub(r'bool check_launch\(.*?\) \{[\s\S]*?\}\n\n', '', text)

with open('src/kernels/sycl/elementwise.dp.cpp', 'w') as f:
    f.write(text)
