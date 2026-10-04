import re

with open('src/kernels/sycl/quantize_act.dp.cpp', 'r') as f:
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
text = text.replace('item_ct1', 'item')
text = text.replace('sycl::ext::oneapi::this_work_item::get_nd_item<3>()', 'sycl::ext::oneapi::this_work_item::get_nd_item<1>()')

# Dimensions (3D to 1D)
text = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1,\s*1,\s*(.*?)\)\s*\*\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)

# Queue and errors
def fix_func(func_name, kernel_name, extra_args=[]):
    global text
    # pattern: void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) { ... }
    # replace queue launch and errors
    
    # 1. find queue launch
    text = re.sub(r'(?:\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->\s*parallel_for|\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->parallel_for)', 'q->parallel_for', text)
    
    # 2. Add queue declaration at the start of functions
    text = re.sub(r'(void ' + func_name + r'\(.*?\) \{)\n(\s*if \(n <= 0\) return;\n)', 
                  r'\1\n\2    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text)
                  
    # 3. Remove DPCT errors
    text = re.sub(r'/\*[\s\S]*?\*/\n\s*const dpct::err0 e = 0;\n', '', text)
    text = re.sub(r'if \(stream == nullptr\) dpct::get_current_device\(\)\.queues_wait_and_throw\(\);\n', '', text)
    
    # 4. Remove dpct::has_capability_or_fail block
    text = re.sub(r'\{\s*dpct::has_capability_or_fail\([\s\S]*?\{\s*sycl::aspect::fp64\}\);\s*q->parallel_for', 'q->parallel_for', text)
    text = re.sub(r'q->parallel_for\(([\s\S]*?)\);\s*\}', r'q->parallel_for(\1);', text)

fix_func('quantize_q8_0', 'quantize_q8_0_kernel')
fix_func('quantize_q8_0_scaled', 'quantize_q8_0_scaled_kernel')
fix_func('dequant_q8_0', 'dequant_q8_0_kernel')
fix_func('quantize_q8_K', 'quantize_q8_K_kernel')
fix_func('dequant_q8_K', 'dequant_q8_K_kernel')

with open('src/kernels/sycl/quantize_act.dp.cpp', 'w') as f:
    f.write(text)

