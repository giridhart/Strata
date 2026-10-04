import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
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

# Queue and errors
def fix_func(func_name, kernel_name, extra_args=[]):
    global text
    # 1. find queue launch
    text = re.sub(r'(?:\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->\s*parallel_for|\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->parallel_for)', 'q->parallel_for', text)
    
    # 2. Add queue declaration at the start of functions
    text = re.sub(r'(void ' + func_name + r'\(.*?\) \{)\n', 
                  r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text)

# dpct capabilities
text = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\{sycl::aspect::fp64\}\);\s*', '', text)

# memcpys
text = re.sub(r'ck\(DPCT_CHECK_ERROR\(st->memcpy\((.*?),\s*(.*?),\s*(.*?)\)\)\s*,\s*".*?"\);', r'q->memcpy(\1, \2, \3).wait();', text)
text = re.sub(r'ck\(DPCT_CHECK_ERROR\(st->memcpy\((.*?),\s*(.*?),\s*(.*?)\)\.wait\(\)\)\s*,\s*".*?"\);', r'q->memcpy(\1, \2, \3).wait();', text)

# remove dpct::err0
text = re.sub(r'/\*[\s\S]*?\*/\n\s*const dpct::err0 e = 0;\n', '', text)
text = re.sub(r'void ck\(dpct::err0 e, const char \*what\) \{[\s\S]*?\}', '', text)

# device count
text = re.sub(r'int n;\n\s*return DPCT_CHECK_ERROR\(n = dpct::device_count\(\)\) == 0 && n > 0;', r'return true;', text)

# queue_ptr
text = re.sub(r'dpct::queue_ptr st = stream \? \(dpct::queue_ptr\)stream : &dpct::get_default_queue\(\);', '', text)

text = re.sub(r'st->submit', 'q->submit', text)
text = re.sub(r'st->memcpy', 'q->memcpy', text)
text = re.sub(r'st->memset', 'q->memset', text)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

