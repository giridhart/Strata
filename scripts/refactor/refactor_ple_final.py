import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# Headers
text = text.replace('#include <dpct/dpct.hpp>', '#include <cstdint>')

# Macros
text = text.replace('__dpct_inline__', 'inline')
text = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', text)
text = re.sub(r'const dpct::err0 e = 0;\n\s*ck\(e,\s*"[^"]+"\);\n', '', text)
text = re.sub(r'/\*[\s\S]*?\*/\n', '', text)
text = re.sub(r'void ck\(dpct::err0 e, const char \*what\) \{[\s\S]*?\}\n\n', '', text)
text = re.sub(r'ck\(0, ".*?"\);', '', text)
text = re.sub(r'return DPCT_CHECK_ERROR\(n = dpct::device_count\(\)\) == 0 && n > 0;', r'return true;', text)
text = re.sub(r'(bool ple_available\(\) \{)\n[\s\S]*?\}', r'\1\n    return true;\n}', text)

# Queue and errors
def fix_func(func_name):
    global text
    # matching the function signature spanning multiple lines
    pattern = r'(void ' + func_name + r'\([^)]+\)(?: try)?\s*\{)'
    text = re.sub(pattern, r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)
    
fix_func('ple_flush')
fix_func('ple_write_block')
fix_func('ple_history_advance')
fix_func('ple_block')
fix_func('ple_inference')

text = re.sub(r'\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)\s*->', 'q->', text)
text = re.sub(r'dpct::queue_ptr st = stream \? \(dpct::queue_ptr\)stream : &dpct::get_default_queue\(\);', '', text)
text = re.sub(r'st->', 'q->', text)

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
text = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\s*\*\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)

text = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>(\1), sycl::range<1>(\2))', text)

# Just in case there is some stray range<3> left
text = re.sub(r'sycl::range<3>\(1,\s*1,\s*(.*?)\)', r'sycl::range<1>(\1)', text)
text = re.sub(r'sycl::nd_range<3>', r'sycl::nd_range<1>', text)
text = re.sub(r'sycl::nd_item<3>', r'sycl::nd_item<1>', text)

# memcpys
text = re.sub(r'ck\(DPCT_CHECK_ERROR\(q->memcpy\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)\s*,\s*"[^"]+"\);', r'q->memcpy(\1, \2, \3).wait();', text)
text = re.sub(r'ck\(DPCT_CHECK_ERROR\(q->memcpy\(([^,]+),\s*([^,]+),\s*([^)]+)\)\.wait\(\)\)\s*,\s*"[^"]+"\);', r'q->memcpy(\1, \2, \3).wait();', text)
text = re.sub(r'ck\(DPCT_CHECK_ERROR\(\s*q->memcpy\(([^,]+),\s*([^,]+),\s*([^)]+)\)\s*\)\s*,\s*"[^"]+"\);', r'q->memcpy(\1, \2, \3).wait();', text)
text = re.sub(r'ck\(DPCT_CHECK_ERROR\(\s*q->memcpy\(([^,]+),\s*([^,]+),\s*([^)]+)\)\.wait\(\)\s*\)\s*,\s*"[^"]+"\);', r'q->memcpy(\1, \2, \3).wait();', text)

# Multi-line memcpys inside ple_block which have trailing things because of the regex missing them.
text = re.sub(r'if \(out\.key\)(.|\n)*?\}  // namespace strata::kernels', r'''if (out.key) q->memcpy(out.key, normalized_key, hc_dim * sizeof(float)).wait();
        if (out.value) q->memcpy(out.value, d_value, n_embd * sizeof(float)).wait();
        if (out.gate) q->memcpy(out.gate, d_gate, hc * sizeof(float)).wait();
        if (out.gated) q->memcpy(out.gated, d_gated, hc_dim * sizeof(float)).wait();
        if (out.normalized) q->memcpy(out.normalized, d_norm, hc_dim * sizeof(float)).wait();
        if (out.conv) q->memcpy(out.conv, d_conv, hc_dim * sizeof(float)).wait();
    }
}
}  // namespace strata::kernels''', text)

# clean try blocks around ple_flush and ple_write_block
text = re.sub(r'catch \(sycl::exception const &exc\) \{[\s\S]*?\}\n', '', text)
text = re.sub(r'void ple_flush(.*?) try \{', r'void ple_flush\1 {', text)
text = re.sub(r'void ple_write_block(.*?) try \{', r'void ple_write_block\1 {', text)
text = re.sub(r'void ple_history_advance(.*?) try \{', r'void ple_history_advance\1 {', text)
text = re.sub(r'uint64_t ple_block_scratch_bytes\(\) try \{', r'uint64_t ple_block_scratch_bytes() {', text)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

