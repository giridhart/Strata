import re

with open('src/kernels/sycl/elementwise.dp.cpp', 'r') as f:
    text = f.read()

# Headers
text = text.replace('#include <dpct/dpct.hpp>', '#include <cstdint>\n#include <algorithm>')
text = text.replace('__dpct_inline__', 'inline')

# Delete sync_if_needed completely
text = re.sub(r'void sync_if_needed\(void \*stream, const char \*what\) try \{[\s\S]*?\} catch \(sycl::exception const &exc\) \{\n\s*std::cerr << exc\.what\(\) << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;\n\s*std::exit\(1\);\n\}', '', text)
text = re.sub(r'void sync_if_needed\(void \*stream, const char \*what\) \{[\s\S]*?\}', '', text)
text = re.sub(r'sync_if_needed\(stream, "[^"]+"\);', '', text)

# Delete check_launch completely
text = re.sub(r'bool check_launch\(const char\* what\) \{[\s\S]*?\}', 'bool check_launch(const char* what) { return true; }', text)

text = re.sub(r'#include <cmath>', '#include <cmath>\n\n', text)
text = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', text)

def inject_q(func_name):
    global text
    # \b to match exact function name, and handle multiple lines
    pattern = r'(void\s+' + func_name + r'\s*\([^)]*?stream\)(?:\s*try)?\s*\{)'
    text = re.sub(pattern, r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

funcs = ['embedding_gather', 'gdn_gate', 'scale_inplace', 'add_inplace', 'f32_to_f16_bulk', 'f32_to_bf16_bulk', 'silu_inplace', 'doorbell_wait', 'copy_rows_from_mapped', 'copy_from_mapped', 'doorbell_publish', 'copy_i32_from_mapped', 'doorbell_ring', 'rms_norm_weighted']
for f in funcs:
    inject_q(f)

# stream ? (sycl::queue*)stream : &dpct::get_default_queue() -> q
text = re.sub(r'\(\s*stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\s*\)->', 'q->', text)
text = re.sub(r'\(\s*stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\s*\)', 'q', text)

# Try catches
text = re.sub(r'catch \(sycl::exception const &exc\) \{\n\s*std::cerr << exc\.what\(\) << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;\n\s*std::exit\(1\);\n\}', '', text)

for f in funcs:
    # also remove `try {` to `{` for these functions
    pattern = r'(void\s+' + f + r'\s*\([^)]*?stream\))\s*try\s*\{'
    text = re.sub(pattern, r'\1 {', text, flags=re.DOTALL)

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

# fix hardware checking
text = text.replace('sycl::info::device::sub_group_sizes', 'sycl::info::device::sub_group_sizes') # keep
text = re.sub(r'q\.get_device\(\)\.get_info<sycl::info::device::sub_group_sizes>\(\)', r'q->get_device().get_info<sycl::info::device::sub_group_sizes>()', text)

# fix atomics
text = text.replace('dpct::atomic_fetch_add<sycl::access::address_space::generic_space>', 'sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>')

with open('src/kernels/sycl/elementwise.dp.cpp', 'w') as f:
    f.write(text)

