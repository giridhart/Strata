import sys
import re

if len(sys.argv) < 2:
    print("Usage: python3 refactor_generic.py <file.dp.cpp>")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, 'r') as f:
    text = f.read()

# 1. Headers & Macros
text = text.replace('#include <dpct/dpct.hpp>', '#include <cstdint>\n#include <algorithm>')
text = text.replace('__dpct_inline__', 'inline')
text = re.sub(r'#include <cmath>', '#include <cmath>\n\n', text)

# 2. Wipe DPCT specific error handling & sync wrappers
text = re.sub(r'const dpct::err0 e = 0;\n\s*', '', text)
text = re.sub(r'const dpct::err0 e =\n\s*DPCT_CHECK_ERROR\(dpct::get_current_device\(\)\.queues_wait_and_throw\(\)\);\n\s*', '', text)
text = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', text)
text = re.sub(r'void sync_if_needed\(void \*stream, const char \*what\) try \{[\s\S]*?std::exit\(1\);\n\}', '', text)
text = re.sub(r'void sync_if_needed\(void \*stream, const char \*what\) \{[\s\S]*?\}', '', text)
text = re.sub(r'sync_if_needed\(stream, "[^"]+"\);', '', text)
text = re.sub(r'bool check_launch\(const char\* what\) \{[\s\S]*?\}', 'bool check_launch(const char* what) { return true; }', text)
text = re.sub(r'if \(stream == nullptr\) dpct::get_current_device\(\).queues_wait_and_throw\(\);\n', 'if (!stream) default_queue.wait();\n', text)

# Try catches globally (dpct often wraps everything)
catch_block = """catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}"""
text = text.replace(catch_block, '')

# 3. Dynamic Injection of sycl::queue* q into stream functions
funcs_with_stream = re.findall(r'void\s+(\w+)\s*\([^)]*?stream[^)]*?\)\s*(?:try\s*)?\{', text)
funcs_with_stream = list(set(funcs_with_stream))

for f in funcs_with_stream:
    pattern = r'(void\s+' + f + r'\s*\([^)]*?stream[^)]*\))(?:\s*try)?\s*\{'
    text = re.sub(pattern, r'\1 {\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

funcs_without_stream = re.findall(r'void\s+(\w+)\s*\([^)]*?\)\s*(?:try\s*)?\{', text)
funcs_without_stream = list(set(funcs_without_stream))

for f in funcs_without_stream:
    if f not in funcs_with_stream:
        # Check if the function body uses dpct::get...
        pattern = r'(void\s+' + f + r'\s*\([^)]*?\))\s*try\s*\{'
        text = re.sub(pattern, r'\1 {', text, flags=re.DOTALL)

# Queues
text = re.sub(r'\(\s*stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\s*\)->', 'q->', text)
text = re.sub(r'\(\s*stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\s*\)', 'q', text)
text = text.replace('dpct::get_in_order_queue()', '([](){ static sycl::queue q; return q; }())')
text = text.replace('dpct::get_default_queue()', '([](){ static sycl::queue q; return q; }())')
text = text.replace('(*([](){ static sycl::queue q; return q; }()))->', '([](){ static sycl::queue q; return q; }())->')


# 4. Dimensions (3D to 1D)
text = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\s*\*\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)

text = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>(\1), sycl::range<1>(\2))', text)

text = re.sub(r'sycl::range<3>\(1,\s*1,\s*(.*?)\)', r'sycl::range<1>(\1)', text)
text = re.sub(r'sycl::nd_range<3>', r'sycl::nd_range<1>', text)
text = re.sub(r'sycl::nd_item<3>', r'sycl::nd_item<1>', text)

# 5. Items
text = text.replace('sycl::nd_item<3> item_ct1', 'sycl::nd_item<1> item')
text = text.replace('item_ct1.get_group(2)', 'item.get_group(0)')
text = text.replace('item_ct1.get_local_range(2)', 'item.get_local_range(0)')
text = text.replace('item_ct1.get_local_id(2)', 'item.get_local_id(0)')
text = text.replace('item_ct1.get_group_range(2)', 'item.get_group_range(0)')
text = text.replace('item_ct1', 'item')

# 6. Built-ins and Subgroups
text = text.replace('sycl::ext::oneapi::this_work_item::get_nd_item<3>()', 'sycl::ext::oneapi::this_work_item::get_nd_item<1>()')
text = text.replace('dpct::select_from_sub_group', 'sycl::select_from_group')
text = text.replace('dpct::shift_sub_group_left', 'sycl::shift_group_left')
text = text.replace('sycl::ext::oneapi::this_work_item::get_sub_group()', 'item.get_sub_group()')
text = re.sub(r'q\.get_device\(\)\.get_info<sycl::info::device::sub_group_sizes>\(\)', r'q->get_device().get_info<sycl::info::device::sub_group_sizes>()', text)

# 7. Atomics
text = text.replace('dpct::atomic_fetch_add<sycl::access::address_space::generic_space>', 'sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>')

with open(filename, 'w') as f:
    f.write(text)

