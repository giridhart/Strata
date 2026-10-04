import re
import sys

def refactor_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove dpct include
    content = re.sub(r'#include <dpct/dpct.hpp>\n', '', content)
    
    # Remove ck function and calls
    content = re.sub(r'void ck\(dpct::err0 e, const char \*what\) \{\n?\}\n', '', content)
    content = re.sub(r'ck\(DPCT_CHECK_ERROR\((.*?)\),\n?\s*".*?"\);', r'\1;', content)
    content = re.sub(r'ck\(0, ".*?"\);', '', content)
    
    # Remove dpct error checking
    content = re.sub(r'DPCT_CHECK_ERROR\((.*?)\)', r'\1', content)
    
    # Replace default queue fallback
    content = re.sub(r'\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)', r'q', content)
    content = re.sub(r'dpct::queue_ptr st = stream \? \(dpct::queue_ptr\)stream : &dpct::get_default_queue\(\);', 
                     r'static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', content)
    content = re.sub(r'\bst->', 'q->', content)
    
    # Device count
    content = re.sub(r'int n = 0;\n\s*return \(n = dpct::device_count\(\)\) == 0 && n > 0;', 
                     r'return sycl::device::get_devices(sycl::info::device_type::gpu).size() > 0;', content)
                     
    # Remove has_capability_or_fail blocks
    content = re.sub(r'\{\n\s*dpct::has_capability_or_fail[^\}]*\}\n', '', content, flags=re.MULTILINE)
    
    # Remove DPCT comments
    content = re.sub(r'/\*\n\s*DPCT.*?\n\s*\*/\n*', '', content, flags=re.DOTALL)
    
    # Replace 3D range with 1D range
    content = re.sub(r'sycl::nd_range<3>\(\n?\s*sycl::range<3>\(1, 1, (.*?)\) \*\n?\s*sycl::range<3>\(1, 1, (.*?)\),\n?\s*sycl::range<3>\(1, 1, (.*?)\)\)',
                     r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', content)
    
    content = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1, 1, (.*?)\) \*\n?\s*sycl::range<3>\(1, 1, (.*?)\),\n?\s*sycl::range<3>\(1, 1, (.*?)\)\)',
                     r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', content)
                     
    # For nd_item and local_accessor
    content = re.sub(r'sycl::nd_item<3>', r'sycl::nd_item<1>', content)
    content = re.sub(r'\.get_group\(2\)', r'.get_group(0)', content)
    content = re.sub(r'\.get_local_id\(2\)', r'.get_local_id(0)', content)
    content = re.sub(r'\.get_local_range\(2\)', r'.get_local_range(0)', content)
    
    # For subgroup primitives
    content = re.sub(r'dpct::shift_sub_group_left\(\n?\s*sycl::ext::oneapi::this_work_item::get_sub_group\(\), (.*?), (.*?)\)',
                     r'sycl::shift_group_left(item.get_sub_group(), \1, \2)', content)

    content = re.sub(r'dpct::shift_sub_group_left\(\n?\s*item\.get_sub_group\(\), (.*?), (.*?)\)',
                     r'sycl::shift_group_left(item.get_sub_group(), \1, \2)', content)
                     
    # Subgroup get
    content = re.sub(r'sycl::ext::oneapi::this_work_item::get_sub_group\(\)', r'item.get_sub_group()', content)
    
    # nd_item instance name
    content = re.sub(r'item_ct1', r'item', content)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        refactor_file(arg)
