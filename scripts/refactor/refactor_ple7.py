import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

text = re.sub(r'void ple_block\(.*?\) \{', r'\g<0>\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', text)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

