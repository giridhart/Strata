import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# fix missing queues
text = re.sub(r'void ple_flush\(.*?\) \{', r'\g<0>\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', text)
text = re.sub(r'void ple_write_block\(.*?\) \{', r'\g<0>\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', text)
text = re.sub(r'void ple_inference\(.*?\) \{', r'\g<0>\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', text)
text = re.sub(r'q->submit', 'q->submit', text)

text = re.sub(r'ck\(DPCT_CHECK_ERROR\(q->memcpy', r'q->memcpy', text)
text = re.sub(r'ck\(0, ".*?"\);', '', text)

text = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>(\1), sycl::range<1>(\2))', text)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

