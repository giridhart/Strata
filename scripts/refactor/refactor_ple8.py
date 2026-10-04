import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# Using re.DOTALL to match across newlines
text = re.sub(r'(void ple_block\([^)]+\)\s*\{)', r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

