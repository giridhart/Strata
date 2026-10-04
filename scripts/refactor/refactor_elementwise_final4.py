import re

with open('src/kernels/sycl/elementwise.dp.cpp', 'r') as f:
    text = f.read()

# fix `*q\n        ->parallel_for`
text = re.sub(r'\*q\s*->', r'q->', text)
text = re.sub(r'\*q', r'q', text)

# re-inject q into copy_rows_from_mapped
text = re.sub(r'(void copy_rows_from_mapped\([^)]+\)(?: try)?\s*\{)', r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

with open('src/kernels/sycl/elementwise.dp.cpp', 'w') as f:
    f.write(text)
