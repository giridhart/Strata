import re

with open('src/kernels/sycl/elementwise.dp.cpp', 'r') as f:
    text = f.read()

# fix `*q\n        ->parallel_for`
text = re.sub(r'\*q\s*->', r'q->', text)
text = re.sub(r'\*q', r'q', text)

# re-inject q into sample_top_k and mix_moe if missing
if 'q->submit([&](sycl::handler &cgh)' in text and 'sycl::queue* q' not in text.split('mix_moe')[1].split('{')[1].split('q->submit')[0]:
    text = re.sub(r'(void mix_moe\([^)]+\)(?: try)?\s*\{)', r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

if 'q->parallel_for' in text and 'sycl::queue* q' not in text.split('sample_top_k')[1].split('{')[1].split('q->parallel_for')[0]:
    text = re.sub(r'(void sample_top_k\([^)]+\)(?: try)?\s*\{)', r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

with open('src/kernels/sycl/elementwise.dp.cpp', 'w') as f:
    f.write(text)

