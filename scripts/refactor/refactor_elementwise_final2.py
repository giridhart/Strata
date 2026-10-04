import re

with open('src/kernels/sycl/elementwise.dp.cpp', 'r') as f:
    text = f.read()

# fix (*q)-> to q->
text = text.replace('(*q)->', 'q->')
text = text.replace('(*q)', '*q')

# remove dpct::has_capability_or_fail completely
text = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', text)

# fix missing q in sample_top_k and mix_moe if missed
# sample_top_k takes `void* stream`, mix_moe takes `void* stream`.
def inject_q(func_name):
    global text
    pattern = r'(void ' + func_name + r'\([^)]+\)(?: try)?\s*\{)'
    text = re.sub(pattern, r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

inject_q('sample_top_k')
inject_q('mix_moe')

text = re.sub(r'q->submit', 'q->submit', text)

with open('src/kernels/sycl/elementwise.dp.cpp', 'w') as f:
    f.write(text)

