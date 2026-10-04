import re

with open('src/kernels/sycl/quantize_act.dp.cpp', 'r') as f:
    text = f.read()

# remove dpct::err0
text = re.sub(r'const dpct::err0 e = 0;\n', '', text)

# remove wait_and_throw
text = re.sub(r'if \(stream == nullptr\) dpct::get_current_device\(\).queues_wait_and_throw\(\);\n', 'if (!stream) default_queue.wait();\n', text)

# fix missing q injection
def inject_q(func_name):
    global text
    pattern = r'(void\s+' + func_name + r'\s*\([^)]*?stream\)(?:\s*try)?\s*\{)'
    text = re.sub(pattern, r'\1\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n', text, flags=re.DOTALL)

inject_q('quantize_q8_0')

with open('src/kernels/sycl/quantize_act.dp.cpp', 'w') as f:
    f.write(text)
