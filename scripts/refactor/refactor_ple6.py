import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# Fix undeclared `q` in ple_inference
text = re.sub(r'void ple_inference\(.*?\) \{', r'\g<0>\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', text)

# Remove extraneous string literals from ck replacements
text = re.sub(r'\)\.wait\(\);\n\s*"[^"]+"\);', r').wait();', text)

# I also need to verify ple_inference's signature and q.
with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)
