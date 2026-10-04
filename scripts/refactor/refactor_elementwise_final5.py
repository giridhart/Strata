import re
with open('src/kernels/sycl/elementwise.dp.cpp', 'r') as f:
    text = f.read()

# completely wipe sync_if_needed
text = re.sub(r'void sync_if_needed\(void \*stream, const char \*what\) try \{[\s\S]*?std::exit\(1\);\n\}', '', text)
text = re.sub(r'void sync_if_needed\(void \*stream, const char \*what\) \{[\s\S]*?\}', '', text)
text = re.sub(r'sync_if_needed\(stream, "[^"]+"\);', '', text)

with open('src/kernels/sycl/elementwise.dp.cpp', 'w') as f:
    f.write(text)

