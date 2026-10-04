import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# fix ple_block_available
text = re.sub(r'bool ple_block_available\(\) try \{[\s\S]*?\}', r'bool ple_block_available() {\n    return true;\n}', text)

# fix st -> q
text = re.sub(r'\bst\b', 'q', text)

# fix extra }
text = re.sub(r'\}\n\}  // namespace strata::kernels', r'}  // namespace strata::kernels', text)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

