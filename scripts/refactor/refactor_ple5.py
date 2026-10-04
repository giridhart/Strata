import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# Replace all sycl::nd_range<3> -> <1>
text = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\s*\*\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)

text = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>(\1), sycl::range<1>(\2))', text)

# Just in case there is some stray range<3> left
text = re.sub(r'sycl::range<3>\(1,\s*1,\s*(.*?)\)', r'sycl::range<1>(\1)', text)
text = re.sub(r'sycl::nd_range<3>', r'sycl::nd_range<1>', text)
text = re.sub(r'sycl::nd_item<3>', r'sycl::nd_item<1>', text)

# fix memcpy
text = re.sub(r'q->memcpy\(out\.value, d_value, n_embd \* sizeof\(float\)\)\),', r'q->memcpy(out.value, d_value, n_embd * sizeof(float)).wait();', text)
text = re.sub(r'q->memcpy\(out\.gate, d_gate, hc \* sizeof\(float\)\)\),', r'q->memcpy(out.gate, d_gate, hc * sizeof(float)).wait();', text)
text = re.sub(r'q->memcpy\(out\.gated, d_gated, hc_dim \* sizeof\(float\)\)\),', r'q->memcpy(out.gated, d_gated, hc_dim * sizeof(float)).wait();', text)
text = re.sub(r'q->memcpy\(out\.normalized, d_norm, hc_dim \* sizeof\(float\)\)\),', r'q->memcpy(out.normalized, d_norm, hc_dim * sizeof(float)).wait();', text)
text = re.sub(r'q->memcpy\(out\.conv, d_conv, hc_dim \* sizeof\(float\)\)\),', r'q->memcpy(out.conv, d_conv, hc_dim * sizeof(float)).wait();', text)

# also remove remaining `ck(0, "launch");`
text = re.sub(r'ck\(0, "launch"\);', '', text)


with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

