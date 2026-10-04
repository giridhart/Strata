import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    content = f.read()

# Replace memcpy block
old_memcpy = """        if (out.key) ck(DPCT_CHECK_ERROR(q->memcpy(out.key, normalized_key,
                                                hc_dim * sizeof(float))),
                    "key");
    if (out.value)
                ck(DPCT_CHECK_ERROR(
               q->memcpy(out.value, d_value, n_embd * sizeof(float))),
           "value");
        if (out.gate)
        q->memcpy(out.gate, d_gate, hc * sizeof(float));
    if (out.gated)
                ck(DPCT_CHECK_ERROR(
               q->memcpy(out.gated, d_gated, hc_dim * sizeof(float))),
           "gated");
    if (out.normalized)
                ck(DPCT_CHECK_ERROR(
               q->memcpy(out.normalized, d_norm, hc_dim * sizeof(float))),
           "norm");
    if (out.conv)
                ck(DPCT_CHECK_ERROR(
               q->memcpy(out.conv, d_conv, hc_dim * sizeof(float))),
           "conv");"""

new_memcpy = """    if (out.key) q->memcpy(out.key, normalized_key, hc_dim * sizeof(float));
    if (out.value) q->memcpy(out.value, d_value, n_embd * sizeof(float));
    if (out.gate) q->memcpy(out.gate, d_gate, hc * sizeof(float));
    if (out.gated) q->memcpy(out.gated, d_gated, hc_dim * sizeof(float));
    if (out.normalized) q->memcpy(out.normalized, d_norm, hc_dim * sizeof(float));
    if (out.conv) q->memcpy(out.conv, d_conv, hc_dim * sizeof(float));"""

if old_memcpy in content:
    content = content.replace(old_memcpy, new_memcpy)

# Remove has_capability
content = re.sub(r'\s*\{\n\s*dpct::has_capability_or_fail\(dpct::get_current_device\(\),\n\s*\{sycl::aspect::fp64\}\);\n', '\n', content)
content = content.replace('});\n        }', '});')

# Fix device count check
content = content.replace('return n = dpct::device_count() == 0 && n > 0;', 'return sycl::device::get_devices(sycl::info::device_type::gpu).size() > 0;')

# Fix barrier
content = content.replace('item.barrier();', 'item.barrier(sycl::access::fence_space::local_space);')

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(content)
