import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# I will just find the block of memcpys at the end of ple_block and replace it.
# The original code looks like this:
"""
    if (out.key) q->memcpy(out.key, normalized_key,
                                                hc_dim * sizeof(float))),
"""

text = re.sub(r'if \(out\.key\)(.|\n)*?\}', r'''if (out.key) q->memcpy(out.key, normalized_key, hc_dim * sizeof(float)).wait();
        if (out.value) q->memcpy(out.value, d_value, n_embd * sizeof(float)).wait();
        if (out.gate) q->memcpy(out.gate, d_gate, hc * sizeof(float)).wait();
        if (out.gated) q->memcpy(out.gated, d_gated, hc_dim * sizeof(float)).wait();
        if (out.normalized) q->memcpy(out.normalized, d_norm, hc_dim * sizeof(float)).wait();
        if (out.conv) q->memcpy(out.conv, d_conv, hc_dim * sizeof(float)).wait();
    }
}''', text)

with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)
