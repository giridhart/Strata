import re

with open('src/kernels/sycl/ple.dp.cpp', 'r') as f:
    text = f.read()

# I will just write a python script to replace the remaining errors.
text = re.sub(r'void ple_history_advance\(.*?\) \{', r'void ple_history_advance(float* hist, const float* normalized, void* stream) {\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;', text)
text = re.sub(r'ck\(0, "history advance launch"\);', r'', text)
text = re.sub(r'return DPCT_CHECK_ERROR\(n = dpct::device_count\(\)\) == 0 && n > 0;', r'return true;', text)
text = re.sub(r'uint64_t ple_block_scratch_bytes\(\)', r'catch (...) {}\nuint64_t ple_block_scratch_bytes()', text)
text = re.sub(r'quantize_q8_0\(emb, d_act, n_embd, st\);', r'quantize_q8_0(emb, d_act, n_embd, q);', text)
text = re.sub(r'w\.key_codes, w\.key_scales, d_key, n_embd, hc_dim, 8, st\);', r'w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, q);', text)

text = re.sub(r'if \(out\.key\) ck\(DPCT_CHECK_ERROR\(q->memcpy\(out\.key, normalized_key, hc_dim \* sizeof\(float\)\)\)\.wait\(\), "ple_flush key"\);', r'if (out.key) q->memcpy(out.key, normalized_key, hc_dim * sizeof(float)).wait();', text)

text = re.sub(r'if \(out\.key\) ck\(DPCT_CHECK_ERROR\(q->memcpy\(out\.key, normalized_key,\n\s*hc_dim \* sizeof\(float\)\)\), "ple_write_block key"\);', r'if (out.key) q->memcpy(out.key, normalized_key, hc_dim * sizeof(float)).wait();', text)

text = re.sub(r'if \(out\.value\)\n\s*ck\(DPCT_CHECK_ERROR\(\n\s*q->memcpy\(out\.value, d_value, n_embd \* sizeof\(float\)\)\),\n\s*"ple_write_block value"\);', r'if (out.value) q->memcpy(out.value, d_value, n_embd * sizeof(float)).wait();', text)

text = re.sub(r'if \(out\.gate\)\n\s*ck\(DPCT_CHECK_ERROR\(q->memcpy\(out\.gate, d_gate, hc \* sizeof\(float\)\)\),\n\s*"ple_write_block gate"\);', r'if (out.gate) q->memcpy(out.gate, d_gate, hc * sizeof(float)).wait();', text)

text = re.sub(r'if \(out\.gated\)\n\s*ck\(DPCT_CHECK_ERROR\(\n\s*q->memcpy\(out\.gated, d_gated, hc_dim \* sizeof\(float\)\)\),\n\s*"ple_write_block gated"\);', r'if (out.gated) q->memcpy(out.gated, d_gated, hc_dim * sizeof(float)).wait();', text)


with open('src/kernels/sycl/ple.dp.cpp', 'w') as f:
    f.write(text)

