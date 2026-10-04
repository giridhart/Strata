with open('src/kernels/sycl/bf16_gemv.dp.cpp', 'r') as f:
    text = f.read()

text = text.replace('''    if (n_in <= 0 || n_out <= 0) return;
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;''', 
'''    if (n_in <= 0 || n_out <= 0) return;
    static sycl::queue default_queue;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;''')

with open('src/kernels/sycl/bf16_gemv.dp.cpp', 'w') as f:
    f.write(text)
