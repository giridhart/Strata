import re

with open('src/kernels/sycl/bf16_gemv.dp.cpp', 'r') as f:
    text = f.read()

# 1. Strip headers
text = re.sub(r'#include <dpct/dpct.hpp>\n', '', text)

# 2. Fix finish() function and error handling
text = re.sub(r'inline void finish\(void \*stream, const char \*what\) try \{.*?\n\}  // namespace',
              r'}  // namespace', text, flags=re.DOTALL)
text = re.sub(r'\s*finish\(stream, ".*?"\);', '', text)

# 3. Replace queue logic
text = re.sub(r'\(stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\)', r'q', text)

# Insert queue declaration at the start of both bf16_gemv and bf16_gemv_split
queue_decl = "    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;"
text = re.sub(r'(void bf16_gemv\(.*?\) \{)\n(    if \(n_in <= 0 \|\| n_out <= 0\) return;\n)', 
              r'\1\n\2' + queue_decl + '\n', text)
text = re.sub(r'(void bf16_gemv_split\(.*?\) \{)\n(    if \(n_in <= 0 \|\| n_out <= 0\) return;\n)', 
              r'\1\n\2' + queue_decl + '\n', text)

# 4. 3D to 1D ranges
text = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1, 1, (.*?)\) \*\n?\s*sycl::range<3>\(1, 1, (.*?)\),\n?\s*sycl::range<3>\(1, 1, (.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)
text = re.sub(r'sycl::nd_range<3>\(sycl::range<3>\(1, 1, (.*?)\) \*\s*sycl::range<3>\(1, 1, (.*?)\),\s*sycl::range<3>\(1, 1, (.*?)\)\)', 
              r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', text)

# 5. Fix kernel items
text = text.replace('sycl::nd_item<3>', 'sycl::nd_item<1>')
text = text.replace('item_ct1.get_group(2)', 'item.get_group(0)')
text = text.replace('item_ct1.get_local_id(2)', 'item.get_local_id(0)')
text = text.replace('item_ct1.get_local_range(2)', 'item.get_local_range(0)')
text = text.replace('item_ct1', 'item')

# 6. Fix subgroup shifts
text = re.sub(r'dpct::shift_sub_group_left\(\s*sycl::ext::oneapi::this_work_item::get_sub_group\(\),\s*acc,\s*off\)', 
              r'sycl::shift_group_left(item.get_sub_group(), acc, off)', text)

# 7. Fix shared memory in bf16_gemv_split_kernel
text = text.replace('uint8_t *dpct_local', 'float *scratch')
text = re.sub(r'auto scratch = \(float \*\)dpct_local;\n', '', text)

# Update submit block
submit_block_old = """    q->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:924: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>((size_t)threads_per_row * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((grid) * (threads_per_row)), sycl::range<1>(threads_per_row)),
            [=](sycl::nd_item<1> item) {
                bf16_gemv_split_kernel(
                    x, w, y, n_in, n_out, threads_per_row,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });"""

submit_block_new = """    q->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> scratch_acc(
            sycl::range<1>(threads_per_row), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(grid * threads_per_row), sycl::range<1>(threads_per_row)),
            [=](sycl::nd_item<1> item) {
                bf16_gemv_split_kernel(
                    x, w, y, n_in, n_out, threads_per_row,
                    scratch_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });"""

# We'll just replace the entire submit block manually using regex or string replace.
# Let's replace the whole chunk:
pattern = r'q->submit\(\[&\]\(sycl::handler &cgh\) \{.*?\}\);\n    \}\);'
text = re.sub(r'q->submit\(\[&\]\(sycl::handler &cgh\) \{[\s\S]*?\}\);\n    \}\);', submit_block_new, text)

# Wait, `cgh.parallel_for` has a closing `});` and then `q->submit` has `});`.
# The pattern should be `q->submit.*?\}\);`
import re
text = re.sub(r'q->submit\(\[&\]\(sycl::handler &cgh\) \{[\s\S]*?\}\);\n\s*\}\);', submit_block_new, text)

# Also remove DPCT comments
text = re.sub(r'/\*\n\s*DPCT.*?\n\s*\*/\n', '', text, flags=re.DOTALL)
text = re.sub(r'/\*\n\s*DPCT.*?\*/\n', '', text, flags=re.DOTALL)

with open('src/kernels/sycl/bf16_gemv.dp.cpp', 'w') as f:
    f.write(text)

