import re
import sys

def refactor_iq_kernels(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Header replacements
    content = content.replace('#include <dpct/dpct.hpp>\n', '')

    # 2. Check function
    content = re.sub(r'void check\(const char\* what\) \{[\s\S]*?\}', 'void check(const char* /*what*/) {\n}', content)

    # 3. Add helper functions
    helpers = '''
inline int dp4a(int a, int b, int c) {
    auto va = sycl::vec<int, 1>(a).as<sycl::vec<int8_t, 4>>().convert<int>();
    auto vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>().convert<int>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

inline unsigned int byte_permute(unsigned int a, unsigned int b, unsigned int s) {
    const uint64_t v = ((uint64_t)b << 32) | a;
    return (unsigned int)((((v >> ((s & 0x7) * 8)) & 0xff)) |
           ((((v >> (((s >> 4) & 0x7) * 8)) & 0xff) << 8)) |
           ((((v >> (((s >> 8) & 0x7) * 8)) & 0xff) << 16)) |
           ((((v >> (((s >> 12) & 0x7) * 8)) & 0xff) << 24)));
}

template <typename VecT, class BinaryOperation>
inline unsigned int vectorized_binary(unsigned int a, unsigned int b, const BinaryOperation binary_op) {
    sycl::vec<unsigned int, 1> v0{a}, v1{b};
    auto v2 = v0.as<VecT>();
    auto v3 = v1.as<VecT>();
    VecT v4;
    for (size_t i = 0; i < v4.size(); ++i) {
        v4[i] = binary_op(v2[i], v3[i]);
    }
    return v4.template as<sycl::vec<unsigned int, 1>>()[0];
}

static sycl::queue default_queue{sycl::property::queue::in_order()};
'''
    # Insert helpers after namespace strata::kernels {\nnamespace {\n
    content = content.replace('namespace strata::kernels {\nnamespace {\n', 'namespace strata::kernels {\nnamespace {\n' + helpers)

    # 4. __dpct_inline__ -> inline
    content = content.replace('__dpct_inline__', 'inline')

    # 5. dpct::byte_level_permute -> byte_permute
    content = content.replace('dpct::byte_level_permute', 'byte_permute')

    # 6. dpct::vectorized_binary -> vectorized_binary
    content = content.replace('dpct::vectorized_binary', 'vectorized_binary')

    # 7. #define ggml_cuda_dp4a(a, b, c) dpct::dp4a((a), (b), (c)) -> dp4a
    content = content.replace('#define ggml_cuda_dp4a(a, b, c) dpct::dp4a((a), (b), (c))', '#define ggml_cuda_dp4a(a, b, c) dp4a((a), (b), (c))')

    # 8. warp_sum
    old_warp_sum = '''__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1096:866: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int o = 16; o > 0; o >>= 1) v += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}'''
    new_warp_sum = '''inline float warp_sum(float v) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += sycl::permute_group_by_xor(sg, v, o);
    }
    return v;
}'''
    # Notice __dpct_inline__ was already replaced by inline earlier
    old_warp_sum2 = '''inline float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1096:866: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int o = 16; o > 0; o >>= 1) v += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}'''
    content = content.replace(old_warp_sum2, new_warp_sum)

    # 9. In quantize_q8_1_kernel:
    old_quant_loop = '''#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = sycl::fmax(
            amax,
            dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), amax, o));
        sum += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), sum, o);
    }'''
    new_quant_loop = '''    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        sum += sycl::permute_group_by_xor(sg, sum, o);
    }'''
    content = content.replace(old_quant_loop, new_quant_loop)

    # 10. Queue ptr & default queue
    content = content.replace('dpct::queue_ptr s', 'sycl::queue *s')
    content = content.replace('dpct::queue_ptr', 'sycl::queue*')
    content = content.replace('&dpct::get_default_queue()', '&default_queue')

    # 11. dpct::has_capability_or_fail
    content = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', content)

    # 12. dpct::dim3 replacements
    # launch_gu(dpct::dim3 grid, ...) -> launch_gu(sycl::range<3> grid, ...)
    content = content.replace('void launch_gu(dpct::dim3 grid,', 'void launch_gu(sycl::range<3> grid,')
    content = content.replace('void launch_down(dpct::dim3 grid,', 'void launch_down(sycl::range<3> grid,')

    # In launch_mmvq:
    # const dpct::dim3 grid((unsigned)((n_out + 3) / 4)), block(32, 4);
    # In DPCT block(32, 4) was sycl::range<3>(1, 4, 32). grid was sycl::range<3>(1, 1, (n_out+3)/4).
    content = content.replace(
        'const dpct::dim3 grid((unsigned)((n_out + 3) / 4)), block(32, 4);',
        'const sycl::range<3> grid(1, 1, (unsigned)((n_out + 3) / 4)), block(1, 4, 32);'
    )

    # In native_expert_grouped:
    # const dpct::dim3 ggu((unsigned)((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned)cap_groups);
    # In DPCT dim3(x, y) was range<3>(1, y, x).
    content = content.replace(
        'const dpct::dim3 ggu((unsigned)((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS),\n                         (unsigned)cap_groups);',
        'const sycl::range<3> ggu(1, (size_t)cap_groups, (size_t)((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS));'
    )
    # const dpct::dim3 gd((unsigned)((L.n_embd + 7) / 8), (unsigned)cap_groups);
    content = content.replace(
        'const dpct::dim3 gd((unsigned)((L.n_embd + 7) / 8), (unsigned)cap_groups);',
        'const sycl::range<3> gd(1, (size_t)cap_groups, (size_t)((L.n_embd + 7) / 8));'
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_iq_kernels('src/kernels/sycl/iq_kernels.dp.cpp')
    print('Refactoring script completed.')
