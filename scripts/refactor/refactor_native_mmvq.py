import re

with open("src/kernels/sycl/native_mmvq.dp.cpp", "r") as f:
    content = f.read()

# 1. Remove #include <dpct/dpct.hpp>
content = content.replace("#include <dpct/dpct.hpp>\n", "")

# 2. Add pure SYCL helper functions after headers
helpers = """
static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

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

inline int vsubss4(int a, int b) {
    auto va = sycl::vec<int, 1>(a).as<sycl::vec<int8_t, 4>>();
    auto vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>();
    auto vr = sycl::sub_sat(va, vb);
    return vr.as<sycl::vec<int, 1>>()[0];
}
"""

content = content.replace("namespace strata::kernels {\nnamespace {", "namespace strata::kernels {\nnamespace {" + helpers)

# 3. Replace warp_sum and warp_max
old_warp = """__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        /*
        DPCT1096:836: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        x += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset);
    }
    return x;
}

__dpct_inline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = sycl::fmax(
            x,
            dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset));
    }
    return x;
}"""

new_warp = """inline float warp_sum(float x) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += sycl::permute_group_by_xor(sg, x, offset);
    }
    return x;
}

inline float warp_max(float x) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, offset));
    }
    return x;
}"""

content = content.replace(old_warp, new_warp)

# 4. Replace __dpct_inline__ with inline
content = content.replace("__dpct_inline__", "inline")

# 5. Replace dpct::byte_level_permute with byte_permute
content = content.replace("dpct::byte_level_permute", "byte_permute")

# 6. Replace dpct::dp4a with dp4a
content = content.replace("dpct::dp4a", "dp4a")

# 7. Replace dpct::dim3 threads(WARP, WARPS);
content = content.replace("const dpct::dim3 threads(WARP, WARPS);", "const sycl::range<3> threads(1, WARPS, WARP);")

# 8. Replace vectorized_binary / sub_sat
old_sub1 = """        const int vi =
            dpct::vectorized_binary<sycl::char4>(vil, vih, dpct::sub_sat());"""
new_sub1 = """        const int vi = vsubss4(vil, vih);"""
content = content.replace(old_sub1, new_sub1)

old_sub2 = """        const int vi = dpct::vectorized_binary<sycl::char4>(
            vil | vih, 0x20202020, dpct::sub_sat());"""
new_sub2 = """        const int vi = vsubss4(vil | vih, 0x20202020);"""
content = content.replace(old_sub2, new_sub2)

# 9. Remove dpct::has_capability_or_fail
content = re.sub(r'dpct::has_capability_or_fail\([^;]+;\s*', '', content)

# 10. Replace dpct::queue_ptr
content = content.replace("dpct::queue_ptr", "sycl::queue*")
content = content.replace("const auto s = static_cast<sycl::queue*>(stream);",
                          "sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();")
content = content.replace("((sycl::queue *)(static_cast<sycl::queue*>(stream)))",
                          "(stream ? static_cast<sycl::queue*>(stream) : &default_queue())")

# 11. Remove error check with get_error_string_dummy
old_err = """    const auto error = 0;
    /*
    DPCT1000:422: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error != 0) {
        /*
        DPCT1009:423: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:420: The statement could not be removed.
        */
        throw std::runtime_error(std::string("native MMVQ launch: ") +
                                 dpct::get_error_string_dummy(error));
    }"""
content = content.replace(old_err, "")

with open("src/kernels/sycl/native_mmvq.dp.cpp", "w") as f:
    f.write(content)

print("Refactored native_mmvq.dp.cpp")
