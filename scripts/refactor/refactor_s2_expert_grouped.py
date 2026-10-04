#!/usr/bin/env python3
import re

with open("src/kernels/sycl/s2_expert_grouped.dp.cpp", "r") as f:
    code = f.read()

# 1. Remove #include <dpct/dpct.hpp>
code = code.replace("#include <dpct/dpct.hpp>\n", "")

# 2. Add pure SYCL math helpers right after namespace strata::kernels { \n namespace {
helpers = """
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

template <typename T>
inline T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta, int logical_sub_group_size = 32) {
    unsigned int id = g.get_local_linear_id();
    unsigned int end_index = (id / logical_sub_group_size + 1) * logical_sub_group_size;
    T result = sycl::shift_group_left(g, x, delta);
    if ((id + delta) >= end_index) {
        result = x;
    }
    return result;
}
"""

code = code.replace("namespace strata::kernels {\nnamespace {", "namespace strata::kernels {\nnamespace {" + helpers)

# 3. Basic replacements
code = code.replace("__dpct_inline__", "inline")
code = code.replace("dpct::dp4a", "dp4a")
code = code.replace("dpct::byte_level_permute", "byte_permute")
code = code.replace("dpct::shift_sub_group_left", "shift_sub_group_left")
code = code.replace("dpct::queue_ptr", "sycl::queue*")

# 4. Remove dpct::err0
code = code.replace("const dpct::err0 e = 0;\n", "")

# 5. In moe_hit_select:
code = code.replace(
    "(stream ? (sycl::queue*)stream : &dpct::get_default_queue())\n        ->parallel_for(",
    "static sycl::queue default_queue{sycl::property::queue::in_order()};\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n    q->parallel_for("
)
code = code.replace(
    'check("moe_hit_select", stream);',
    'if (!stream) default_queue.wait();\n    check("moe_hit_select", stream);'
)

# In moe_hit_select_multi:
code = code.replace(
    "(stream ? (sycl::queue*)stream : &dpct::get_default_queue())->submit([&](sycl::handler &cgh) {",
    "static sycl::queue default_queue{sycl::property::queue::in_order()};\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n    q->submit([&](sycl::handler &cgh) {"
)
code = code.replace(
    'check("moe_hit_select_multi", stream);',
    'if (!stream) default_queue.wait();\n    check("moe_hit_select_multi", stream);'
)
code = code.replace(
    "warp_count_acc_ct1\n                        .get_multi_ptr<sycl::access::decorated::no>()\n                        .get()",
    "&warp_count_acc_ct1[0]"
)

# In moe_group_resident:
code = code.replace(
    "(stream ? (sycl::queue*)stream : &dpct::get_default_queue())->submit([&](sycl::handler &cgh) {",
    "static sycl::queue default_queue{sycl::property::queue::in_order()};\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;\n    q->submit([&](sycl::handler &cgh) {"
)
code = code.replace(
    'check("moe_group_resident", stream);',
    'if (!stream) default_queue.wait();\n    check("moe_group_resident", stream);'
)
code = code.replace(
    "e_s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()\n                        .get(),\n                    first_s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()\n                        .get(),\n                    size_s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()\n                        .get(),\n                    gidx_s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()\n                        .get(),\n                    gstart_s_acc_ct1\n                        .get_multi_ptr<sycl::access::decorated::no>()\n                        .get()",
    "&e_s_acc_ct1[0], &first_s_acc_ct1[0], &size_s_acc_ct1[0], &gidx_s_acc_ct1[0], &gstart_s_acc_ct1[0]"
)

# In moe_hit_add:
old_moe_hit_add = """void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const dpct::dim3 grid(
        (unsigned)((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8),
        (unsigned)cap);
    (stream ? (sycl::queue*)stream : &dpct::get_default_queue())
        ->parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           add_hits_kernel(parts, hit_out, dst, count,
                                           (int)n_embd);
                       });
    check("moe_hit_add", stream);
}"""

new_moe_hit_add = """void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const size_t grid_x = (size_t)std::min((n_embd + 255) / 256, (int64_t)8);
    q->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t)cap, grid_x * 256),
                                     sycl::range<3>(1, 1, 256)),
                   [=](sycl::nd_item<3> item_ct1) {
                       add_hits_kernel(parts, hit_out, dst, count,
                                       (int)n_embd);
                   });
    if (!stream) default_queue.wait();
    check("moe_hit_add", stream);
}"""
code = code.replace(old_moe_hit_add, new_moe_hit_add)

# In moe_grouped_s2:
# Replace grid definitions and ranges
old_moe_grouped = """    {
        const dpct::dim3 grid((unsigned)(2 * FF / GU_ROWS), (unsigned)cap_groups);
        if (fast)
            cs->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:766: '8 * GMAX * NC' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                sycl::local_accessor<int, 1> xs_w_acc_ct1(
                    sycl::range<1>(5120 /*8 * GMAX * NC*/), cgh);
                /*
                DPCT1101:767: 'GMAX * NC' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<sycl::int2, 1> xs_dh_acc_ct1(
                    sycl::range<1>(640 /*GMAX * NC*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gu_grouped_t_kernel(
                            grp_ptr, grp_start, n_groups, ent_tok, x_q8_0,
                            x_scales, gate_up, (int)cap_entries,
                            xs_w_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            xs_dh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        else
            cs->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:768: 'GMAX' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:769: 'H / 4' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<int[8 /*GMAX*/][640 /*H / 4*/], 0>
                    xs_q_acc_ct1(cgh);
                /*
                DPCT1101:770: 'GMAX' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:771: 'H / 32' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[8 /*GMAX*/][80 /*H / 32*/], 0>
                    xs_d_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gu_grouped_kernel(grp_ptr, grp_start, n_groups,
                                              ent_tok, x_q8_0, x_scales,
                                              gate_up, (int)cap_entries,
                                              xs_q_acc_ct1, xs_d_acc_ct1);
                        });
            });
        check("moe_grouped_s2/gu", stream);
    }"""

new_moe_grouped = """    {
        const sycl::range<3> grid_range(1, (size_t)cap_groups, (size_t)(2 * FF / GU_ROWS) * 256);
        if (fast)
            cs->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> xs_w_acc_ct1(sycl::range<1>(5120), cgh);
                sycl::local_accessor<sycl::int2, 1> xs_dh_acc_ct1(sycl::range<1>(640), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_range, sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                        gu_grouped_t_kernel(
                            grp_ptr, grp_start, n_groups, ent_tok, x_q8_0,
                            x_scales, gate_up, (int)cap_entries,
                            &xs_w_acc_ct1[0], &xs_dh_acc_ct1[0]);
                    });
            });
        else
            cs->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int[8][640], 0> xs_q_acc_ct1(cgh);
                sycl::local_accessor<float[8][80], 0> xs_d_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_range, sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            gu_grouped_kernel(grp_ptr, grp_start, n_groups,
                                              ent_tok, x_q8_0, x_scales,
                                              gate_up, (int)cap_entries,
                                              xs_q_acc_ct1, xs_d_acc_ct1);
                        });
            });
        check("moe_grouped_s2/gu", stream);
    }"""
code = code.replace(old_moe_grouped, new_moe_grouped)

old_moe_grouped_down = """    {
        const dpct::dim3 grid((unsigned)(H / D_ROWS), (unsigned)cap_groups);
        const float* hs = x_scales != nullptr ? h_scales : nullptr;
        if (fast)
            cs->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:772: '8 * GMAX * NC' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                sycl::local_accessor<int, 1> hs_w_acc_ct1(
                    sycl::range<1>(1280 /*8 * GMAX * NC*/), cgh);
                /*
                DPCT1101:773: 'GMAX * NC' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<sycl::int2, 1> hs_dh_acc_ct1(
                    sycl::range<1>(160 /*GMAX * NC*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        down_grouped_t_kernel(
                            grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs,
                            out,
                            hs_w_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            hs_dh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        else
            cs->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:774: 'GMAX' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:775: 'FF / 4' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<int[8 /*GMAX*/][160 /*FF / 4*/], 0>
                    hs_q_acc_ct1(cgh);
                /*
                DPCT1101:776: 'GMAX' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:777: 'FF / 32' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[8 /*GMAX*/][20 /*FF / 32*/], 0>
                    hs_d_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            down_grouped_kernel(grp_ptr, grp_start, n_groups,
                                                ent_dst, h_q8_0, hs, out,
                                                hs_q_acc_ct1, hs_d_acc_ct1);
                        });
            });
        check("moe_grouped_s2/down", stream);
    }"""

new_moe_grouped_down = """    {
        const sycl::range<3> grid_range_d(1, (size_t)cap_groups, (size_t)(H / D_ROWS) * 256);
        const float* hs = x_scales != nullptr ? h_scales : nullptr;
        if (fast)
            cs->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> hs_w_acc_ct1(sycl::range<1>(1280), cgh);
                sycl::local_accessor<sycl::int2, 1> hs_dh_acc_ct1(sycl::range<1>(160), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_range_d, sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                        down_grouped_t_kernel(
                            grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs,
                            out, &hs_w_acc_ct1[0], &hs_dh_acc_ct1[0]);
                    });
            });
        else
            cs->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int[8][160], 0> hs_q_acc_ct1(cgh);
                sycl::local_accessor<float[8][20], 0> hs_d_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_range_d, sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            down_grouped_kernel(grp_ptr, grp_start, n_groups,
                                                ent_dst, h_q8_0, hs, out,
                                                hs_q_acc_ct1, hs_d_acc_ct1);
                        });
            });
        check("moe_grouped_s2/down", stream);
    }"""
code = code.replace(old_moe_grouped_down, new_moe_grouped_down)

# Replace all remaining dpct::get_default_queue() occurrences
code = code.replace(
    "sycl::queue* cs = stream ? (sycl::queue*)stream : &dpct::get_default_queue();",
    "static sycl::queue default_queue{sycl::property::queue::in_order()};\n    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;"
)
code = code.replace(
    "sycl::queue* cs = stream ? (dpct::queue_ptr)stream : &dpct::get_default_queue();",
    "static sycl::queue default_queue{sycl::property::queue::in_order()};\n    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;"
)

# Quantize calls: pass cs
code = code.replace(
    "if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);",
    "if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, cs);\n    else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, cs);"
)
code = code.replace(
    "else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);",
    ""
)

code = code.replace(
    "if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);",
    "if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, cs);\n    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, cs);"
)
code = code.replace(
    "else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);",
    ""
)

code = code.replace(
    "if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);",
    "if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, cs);\n    else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, cs);"
)
code = code.replace(
    "else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);",
    ""
)

# In moe_hit_grouped_s2, add wait
code = code.replace(
    'check("moe_hit_grouped_s2/down", stream);',
    'check("moe_hit_grouped_s2/down", stream);\n    if (!stream) default_queue.wait();'
)
code = code.replace(
    'check("moe_hit_grouped_s2_dev/down", stream);',
    'check("moe_hit_grouped_s2_dev/down", stream);\n    if (!stream) default_queue.wait();'
)
code = code.replace(
    'check("moe_hit_grouped_s2_multi/down", stream);',
    'check("moe_hit_grouped_s2_multi/down", stream);\n    if (!stream) default_queue.wait();'
)
code = code.replace(
    'check("moe_grouped_s2/down", stream);',
    'check("moe_grouped_s2/down", stream);\n    if (!stream) default_queue.wait();'
)

# In moe_hit_grouped_s2_cpu_order:
code = code.replace(
    "void moe_hit_grouped_s2_cpu_order(const uint8_t *blob_base,\n                                  const int32_t *slot_index,\n                                  const int32_t *dst_index, int64_t n_hits,\n                                  int64_t blob_bytes, const uint8_t *x_q8_0,\n                                  void *scratch, float *out, void *stream,\n                                  const float *x_scales,\n                                  float *gate_up_trace) try {",
    "void moe_hit_grouped_s2_cpu_order(const uint8_t *blob_base,\n                                  const int32_t *slot_index,\n                                  const int32_t *dst_index, int64_t n_hits,\n                                  int64_t blob_bytes, const uint8_t *x_q8_0,\n                                  void *scratch, float *out, void *stream,\n                                  const float *x_scales,\n                                  float *gate_up_trace) {"
)

old_trace = """    if (gate_up_trace != nullptr &&
        /*
        DPCT1124:668: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(cs->memcpy(
            gate_up_trace, gu, (size_t)n_hits * 2 * FF * sizeof(float))) != 0) {
        std::fprintf(stderr, "cpu_order/gate_up_trace copy failed\\n");
        std::exit(1);
    }"""
new_trace = """    if (gate_up_trace != nullptr) {
        cs->memcpy(gate_up_trace, gu, (size_t)n_hits * 2 * FF * sizeof(float)).wait();
    }"""
code = code.replace(old_trace, new_trace)

code = code.replace(
    'check("cpu_order/down", stream);\n}\ncatch (sycl::exception const &exc) {\n  std::cerr << exc.what() << "Exception caught at file:" << __FILE__\n            << ", line:" << __LINE__ << std::endl;\n  std::exit(1);\n}',
    'check("cpu_order/down", stream);\n    if (!stream) default_queue.wait();\n}'
)

# Remove all DPCT comments
code = re.sub(r'/\*\s*DPCT\d+:[^\*]*\*/', '', code)

with open("src/kernels/sycl/s2_expert_grouped.dp.cpp", "w") as f:
    f.write(code)

print("Updated s2_expert_grouped.dp.cpp written.")
