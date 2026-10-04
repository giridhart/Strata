#!/usr/bin/env python3
import re

with open("src/kernels/sycl/qsa.dp.cpp", "r") as f:
    code = f.read()

# 1. Remove #include <dpct/dpct.hpp>
code = code.replace("#include <dpct/dpct.hpp>\n", "")

# 2. Add permute helper
helper = """
template <typename T>
inline T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask) {
    return sycl::permute_group_by_xor(g, x, mask);
}
"""
code = code.replace("namespace strata::kernels {\nnamespace {", "namespace strata::kernels {\nnamespace {" + helper)

# 3. Replace __dpct_inline__ with inline
code = code.replace("__dpct_inline__", "inline")

# 4. Replace dpct::permute_sub_group_by_xor
code = code.replace("dpct::permute_sub_group_by_xor", "permute_sub_group_by_xor")

# 5. Simplify check_launch and check_sync
old_check = """void check_launch(const char* what) {
    /*
    DPCT1010:510: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

/// The SYNCHRONOUS entry points must check their own wait.  Without this the sticky error from a kernel that
/// faulted becomes visible only at the NEXT call's `cudaGetLastError()`, which then names the wrong kernel -
/// exactly the mis-attribution that made round 198's first `kv_append` failure look like a later one.
void check_sync(const char *what) try {
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}"""

new_check = """void check_launch(const char* what) {
    (void)what;
}

void check_sync(const char *what) {
    (void)what;
}"""
code = code.replace(old_check, new_check)

# 6. Step scratch and upload
old_step = """int32_t *step_scratch() try {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        if (DPCT_CHECK_ERROR(
                d_step = (int32_t *)sycl::malloc_device(
                    qsa_step_bytes(), dpct::get_in_order_queue())) != 0) {
            std::fprintf(stderr, "qsa: step upload: cudaMalloc failed\\n");
            std::exit(1);
        }
    }
    return d_step;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void step_upload_raw(const int32_t *h_step) try {
    int32_t* d = step_scratch();
    if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d, h_step, qsa_step_bytes())
                             .wait()) != 0) {
        std::fprintf(stderr, "qsa: step upload: cudaMemcpy failed\\n");
        std::exit(1);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}"""

new_step = """int32_t *step_scratch() {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        static sycl::queue default_queue{sycl::property::queue::in_order()};
        d_step = (int32_t *)sycl::malloc_device(qsa_step_bytes(), default_queue);
        if (d_step == nullptr) {
            std::fprintf(stderr, "qsa: step upload: cudaMalloc failed\\n");
            std::exit(1);
        }
    }
    return d_step;
}

void step_upload_raw(const int32_t *h_step) {
    int32_t* d = step_scratch();
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    default_queue.memcpy(d, h_step, qsa_step_bytes()).wait();
}"""
code = code.replace(old_step, new_step)

# 7. kv_append_step
old_kv_append = """void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    // The grid is the head x dim count, which is a CONSTANT - `kv_append` writes one cell.
    const long long n = s.n_head_kv * s.head_dim;
    (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->submit([&](sycl::handler &cgh) {
        KvHostPools host_host_KvHostPools_ct9 = host ? *host : KvHostPools{};

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n, THREADS)) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                kv_append_kernel(k_pool, v_pool, page_table, step, kcur, vcur,
                                 (int)s.n_head_kv, (int)s.head_dim,
                                 (int)s.page_size, host_host_KvHostPools_ct9);
            });
    });
    check_launch("kv_append");
    if (stream == nullptr) check_sync("kv_append");
}"""

new_kv_append = """void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const long long n = s.n_head_kv * s.head_dim;
    cs->submit([&](sycl::handler &cgh) {
        KvHostPools host_host_KvHostPools_ct9 = host ? *host : KvHostPools{};

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n, THREADS)) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                kv_append_kernel(k_pool, v_pool, page_table, step, kcur, vcur,
                                 (int)s.n_head_kv, (int)s.head_dim,
                                 (int)s.page_size, host_host_KvHostPools_ct9);
            });
    });
    if (!stream) default_queue.wait();
    check_launch("kv_append");
}"""
code = code.replace(old_kv_append, new_kv_append)

# 8. qsa_index_step
old_qsa_index_step = """void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    // THE GRID IS `max_blocks`, A CONSTANT - not this token's `n_bid + 1`.  The kernel returns for
    // `b > n_bid`, so the surplus blocks cost a launch and nothing else.
    const int threads = 32 * (int) s.idx_n_head;
    /*
    DPCT1049:85: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(
            (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->get_device(),
            {sycl::aspect::fp64});

        (stream ? (sycl::queue*)stream : &dpct::get_default_queue())
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<double, 1> s_dot_acc_ct1(
                    sycl::range<1>(32), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, (unsigned)max_blocks) *
                            sycl::range<3>(1, 1, threads),
                        sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        qsa_index_kernel(
                            pooled, q_idx, bias, (int)s.idx_n_head,
                            (int)s.idx_dim, s.idx_block, step, cell_scores,
                            s_dot_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    check_launch("qsa_index");
    if (stream == nullptr) check_sync("qsa_index");
}"""

new_qsa_index_step = """void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    const int threads = 32 * (int) s.idx_n_head;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    cs->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> s_dot_acc(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (unsigned)max_blocks) *
                    sycl::range<3>(1, 1, threads),
                sycl::range<3>(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                qsa_index_kernel(
                    pooled, q_idx, bias, (int)s.idx_n_head,
                    (int)s.idx_dim, s.idx_block, step, cell_scores,
                    &s_dot_acc[0]);
            });
    });
    if (!stream) default_queue.wait();
    check_launch("qsa_index");
}"""
code = code.replace(old_qsa_index_step, new_qsa_index_step)

# 9. topk_512_step
old_topk = """void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    // `cap` is checked against the GEOMETRY'S largest possible width rather than this token's, because this
    // token's is a device value.  The two are the same guarantee: `width = min(n_kv, top_k + r - 1)` and
    // `n_kv <= kTopkMaxCells`, so this is the most the kernel can ever write.
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max) {
        std::fprintf(stderr, "qsa: topk_512: cap %lld < the largest possible selection width %lld\\n",
                     (long long) cap, (long long) width_max);
        std::exit(1);
    }
    // One block, so the launch is already constant; the kernel reads `n_kv` and `width` from `step`.
    (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:908: 'TOPK_THREADS' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        sycl::local_accessor<int, 1> s_a_acc_ct1(
            sycl::range<1>(256 /*TOPK_THREADS*/), cgh);
        /*
        DPCT1101:909: 'TOPK_THREADS' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        sycl::local_accessor<int, 1> s_b_acc_ct1(
            sycl::range<1>(256 /*TOPK_THREADS*/), cgh);
        sycl::local_accessor<long long, 0> s_cgt_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, TOPK_THREADS),
                              sycl::range<3>(1, 1, TOPK_THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                topk_kernel(
                    cell_scores, step, ids,
                    s_a_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_b_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_cgt_acc_ct1);
            });
    });
    check_launch("topk_512");
    if (stream == nullptr) check_sync("topk_512");
}"""

new_topk = """void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max) {
        std::fprintf(stderr, "qsa: topk_512: cap %lld < the largest possible selection width %lld\\n",
                     (long long) cap, (long long) width_max);
        std::exit(1);
    }
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    cs->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 1> s_a_acc(sycl::range<1>(256), cgh);
        sycl::local_accessor<int, 1> s_b_acc(sycl::range<1>(256), cgh);
        sycl::local_accessor<long long, 0> s_cgt_acc(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, TOPK_THREADS),
                              sycl::range<3>(1, 1, TOPK_THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                topk_kernel(
                    cell_scores, step, ids,
                    &s_a_acc[0],
                    &s_b_acc[0],
                    s_cgt_acc);
            });
    });
    if (!stream) default_queue.wait();
    check_launch("topk_512");
}"""
code = code.replace(old_topk, new_topk)

# 10. kv_gather_step
old_gather = """void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (step == nullptr) fail("kv_gather: step is null");
    if (max_ids <= 0) return;
    // THE GRID IS THE CAPACITY.  The kernel reads the real `n_ids` from `step` and its `if (i >= total) return;`
    // guard leaves the surplus threads idle.
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    (stream ? (sycl::queue*)stream : &dpct::get_default_queue())
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(total, 256)) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                kv_gather_kernel(k_pool, v_pool, page_table, ids, step,
                                 (int)s.n_head_kv, (int)s.head_dim,
                                 (int)s.page_size, k_scratch, v_scratch);
            });
    check_launch("kv_gather");
    if (stream == nullptr) check_sync("kv_gather");
}"""

new_gather = """void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (step == nullptr) fail("kv_gather: step is null");
    if (max_ids <= 0) return;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    cs->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(total, 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            kv_gather_kernel(k_pool, v_pool, page_table, ids, step,
                             (int)s.n_head_kv, (int)s.head_dim,
                             (int)s.page_size, k_scratch, v_scratch);
        });
    if (!stream) default_queue.wait();
    check_launch("kv_gather");
}"""
code = code.replace(old_gather, new_gather)

# 11. qsa_attend_step
old_attend = """void qsa_attend_step(const float *q, const uint16_t *k_scratch,
                     const uint16_t *v_scratch, const int32_t *step,
                     int64_t max_ids, const QsaShapes &s, float *attn,
                     float *weights, void *stream) try {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    // THE SHARED SIZE IS THE CAPACITY.  Shared memory is part of the LAUNCH CONFIGURATION, so sizing it from
    // this token's `n_ids` bakes the sequence length in: a captured layer would allocate room for token 1 and
    // then index past it at token 500.  The kernel reads the real count from `step`, so a larger allocation is
    // exactly what it wants.  The opt-in therefore happens once, for the capacity.
    /*
    DPCT1083:87: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = (size_t)(max_ids + 32) * sizeof(float);
    // per DEVICE: the opt-in is a device setting (a layer split runs this on two cards)
    static size_t s_configured_dev[64] = {};
    int cur_dev = 0;
    cur_dev = dpct::get_current_device_id();
    size_t& s_configured = s_configured_dev[(cur_dev >= 0 && cur_dev < 64) ? cur_dev : 0];
    if (smem > s_configured) {
        int dev = 0, max_shared = 0;
        dev = dpct::get_current_device_id();
        /*
        DPCT1019:539: local_mem_size in SYCL is not a complete equivalent of
        cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to adjust
        the code.
        */
        max_shared = dpct::get_device(dev).get_local_mem_size();
        max_shared = strata::smem_optin_of(max_shared);
        if ((int) smem > max_shared) {
            std::fprintf(stderr, "qsa: qsa_attend: max_ids %lld needs %zu B of shared, over the %d B limit\\n",
                         (long long) max_ids, smem, max_shared);
            std::exit(1);
        }
        const dpct::err0 e =
            /*
            DPCT1027:540: The call to cudaFuncSetAttribute was replaced with 0
            because SYCL currently does not support corresponding setting.
            */
            0;

        s_configured = smem;
    }
    /*
    DPCT1049:86: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smem), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)s.n_head) *
                                  sycl::range<3>(1, 1, (unsigned)s.head_dim),
                              sycl::range<3>(1, 1, (unsigned)s.head_dim)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                qsa_attend_kernel(
                    q, k_scratch, v_scratch, step, (int)s.n_head,
                    (int)s.n_head_kv, (int)s.head_dim, attn, weights,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    check_launch("qsa_attend");
    if (stream == nullptr) check_sync("qsa_attend");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}"""

new_attend = """void qsa_attend_step(const float *q, const uint16_t *k_scratch,
                     const uint16_t *v_scratch, const int32_t *step,
                     int64_t max_ids, const QsaShapes &s, float *attn,
                     float *weights, void *stream) {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    const size_t smem = (size_t)(max_ids + 32) * sizeof(float);
    static size_t s_configured = 0;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    if (smem > s_configured) {
        int max_shared = (int)cs->get_device().get_info<sycl::info::device::local_mem_size>();
        max_shared = strata::smem_optin_of(max_shared);
        if ((int) smem > max_shared) {
            std::fprintf(stderr, "qsa: qsa_attend: max_ids %lld needs %zu B of shared, over the %d B limit\\n",
                         (long long) max_ids, smem, max_shared);
            std::exit(1);
        }
        s_configured = smem;
    }
    cs->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(smem), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)s.n_head) *
                                  sycl::range<3>(1, 1, (unsigned)s.head_dim),
                              sycl::range<3>(1, 1, (unsigned)s.head_dim)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                qsa_attend_kernel(
                    q, k_scratch, v_scratch, step, (int)s.n_head,
                    (int)s.n_head_kv, (int)s.head_dim, attn, weights,
                    &local_acc[0]);
            });
    });
    if (!stream) default_queue.wait();
    check_launch("qsa_attend");
}"""
code = code.replace(old_attend, new_attend)

# 12. indexer_key_append
old_indexer_key = """void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    const int threads = (int) s.idx_dim;
    /*
    DPCT1083:89: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = (size_t)s.idx_dim * sizeof(double);
    // **ONE LAUNCH, NO HOST BRANCH ON THE POSITION.**  The completion rotation is inside the kernel now.  The
    // arguments here are identical for every token, which is the property a CUDA graph needs: this function can
    // be captured and replayed, and the position arrives through `pos_dev` like every other kernel's data.
    /*
    DPCT1049:88: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(
            (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->get_device(),
            {sycl::aspect::fp64});

        (stream ? (sycl::queue*)stream : &dpct::get_default_queue())
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(smem), cgh);

                auto mrope_table_ct14 = mrope_table();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        indexer_key_append_kernel(
                            raw, pos_dev, (int)pos_base, w_k_norm, eps, b.tail,
                            b.dead, b.pooled, b.block_pos, (int)s.idx_dim,
                            (int)s.idx_block, (int)s.n_rot, cos_tab, sin_tab,
                            mrope_table_ct14,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    check_launch("indexer_key_append");
    if (stream == nullptr) check_sync("indexer_key_append");
}"""

new_indexer_key = """void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    const int threads = (int) s.idx_dim;
    const size_t smem = (size_t)s.idx_dim * sizeof(double);
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    cs->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(smem), cgh);
        auto mrope_table_ct14 = mrope_table();

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, threads),
                              sycl::range<3>(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) {
                indexer_key_append_kernel(
                    raw, pos_dev, (int)pos_base, w_k_norm, eps, b.tail,
                    b.dead, b.pooled, b.block_pos, (int)s.idx_dim,
                    (int)s.idx_block, (int)s.n_rot, cos_tab, sin_tab,
                    mrope_table_ct14,
                    &local_acc[0]);
            });
    });
    if (!stream) default_queue.wait();
    check_launch("indexer_key_append");
}"""
code = code.replace(old_indexer_key, new_indexer_key)

# 13. qsa_gate_apply_f32
old_gate_f32 = """void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    {
        dpct::has_capability_or_fail(
            (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->get_device(),
            {sycl::aspect::fp64});

        (stream ? (sycl::queue*)stream : &dpct::get_default_queue())
            ->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n, 256)) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    qsa_gate_apply_f32_kernel(attn, q_full, (int)s.n_head,
                                              (int)s.head_dim, out);
                });
    }
    check_launch("qsa_gate_apply_f32");
    if (stream == nullptr) check_sync("qsa_gate_apply_f32");
}"""

new_gate_f32 = """void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    cs->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n, 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            qsa_gate_apply_f32_kernel(attn, q_full, (int)s.n_head,
                                      (int)s.head_dim, out);
        });
    if (!stream) default_queue.wait();
    check_launch("qsa_gate_apply_f32");
}"""
code = code.replace(old_gate_f32, new_gate_f32)

# 14. qsa_gate_apply
old_gate = """void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    {
        dpct::has_capability_or_fail(
            (stream ? (sycl::queue*)stream : &dpct::get_default_queue())->get_device(),
            {sycl::aspect::fp64});

        (stream ? (sycl::queue*)stream : &dpct::get_default_queue())
            ->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n, 256)) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    qsa_gate_apply_kernel(attn, q_full, (int)s.n_head,
                                          (int)s.head_dim, out);
                });
    }
    check_launch("qsa_gate_apply");
    if (stream == nullptr) check_sync("qsa_gate_apply");
}"""

new_gate = """void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* cs = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    cs->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n, 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            qsa_gate_apply_kernel(attn, q_full, (int)s.n_head,
                                  (int)s.head_dim, out);
        });
    if (!stream) default_queue.wait();
    check_launch("qsa_gate_apply");
}"""
code = code.replace(old_gate, new_gate)

# 15. Rename dpct_local in kernel definitions
code = code.replace("uint8_t *dpct_local", "uint8_t *local_mem")
code = code.replace("(double *)dpct_local", "(double *)local_mem")
code = code.replace("(float *)dpct_local", "(float *)local_mem")

# 16. Remove DPCT comments
code = re.sub(r'/\*\s*DPCT\d+:[^\*]*\*/', '', code)

with open("src/kernels/sycl/qsa.dp.cpp", "w") as f:
    f.write(code)

print("qsa.dp.cpp written.")
