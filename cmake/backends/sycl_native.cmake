# Native Intel SYCL (Xe2 / Arc B580) test runner and Google Benchmark performance suite.

if(STRATA_ENABLE_SYCL AND STRATA_BUILD_TESTS)
  add_custom_target(check-sycl
    COMMAND ${CMAKE_CTEST_COMMAND} -R sycl_ --output-on-failure
    DEPENDS sycl_native_qsa_score sycl_native_router sycl_native_ple_postops
            sycl_native_bf16 sycl_native_gdn sycl_native_gdn_preprocess
            sycl_native_flash_attn sycl_native_mmvq sycl_model_shapes sycl_fused_gdn
            sycl_dequant_bf16 sycl_verify_kernels sycl_spec_tree sycl_draft_head sycl_graph strata-device
            sycl_native_gr_norm sycl_xe2_subgroup_gemv sycl_xmx_joint_matrix
            sycl_qsa_negative_cell_guard sycl_queue_sync_test sycl_subgroup_divergence_test
            s_gemv_pair_parity quantize_act_images_parity bf16_gemv_pair_parity
            kv_gather_wide_parity qsa_scores_rows_parity prefill_mmq_tab_parity
            gdn_chunk_parity trellis_parity finish_test gemv_rows_test numerics_test
            hc_mix_split_test q8_quant_multi_test sycl_igc_workaround_test
            sycl_level_zero_memory_test prefill_gemm_parity sycl_iq_mmq_parity
            sycl_iq_fused_bench sycl_p2_kernels_bench
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Intel SYCL tests via ctest"
    USES_TERMINAL
  )
  add_custom_target(sycl-test DEPENDS check-sycl)
endif()

# ---- Google Benchmark performance tuning suite ----
if(STRATA_BUILD_PERF)
  find_package(benchmark CONFIG QUIET)
  if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      google_benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(RUN_HAVE_STD_REGEX 0 CACHE INTERNAL "")
    set(RUN_HAVE_STEADY_CLOCK 0 CACHE INTERNAL "")
    set(RUN_HAVE_PTHREAD_AFFINITY 0 CACHE INTERNAL "")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
      add_compile_options(-Wno-c2y-extensions)
    endif()
    FetchContent_MakeAvailable(google_benchmark)
    if(TARGET benchmark)
      set_target_properties(benchmark PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    endif()
    if(TARGET benchmark_main)
      set_target_properties(benchmark_main PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    endif()
  endif()

  if(STRATA_ENABLE_SYCL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf/sycl/perf_xe2_gemv.cpp")
    add_executable(perf_xe2_gemv perf/sycl/perf_xe2_gemv.cpp)
    target_include_directories(perf_xe2_gemv PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(perf_xe2_gemv PRIVATE strata_core strata_kernels benchmark::benchmark)

    set(_strata_perf_targets perf_xe2_gemv)

    foreach(_bm
      perf_xe2_xmx_tiles
      perf_qsa_attention
      perf_moe_accumulate
      perf_dequant_dpas
      perf_s_gemv_pair
      perf_quantize_act
      perf_bf16_gemv_pair
      perf_kv_gather_wide
    )
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf/sycl/${_bm}.cpp")
        add_executable(${_bm} perf/sycl/${_bm}.cpp)
        target_include_directories(${_bm} PRIVATE ${_strata_gpu_include_directories})
        target_link_libraries(${_bm} PRIVATE strata_core strata_kernels benchmark::benchmark)
        list(APPEND _strata_perf_targets ${_bm})
      endif()
    endforeach()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf/sycl/perf_iq1m_lut.cpp")
      add_executable(perf_iq1m_lut perf/sycl/perf_iq1m_lut.cpp)
      target_include_directories(perf_iq1m_lut PRIVATE ${_strata_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
      target_link_libraries(perf_iq1m_lut PRIVATE strata_core strata_kernels benchmark::benchmark)
      list(APPEND _strata_perf_targets perf_iq1m_lut)
    endif()

    # Tier 2 Performance Benchmarks for Intel Xe2 / Arc B580
    foreach(_bm
      perf_fused_gdn
      perf_pcie_overlap
      perf_graph_replay
      perf_kv_decode
      perf_prefill_gemm
      perf_mmq_direct
      perf_gdn_chunk
      perf_trellis
      perf_speculative_verify
    )
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf/sycl/${_bm}.cpp")
        add_executable(${_bm} perf/sycl/${_bm}.cpp)
        target_include_directories(${_bm} PRIVATE ${_strata_gpu_include_directories})
        target_link_libraries(${_bm} PRIVATE strata_core strata_kernels strata_prefill benchmark::benchmark)
        list(APPEND _strata_perf_targets ${_bm})
      endif()
    endforeach()

    # Tier 3 & Tier 4 Performance Benchmarks
    foreach(_bm
      perf_residual_pipeline
      perf_layer_pipeline
      perf_moe_doorbell_stream
      perf_paged_kv_traffic
      perf_lmhead_sampler
      perf_fwht_kv4
      perf_ple_ngram
      perf_embed_overlap
      perf_finish
      perf_gemv_rows
      perf_numerics
      perf_hc_mix_split
      perf_q8_quant_multi
    )
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf/sycl/${_bm}.cpp")
        add_executable(${_bm} perf/sycl/${_bm}.cpp)
        target_include_directories(${_bm} PRIVATE ${_strata_gpu_include_directories})
        target_link_libraries(${_bm} PRIVATE strata_core strata_kernels benchmark::benchmark)
        list(APPEND _strata_perf_targets ${_bm})
      endif()
    endforeach()

    add_custom_target(perf-sycl
      COMMAND perf_xe2_gemv
      DEPENDS ${_strata_perf_targets}
      COMMENT "Running Intel Xe2 Performance Tuning Suite"
      USES_TERMINAL
    )

    add_custom_target(perf-report
      COMMAND ${CMAKE_COMMAND} -E env python3 "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_xe2_perf_benchmarks.py" --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
      DEPENDS ${_strata_perf_targets}
      COMMENT "Running Automated Intel Xe2 / Arc B580 Performance Benchmarks and Generating Markdown Report"
      USES_TERMINAL
    )
  endif()
endif()
