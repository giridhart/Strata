sed -i 's/cudaGraph_t/dpct::experimental::command_graph_ptr/g' src/kernels/sycl/shared_expert_parity.dp.cpp
sed -i 's/cudaGraphExec_t/dpct::experimental::command_graph_exec_ptr/g' src/kernels/sycl/shared_expert_parity.dp.cpp
sed -i 's/check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)/stream->begin_record()/g' src/kernels/sycl/shared_expert_parity.dp.cpp
sed -i 's/check(cudaStreamEndCapture(stream, &graph)/stream->end_record(\&graph)/g' src/kernels/sycl/shared_expert_parity.dp.cpp
sed -i 's/check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0)/executable = graph->instantiate()/g' src/kernels/sycl/shared_expert_parity.dp.cpp
sed -i 's/check(cudaGraphLaunch(executable, stream)/stream->ext_oneapi_graph(*executable)/g' src/kernels/sycl/shared_expert_parity.dp.cpp
