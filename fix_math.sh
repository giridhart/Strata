sed -i 's|const float iscale = scale_num / max;|const float iscale = (float)((double)scale_num / (double)max);|g' src/kernels/sycl/quantize_act_parity.dp.cpp
sed -i 's|const int v = nearest_int_host(iscale \* xb\[j\]);|volatile float prod = iscale * xb[j]; const int v = nearest_int_host(prod);|g' src/kernels/sycl/quantize_act_parity.dp.cpp
