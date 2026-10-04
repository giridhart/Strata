SHELL := /bin/bash
%:
	source /opt/intel/oneapi/setvars.sh --force && source /opt/intel/oneapi/compiler/2025.3/env/vars.sh && source /opt/intel/oneapi/mkl/2025.3/env/vars.sh && $(MAKE) -j8 -C build_sycl $@
