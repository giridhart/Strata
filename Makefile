SHELL := /bin/bash
%:
	source /opt/intel/oneapi/setvars.sh --force && $(MAKE) -j8 -C build_sycl $@
