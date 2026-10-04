#ifdef SYCL_LANGUAGE_VERSION
#pragma once
// Multi-GPU: make `device` current for a scope and restore the caller's device after it.  An object that owns CUDA
// streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the
// caller's thread may be on any device.  A negative device, or the device already current, does nothing.

#include <sycl/sycl.hpp>

namespace strata::core {

inline int& thread_active_device() {
    thread_local int active_device = 0;
    return active_device;
}

struct OnDevice {
    int previous = -1;
    explicit OnDevice(int device) {
        if (device >= 0) {
            int& cur = thread_active_device();
            if (cur != device) {
                previous = cur;
                cur = device;
            }
        }
    }
    ~OnDevice() {
        if (previous >= 0) {
            thread_active_device() = previous;
        }
    }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

}  // namespace strata::core

#else
#pragma once
// Multi-GPU: make `device` current for a scope and restore the caller's device after it.  An object that owns CUDA
// streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the
// caller's thread may be on any device.  A negative device, or the device already current, does nothing.

#include <cuda_runtime.h>

namespace strata::core {

struct OnDevice {
    int previous = -1;
    explicit OnDevice(int device) {
        int cur = 0;
        if (device >= 0 && cudaGetDevice(&cur) == cudaSuccess && cur != device && cudaSetDevice(device) == cudaSuccess)
            previous = cur;
    }
    ~OnDevice() { if (previous >= 0) cudaSetDevice(previous); }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

}  // namespace strata::core

#endif
