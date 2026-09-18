// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/device_memory.h>

#include <pbrt/gpu/util.h>

#include <cuda_runtime.h>

namespace pbrt {

void *AllocateDeviceBytes(size_t bytes) {
    if (bytes == 0)
        return nullptr;
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void FreeDeviceBytes(void *p) {
    if (p)
        CUDA_CHECK(cudaFree(p));
}

}  // namespace pbrt
