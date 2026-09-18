// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_DEVICE_MEMORY_H
#define PBRT_GPU_DEVICE_MEMORY_H

#include <pbrt/pbrt.h>

#ifdef PBRT_BUILD_GPU_RENDERER

namespace pbrt {

// Allocates |bytes| of device memory with cudaMalloc(). Implemented in
// device_memory.cpp so that the CUDA runtime headers stay out of this header.
void *AllocateDeviceBytes(size_t bytes);
void FreeDeviceBytes(void *p);

// Owning handle for a device allocation. cudaFree() is host-only, so
// construction and destruction happen on the CPU; kernels receive the raw
// pointer from get().
template <typename T>
class DevicePtr {
  __device__ __host__ explicit DevicePtr(T *ptr) : p(ptr) {}
  private:
    T *p = nullptr;
};

}  // namespace pbrt

#endif  // PBRT_BUILD_GPU_RENDERER

#endif  // PBRT_GPU_DEVICE_MEMORY_H
