Metal Backend {#metal_backend}
=============

[TOC]

# Supported configuration

The Metal backend provides native GPU execution on macOS. Its initial support
target is macOS 12 or later on Apple silicon (arm64, M1 or newer). The backend
has been validated with a native arm64 Release build on an Apple M4 system.
Intel-based Macs and non-macOS Apple platforms are not currently validated.

The backend uses the system-default Metal device. When both Metal and OpenCL
are installed, the Unified backend selects Metal first. Applications can still
select OpenCL explicitly with af::setBackend(AF_BACKEND_OPENCL).

# Building

Building the Metal backend requires Xcode or the Xcode Command Line Tools, a
C++17 compiler, and Apple's header-only
[metal-cpp](https://developer.apple.com/metal/cpp/) interface. The metal-cpp
headers are not vendored by ArrayFire. Download and extract them from Apple,
then configure ArrayFire with the directory containing `Metal.hpp`:

    cmake -S . -B build \
        -DAF_BUILD_METAL=ON \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DMETALCPP_INCLUDE_DIR=/absolute/path/to/metal-cpp/Metal
    cmake --build build

The single-header metal-cpp distribution is also supported by setting
`METALCPP_INCLUDE_DIR` to the directory containing the generated `Metal.hpp`.

Applications can link directly to `ArrayFire::afmetal`, or link to
`ArrayFire::af` and select AF_BACKEND_METAL through the Unified backend.

# Current limitations

The backend implements the ArrayFire API subject to Metal device resource
limits and the following backend-specific constraints:

* Only the system-default Metal device is currently exposed.
* Pinned host-memory allocation is not supported.
* Metal shaders do not provide native double-precision arithmetic. Operations
  on double and double-complex arrays use host-side or library fallbacks where
  implemented and should not be expected to provide GPU FP64 performance.
* Two-dimensional morphology masks larger than 19 by 19 are not supported for
  non-boolean input types.
* Metal kernels are compiled from embedded source when the backend is first
  initialized, so the first operation can include pipeline compilation time.

Array sizes and dispatch dimensions must also fit the limits reported by the
active Metal device. Allocation or pipeline failures are reported through the
normal ArrayFire error mechanism.
