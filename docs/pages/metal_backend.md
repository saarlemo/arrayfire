Metal Backend {#metal_backend}
=============

[TOC]

# Supported configuration

The Metal backend provides native GPU execution on macOS. Its initial support
target is macOS 12 or later on Apple silicon (arm64, M1 or newer). The backend
has been validated with a native arm64 Release build on an Apple M4 system.
Intel-based Macs and non-macOS Apple platforms are not currently validated.

The backend enumerates the Metal devices returned by Apple's Metal runtime.
`AF_METAL_DEFAULT_DEVICE` selects the initial ArrayFire device, and an invalid
value falls back to device 0. When the Unified backend is used, Metal has
priority over OpenCL, after CUDA and oneAPI; applications can still select a
backend explicitly with `af::setBackend`.

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

# Interoperability

Include `af/metal.h` to access the Metal device and command queue used by
ArrayFire. The C++ helpers use forward-declared `metal-cpp` types, so including
the header does not itself include `Metal.hpp`.

```cpp
#include <Metal.hpp>
#include <af/metal.h>

MTL::Device* device = afmtl::getDevice();
MTL::CommandQueue* queue = afmtl::getQueue();

af::array input = af::randu(1024, f32);
afmtl::BufferView inputBuffer = afmtl::getBuffer(input);

MTL::CommandBuffer* commands = queue->commandBuffer();
// Encode custom work using inputBuffer.buffer and inputBuffer.offset.
afmtl::submit(commands);
af::sync();
```

`afmtl::getBuffer` returns a move-only `BufferView` and the byte offset needed
when the array is a view into a larger buffer. Managed memory remains locked
against reuse and external buffers are retained until the `BufferView` is
destroyed.

As with CUDA and OpenCL, obtaining an ArrayFire device pointer or Metal buffer
does not wait for queued work to finish. Encode dependent native work on the
queue returned by `afmtl::getQueue`, or call `af::sync()` before accessing a
shared buffer from the host or from an independently managed command queue.

Command buffers passed to `afmtl::submit` are committed and tracked by
ArrayFire. This preserves ordering with ArrayFire operations and makes
`af::sync()` wait for the custom work. Do not commit a command buffer before
passing it to `afmtl::submit`.

`afmtl::array` creates an ArrayFire array backed by a contiguous
`MTL::Buffer`. ArrayFire retains the buffer, so the caller's ownership is
unchanged.

# Current limitations

The backend implements the supported ArrayFire API paths subject to Metal
device resource limits and the following backend-specific constraints:

* `f64` and `c64` arrays are rejected with `AF_ERR_NO_DBL`; use a backend with
  double-precision support when those types are required.
* `f16` is not implemented for every ArrayFire operation, even though Metal
  half-precision array storage is available.
* Two-dimensional morphology masks larger than 19 by 19 are not supported for
  non-boolean input types.
* FFT requires macOS 14 or newer. Matrix inverse requires macOS 13 or newer
  for real single precision and macOS 14 or newer for complex single
  precision.
* A first use can include pipeline compilation time when no compatible Metal
  binary archive exists. Builds without an embedded metallib also compile the
  built-in MSL source when a device module is first initialized.

Array sizes and dispatch dimensions must also fit the limits reported by the
active Metal device. Allocation or pipeline failures are reported through the
normal ArrayFire error mechanism.
