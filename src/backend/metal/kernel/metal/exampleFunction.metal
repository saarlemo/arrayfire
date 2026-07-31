/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <metal_stdlib>

using namespace metal;

struct ExampleFunctionParams {
    ulong dims[4];
    long outputStrides[4];
    long leftStrides[4];
    long rightStrides[4];
};

#define DEFINE_EXAMPLE_FUNCTION(NAME, TYPE)                                  \
kernel void NAME(const device TYPE* left [[buffer(0)]],                      \
                 const device TYPE* right [[buffer(1)]],                     \
                 device TYPE* output [[buffer(2)]],                          \
                 constant ExampleFunctionParams& p [[buffer(3)]],            \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];      \
    if (gid >= total) return;                                                \
    ulong q = gid;                                                          \
    long outputIndex = 0;                                                   \
    long leftIndex = 0;                                                     \
    long rightIndex = 0;                                                    \
    for (uint d = 0; d < 4; ++d) {                                         \
        const ulong coordinate = q % p.dims[d];                             \
        q /= p.dims[d];                                                     \
        outputIndex += long(coordinate) * p.outputStrides[d];               \
        leftIndex += long(coordinate) * p.leftStrides[d];                   \
        rightIndex += long(coordinate) * p.rightStrides[d];                 \
    }                                                                       \
    output[outputIndex] = left[leftIndex] + right[rightIndex];              \
}

DEFINE_EXAMPLE_FUNCTION(example_function_float, float)
DEFINE_EXAMPLE_FUNCTION(example_function_cfloat, float2)
DEFINE_EXAMPLE_FUNCTION(example_function_int, int)
DEFINE_EXAMPLE_FUNCTION(example_function_uint, uint)
DEFINE_EXAMPLE_FUNCTION(example_function_char, char)
DEFINE_EXAMPLE_FUNCTION(example_function_uchar, uchar)
